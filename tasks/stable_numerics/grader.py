"""
Grader for numerically stable log_softmax and cross_entropy task.
Eases overall difficulty via randomized test subsets & mixed difficulty profiles.
"""
from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List, Sequence

import numpy as np


# =========================
# Config & tolerances
# =========================

# Base tolerances
_BASE_TOL = {
    np.float32: dict(rtol=1.5e-3, atol=2e-4, sum_rtol=1e-3, sum_atol=2e-4),
    np.float16: dict(rtol=1.5e-2, atol=3e-3, sum_rtol=2e-2, sum_atol=4e-3),
}

# Strict tolerances used only in hard profile for certain invariance checks
_STRICT_TOL = {
    np.float32: dict(rtol=8e-4, atol=2e-5, sum_rtol=5e-4, sum_atol=1e-4),
    np.float16: dict(rtol=1.0e-2, atol=2e-3, sum_rtol=1.0e-2, sum_atol=2e-3),
}


@dataclass
class DifficultyProfile:
    name: str
    include_extremes: bool          # big dims, large shifts, sigma=100, F/neg strides, strict checks
    n_logsm_cases: int              # how many log_softmax cases to sample
    n_ce_cases: int                 # how many CE cases to sample
    visible_frac: float             # fraction of cases marked as visible
    use_strict_hidden_checks: bool  # apply strict extra checks
    allow_forder_and_neg_strides: bool  # include F-order/negative-stride inputs in hidden checks


def _choose_profile_from_env() -> DifficultyProfile:
    mode = os.getenv("RL_ENV_DIFFICULTY", "mixed").lower().strip()
    rng = np.random.default_rng()

    if mode == "easy":
        return DifficultyProfile(
            name="easy",
            include_extremes=False,
            n_logsm_cases=6,
            n_ce_cases=4,
            visible_frac=0.35,
            use_strict_hidden_checks=False,
            allow_forder_and_neg_strides=False,
        )
    if mode == "hard":
        return DifficultyProfile(
            name="hard",
            include_extremes=True,
            n_logsm_cases=10,
            n_ce_cases=6,
            visible_frac=0.15,
            use_strict_hidden_checks=True,
            allow_forder_and_neg_strides=True,
        )

    #mixed: 70% chance to enable hard profile; otherwise close to easy
    
    if rng.random() < 0.70:
        return DifficultyProfile(
            name="mixed-hard",
            include_extremes=True,
            n_logsm_cases=9,
            n_ce_cases=6,
            visible_frac=0.25,
            use_strict_hidden_checks=True,
            allow_forder_and_neg_strides=True,
        )
    else:
        return DifficultyProfile(
            name="mixed-easy",
            include_extremes=False,
            n_logsm_cases=7,
            n_ce_cases=5,
            visible_frac=0.30,
            use_strict_hidden_checks=False,
            allow_forder_and_neg_strides=False,
        )


# =========================
# Reference implementations
# =========================

def _reference_logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_64 = x.astype(np.float64)
    x_max = np.max(x_64, axis=axis, keepdims=True)
    x_shifted = x_64 - x_max
    exp_shifted = np.exp(x_shifted)
    log_sum = np.log(np.sum(exp_shifted, axis=axis, keepdims=True))
    return x_max + log_sum


def _reference_log_softmax(x: np.ndarray) -> np.ndarray:
    x_64 = x.astype(np.float64)
    log_sum_exp = _reference_logsumexp(x_64, axis=-1)
    return (x_64 - log_sum_exp).astype(x.dtype)


def _reference_cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> float:
    logits_64 = logits.astype(np.float64)
    log_softmax_64 = _reference_log_softmax(logits_64)

    if logits.ndim == 1:
        target = int(target_idx)
        ce = -log_softmax_64[target]
    else:
        targets = np.array(target_idx, dtype=np.int32)
        batch_indices = np.arange(len(targets))
        ce = -np.mean(log_softmax_64[batch_indices, targets])

    return float(ce)


# =========================
# Extra hidden checks
# =========================

def _check_extra_hidden(
    user_impl,
    logits: np.ndarray,
    y_logsm: np.ndarray,
    target_idx,
    axis: int,
    strict: bool,
    allow_forder_and_neg: bool,
) -> List[str]:
    """
    Additional checks to catch “almost correct” solutions.
    Returns a list of problems (empty => everything is ok).
    """
    problems: List[str] = []
    dtype = logits.dtype
    tol_base = _BASE_TOL.get(dtype, _BASE_TOL[np.float32])
    tol = _STRICT_TOL.get(dtype, _STRICT_TOL[np.float32]) if strict else tol_base

    # 1) finite
    if not np.isfinite(y_logsm).all():
        problems.append("llog_softmax contains non-final values")

    # 2) dtype preserve (в строгом режиме обязательно; иначе — soft)
    if y_logsm.dtype != dtype:
        if strict:
            problems.append(f"log_softmax changes dtype: {y_logsm.dtype} != {dtype}")
        else:
            # мягко: позволим fp16->fp32, но не иные преобразования
            if not (dtype == np.float16 and y_logsm.dtype == np.float32):
                problems.append(f"log_softmax changes dtype (soft check): {y_logsm.dtype} != {dtype}")

    # 3) no-mutation
    _before = logits.copy(order="K")
    _ = user_impl.log_softmax(logits)
    if not np.array_equal(logits, _before):
        problems.append("input logits are mutated in-place")

    # 4) shift invariance: константа меньше для fp16
    c_value = 20.0 if dtype == np.float16 else 100.0
    c = np.array(c_value, dtype=dtype)
    try:
        y_shift = user_impl.log_softmax(logits + c)
        if not np.allclose(y_logsm.astype(np.float64), y_shift.astype(np.float64),
                           rtol=tol["rtol"], atol=tol["atol"]):
            problems.append("invariance to adding a constant is broken")
    except Exception as e:
        problems.append(f"error when checking invariance: {e}")

    # 5) normalization
    try:
        s = np.sum(np.exp(y_logsm.astype(np.float64)), axis=axis, keepdims=True)
        if not np.allclose(s, 1.0, rtol=tol["sum_rtol"], atol=tol["sum_atol"]):
            problems.append("exp(log_softmax) does not sum to 1 on the class axis")
    except Exception as e:
        problems.append(f"error when checking normalization: {e}")

    # 6) CE equivalence
    try:
        if logits.ndim == 1:
            ce = user_impl.cross_entropy(logits, int(target_idx))
            if isinstance(ce, np.ndarray):
                problems.append("cross_entropy(1D) should return a scalar")
            else:
                ref = -y_logsm[int(target_idx)].astype(np.float64)
                if not np.allclose(float(ce), float(ref), rtol=tol["rtol"], atol=tol["atol"]):
                    problems.append("cross_entropy != -log_softmax[target] (1D)")
        else:
            idx = target_idx
            ce = user_impl.cross_entropy(logits, idx)
            ref_vec = -y_logsm[np.arange(logits.shape[0]), idx].astype(np.float64)

            if isinstance(ce, np.ndarray):
                if ce.shape != (logits.shape[0],):
                    problems.append(f"cross_entropy(2D) wrong shape: {ce.shape}")
                else:
                    if not np.allclose(ce.astype(np.float64), ref_vec,
                                       rtol=tol["rtol"], atol=tol["atol"]):
                        problems.append("cross_entropy per-sample deviates from -log_softmax")
            else:
                ref_mean = float(np.mean(ref_vec))
                if not np.allclose(float(ce), ref_mean, rtol=tol["rtol"], atol=tol["atol"]):
                    problems.append("cross_entropy scalar != mean per-sample")
    except Exception as e:
        problems.append(f"error when checking cross_entropy: {e}")

    # 7) F-order / negative strides (только если разрешено)
    if allow_forder_and_neg and logits.ndim >= 1:
        try:
            # Fortran-order copy
            lf = np.asfortranarray(logits)
            y_f = user_impl.log_softmax(lf)
            if not np.isfinite(y_f).all():
                problems.append("log_softmax(F-order) returned non-finite values")

            # Negative strides via slicing
            ln = logits[..., ::-1]
            y_n = user_impl.log_softmax(ln)
            if not np.isfinite(y_n).all():
                problems.append("log_softmax(negative-strides) returned non-finite values")
        except Exception as e:
            problems.append(f"error F-order/neg-strides: {e}")

    return problems


# =========================
# Test case generation
# =========================

class TestCase:
    def __init__(
        self,
        name: str,
        func: str,  # "log_softmax" or "cross_entropy"
        input_data: Dict[str, Any],
        expected_result: np.ndarray | float,
        dtype: np.dtype,
        rtol: float,
        atol: float,
        is_visible: bool = False,
    ):
        self.name = name
        self.func = func
        self.input_data = input_data
        self.expected_result = expected_result
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol
        self.is_visible = is_visible


def _sample(seq: Sequence, rng: np.random.Generator, k: int) -> List:
    if k <= 0:
        return []
    idx = rng.choice(len(seq), size=min(k, len(seq)), replace=False)
    return [seq[i] for i in np.atleast_1d(idx)]


def _generate_test_cases(rng: np.random.Generator, profile: DifficultyProfile) -> List[TestCase]:
    test_cases: List[TestCase] = []

    # Base pools
    dims_base = [8, 64, 512]
    batches_base = [1, 7, 32]
    sigmas_base = [1.0, 10.0]
    shifts_base = [0, 500, -500]
    dtypes = [np.float32, np.float16]

    # Extreme pools (used only if include_extremes=True)
    dims_ext = [2048, 4096]
    batches_ext = [1, 32]
    sigmas_ext = [10.0, 100.0]
    shifts_ext = [0, 800, -800, 1000, -1000]

    # How many to draw
    n_logsm = profile.n_logsm_cases
    n_ce = profile.n_ce_cases

    # Build candidate tuples (dim, batch, sigma, shift, dtype)
    candidates: List[tuple] = []

    def add_pool(dims, batches, sigmas, shifts):
        for dim in dims:
            for batch in batches if dim >= 8 else [1]:
                for sigma in sigmas:
                    for shift in shifts:
                        for dt in dtypes:
                            candidates.append((dim, batch, sigma, shift, dt))

    add_pool(dims_base, batches_base, sigmas_base, shifts_base)
    if profile.include_extremes:
        add_pool(dims_ext, batches_ext, sigmas_ext, shifts_ext)

    # Sample LS cases
    ls_cases = _sample(candidates, rng, n_logsm)
    # Sample CE cases (reuse candidates but different draw)
    ce_cases = _sample(candidates, rng, n_ce)

    # Build TestCase objects
    visible_quota_ls = max(1, int(round(profile.visible_frac * len(ls_cases))))
    visible_quota_ce = max(1, int(round(profile.visible_frac * len(ce_cases))))
    vis_count_ls = 0
    vis_count_ce = 0

    # log_softmax
    for i, (dim, batch, sigma, shift, dt) in enumerate(ls_cases):
        rtol = _BASE_TOL[dt]["rtol"]
        atol = _BASE_TOL[dt]["atol"]

        if batch == 1:
            x = rng.normal(shift, sigma, size=(dim,)).astype(dt)
        else:
            x = rng.normal(shift, sigma, size=(batch, dim)).astype(dt)

        # Occasionally insert near-constant vectors (edge)
        if rng.random() < (0.20 if profile.include_extremes else 0.10):
            if batch == 1:
                x = np.full((dim,), shift, dtype=dt) + rng.normal(0, 1e-12, size=(dim,)).astype(dt)
            else:
                x = np.full((batch, dim), shift, dtype=dt) + rng.normal(0, 1e-12, size=(batch, dim)).astype(dt)

        expected = _reference_log_softmax(x)
        is_visible = vis_count_ls < visible_quota_ls
        vis_count_ls += 1 if is_visible else 0

        test_cases.append(TestCase(
            name=f"log_softmax_dim{dim}_batch{batch}_σ{sigma}_shift{shift}_{dt.__name__}",
            func="log_softmax",
            input_data={"x": x},
            expected_result=expected,
            dtype=dt,
            rtol=rtol,
            atol=atol,
            is_visible=is_visible,
        ))

    # cross_entropy
    for i, (dim, batch, sigma, shift, dt) in enumerate(ce_cases):
        rtol = _BASE_TOL[dt]["rtol"]
        atol = _BASE_TOL[dt]["atol"]

        # 1D
        logits_1d = rng.normal(0, sigma, size=(dim,)).astype(dt)
        target_1d = int(rng.integers(0, dim))
        expected_1d = _reference_cross_entropy(logits_1d, target_1d)
        is_visible = vis_count_ce < visible_quota_ce
        vis_count_ce += 1 if is_visible else 0

        test_cases.append(TestCase(
            name=f"cross_entropy_1d_dim{dim}_σ{sigma}_{dt.__name__}",
            func="cross_entropy",
            input_data={"logits": logits_1d, "target_idx": target_1d},
            expected_result=expected_1d,
            dtype=dt,
            rtol=rtol,
            atol=atol,
            is_visible=is_visible,
        ))

        # 2D
        if batch == 1:
            batch_use = 7  # to make sense of 2D
        else:
            batch_use = batch

        logits_2d = rng.normal(0, sigma, size=(batch_use, dim)).astype(dt)
        targets_2d = rng.integers(0, dim, size=(batch_use,))
        expected_2d = _reference_cross_entropy(logits_2d, targets_2d)
        is_visible2 = (vis_count_ce < visible_quota_ce)
        vis_count_ce += 1 if is_visible2 else 0

        test_cases.append(TestCase(
            name=f"cross_entropy_2d_dim{dim}_batch{batch_use}_σ{sigma}_{dt.__name__}",
            func="cross_entropy",
            input_data={"logits": logits_2d, "target_idx": targets_2d},
            expected_result=expected_2d,
            dtype=dt,
            rtol=rtol,
            atol=atol,
            is_visible=is_visible2,
        ))

    return test_cases


# =========================
# Loader & runner
# =========================

def _load_user_implementation():
    impl_path = Path(__file__).parent / "user_impl.py"
    spec = importlib.util.spec_from_file_location("user_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_impl"] = module
    spec.loader.exec_module(module)
    return module


def _run_test_case(test_case: TestCase, user_impl, profile: DifficultyProfile) -> Tuple[bool, str]:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            if test_case.func == "log_softmax":
                func = user_impl.log_softmax
                result = func(**test_case.input_data)
            elif test_case.func == "cross_entropy":
                func = user_impl.cross_entropy
                result = func(**test_case.input_data)
            else:
                return False, f"Unknown function: {test_case.func}"

            # NaN/Inf
            if isinstance(result, np.ndarray):
                if not np.isfinite(result).all():
                    return False, "Result contains NaN/Inf"
            else:
                if not np.isfinite(result):
                    return False, "Result is NaN/Inf"

            # Accuracy
            if test_case.func == "cross_entropy":
                if isinstance(result, np.ndarray) and result.ndim > 0:
                    result = float(np.mean(result))
                result = float(result)
                expected = float(test_case.expected_result)
                if not np.isclose(result, expected, rtol=test_case.rtol, atol=test_case.atol):
                    return False, f"Accuracy mismatch: got {result:.6f}, expected {expected:.6f}"
            else:
                if not np.allclose(result, test_case.expected_result, rtol=test_case.rtol, atol=test_case.atol):
                    max_diff = float(np.max(np.abs(result.astype(np.float64) - test_case.expected_result.astype(np.float64))))
                    return False, f"Accuracy mismatch: max diff {max_diff:.6f}"

            # Extra hidden checks — apply only to hidden
            if not test_case.is_visible:
                if test_case.func == "log_softmax":
                    logits = test_case.input_data["x"]
                    y_logsm = result
                    if logits.ndim == 1:
                        target_idx = 0
                    else:
                        target_idx = np.zeros(logits.shape[0], dtype=int)
                    extra = _check_extra_hidden(
                        user_impl,
                        logits,
                        y_logsm,
                        target_idx,
                        axis=-1,
                        strict=profile.use_strict_hidden_checks,
                        allow_forder_and_neg=profile.allow_forder_and_neg_strides,
                    )
                    if extra and profile.use_strict_hidden_checks:
                        return False, f"Extra checks failed: {'; '.join(extra)}"
                    # In non-strict profile — only critical errors will be caught earlier; here we softly ignore extra
                else:
                    # CE: call log_softmax and run the same checks
                    logits = test_case.input_data["logits"]
                    try:
                        y_logsm = user_impl.log_softmax(logits)
                        extra = _check_extra_hidden(
                            user_impl,
                            logits,
                            y_logsm,
                            test_case.input_data["target_idx"],
                            axis=-1,
                            strict=profile.use_strict_hidden_checks,
                            allow_forder_and_neg=profile.allow_forder_and_neg_strides,
                        )
                        if extra and profile.use_strict_hidden_checks:
                            return False, f"Extra checks failed: {'; '.join(extra)}"
                    except Exception:
                        pass

            return True, "OK"

        except Exception as e:
            return False, f"Exception: {str(e)}"


def run_visible_tests(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    profile = _choose_profile_from_env()
    test_cases = _generate_test_cases(rng, profile)
    visible_tests = [tc for tc in test_cases if tc.is_visible]

    try:
        user_impl = _load_user_implementation()
    except Exception as e:
        return {
            "passed": False,
            "summary": f"Failed to load implementation: {str(e)}",
            "total": len(visible_tests),
            "passed_count": 0,
        }

    passed_count = 0
    failed_tests = []

    for tc in visible_tests:
        passed, msg = _run_test_case(tc, user_impl, profile)
        if passed:
            passed_count += 1
        else:
            failed_tests.append(f"{tc.name}: {msg}")

    passed = passed_count == len(visible_tests)
    summary = f"Visible tests: {passed_count}/{len(visible_tests)} passed"
    if failed_tests and len(failed_tests) <= 5:
        summary += f". Failures: {', '.join(failed_tests[:5])}"

    return {
        "passed": passed,
        "summary": summary,
        "total": len(visible_tests),
        "passed_count": passed_count,
    }


def run_full_tests(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    profile = _choose_profile_from_env()
    test_cases = _generate_test_cases(rng, profile)

    try:
        user_impl = _load_user_implementation()
    except Exception as e:
        return {"passed": False, "summary": f"Failed to load implementation: {str(e)}"}

    passed_count = 0
    total_count = len(test_cases)

    for tc in test_cases:
        ok, _ = _run_test_case(tc, user_impl, profile)
        if ok:
            passed_count += 1

    passed = passed_count == total_count
    summary = f"Full tests: {passed_count}/{total_count} passed (profile={profile.name})"

    return {"passed": passed, "summary": summary}


def grade_submission(seed: int = 42) -> bool:
    result = run_full_tests(seed)
    return result["passed"]