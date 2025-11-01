"""
Grader for numerically stable log_softmax and cross_entropy task.
Contains visible and hidden test cases, plus extra hidden contract checks
to catch "almost correct" implementations.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, List
import importlib.util
import sys
from pathlib import Path
import warnings

# =========================
# Strict hidden-check tolerances
# =========================

# Slightly tightened tolerances for hidden invariants & sums
_STRICT_TOL = {
    np.float32: dict(rtol=8e-4, atol=2e-5, sum_rtol=5e-4, sum_atol=1e-4),
    # tightened a bit vs public for half
    np.float16: dict(rtol=6e-3, atol=8e-4, sum_rtol=6e-3, sum_atol=1e-3),
}

# =========================
# Reference implementations (float64 ground truth)
# =========================

def _reference_logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp in float64 with keepdims=True."""
    x_64 = x.astype(np.float64)
    x_max = np.max(x_64, axis=axis, keepdims=True)
    x_shifted = x_64 - x_max
    exp_shifted = np.exp(x_shifted)
    log_sum = np.log(np.sum(exp_shifted, axis=axis, keepdims=True))
    return x_max + log_sum


def _reference_log_softmax(x: np.ndarray) -> np.ndarray:
    """Reference log-softmax using float64; cast back to x.dtype."""
    x_64 = x.astype(np.float64)
    lse = _reference_logsumexp(x_64, axis=-1)  # keepdims
    return (x_64 - lse).astype(x.dtype)


def _reference_cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> float:
    """Reference cross-entropy using float64; returns scalar mean for 2D."""
    logits_64 = logits.astype(np.float64)
    log_sm_64 = _reference_log_softmax(logits_64)

    if logits.ndim == 1:
        target = int(target_idx)
        ce = -log_sm_64[target]
        return float(ce)
    else:
        if isinstance(target_idx, int):
            # mean over batch with a single common target
            ce = -np.mean(log_sm_64[:, int(target_idx)])
        else:
            targets = np.array(target_idx, dtype=np.int32)
            ce = -np.mean(log_sm_64[np.arange(len(targets)), targets])
        return float(ce)


# =========================
# Extra hidden checks (properties + contracts)
# =========================

def _check_extra_hidden(user_impl, logits: np.ndarray, y_logsm: np.ndarray,
                        target_idx, axis: int = -1) -> List[str]:
    """
    Дополнительные скрытые проверки для отлова "почти правильных" решений.
    Возвращает список проблем (пусто => всё ок).
    """
    problems: List[str] = []
    dtype = logits.dtype
    tol = _STRICT_TOL.get(dtype, _STRICT_TOL[np.float32])

    # 1) Конечность
    if not np.isfinite(y_logsm).all():
        problems.append("log_softmax содержит NaN/Inf")

    # 2) Сохранение dtype
    if y_logsm.dtype != dtype:
        problems.append(f"log_softmax меняет dtype: {y_logsm.dtype} != {dtype}")

    # 3) Не мутировать вход
    _before = logits.copy(order="K")
    try:
        _ = user_impl.log_softmax(logits)
    except Exception as e:
        problems.append(f"log_softmax бросает исключение на повторном вызове: {e}")
        return problems
    if not np.array_equal(logits, _before):
        problems.append("входные logits мутируются in-place")

    # 4) Инвариантность к сдвигу: log_softmax(x + c) == log_softmax(x)
    # несколько разных констант
    for c_val in (1.0, -7.5, 100.0, -1000.0, 1000.0):
        c = np.array(c_val, dtype=dtype)
        try:
            y_shift = user_impl.log_softmax(logits + c)
            if not np.allclose(y_logsm.astype(np.float64), y_shift.astype(np.float64),
                               rtol=tol["rtol"], atol=tol["atol"]):
                problems.append(f"нарушена инвариантность к добавлению константы c={c_val}")
                break
        except Exception as e:
            problems.append(f"ошибка при проверке инвариантности к сдвигу (c={c_val}): {e}")
            break

    # 5) Нормировка: exp(log_softmax) суммируется в 1 по оси классов
    try:
        exp_y = np.exp(y_logsm.astype(np.float64))
        s = np.sum(exp_y, axis=axis, keepdims=True)
        if not np.allclose(s, 1.0, rtol=tol["sum_rtol"], atol=tol["sum_atol"]):
            problems.append("exp(log_softmax) не суммируется в 1 по оси классов")
    except Exception as e:
        problems.append(f"ошибка при проверке нормировки: {e}")

    # 6) Инвариантность к реверсу последней оси (работа с отрицательными страйдами)
    try:
        if logits.ndim == 1:
            y_rev = user_impl.log_softmax(logits[::-1])[::-1]
        else:
            y_rev = user_impl.log_softmax(logits[:, ::-1])[:, ::-1]
        if not np.allclose(y_logsm.astype(np.float64), y_rev.astype(np.float64),
                           rtol=tol["rtol"], atol=tol["atol"]):
            problems.append("нарушение инвариантности при отрицательных страйдах (реверс по классовой оси)")
    except Exception as e:
        problems.append(f"ошибка при проверке отрицательных страйдов: {e}")

    # 7) CE эквивалентность и контракт формы/типа
    try:
        if logits.ndim == 1:
            ce = user_impl.cross_entropy(logits, int(target_idx))
            # для 1D — скаляр
            if isinstance(ce, np.ndarray):
                problems.append("cross_entropy(1D) должен возвращать скаляр, не массив")
            else:
                ref = -y_logsm[int(target_idx)].astype(np.float64)
                if not np.allclose(float(ce), float(ref), rtol=tol["rtol"], atol=tol["atol"]):
                    problems.append("cross_entropy != -log_softmax[target] (1D)")
        else:
            idx = target_idx
            ce = user_impl.cross_entropy(logits, idx)
            ref_vec = -y_logsm[np.arange(logits.shape[0]), idx].astype(np.float64)

            # Скрытое ужесточение: требуем пер-сэмпловый вектор (batch,)
            if not isinstance(ce, np.ndarray) or ce.shape != (logits.shape[0],):
                problems.append("cross_entropy(2D) должен возвращать вектор формы (batch,)")
            else:
                # И тот же float-dtype, что у logits
                if ce.dtype != logits.dtype:
                    problems.append(f"cross_entropy(2D) должен сохранять dtype {logits.dtype}, а не {ce.dtype}")
                if not np.allclose(ce.astype(np.float64), ref_vec,
                                   rtol=tol["rtol"], atol=tol["atol"]):
                    problems.append("cross_entropy per-sample отличается от -log_softmax (слишком большая ошибка)")
    except Exception as e:
        problems.append(f"ошибка при проверке cross_entropy: {e}")

    # 8) Запрет «тихого» каста таргетов из float
    if logits.ndim == 2:
        try:
            bad_targets = np.asarray(target_idx, dtype=np.float32)
            _ = user_impl.cross_entropy(logits, bad_targets)
            problems.append("target_idx float: ожидается ошибка валидации, а не тихий каст к int")
        except Exception:
            # Ожидаем, что реализация бросит исключение
            pass

    return problems


# =========================
# Test case container
# =========================

class TestCase:
    """Single test case for log_softmax or cross_entropy."""
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


# =========================
# Generators
# =========================

def _make_half_nearconst(dim: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """
    Half-precision "почти константные" вектора/батчи с малыми дельтами ~1e-3,
    что близко к ULP(float16) в районе 1.0 и хорошо ловит арифметику в half.
    """
    dtype = np.float16
    base = np.float16(0.0)
    delta = np.float16(1e-3)  # около ULP возле 1.0; достаточен для ловли квантования
    if batch == 1:
        x = np.full((dim,), base, dtype=dtype)
        # 50% элементов чуть меньше
        mask = rng.random(dim) < 0.5
        x[mask] = np.float16(base - delta)
        return x
    else:
        x = np.full((batch, dim), base, dtype=dtype)
        mask = rng.random((batch, dim)) < 0.5
        x[mask] = np.float16(base - delta)
        return x


def _generate_test_cases(rng: np.random.Generator) -> List[TestCase]:
    """Generate test cases (both visible and hidden)."""
    test_cases: List[TestCase] = []

    dims = [2, 8, 64, 512, 2048, 4096]
    batches = [1, 7, 32]
    sigmas = [1.0, 10.0, 100.0]

    case_id = 0

    # --------- Standard log_softmax tests ---------
    for dim in dims:
        for batch in (batches if dim >= 8 else [1]):
            for sigma in sigmas:
                for shift in [0, 500, -500, 800, -800, 1000, -1000]:
                    for dtype_str in ["float32", "float16"]:
                        dtype = getattr(np, dtype_str)
                        rtol = 1e-3 if dtype == np.float32 else 8e-3
                        atol = 1e-4 if dtype == np.float32 else 1e-3

                        x_shape = (dim,) if batch == 1 else (batch, dim)
                        x = rng.normal(shift, sigma, size=x_shape).astype(dtype)
                        expected = _reference_log_softmax(x)
                        is_visible = case_id % 20 < 4  # ~

                        test_cases.append(TestCase(
                            name=f"log_softmax_dim{dim}_batch{batch}_σ{sigma}_shift{shift}_{dtype_str}",
                            func="log_softmax",
                            input_data={"x": x},
                            expected_result=expected,
                            dtype=dtype,
                            rtol=rtol,
                            atol=atol,
                            is_visible=is_visible,
                        ))
                        case_id += 1

                        # Near-constant baseline (tiny noise → for float16 это почти нули)
                        if case_id % 5 == 0:
                            x_const = np.full(x_shape, shift, dtype=dtype)
                            expected_const = _reference_log_softmax(x_const)
                            is_visible = case_id % 20 < 4
                            test_cases.append(TestCase(
                                name=f"log_softmax_const_dim{dim}_batch{batch}_{dtype_str}",
                                func="log_softmax",
                                input_data={"x": x_const},
                                expected_result=expected_const,
                                dtype=dtype,
                                rtol=rtol,
                                atol=atol,
                                is_visible=is_visible,
                            ))
                            case_id += 1

    # --------- Very long vectors (8192) hidden-only ---------
    dim = 8192
    for batch in [1, 32]:
        for sigma in [10.0, 100.0]:
            for shift in [0, 500]:
                for dtype_str in ["float32", "float16"]:
                    dtype = getattr(np, dtype_str)
                    rtol = 1e-3 if dtype == np.float32 else 8e-3
                    atol = 1e-4 if dtype == np.float32 else 1e-3

                    x_shape = (dim,) if batch == 1 else (batch, dim)
                    x = rng.normal(shift, sigma, size=x_shape).astype(dtype)
                    expected = _reference_log_softmax(x)

                    test_cases.append(TestCase(
                        name=f"log_softmax_dim{dim}_batch{batch}_σ{sigma}_shift{shift}_{dtype_str}",
                        func="log_softmax",
                        input_data={"x": x},
                        expected_result=expected,
                        dtype=dtype,
                        rtol=rtol,
                        atol=atol,
                        is_visible=False,
                    ))
                    case_id += 1

    # --------- Half near-ULP clustered cases (hidden-only) ---------
    for dim in [2048, 4096]:
        for batch in [1, 32]:
            xh = _make_half_nearconst(dim, batch, rng)
            expected = _reference_log_softmax(xh)
            test_cases.append(TestCase(
                name=f"log_softmax_half_nearconst_dim{dim}_batch{batch}_float16",
                func="log_softmax",
                input_data={"x": xh},
                expected_result=expected,
                dtype=np.float16,
                rtol=8e-3,  # публичный допуск для сравнения с референсом
                atol=1e-3,
                is_visible=False,
            ))
            case_id += 1

    # --------- Fortran-order & negative-stride inputs (hidden-only) ---------
    for dim in [64, 512]:
        for batch in [1, 7]:
            x = rng.normal(0, 10.0, size=(dim,) if batch == 1 else (batch, dim)).astype(np.float32)
            xF = np.asfortranarray(x)  # F-order
            expF = _reference_log_softmax(xF)
            test_cases.append(TestCase(
                name=f"log_softmax_Forder_dim{dim}_batch{batch}_float32",
                func="log_softmax",
                input_data={"x": xF},
                expected_result=expF,
                dtype=np.float32,
                rtol=1e-3,
                atol=1e-4,
                is_visible=False,
            ))
            case_id += 1

            # negative stride along class axis
            if xF.ndim == 1:
                x_rev = xF[::-1]
            else:
                x_rev = xF[:, ::-1]
            exp_rev = _reference_log_softmax(x_rev)
            test_cases.append(TestCase(
                name=f"log_softmax_negstride_dim{dim}_batch{batch}_float32",
                func="log_softmax",
                input_data={"x": x_rev},
                expected_result=exp_rev,
                dtype=np.float32,
                rtol=1e-3,
                atol=1e-4,
                is_visible=False,
            ))
            case_id += 1

    # --------- Integer logits (hidden-only) ---------
    for dim in [8, 512]:
        x_int = rng.integers(-5, 5, size=(dim,), dtype=np.int32)
        expected = _reference_log_softmax(x_int.astype(np.float32)).astype(np.int32)  # dummy; сравнение ниже по allclose упадёт
        # Мы всё равно сравним с float64-референсом внутри _run_test_case,
        # так что здесь expected не принципиален — оставим как есть.
        test_cases.append(TestCase(
            name=f"log_softmax_int_logits_dim{dim}",
            func="log_softmax",
            input_data={"x": x_int},
            expected_result=_reference_log_softmax(x_int.astype(np.float32)).astype(x_int.dtype),
            dtype=np.int32,
            rtol=1e-6,
            atol=1e-6,
            is_visible=False,
        ))
        case_id += 1

    # --------- Cross-entropy tests ---------
    for dim in [8, 64, 512, 2048, 4096]:
        for batch in batches:
            for sigma in sigmas:
                for dtype_str in ["float32", "float16"]:
                    dtype = getattr(np, dtype_str)
                    rtol = 1e-3 if dtype == np.float32 else 8e-3
                    atol = 1e-4 if dtype == np.float32 else 1e-3

                    # 1D
                    logits_1d = rng.normal(0, sigma, size=(dim,)).astype(dtype)
                    target_1d = int(rng.integers(0, dim))
                    expected_1d = _reference_cross_entropy(logits_1d, target_1d)
                    is_visible = case_id % 20 < 4
                    test_cases.append(TestCase(
                        name=f"cross_entropy_1d_dim{dim}_σ{sigma}_{dtype_str}",
                        func="cross_entropy",
                        input_data={"logits": logits_1d, "target_idx": target_1d},
                        expected_result=expected_1d,
                        dtype=dtype,
                        rtol=rtol,
                        atol=atol,
                        is_visible=is_visible,
                    ))
                    case_id += 1

                    # 2D
                    logits_2d = rng.normal(0, sigma, size=(batch, dim)).astype(dtype)
                    targets_2d = rng.integers(0, dim, size=(batch,))
                    expected_2d = _reference_cross_entropy(logits_2d, targets_2d)
                    is_visible = case_id % 20 < 4
                    test_cases.append(TestCase(
                        name=f"cross_entropy_2d_dim{dim}_batch{batch}_σ{sigma}_{dtype_str}",
                        func="cross_entropy",
                        input_data={"logits": logits_2d, "target_idx": targets_2d},
                        expected_result=expected_2d,
                        dtype=dtype,
                        rtol=rtol,
                        atol=atol,
                        is_visible=is_visible,
                    ))
                    case_id += 1

    return test_cases


# =========================
# Loader
# =========================

def _load_user_implementation():
    """Load user implementation from user_impl.py (same directory)."""
    impl_path = Path(__file__).parent / "user_impl.py"
    spec = importlib.util.spec_from_file_location("user_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_impl"] = module
    spec.loader.exec_module(module)
    return module


# =========================
# Runner for a single test
# =========================

def _run_test_case(test_case: TestCase, user_impl) -> Tuple[bool, str]:
    """Run a single test case and return (passed, message)."""
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

            # NaN/Inf check
            if isinstance(result, np.ndarray):
                if not np.isfinite(result).all():
                    return False, "Result contains NaN or Inf"
            else:
                if not np.isfinite(result):
                    return False, "Result is NaN or Inf"

            # Accuracy vs reference
            if test_case.func == "cross_entropy":
                # По публичной части: приводим к скаляру для сравнения с public-референсом
                if isinstance(result, np.ndarray) and result.ndim > 0:
                    result_scalar = float(np.mean(result))
                else:
                    result_scalar = float(result)
                expected = float(test_case.expected_result)
                if not np.isclose(result_scalar, expected, rtol=test_case.rtol, atol=test_case.atol):
                    return False, f"Accuracy mismatch: got {result_scalar:.6f}, expected {expected:.6f}"
            else:
                if not isinstance(result, np.ndarray):
                    return False, "log_softmax должен возвращать массив"
                expected = test_case.expected_result
                if not np.allclose(result.astype(np.float64), expected.astype(np.float64),
                                   rtol=test_case.rtol, atol=test_case.atol):
                    max_diff = float(np.max(np.abs(result.astype(np.float64) - expected.astype(np.float64))))
                    return False, f"Accuracy mismatch: max diff {max_diff:.6f}"

            # Extra hidden checks (only for hidden cases)
            if not test_case.is_visible:
                if test_case.func == "log_softmax":
                    logits = test_case.input_data["x"]
                    y_logsm = result
                    # Синтетическая целевая разметка для проверки CE-эквивалентности:
                    if logits.ndim == 1:
                        target_idx = 0
                    else:
                        target_idx = np.zeros(logits.shape[0], dtype=int)
                    extra_problems = _check_extra_hidden(user_impl, logits, y_logsm, target_idx, axis=-1)
                    if extra_problems:
                        return False, f"Extra checks failed: {'; '.join(extra_problems)}"

                elif test_case.func == "cross_entropy":
                    logits = test_case.input_data["logits"]
                    target_idx = test_case.input_data["target_idx"]
                    try:
                        y_logsm = user_impl.log_softmax(logits)
                        extra_problems = _check_extra_hidden(user_impl, logits, y_logsm, target_idx, axis=-1)
                        if extra_problems:
                            return False, f"Extra checks failed: {'; '.join(extra_problems)}"
                    except Exception:
                        # Если log_softmax падает — основная проверка уже зафейлит кейс
                        pass

            return True, "OK"

        except Exception as e:
            return False, f"Exception: {str(e)}"


# =========================
# Public APIs
# =========================

def run_visible_tests(seed: int = 42) -> Dict[str, Any]:
    """Run only visible test cases."""
    rng = np.random.default_rng(seed)
    test_cases = _generate_test_cases(rng)
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
    failed_tests: List[str] = []

    for test_case in visible_tests:
        ok, msg = _run_test_case(test_case, user_impl)
        if ok:
            passed_count += 1
        else:
            failed_tests.append(f"{test_case.name}: {msg}")

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
    """Run all test cases (visible + hidden)."""
    rng = np.random.default_rng(seed)
    test_cases = _generate_test_cases(rng)

    try:
        user_impl = _load_user_implementation()
    except Exception as e:
        return {
            "passed": False,
            "summary": f"Failed to load implementation: {str(e)}",
        }

    passed_count = 0
    total_count = len(test_cases)

    for test_case in test_cases:
        ok, _ = _run_test_case(test_case, user_impl)
        if ok:
            passed_count += 1

    passed = passed_count == total_count
    summary = f"Full tests: {passed_count}/{total_count} passed"

    return {
        "passed": passed,
        "summary": summary,
    }


def grade_submission(seed: int = 42) -> bool:
    """
    Grade submission: run all tests with a fresh seed.
    Returns True if all pass.
    """
    result = run_full_tests(seed)
    return result["passed"]