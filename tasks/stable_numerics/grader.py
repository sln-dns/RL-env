"""
Grader for numerically stable log_softmax and cross_entropy task.
Contains visible and hidden test cases.
"""
import numpy as np
from typing import Tuple, Dict, Any, List
import importlib.util
import sys
from pathlib import Path

# Reference implementation using float64 for high precision
def _reference_logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp in float64."""
    x_64 = x.astype(np.float64)
    x_max = np.max(x_64, axis=axis, keepdims=True)
    x_shifted = x_64 - x_max
    exp_shifted = np.exp(x_shifted)
    log_sum = np.log(np.sum(exp_shifted, axis=axis, keepdims=True))
    return x_max + log_sum  # Keep dimensions for broadcasting


def _reference_log_softmax(x: np.ndarray) -> np.ndarray:
    """Reference log-softmax using float64."""
    x_64 = x.astype(np.float64)
    log_sum_exp = _reference_logsumexp(x_64, axis=-1)  # Already has keepdims=True
    return (x_64 - log_sum_exp).astype(x.dtype)


def _reference_cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> float:
    """Reference cross-entropy using float64."""
    logits_64 = logits.astype(np.float64)
    log_softmax_64 = _reference_log_softmax(logits_64)
    
    if logits.ndim == 1:
        target = int(target_idx)
        ce = -log_softmax_64[target]
    else:
        targets = np.array(target_idx, dtype=np.int32)
        if isinstance(target_idx, int):
            ce = -log_softmax_64[0, target_idx]
        else:
            # Average over batch
            batch_indices = np.arange(len(targets))
            ce = -np.mean(log_softmax_64[batch_indices, targets])
    
    return float(ce)


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


def _generate_test_cases(rng: np.random.Generator) -> List[TestCase]:
    """Generate test cases (both visible and hidden)."""
    test_cases = []
    
    dims = [2, 8, 64, 512, 2048, 4096]
    batches = [1, 7, 32]
    sigmas = [1.0, 10.0, 100.0]
    
    case_id = 0
    
    # Log-softmax tests
    for dim in dims:
        for batch in batches if dim >= 8 else [1]:
            for sigma in sigmas:
                for shift in [0, 500, -500, 800, -800, 1000, -1000]:
                    for dtype_str in ["float32", "float16"]:
                        dtype = getattr(np, dtype_str)
                        rtol = 1e-3 if dtype == np.float32 else 8e-3
                        atol = 1e-4 if dtype == np.float32 else 1e-3
                        
                        # Generate input
                        if batch == 1:
                            x = rng.normal(shift, sigma, size=(dim,)).astype(dtype)
                        else:
                            x = rng.normal(shift, sigma, size=(batch, dim)).astype(dtype)
                        
                        # Reference
                        expected = _reference_log_softmax(x)
                        
                        # Decide visibility: ~20% visible, 80% hidden
                        is_visible = case_id % 10 < 1
                        
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
                        
                        # Near-constant vectors (edge case)
                        if case_id % 5 == 0:
                            if batch == 1:
                                x_const = np.full((dim,), shift, dtype=dtype) + rng.normal(0, 1e-12, size=(dim,)).astype(dtype)
                            else:
                                x_const = np.full((batch, dim), shift, dtype=dtype) + rng.normal(0, 1e-12, size=(batch, dim)).astype(dtype)
                            
                            expected_const = _reference_log_softmax(x_const)
                            is_visible = case_id % 10 < 1
                            
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
       
        # Additional tests for very long vectors (8192)
    dim = 8192
    for batch in [1, 32]:  # only for batch=1 and batch=32
        for sigma in [10.0, 100.0]:  # Only for extremal sigmas
            for shift in [0, 500]:  # Only for certain shifts
                for dtype_str in ["float32", "float16"]:
                    dtype = getattr(np, dtype_str)
                    rtol = 1e-3 if dtype == np.float32 else 8e-3
                    atol = 1e-4 if dtype == np.float32 else 1e-3
                    
                    if batch == 1:
                        x = rng.normal(shift, sigma, size=(dim,)).astype(dtype)
                    else:
                        x = rng.normal(shift, sigma, size=(batch, dim)).astype(dtype)
                    
                    expected = _reference_log_softmax(x)
                    is_visible = False  # All hidden - these are hard tests
                    
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


    # Cross-entropy tests
    for dim in [8, 64, 512, 2048, 4096]:
        for batch in batches:
            for sigma in sigmas:
                for dtype_str in ["float32", "float16"]:
                    dtype = getattr(np, dtype_str)
                    rtol = 1e-3 if dtype == np.float32 else 8e-3
                    atol = 1e-4 if dtype == np.float32 else 1e-3
                    
                    # 1D case
                    logits_1d = rng.normal(0, sigma, size=(dim,)).astype(dtype)
                    target_1d = rng.integers(0, dim)
                    expected_1d = _reference_cross_entropy(logits_1d, target_1d)

                    is_visible = case_id % 10 < 1
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
                    
                    # 2D case
                    logits_2d = rng.normal(0, sigma, size=(batch, dim)).astype(dtype)
                    targets_2d = rng.integers(0, dim, size=(batch,))
                    expected_2d = _reference_cross_entropy(logits_2d, targets_2d)
                    
                    is_visible = case_id % 10 < 1
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


def _load_user_implementation():
    """Load user implementation from user_impl.py."""
    impl_path = Path(__file__).parent / "user_impl.py"
    
    spec = importlib.util.spec_from_file_location("user_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {impl_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_impl"] = module
    spec.loader.exec_module(module)
    
    return module


def _run_test_case(test_case: TestCase, user_impl) -> Tuple[bool, str]:
    """Run a single test case and return (passed, message)."""
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
            
            # Check for NaN/Inf
            if isinstance(result, np.ndarray):
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    return False, f"Result contains NaN or Inf"
            else:
                if np.isnan(result) or np.isinf(result):
                    return False, f"Result is NaN or Inf"
            
            # Check accuracy
            if test_case.func == "cross_entropy":
                # For CE, normalize to scalar if needed
                if isinstance(result, np.ndarray) and result.ndim > 0:
                    result = float(np.mean(result))
                result = float(result)
                expected = float(test_case.expected_result)
                if not np.isclose(result, expected, rtol=test_case.rtol, atol=test_case.atol):
                    return False, f"Accuracy mismatch: got {result:.6f}, expected {expected:.6f}"
            else:
                # For log_softmax, check array match
                if not np.allclose(result, test_case.expected_result, rtol=test_case.rtol, atol=test_case.atol):
                    max_diff = np.max(np.abs(result - test_case.expected_result))
                    return False, f"Accuracy mismatch: max diff {max_diff:.6f}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Exception: {str(e)}"


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
    failed_tests = []
    
    for test_case in visible_tests:
        passed, msg = _run_test_case(test_case, user_impl)
        if passed:
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
        passed, _ = _run_test_case(test_case, user_impl)
        if passed:
            passed_count += 1
    
    passed = passed_count == total_count
    summary = f"Full tests: {passed_count}/{total_count} passed"
    
    return {
        "passed": passed,
        "summary": summary,
    }


def grade_submission(seed: int = 42) -> bool:
    """Grade submission: run all tests with a fresh seed. Returns True if all pass."""
    result = run_full_tests(seed)
    return result["passed"]

