# Numerically Stable Softmax / Log-Softmax / Cross-Entropy

You are given unstable implementations of softmax, log-softmax, and cross-entropy functions. Your task is to fix them by creating numerically stable versions in `tasks/stable_numerics/user_impl.py`.

## Requirements

You must implement two functions:

1. **`log_softmax(x: np.ndarray) -> np.ndarray`**
   - Input: 1D or 2D array (batch, dim). Supports float32 and float16 dtypes.
   - Requirements:
     - Must not produce NaN/Inf on extreme inputs (|x| up to ~1000, identical or near-identical values, long vectors up to 4096 dimensions).
     - Must match high-precision reference (float64) accuracy: rtol≤1e-3, atol≤1e-4 for float32; rtol≤1e-2, atol≤1e-3 for float16.
     - Multiple stable approaches are acceptable: subtracting x.max(axis=...), using logsumexp, or any equivalent transformation.

2. **`cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float`**
   - Input: logits as 1D or 2D array (batch, dim); target as int (for 1D logits) or 1D array of indices (for 2D logits).
   - Requirements:
     - Must compute CE through log-domain (or equivalently stable method).
     - Same accuracy tolerances and no NaN/Inf as above.
     - For 2D logits, you may return either a scalar (averaged over batch) or an array (per-sample losses). The grader accepts both.

**Important**: The grader does not check code structure or require specific formulas—it only compares numerical properties and result quality.

## Test Cases

The test suite includes:
- Various dimensions: dim ∈ {2, 8, 64, 512, 2048, 4096}, batch sizes ∈ {1, 7, 32}
- Different distributions: normal N(0,σ) with σ ∈ {1, 10, 100} and shifts ±500, near-constant vectors, mixtures of large positive/negative values
- Dtypes: float32 and float16
- Some test cases are visible (via `run_tests("visible")`), others are hidden

## Tools Available

- **`write_file(path: str, content: str)`**: Write code to `tasks/stable_numerics/user_impl.py` (only this path is allowed).
- **`run_tests(kind: "visible" | "full")`**: 
  - `"visible"`: Run only visible test cases; returns a summary (count, passed).
  - `"full"`: Run all tests (visible + hidden) and returns {passed: bool} with a brief summary (hidden inputs are not revealed).
- **`python_expression(expression: str)`**: Quick REPL for testing expressions.
- **`submit_answer(answer: Any)`**: Submit your final answer after successfully passing all tests.

## Workflow

1. Use `write_file` to implement your solutions in `user_impl.py`.
2. Test your implementation with `run_tests("visible")` during development.
3. Once confident, run `run_tests("full")` for final verification.
4. If `run_tests("full")` returns `passed: true`, call `submit_answer("OK")`.

## Hint

Consider using stable transformations to shift values before exponentiation to avoid numerical overflow/underflow. Directly computing `exp(x)` for large x values can lead to numerical instability.

