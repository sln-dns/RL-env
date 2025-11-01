Implement numerically stable versions of the following in tasks/stable_numerics/user_impl.py:
  • log_softmax(x: np.ndarray) -> np.ndarray
  • cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float

Behavioral requirements

General
  • Accept 1D or 2D inputs (vector or batch over the last axis).
  • Work with common dtypes (e.g., float32, float16).
  • No input mutation (do not modify the provided arrays in-place).
  • Preserve dtype (return the same dtype as the input).
  • Handle non-C-contiguous memory (e.g., Fortran order, negative strides).

log_softmax
  • Outputs are finite (no NaN/±Inf) on large-magnitude and near-constant inputs.
  • Shift invariance: log_softmax(x + c) == log_softmax(x) for any scalar c.
  • exp(log_softmax(x)).sum(axis=-1) is ~1 (per sample).

cross_entropy
  • Semantics consistent with -log_softmax(logits)[target] per sample.
  • Outputs are finite and numerically stable.
  • For 1D logits: return a scalar loss.
  • For 2D logits: return either a per-sample vector (batch,) or a scalar mean.

The grader evaluates behavior and result quality, not implementation style.
Note: final grading includes additional hidden checks that validate invariants (shift invariance, normalization to 1, no input mutation, dtype preservation, support for non-contiguous memory) and may enforce stricter output/typing contracts. For 2D cross_entropy, prefer returning a per-sample vector of shape (batch,) with the same dtype as logits; implementations that only return a scalar mean may fail some hidden tests.

Tools
  • write_file(path: str, content: str) — write to tasks/stable_numerics/user_impl.py.
  • run_tests(kind: "visible" | "full") — run visible tests, then the full suite.
  • python_expression(expression: str) — ad-hoc checks.
  • submit_answer(answer: Any) — call after passing the full suite.

Workflow
  1. Implement both functions in user_impl.py.
  2. Iterate with run_tests("visible").
  3. Validate with run_tests("full").
  4. If full tests pass, submit_answer("OK").