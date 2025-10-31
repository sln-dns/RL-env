# Numerically Stable Softmax / Log-Softmax / Cross-Entropy Task

## Overview

This task requires implementing numerically stable versions of `log_softmax` and `cross_entropy` functions that work correctly on extreme inputs and with different floating-point precisions (float32 and float16).

## Problem

Naive implementations of softmax and cross-entropy can produce NaN or Inf values when:
- Input values are very large (|x| up to ~1000)
- Vectors have nearly identical values
- Using lower precision (float16)

## Solution Requirements

### `log_softmax(x: np.ndarray) -> np.ndarray`
- Must handle 1D and 2D inputs (single vector or batch)
- Must support float32 and float16 dtypes
- Must not produce NaN/Inf on extreme inputs
- Must match reference accuracy: rtol≤1e-3, atol≤1e-4 for float32; rtol≤1e-2, atol≤1e-3 for float16

### `cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float`
- Must compute through log-domain (stable method)
- Same accuracy requirements as log_softmax
- For 2D logits, can return either scalar (averaged) or array (per-sample)

## Key Techniques

Use stable numerical techniques:
- Subtract maximum value: `x - x.max(axis=...)`
- Log-sum-exp trick: `log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`
- Avoid direct `exp(x)` for large x values


## Local Testing

To test locally (without running the full agent):

```python
from tasks.stable_numerics.grader import run_visible_tests, run_full_tests, grade_submission

# Test with visible cases
result = run_visible_tests(seed=42)
print(result)

# Run full test suite
result = run_full_tests(seed=42)
print(result)

# Grade submission
passed = grade_submission(seed=42)
print(f"Passed: {passed}")
```

Make sure your implementation is in `tasks/stable_numerics/user_impl.py`.

