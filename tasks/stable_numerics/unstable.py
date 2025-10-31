"""
Unstable reference implementations - DO NOT USE THESE!
These are provided as examples of what NOT to do.
They will fail on extreme inputs or with float16 precision.
"""
import numpy as np


def unstable_log_softmax(x: np.ndarray) -> np.ndarray:
    """
    NAIVE, UNSTABLE implementation - will produce NaN/Inf on extreme values.
    DO NOT COPY THIS!
    """
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    return np.log(exp_x / sum_exp)  # This will overflow for large x


def unstable_cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float:
    """
    NAIVE, UNSTABLE implementation - computes softmax then takes log.
    Do not use this implementation!
    """
    # This approach is numerically unstable
    softmax = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    log_probs = np.log(softmax + 1e-10)  # Even with epsilon, this can be unstable
    
    if logits.ndim == 1:
        return -log_probs[target_idx]
    else:
        targets = np.array(target_idx)
        return -np.mean(log_probs[np.arange(len(targets)), targets])

