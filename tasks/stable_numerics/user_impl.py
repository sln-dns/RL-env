import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute log-softmax in a numerically stable way.
    
    Uses the log-sum-exp trick: log_softmax(x) = x - (max(x) + log(sum(exp(x - max(x)))))
    
    Args:
        x: 1D or 2D array (for 2D: shape (batch, dim))
        
    Returns:
        log-softmax of x with same shape and dtype
    """
    # Ensure we have a numpy array
    x = np.asarray(x)
    original_dtype = x.dtype
    
    # Work in float64 for better numerical stability during computation
    x_float64 = x.astype(np.float64)
    
    if x.ndim == 1:
        # 1D case
        x_max = np.max(x_float64)
        # Compute log-softmax using the log-sum-exp trick
        exp_shifted = np.exp(x_float64 - x_max)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = x_max + np.log(sum_exp)
        result = x_float64 - log_sum_exp
    else:
        # 2D case (batch, dim)
        # Find max along dimension 1 (for each sample in batch)
        x_max = np.max(x_float64, axis=1, keepdims=True)
        # Compute log-softmax using the log-sum-exp trick
        exp_shifted = np.exp(x_float64 - x_max)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        log_sum_exp = x_max + np.log(sum_exp)
        result = x_float64 - log_sum_exp
    
    # Convert back to original dtype
    return result.astype(original_dtype)


def cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float:
    """
    Compute cross-entropy loss in a numerically stable way.
    
    Cross-entropy = -log_softmax(logits[target_idx])
    
    Args:
        logits: 1D or 2D array
                - 1D: shape (dim,) - single sample
                - 2D: shape (batch, dim) - multiple samples
        target_idx: int (for 1D logits) or 1D array of ints (for 2D logits)
                   - Indices of target classes
    
    Returns:
        float (for 1D logits) or array/scalar (for 2D logits)
        - For 2D: can return either per-sample losses (array) or average (scalar)
    """
    logits = np.asarray(logits)
    
    # Compute log-softmax
    log_sm = log_softmax(logits)
    
    if logits.ndim == 1:
        # 1D case: single sample
        target_idx = int(target_idx)
        ce = -log_sm[target_idx]
        return float(ce)
    else:
        # 2D case: batch of samples
        target_idx = np.asarray(target_idx, dtype=np.intp)
        batch_size = logits.shape[0]
        
        # Get the log-softmax values at target indices
        ce_per_sample = -log_sm[np.arange(batch_size), target_idx]
        
        # Return per-sample losses as array
        return ce_per_sample
