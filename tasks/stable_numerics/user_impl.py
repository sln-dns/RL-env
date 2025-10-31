import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-softmax implementation.
    
    Args:
        x: Input array, shape (dim,) for 1D or (batch, dim) for 2D
        
    Returns:
        Log-softmax of x, same shape as input
    """
    # Handle both 1D and 2D cases
    if x.ndim == 1:
        # For 1D array, subtract max and compute logsumexp manually
        x_shifted = x - np.max(x)
        logsumexp_val = np.max(x) + np.log(np.sum(np.exp(x_shifted)))
        return x - logsumexp_val
    elif x.ndim == 2:
        # For 2D array (batch, dim), compute per row
        # Subtract max along axis 1 for each row
        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max
        # Compute logsumexp: max + log(sum(exp(x - max)))
        logsumexp_vals = x_max + np.log(np.sum(np.exp(x_shifted), axis=1, keepdims=True))
        return x - logsumexp_vals
    else:
        raise ValueError(f"Expected 1D or 2D array, got {x.ndim}D")


def cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float:
    """
    Numerically stable cross-entropy loss computation.
    
    Args:
        logits: Model outputs, shape (dim,) for single sample or (batch, dim) for batch
        target_idx: Target class index, int for 1D logits or array of ints for 2D logits
        
    Returns:
        Cross-entropy loss:
        - For 1D logits: scalar (same dtype as input)
        - For 2D logits: array of per-sample losses (when target_idx is array) or scalar (when target_idx is int)
    """
    if logits.ndim == 1:
        # Single sample case
        if not isinstance(target_idx, (int, np.integer)):
            raise ValueError("For 1D logits, target_idx must be an int")
        
        # Compute log-softmax
        log_softmax_vals = log_softmax(logits)
        # Cross-entropy is -log_softmax[target_idx]
        return -log_softmax_vals[target_idx]
    
    elif logits.ndim == 2:
        # Batch case
        if isinstance(target_idx, (int, np.integer)):
            # Single target for all samples
            log_softmax_vals = log_softmax(logits)
            # Return per-sample losses
            return -log_softmax_vals[:, target_idx]
        else:
            # Per-sample targets
            target_idx = np.asarray(target_idx)
            log_softmax_vals = log_softmax(logits)
            # Extract log_softmax values for each target
            batch_indices = np.arange(len(target_idx))
            ce_losses = -log_softmax_vals[batch_indices, target_idx]
            # Return per-sample losses as array
            return ce_losses
    else:
        raise ValueError(f"Expected 1D or 2D logits, got {logits.ndim}D")
