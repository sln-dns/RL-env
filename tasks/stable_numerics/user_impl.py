import numpy as np
from typing import Union


def log_softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable log_softmax implementation.
    
    Args:
        x: Input array of shape (n,) or (batch, n)
        
    Returns:
        log_softmax(x) with same shape and dtype as input
    """
    # Preserve dtype
    original_dtype = x.dtype
    
    # Make a copy to avoid mutating input
    x = np.array(x, copy=True)
    
    # Ensure it's at least 1D
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Compute log_softmax with numerical stability
    # For numerical stability, subtract the max along the last axis
    x_max = np.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    
    # Compute log(sum(exp(x - max)))
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    
    # log_softmax(x) = x - log(sum(exp(x)))
    result = x_shifted - log_sum_exp
    
    # Ensure output dtype matches input
    return result.astype(original_dtype)


def cross_entropy(logits: np.ndarray, target_idx: Union[np.ndarray, int]) -> Union[np.ndarray, float]:
    """
    Numerically stable cross-entropy loss.
    
    Args:
        logits: Input logits of shape (n,) or (batch, n)
        target_idx: Target class index (int for 1D, array for 2D batch)
        
    Returns:
        Cross-entropy loss. For 1D logits returns scalar (numpy scalar), 
        for 2D logits returns vector of shape (batch,)
    """
    # Preserve dtype
    original_dtype = logits.dtype
    
    # Make a copy to avoid mutating input
    logits = np.array(logits, copy=True)
    
    # Ensure at least 1D
    if logits.ndim == 0:
        logits = logits.reshape(1)
    
    # Compute log_softmax
    log_probs = log_softmax(logits)
    
    if logits.ndim == 1:
        # 1D case: single sample
        # Ensure target_idx is a scalar
        if isinstance(target_idx, np.ndarray):
            target_idx = int(target_idx.item())
        target_idx = int(target_idx)
        
        # Cross-entropy: -log_softmax(logits)[target_idx]
        loss = -log_probs[target_idx]
        # Return as numpy scalar with the correct dtype
        return loss.astype(original_dtype)
    
    elif logits.ndim == 2:
        # 2D case: batch of samples
        # target_idx should be an array of shape (batch,)
        if isinstance(target_idx, (int, np.integer)):
            # Single target for all samples
            target_idx = np.full(logits.shape[0], target_idx, dtype=np.int64)
        else:
            target_idx = np.asarray(target_idx, dtype=np.int64)
        
        # Use advanced indexing to extract the correct log_probs for each sample
        batch_indices = np.arange(logits.shape[0])
        loss = -log_probs[batch_indices, target_idx]
        
        # Return as vector of shape (batch,) with same dtype as logits
        return loss.astype(original_dtype)
    
    else:
        raise ValueError(f"Expected 1D or 2D logits, got shape {logits.shape}")
