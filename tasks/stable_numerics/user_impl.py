import numpy as np

# TODO: Implement numerically stable log_softmax and cross_entropy functions


def log_softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-softmax function.
    
    Args:
        x: 1D or 2D array (batch, dim). Supports float32 and float16.
    
    Returns:
        Log-softmax of x, same shape and dtype as input.
    """
    raise NotImplementedError("Implement this function")


def cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float:
    """
    Numerically stable cross-entropy loss.
    
    Args:
        logits: 1D or 2D array (batch, dim). Supports float32 and float16.
        target_idx: int (for 1D logits) or 1D array of indices (for 2D logits).
    
    Returns:
        Cross-entropy loss (scalar for 1D, float or array for 2D).
    """
    raise NotImplementedError("Implement this function")
