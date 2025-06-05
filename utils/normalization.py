import numpy as np

def instance_normalize(insatnce: np.ndarray,
                      method: str = "standard",
                      eps: float = 1e-6) -> np.ndarray:
    """
    Returns a Instance normalized data.

    Parameters
    ----------
    insatnce : np.ndarray, shape (time_steps, num_features)
        The sequence data to be normalized.
    method : str, default="standard"
        - "minmax": Scales values within insatnce to [0, 1] based on min/max values.
        - "standard": Standardizes insatnce using mean and standard deviation (mean 0, variance 1).
    eps : float, default=1e-6
        A small value added to the denominator to prevent division by zero.

    Returns
    -------
    insatnce_norm : np.ndarray, same shape as insatnce
        The normalized sequence.
    """
    if method == "minmax":
        insatnce_min = np.min(insatnce, axis=0)           #  minimum value of features
        insatnce_max = np.max(insatnce, axis=0)           #  maximum value of features
        insatnce_norm = (insatnce - insatnce_min) / (insatnce_max - insatnce_min + eps)
    
    elif method == "standard":
        insatnce_mean = np.mean(insatnce, axis=0)         # mean value of all value
        insatnce_std  = np.std(insatnce, axis=0)          # standard deviation for each value
        insatnce_norm = (insatnce - insatnce_mean) / (insatnce_std + eps)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return insatnce_norm