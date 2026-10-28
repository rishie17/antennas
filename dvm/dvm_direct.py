import numpy as np


def direct(N, ph):
    """Scaled DVM  A~_N[k, l] = alpha^(k l),  alpha = exp(-j ph)."""
    k = np.arange(N)
    return np.exp(-1j * ph * np.outer(k, k))
