import numpy as np


def direct(N, ph):
    """Scaled DVM  A~_N[k, l] = alpha^(k l),  alpha = exp(-j ph).
    Beam k delays element l by k*l unit delays and sums: N*N delay/gain blocks, O(N^2).
    This is the ground truth every fast algorithm is checked against (tests/test_g1.py)."""
    k = np.arange(N)
    return np.exp(-1j * ph * np.outer(k, k))
