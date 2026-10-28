"""All simulation constants. Frequencies are handled as fn = f / f_max everywhere."""
import numpy as np

C0 = 299_792_458.0
SEED = 0

N_MAIN = 16
F_MAX_PAPER = 1e9            # paper Sec. VI: f_max = 1 GHz -> tau = 62.5 ps
F_MAX_COURSE = 10e9          # course band 5-10 GHz; identical in normalized fn = f/f_max
PAPER_FNS = (0.15, 0.30, 0.60, 1.00)          # 150/300/600/1000 MHz at f_max = 1 GHz
SWEEP_FNS = np.linspace(0.1, 1.0, 46)
BAND = (0.5, 1.0)            # course band 5-10 GHz in fn; metrics/b* use this band
U_GRID = np.linspace(-1.0, 1.0, 4001)          # u = sin(theta)
B_SWEEP = (2, 3, 4, 5, 6, 8)
N_COMPLEXITY = tuple(2 ** r for r in range(4, 11))   # 16 .. 1024

# Pass/fail thresholds (fixed before running; never tuned)
SQUINT_FREE_DEG = 0.5
POINT_ERR_DEG = 0.5
SLL_DEGRADE_DB = 1.0


def tau(N, f_max):
    return 1.0 / (N * f_max)


def spacing(f_max):
    return C0 / (2 * f_max)   # lambda/2 at f_max -> no grating lobes in band


def phi(N, fn):
    """omega * tau, so alpha = exp(-1j * phi).  omega*tau = 2*pi*f/(N f_max)."""
    return 2 * np.pi * fn / N


def kprimes(N):
    return np.arange(N) - N // 2


def inner_beams(N):
    """Beams used for squint/pointing/SLL statistics: |k'| <= N/2 - 2 (endfire excluded)."""
    k = kprimes(N)
    return np.abs(k) <= N // 2 - 2
