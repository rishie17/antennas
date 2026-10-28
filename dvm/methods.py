"""Beamformer matrices B(fn) (rows = beams k' = -N/2..N/2-1) acting on element signals x."""
import numpy as np

from .array import predelay
from .dvm_approx import advm, advm_b
from .dvm_direct import direct
from .dvm_exact import exact
from .params import phi

_DVM = {'DIRECT': direct, 'EXACT': exact, 'ADVM': advm}


def beam_matrix(method, N, fn, b=None):
    if method == 'PS':
        # Phase-shifter / DFT beamformer: DVM frozen at f_max, applied at all f (squints).
        kp = np.arange(N) - N // 2
        return np.exp(-2j * np.pi * np.outer(kp, np.arange(N)) / N)
    ph = phi(N, fn)
    A = advm_b(N, ph, b) if method == 'ADVM-b' else _DVM[method](N, ph)
    return A * predelay(N, ph)[None, :]          # A @ diag(pre-delay)
