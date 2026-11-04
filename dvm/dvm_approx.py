"""Approximate DVMs.

ADVM   : paper Prop. 2.3 / Algorithms 1-4. Exact factorization with F~_M, F~*_M replaced
         by the ADFT recursion F^_M = P^T blkdiag(F^_{M/2}, F^_{M/2}) H_M down to F^_32
         = S4 S3 S2 S1 S0 (Appendix). N = 16 -> F^_32 directly (Asdvm16).
ADVM-b : OUR extension (not in the paper). Exact radix-2 DVM with every twiddle of the
         F_M / F*_M stages rounded to the nearest 2^b-PSK phase (unit magnitude).
D( is pre-computed from the exact F~_M c in both cases (as in the paper).
"""
import numpy as np

from .adft import S_FACTORS
from .dvm_exact import dft_factors, dvm_factors, to_matrix


def advm_factors(N, ph):
    """Same D^ J F D( F* J^T D^ structure as EXACT; only F changes. The FFT recursion runs
    normally from 2N down to 32, then the 32-point stage is the multiplierless S0..S4 instead
    of an exact DFT. Fewer multipliers, but F^ != F, so the result only approximates A~."""
    assert N >= 16, 'N = 8 needs the 16-point ADFT of ref. [45]; out of scope'
    return dvm_factors(N, ph, dft_factors(2 * N, base=(32, S_FACTORS)))


def advm(N, ph):
    return to_matrix(advm_factors(N, ph), N)


def psk_twiddle(b):
    """Nearest 2^b-PSK point to exp(-2j pi l / M). Ties round toward +phase (floor(x+0.5));
    the arithmetic is exact because M and 2^b are powers of two.

    Intuition: b=2 -> only {1, -j, -1, j} (free: wires and 90-degree shifts); b=3 adds the
    45-degree points (need a 1/sqrt2 gain); once 2^b >= M every twiddle lands exactly on
    the grid and ADVM-b == EXACT."""
    def tw(M, l):                                # same signature as dvm_exact.twiddle_exact
        k = np.floor(-np.asarray(l) * 2 ** b / M + 0.5)
        return np.exp(2j * np.pi * k / 2 ** b)
    return tw


def advm_b_factors(N, ph, b):
    return dvm_factors(N, ph, dft_factors(2 * N, twiddle=psk_twiddle(b)))


def advm_b(N, ph, b):
    return to_matrix(advm_b_factors(N, ph, b), N)
