"""Fast DVM as an explicit list of sparse factors (paper Sec. II, from ref. [6]).

  A~_N = (1/M) * D^_N  J^T  F~*_M  D(  F~_M  J  D^_N,     M = 2N,
  D^_N = diag(alpha^(k^2/2)),  D( = diag(F~_M c),  F~_M = sqrt(M) F_M (unnormalized DFT).

Every algorithm is a list of (tag, sparse matrix) applied right-to-left
(first list item acts first). The same list is used to compute AND to count
operations (dvm/opcount.py), so the counts are of the code that actually runs.
The 1/M scale is a power of two (bit shift) and is applied outside the list.
"""
import numpy as np
import scipy.sparse as sp


def twiddle_exact(M, l):
    return np.exp(-2j * np.pi * l / M)


def dft_factors(M, base=None, twiddle=twiddle_exact):
    """Radix-2 decimation-in-frequency: F_M = P_M^T blkdiag(F_{M/2}, F_{M/2}) H_M.
    base: optional (size, [factors]) to stop the recursion (used for the ADFT)."""
    if base is not None and M == base[0]:
        return [('dft', sp.csr_matrix(S)) for S in base[1]]
    if M == 1:
        return []
    h = M // 2
    I = sp.identity(h, format='csr')
    # H_M = [[I, I], [D`, -D`]] = diag(I, D`) @ [[I, I], [I, -I]]: butterfly, then ONE
    # multiplication per twiddle (writing H as a single factor would count each twiddle twice).
    butterfly = sp.bmat([[I, I], [I, -I]], format='csr')
    T = sp.diags(np.r_[np.ones(h), twiddle(M, np.arange(h))], format='csr')
    sub = dft_factors(h, base, twiddle)
    perm = np.r_[np.arange(0, M, 2), np.arange(1, M, 2)]          # P_M x = [x_even; x_odd]
    PT = sp.csr_matrix((np.ones(M), (perm, np.arange(M))), shape=(M, M))   # P_M^T
    return ([('dft', butterfly), ('dft', T)]
            + [('dft', sp.block_diag([S, S], format='csr')) for _, S in sub] + [('dft', PT)])


def inverse(factors):
    """F* = conjugate transpose of the product -> reverse order, conj-transpose each factor."""
    return [(tag, S.conj().T.tocsr()) for tag, S in reversed(factors)]


def circulant_column(N, ph):
    """c = [1, a^-1/2, ..., a^-(N-1)^2/2, 1, a^-(N-1)^2/2, ..., a^-1/2]."""
    m = np.arange(N)
    half = np.exp(1j * ph * m ** 2 / 2)
    return np.r_[half, 1.0, half[:0:-1]]


def dvm_factors(N, ph, fwd):
    """Factor list for M * A~_N given the forward (approximate) transform factors `fwd`."""
    M = 2 * N
    Dhat = sp.diags(np.exp(-1j * ph * np.arange(N) ** 2 / 2), format='csr')
    J = sp.vstack([sp.identity(N), sp.csr_matrix((N, N))], format='csr')
    Dbreve = sp.diags(np.fft.fft(circulant_column(N, ph)), format='csr')   # pre-computed, exact
    return ([('Dhat', Dhat), ('J', J)] + fwd + [('Dbreve', Dbreve)] + inverse(fwd)
            + [('J', J.T.tocsr()), ('Dhat', Dhat)])


def apply(factors, X):
    for _, S in factors:
        X = S @ X
    return X


def to_matrix(factors, N):
    """Dense N x N matrix of the factored operator, including the 1/M scale."""
    return apply(factors, np.eye(N, dtype=complex)) / (2 * N)


def exact_factors(N, ph):
    return dvm_factors(N, ph, dft_factors(2 * N))


def exact(N, ph):
    return to_matrix(exact_factors(N, ph), N)
