"""32-point approximate DFT (ADFT) transcribed from the Appendix of Perera et al. 2025.

F_hat_32 = S4 @ S3 @ S2 @ S1 @ S0   (paper eq. (1)), explicit matrix in eq. (14).
Entries of every S_i are in {0, ±1, ±j}; entries of F_hat_32 in {±1, ±j, ±1±j}.

Why it matters: a true 32-point DFT needs complex multipliers (twiddles like e^(-j*2pi/32)).
F_hat_32 replaces each DFT entry e^(-j*2pi*k*n/32) by a point of {±1, ±j, ±1±j} with roughly
the same phase (unnormalized: ±1±j has magnitude sqrt2), so the whole thing is
adds, subtractions and 90-degree swaps -- no multipliers. The cost is that F_hat != F.

This file is pure transcription: S0..S4 are typed in from the paper, and the explicit eq. (14)
matrix is typed in separately so tests/test_g2.py can check S4 S3 S2 S1 S0 == eq. (14).
"""
import numpy as np
from scipy.linalg import block_diag as _blkdiag


def _rows(n_cols, spec):
    """spec: list of rows, each a list of signed column indices; '-k' means -1 at col k.
    Column 0 with a minus sign is written as the string '-0'."""
    A = np.zeros((len(spec), n_cols), dtype=complex)
    for r, row in enumerate(spec):
        for item in row:
            s = str(item)
            A[r, int(s.lstrip('-'))] += -1 if s.startswith('-') else 1
    return A



# ---- S0 (eq. 10): 17x17 and 15x15 sum/difference butterflies ----
S0_11 = _rows(17, [[i, 16 - i] for i in range(8)] + [[8]] + [[8 - m, -(8 + m)] for m in range(1, 9)])
S0_22 = _rows(15, [[i, 14 - i] for i in range(7)] + [[7]] + [[7 - m, -(7 + m)] for m in range(1, 8)])
S0 = _blkdiag(S0_11, S0_22)

# ---- S1 (eq. 11) ----
_s1_common = [[1, 7], [2, 6], [3, 5], [4], [3, -5], [2, -6], [1, -7]]
_s1_tail = [[9, 15], [10, 14], [11, 13], [12], [11, -13], [10, -14], [9, -15]]
S1_11 = _rows(16, [[0, 8]] + _s1_common + [['0', -8]] + _s1_tail)
S1_12 = _rows(16, [[8]] + _s1_common + [[-8]] + _s1_tail)
S1_21 = np.diag([0] + [1] * 15).astype(complex)
S1_22 = np.diag([1] + [-1] * 15).astype(complex)
S1 = np.block([[S1_11, S1_12], [S1_21, S1_22]])

# ---- S2 (eq. 12) ----
S2_11 = _rows(16, [
    [0, 2, 4], [1, 3], [0, -2, 4], [0, 1, -3, -4], ['-0', 1, -3, 4],
    [5, 6, -8], [6, 7, 8], [6, -7, 8], [5, -6, 8],
    [9, 10, -12], [10, 11, 12], [10, -11, 12], [9, -10, 12],
    [13, 14, 15], [13, -14, 15], [13, -15]])
S2_22 = _rows(16, [
    ['-0', -12, 14], [1], [2, 4, -8], [3, 5, 7], [4, 6, 8], [3, -5], [4, -6, 8], [3, -7],
    [2, -4, 8], [9], [0, 10, -12], [11, 13, 15], ['-0', 10, 12], [11, -13], [0, 12, 14], [11, -15]])
S2 = _blkdiag(S2_11, S2_22)

# ---- S3 (eq. 13) ----
S3_11 = _blkdiag(np.array([[1, 1], [1, -1]], dtype=complex), np.eye(14))
S3_22 = _rows(16, [
    [0, 9, -13], [1, 5, -7, 8], [1, -2, 7], [3, 4], [3, -4], [1, -5, 6], [1, -5, -6], [1, 2, 7],
    [1, 5, -7, -8], [9, 10, 13, -15], [9, -10, 13, -15], [11, 14], [9, 12, 15], [0, -9, 13],
    [11, -14], [-9, 12, -15]])
S3 = _blkdiag(S3_11, S3_22)

# ---- S4 (Appendix, "bottom of page 16"): row -> [(col, value), ...] ----
_S4_SPEC = {
    0: [(0, 1)], 1: [(19, -1j), (27, 1)], 2: [(6, 1), (10, -1j)], 3: [(23, -1j), (28, -1)],
    4: [(3, 1), (13, 1j)], 5: [(17, -1j), (25, 1)], 6: [(5, -1), (9, -1j)], 7: [(16, -1), (22, -1j)],
    8: [(2, 1), (15, -1j)], 9: [(21, -1j), (29, -1)], 10: [(8, 1), (12, -1j)], 11: [(24, -1j), (26, -1)],
    12: [(4, -1), (14, 1j)], 13: [(18, -1j), (31, -1)], 14: [(7, 1), (11, 1j)], 15: [(20, -1j), (30, -1)],
    16: [(1, 1)], 17: [(20, 1j), (30, -1)], 18: [(7, 1), (11, -1j)],
    # Row 19 is printed with its +j in column 19. That column is already used by rows 1/31,
    # column 18 would otherwise be orphaned, and row 19 must be the conjugate partner of row 13.
    # Typo fix (DECISIONS.md D1); G2 confirms it reproduces eq. (14) exactly.
    19: [(18, 1j), (31, -1)],
    20: [(4, -1), (14, -1j)], 21: [(24, 1j), (26, -1)], 22: [(8, 1), (12, 1j)], 23: [(21, 1j), (29, -1)],
    24: [(2, 1), (15, 1j)], 25: [(16, -1), (22, 1j)], 26: [(5, -1), (9, 1j)], 27: [(17, 1j), (25, 1)],
    28: [(3, 1), (13, -1j)], 29: [(23, 1j), (28, -1)], 30: [(6, 1), (10, 1j)], 31: [(19, 1j), (27, 1)],
}
S4 = np.zeros((32, 32), dtype=complex)
for _r, _entries in _S4_SPEC.items():
    for _c, _v in _entries:
        S4[_r, _c] = _v

S_FACTORS = [S0, S1, S2, S3, S4]   # application order: S0 first

# ---- eq. (14): explicit F_hat_32, transcribed block by block (rows as printed) ----
_F11 = """
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1-j 1-j 1-j -j -j -j -j -j -1-j -1-j -1-j -1 -1
1 1 1-j -j -j -j -1-j -1 -1 -1 -1+j j j j 1+j 1
1 1-j -j -j -1-j -1 -1 -1+j j 1+j 1 1 1-j -j -j -1-j
1 1-j -j -1-j -1 -1+j j 1+j 1 1-j -j -1-j -1 -1+j j 1+j
1 1-j -j -1 -1+j j 1 1-j -j -1-j -1 j 1+j 1 -j -1-j
1 -j -1-j -1 j 1 1-j -j -1 j 1+j 1 -j -1 -1+j j
1 -j -1 -1+j 1+j 1-j -j -1 j 1 -j -1-j -1+j 1+j 1 -j
1 -j -1 j 1 -j -1 j 1 -j -1 j 1 -j -1 j
1 -j -1 1+j 1-j -1-j j 1 -j -1 j 1-j -1-j -1+j 1 -j
1 -j -1+j 1 -j -1 1+j -j -1 j 1-j -1 j 1 -1-j j
1 -1-j j 1 -1-j j 1 -1-j j 1-j -1 j 1-j -1 j 1-j
1 -1-j j 1-j -1 1+j -j -1+j 1 -1-j j 1-j -1 1+j -j -1+j
1 -1-j j -j -1+j 1 -1 1+j -j -1+j 1 -1 1+j -j j 1-j
1 -1 1+j -j j -j -1+j 1 -1 1 -1-j j -j j 1-j -1
1 -1 1 -1-j 1+j -1-j j -j j -j j 1-j -1+j 1-j -1 1
"""
_F12 = """
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
-1 -1 -1 -1+j -1+j -1+j j j j j j 1+j 1+j 1+j 1 1
1 1 1-j -j -j -j -1-j -1 -1 -1 -1+j j j j 1+j 1
-1 -1+j j j 1+j 1 1 1-j -j -1-j -1 -1 -1+j j j 1+j
1 1-j -j -1-j -1 -1+j j 1+j 1 1-j -j -1-j -1 -1+j j 1+j
-1 -1+j j 1 1-j -j -1 -1+j j 1+j 1 -j -1-j -1 j 1+j
1 -j -1-j -1 j 1 1-j -j -1 j 1+j 1 -j -1 -1+j j
-1 j 1 1-j -1-j -1+j j 1 -j -1 j 1+j 1-j -1-j -1 j
1 -j -1 j 1 -j -1 j 1 -j -1 j 1 -j -1 j
-1 j 1 -1-j -1+j 1+j -j -1 j 1 -j -1+j 1+j 1-j -1 j
1 -j -1+j 1 -j -1 1+j -j -1 j 1-j -1 j 1 -1-j j
-1 1+j -j -1 1+j -j -1 1+j -j -1+j 1 -j -1+j 1 -j -1+j
1 -1-j j 1-j -1 1+j -j -1+j 1 -1-j j 1-j -1 1+j -j -1+j
-1 1+j -j j 1-j -1 1 -1-j j 1-j -1 1 -1-j j -j -1+j
1 -1 1+j -j j -j -1+j 1 -1 1 -1-j j -j j 1-j -1
-1 1 -1 1+j -1-j 1+j -j j -j j -j -1+j 1-j -1+j 1 -1
"""
_F21 = """
1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1
1 -1 1 -1+j 1-j -1+j -j j -j j -j 1+j -1-j 1+j -1 1
1 -1 1-j j -j j -1-j 1 -1 1 -1+j -j j -j 1+j -1
1 -1+j -j j -1-j 1 -1 1-j j -1-j 1 -1 1-j j -j 1+j
1 -1+j -j 1+j -1 1-j j -1-j 1 -1+j -j 1+j -1 1-j j -1-j
1 -1+j -j 1 -1+j -j 1 -1+j -j 1+j -1 -j 1+j -1 -j 1+j
1 j -1-j 1 j -1 1-j j -1 -j 1+j -1 -j 1 -1+j -j
1 j -1 1-j 1+j -1+j -j 1 j -1 -j 1+j -1+j -1-j 1 j
1 j -1 -j 1 j -1 -j 1 j -1 -j 1 j -1 -j
1 j -1 -1-j 1-j 1+j j -1 -j 1 j -1+j -1-j 1-j 1 j
1 j -1+j -1 -j 1 1+j j -1 -j 1-j 1 j -1 -1-j -j
1 1+j j -1 -1-j -j 1 1+j j -1+j -1 -j 1-j 1 j -1+j
1 1+j j -1+j -1 -1-j -j 1-j 1 1+j j -1+j -1 -1-j -j 1-j
1 1+j j j -1+j -1 -1 -1-j -j 1-j 1 1 1+j j j -1+j
1 1 1+j j j j -1+j -1 -1 -1 -1-j -j -j -j 1-j 1
1 1 1 1+j 1+j 1+j j j j j j -1+j -1+j -1+j -1 -1
"""
_F22 = """
1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1
-1 1 -1 1-j -1+j 1-j j -j j -j j -1-j 1+j -1-j 1 -1
1 -1 1-j j -j j -1-j 1 -1 1 -1+j -j j -j 1+j -1
-1 1-j j -j 1+j -1 1 -1+j -j 1+j -1 1 -1+j -j j -1-j
1 -1+j -j 1+j -1 1-j j -1-j 1 -1+j -j 1+j -1 1-j j -1-j
-1 1-j j -1 1-j j -1 1-j j -1-j 1 j -1-j 1 j -1-j
1 j -1-j 1 j -1 1-j j -1 -j 1+j -1 -j 1 -1+j -j
-1 -j 1 -1+j -1-j 1-j j -1 -j 1 j -1-j 1-j 1+j -1 -j
1 j -1 -j 1 j -1 -j 1 j -1 -j 1 j -1 -j
-1 -j 1 1+j -1+j -1-j -j 1 j -1 -j 1-j 1+j -1+j -1 -j
1 j -1+j -1 -j 1 1+j j -1 -j 1-j 1 j -1 -1-j -j
-1 -1-j -j 1 1+j j -1 -1-j -j 1-j 1 j -1+j -1 -j 1-j
1 1+j j -1+j -1 -1-j -j 1-j 1 1+j j -1+j -1 -1-j -j 1-j
-1 -1-j -j -j 1-j 1 1 1+j j -1+j -1 -1 -1-j -j -j 1-j
1 1 1+j j j j -1+j -1 -1 -1 -1-j -j -j -j 1-j 1
-1 -1 -1 -1-j -1-j -1-j -j -j -j -j -j 1-j 1-j 1-j 1 1
"""


def _tok(t):
    """'1-j' -> 1-1j, '-j' -> -1j, 'j' -> 1j."""
    t = t.replace('j', '1j') if not any(ch.isdigit() for ch in t.split('j')[0][-1:]) else t
    return complex(t)


F_HAT_32 = np.block([[np.array([[_tok(t) for t in r.split()] for r in b.strip().splitlines()])
                      for b in pair] for pair in ((_F11, _F12), (_F21, _F22))])

# Same matrix from a one-line rule observed in eq. (14): F_hat[k, n] = g((k*n) mod 32),
# with g = row 1 of eq. (14). Used only as an independent cross-check in tests/test_g2.py.
_G = F_HAT_32[1]
F_HAT_32_RULE = _G[np.outer(np.arange(32), np.arange(32)) % 32]
