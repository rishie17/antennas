"""Operation counts read off the factor lists that are actually executed.

gain-delay block = multiplication by a value NOT in {±1, ±j, ±1±j}   (paper Sec. III-B)
  split as: 'delays'      -> D^_N entries (and steering pre-delays, frequency dependent)
            'anti-causal' -> D( entries (pre-computed, paper Table 3)
            'gains'       -> constant twiddles inside F_M / F*_M
adders  = sum over rows of (nnz - 1) in every factor (complex additions)
memory  = number of distinct non-trivial constant coefficients (twiddles) + |D(| entries
Counts are taken at a generic alpha (ph = 1/pi) so no delay accidentally equals ±1/±j.
"""
import numpy as np

# "Free" coefficients: x*(+-1) is a sign flip, x*(+-j) swaps re/im (90-degree shift), and
# x*(1+j) = x + j*x is just an add. None of them needs a real multiplier / gain block.
TRIVIAL = np.array([1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
GENERIC_PH = 1 / np.pi
_KIND = {'Dhat': 'delays', 'predelay': 'delays', 'direct': 'delays', 'Dbreve': 'anticausal'}


def _nontrivial(v):
    """Entries of v that are neither ~0 nor ~a TRIVIAL value (i.e. need a gain-delay block)."""
    v = v[np.abs(v) > 1e-12]
    return v[np.min(np.abs(v[:, None] - TRIVIAL[None, :]), axis=1) > 1e-9]


def count(factors):
    out = dict(gains=0, delays=0, anticausal=0, adders=0)
    consts = []
    for tag, S in factors:
        S = S.tocsr()
        S.eliminate_zeros()
        nt = _nontrivial(S.data)
        out[_KIND.get(tag, 'gains')] += nt.size          # bucket by what the factor is (tag)
        if _KIND.get(tag, 'gains') == 'gains':
            consts.append(nt)
        # a row with n nonzeros sums n inputs -> n-1 additions (indptr diff = nnz per row)
        out['adders'] += int(np.maximum(np.diff(S.indptr) - 1, 0).sum())
    out['gain_delay'] = out['gains'] + out['delays'] + out['anticausal']
    consts = np.concatenate(consts) if consts else np.array([])
    out['memory'] = len(np.unique(np.round(consts, 9))) + out['anticausal']
    return out
