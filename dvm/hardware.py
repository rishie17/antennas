"""Hardware mapping of the executed factor lists (signal-flow graph -> RF components),
physical delay-line lengths, and dispersive replacements of the ideal delays alpha^p.

Component rules (applied to every factor of the list that dvm/opcount.py also counts):
  delay factor (D^ or DIRECT entry alpha^p, p != 0)  -> delay line of length p*tau
  D( entry (frequency-dependent, pre-computed)        -> frequency-dependent gain block
  other non-trivial constant (twiddle)                -> constant gain block
  2 rows sharing a 2-column support, unit entries,
    orthogonal (e.g. [[1,1],[1,-1]])                  -> 180 deg hybrid (real) / 90 deg hybrid (has +-j)
  remaining row with nnz inputs                       -> nnz-1 two-way combiners
  remaining coefficient -1 / +-j                      -> 180 deg inverter / fixed 90 deg shifter
  column feeding u > 1 components                     -> u-1 two-way splitters
  permutations, zero-padding J, entries 1             -> wires
Invariant (tested): combiners + 2*hybrids == opcount adders, delay_lines == opcount delays,
gain_const == opcount gains, gain_freqdep == opcount anticausal.
"""
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

from .array import predelay
from .dvm_approx import advm_factors
from .dvm_direct import direct
from .dvm_exact import exact, exact_factors
from .opcount import GENERIC_PH, _nontrivial
from .params import C0, F_MAX_COURSE, tau

F_CENTER = 7.5e9
FC_WR90 = C0 / (2 * 22.86e-3)          # WR-90 TE10 cutoff, a = 22.86 mm -> 6.557 GHz
MEDIA = {'air': ('tem', 1.0), 'stripline': ('tem', 4.4),
         'WR-90': ('wg', FC_WR90), 'WG-fc4.5': ('wg', 4.5e9)}   # (kind, eps_r or f_cutoff)
BUILD = {'DIRECT': lambda N, ph: [('direct', sp.csr_matrix(direct(N, ph)))],
         'EXACT': exact_factors, 'ADVM': advm_factors}
_DELAY_TAGS = {'Dhat', 'direct'}
_PH_SMALL = 1e-7     # alpha^p at tiny omega*tau: p = -angle/ph without 2*pi wrapping (p <= (N-1)^2)


def _hybrids(S):
    """[(rows, cols, '180'|'90')] for orthogonal unit-modulus 2x2 blocks in a constant factor."""
    by_support = defaultdict(list)
    for r in range(S.shape[0]):
        cols = S.indices[S.indptr[r]:S.indptr[r + 1]]
        if len(cols) == 2:
            by_support[tuple(sorted(cols))].append(r)
    out = []
    for cols, rows in by_support.items():
        if len(rows) != 2:
            continue
        B = S[rows][:, list(cols)].toarray()
        if np.allclose(np.abs(B), 1) and abs(np.vdot(B[0], B[1])) < 1e-9:
            out.append((rows, cols, '180' if np.allclose(B.imag, 0) else '90'))
    return out


def sfg(method, N):
    """Component counts and the multiset of delay lengths (units of tau) for one method."""
    c = Counter(delay_lines=0, combiners=0, splitters=0, hyb180=0, hyb90=0, shift90=0,
                inverters=0, gain_const=0, gain_freqdep=0)
    lengths = []
    # Build the factor list twice: at a generic omega*tau (to tell which entries are
    # non-trivial) and at a tiny omega*tau (where angle(alpha^p) = -p*ph has no 2pi wrap, so the
    # delay length p can be read straight off the phase).
    for (tag, S), (_, Ss) in zip(BUILD[method](N, GENERIC_PH), BUILD[method](N, _PH_SMALL)):
        S = S.tocsr()
        S.eliminate_zeros()
        if tag in _DELAY_TAGS:
            p = np.round(-2 * np.angle(Ss.tocsr().data) / _PH_SMALL) / 2
            lengths += list(p[p != 0])
            c['delay_lines'] += int((p != 0).sum())
        elif tag == 'Dbreve':
            c['gain_freqdep'] += _nontrivial(S.data).size
        hyb = [] if tag in _DELAY_TAGS else _hybrids(S)
        in_hyb = {r for rows, _, _ in hyb for r in rows}
        for _, _, kind in hyb:
            c['hyb' + kind] += 1
        nnz = np.diff(S.indptr)
        c['combiners'] += int(sum(max(n - 1, 0) for r, n in enumerate(nnz) if r not in in_hyb))
        uses = np.bincount(S.indices, minlength=S.shape[1])
        for _, cols, _ in hyb:
            uses[list(cols)] -= 1                  # a hybrid takes each of its inputs once
        c['splitters'] += int(np.maximum(uses - 1, 0).sum())
        if tag in _DELAY_TAGS or tag == 'Dbreve':
            continue
        for r in range(S.shape[0]):
            if r in in_hyb:
                continue
            for v in S.data[S.indptr[r]:S.indptr[r + 1]]:
                if np.isclose(v, -1):
                    c['inverters'] += 1
                elif np.isclose(abs(v.imag), 1) and np.isclose(v.real, 0):
                    c['shift90'] += 1
                elif _nontrivial(np.array([v])).size:
                    c['gain_const'] += 1
                else:
                    assert np.isclose(v, 1), f'unmapped coefficient {v}'   # +-1+-j never occurs alone
    c['gain_blocks'] = c['gain_const'] + c['gain_freqdep']
    return dict(c), np.array(sorted(lengths))


def unit_length(medium, N, f_max=F_MAX_COURSE, f0=F_CENTER, design='group'):
    """Physical length (m) of a tau-long line. Waveguides: tau of GROUP delay at f0 (design='group')
    or of PHASE delay at f0 (design='phase')."""
    kind, x = MEDIA[medium]
    if kind == 'tem':
        return tau(N, f_max) * C0 / np.sqrt(x)
    r = np.sqrt(1 - (x / f0) ** 2)
    return tau(N, f_max) * C0 * (r if design == 'group' else 1 / r)


def phase_per_tau(medium, N, f, f_max=F_MAX_COURSE, design='group'):
    """beta(f) * L_tau (rad): the line's replacement for omega*tau. NaN below cutoff (evanescent)."""
    kind, x = MEDIA[medium]
    f = np.asarray(f, dtype=float)
    k0 = 2 * np.pi * f / C0
    if kind == 'tem':
        beta = k0 * np.sqrt(x)
    else:
        with np.errstate(invalid='ignore'):
            beta = np.where(f > x, k0 * np.sqrt(1 - (x / f) ** 2), np.nan)
    return beta * unit_length(medium, N, f_max, design=design)


def dispersive_beam_matrix(N, ph_line):
    """EXACT beamformer with EVERY alpha^p (D^, the alpha^(-m^2/2) inside D(, pre-delays) replaced by a
    line of length p*L_tau. All lines share one medium, so alpha^p -> exp(-j*ph_line*p)."""
    return exact(N, ph_line) * predelay(N, ph_line)[None, :]


if __name__ == '__main__':
    for m in BUILD:
        print(m, sfg(m, 16)[0])
