"""Regenerate every CSV, figure and .tex file:   python scripts/run_all.py"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dvm.array import array_factor, u_target_dvm, u_target_ps          # noqa: E402
from dvm.dvm_approx import advm, advm_b, advm_b_factors, advm_factors  # noqa: E402
from dvm.dvm_direct import direct                                     # noqa: E402
from dvm.dvm_exact import exact_factors                               # noqa: E402
from dvm.io import write_csv                                          # noqa: E402
from dvm.methods import beam_matrix                                   # noqa: E402
from dvm.metrics import pattern_error_db, peaks, rel_spec_err, sll_db, u_to_deg  # noqa: E402
from dvm.opcount import GENERIC_PH, count                             # noqa: E402
from dvm.params import (B_SWEEP, BAND, N_COMPLEXITY, N_MAIN, PAPER_FNS, POINT_ERR_DEG,  # noqa: E402
                        SEED, SLL_DEGRADE_DB, SQUINT_FREE_DEG, SWEEP_FNS, U_GRID,
                        inner_beams, kprimes, phi)

RES = ROOT / 'results'
FNS = np.unique(np.round(np.r_[SWEEP_FNS, PAPER_FNS], 10))
METHODS = [('PS', ''), ('DIRECT', ''), ('EXACT', ''), ('ADVM', '')] + [('ADVM-b', b) for b in B_SWEEP]

LEMMA_32 = {16: 62, 32: 186, 64: 498, 128: 1250, 256: 3010, 512: 7042, 1024: 16130}
TABLE1_SDVM = {16: 130, 32: 322, 64: 770, 128: 1794, 256: 4098, 512: 9218, 1024: 20482}


def beam_rows(method, b, N, fns):
    rows = []
    for fn in fns:
        AF = array_factor(beam_matrix(method, N, fn, b or None), fn, U_GRID)
        up, pk, ipk = peaks(AF, U_GRID)
        sll = sll_db(AF, ipk)
        ut = u_target_dvm(N)                       # intended direction, same for every method
        ups = u_target_ps(N, fn)                   # where PS analytically squints to
        for i, k in enumerate(kprimes(N)):
            rows.append(dict(method=method, b=b, N=N, fn=round(float(fn), 10), kprime=int(k),
                             inner=int(inner_beams(N)[i]), theta_target=u_to_deg(ut[i]),
                             theta_ps_analytic=u_to_deg(ups[i]) if abs(ups[i]) <= 1 else np.nan,
                             theta_peak=u_to_deg(up[i]),
                             point_err=abs(u_to_deg(up[i]) - u_to_deg(ut[i])),
                             peak_db=20 * np.log10(pk[i]), sll_db=sll[i]))
    return rows


def summarize(rows, lo, hi, band_name):
    """Per (method, b): worst case over the inner beams and the frequencies in [lo, hi]."""
    exact_sll = {(r['fn'], r['kprime']): r['sll_db'] for r in rows if r['method'] == 'EXACT'}
    out = []
    for method, b in METHODS:
        R = [r for r in rows if r['method'] == method and r['b'] == b and r['inner']
             and lo - 1e-9 <= r['fn'] <= hi + 1e-9]
        spread, gvar = [], []
        for k in sorted({r['kprime'] for r in R}):
            Rk = [r for r in R if r['kprime'] == k]
            th = np.array([r['theta_peak'] for r in Rk])
            pk = np.array([r['peak_db'] for r in Rk])
            spread.append(th.max() - th.min())
            gvar.append(pk.max() - pk.min())
        dsll = np.array([r['sll_db'] - exact_sll[(r['fn'], r['kprime'])] for r in R])
        sll = np.array([r['sll_db'] for r in R])
        perr = np.array([r['point_err'] for r in R])
        s = dict(method=method, b=b, band=band_name, fn_lo=lo, fn_hi=hi,
                 squint_spread_deg=max(spread), max_point_err_deg=np.nanmax(perr),
                 max_gain_var_db=max(gvar), worst_sll_db=np.nanmax(sll),
                 worst_sll_delta_db=np.nanmax(dsll) if np.isfinite(dsll).any() else np.nan)
        s['meets_criteria'] = int(s['squint_spread_deg'] < SQUINT_FREE_DEG and s['max_point_err_deg'] < POINT_ERR_DEG
                                  and s['worst_sll_delta_db'] < SLL_DEGRADE_DB)
        out.append(s)
    return out


def complexity_rows():
    rows = []
    for N in N_COMPLEXITY:
        import scipy.sparse as sp
        cs = {'DIRECT': count([('direct', sp.csr_matrix(direct(N, GENERIC_PH)))]),
              'EXACT': count(exact_factors(N, GENERIC_PH)),
              'ADVM': count(advm_factors(N, GENERIC_PH))}
        paper = {'DIRECT': (N - 1) ** 2, 'EXACT': TABLE1_SDVM[N], 'ADVM': LEMMA_32[N]}
        for m, c in cs.items():
            rows.append(dict(N=N, method=m, b='', **c, paper_gain_delay=paper[m], steering_predelays=N - 1,
                             red_vs_direct_pct=100 * (1 - c['gain_delay'] / cs['DIRECT']['gain_delay']),
                             red_vs_exact_pct=100 * (1 - c['gain_delay'] / cs['EXACT']['gain_delay'])))
    for b in B_SWEEP:
        c = count(advm_b_factors(N_MAIN, GENERIC_PH, b))
        rows.append(dict(N=N_MAIN, method='ADVM-b', b=b, **c, paper_gain_delay='', steering_predelays=N_MAIN - 1,
                         red_vs_direct_pct=100 * (1 - c['gain_delay'] / (N_MAIN - 1) ** 2),
                         red_vs_exact_pct=100 * (1 - c['gain_delay'] / TABLE1_SDVM[N_MAIN])))
    return rows


def specnorm_rows():
    rows = []
    for fn in SWEEP_FNS:
        A = direct(N_MAIN, phi(N_MAIN, fn))
        rows.append(dict(method='ADVM', b='', N=N_MAIN, fn=fn, rel_err=rel_spec_err(A, advm(N_MAIN, phi(N_MAIN, fn)))))
        for b in B_SWEEP + (20,):
            rows.append(dict(method='ADVM-b', b=b, N=N_MAIN, fn=fn,
                             rel_err=rel_spec_err(A, advm_b(N_MAIN, phi(N_MAIN, fn), b))))
    return rows


def scaling_rows():
    """ADVM vs EXACT for N = 16, 32, 64 at the paper frequencies (building-block recursion)."""
    rows = []
    for N in (16, 32, 64):
        inner = inner_beams(N)
        for fn in PAPER_FNS:
            AFa = array_factor(beam_matrix('ADVM', N, fn), fn, U_GRID)
            AFe = array_factor(beam_matrix('EXACT', N, fn), fn, U_GRID)
            upa, _, ia = peaks(AFa, U_GRID)
            _, _, ie = peaks(AFe, U_GRID)
            perr = np.abs(u_to_deg(upa) - u_to_deg(u_target_dvm(N)))[inner]
            dsll = (sll_db(AFa, ia) - sll_db(AFe, ie))[inner]
            rows.append(dict(N=N, fn=fn, max_pattern_err_db=pattern_error_db(AFa, AFe)[inner].max(),
                             max_point_err_deg=perr.max(), worst_sll_delta_db=np.nanmax(dsll),
                             rel_spec_err=rel_spec_err(direct(N, phi(N, fn)), advm(N, phi(N, fn)))))
    return rows


def main():
    np.random.seed(SEED)   # nothing below is random; seeded anyway per project rules
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    (RES / 'figs').mkdir(exist_ok=True)

    beams = [r for m, b in METHODS for r in beam_rows(m, b, N_MAIN, FNS)]
    write_csv(RES / 'beams.csv', beams)
    summ = summarize(beams, *BAND, 'course_5-10GHz') + summarize(beams, 0.1, 1.0, 'full_0.1-1')
    write_csv(RES / 'summary_methods.csv', summ)

    course = [s for s in summ if s['band'] == 'course_5-10GHz']
    passing = [s['b'] for s in course if s['method'] == 'ADVM-b' and s['meets_criteria']]
    write_csv(RES / 'bstar.csv', [dict(N=N_MAIN, band='course_5-10GHz', b_star=min(passing) if passing else 'none',
                                       b_values_tested=' '.join(map(str, B_SWEEP)))])
    write_csv(RES / 'complexity.csv', complexity_rows())
    write_csv(RES / 'specnorm.csv', specnorm_rows())
    write_csv(RES / 'scaling.csv', scaling_rows())
    print(f'CSVs done in {time.time() - t0:.1f}s')

    import make_figs
    import make_tex
    make_figs.main()
    make_tex.main()
    print(f'run_all finished in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
