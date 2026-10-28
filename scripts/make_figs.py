"""Figures F1-F6 -> results/figs/*.png (300 dpi) + *.pdf, 3.5 in wide (IEEE column)."""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dvm.array import array_factor                                    # noqa: E402
from dvm.io import col, read_csv, select                              # noqa: E402
from dvm.methods import beam_matrix                                   # noqa: E402
from dvm.params import (B_SWEEP, BAND, F_MAX_PAPER, N_MAIN, PAPER_FNS, POINT_ERR_DEG,  # noqa: E402
                        SLL_DEGRADE_DB, SQUINT_FREE_DEG, U_GRID)

RES = ROOT / 'results'
FIGS = RES / 'figs'
W = 3.5
# Fixed method identity across every figure (colour + line style, never colour alone).
STYLE = {'EXACT': dict(color='#2a78d6', ls='-'), 'PS': dict(color='#eb6834', ls='--'),
         'ADVM': dict(color='#1baf7a', ls=':'), 'DIRECT': dict(color='#7a7a74', ls='-')}
INK, MUTED = '#2b2b29', '#8a8a84'

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6, 'lines.linewidth': 1.0,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def band_shade(ax):
    ax.axvspan(*BAND, color='#f1f0ec', zorder=0, lw=0)


def f1_peak_vs_freq(beams):
    fig, ax = plt.subplots(figsize=(W, 2.6))
    band_shade(ax)
    for k in (-6, -4, -2, 0, 2, 4, 6):
        for m in ('PS', 'EXACT'):
            R = select(beams, method=m, N=N_MAIN, kprime=k)
            y = col(R, 'theta_peak')
            if m == 'PS':      # stop once the squinted beam has left visible space (|sin| > 1)
                y = np.where(np.isnan(col(R, 'theta_ps_analytic')), np.nan, y)
            ax.plot(col(R, 'fn'), y, **STYLE[m], label=m if k == 0 else None)
        R = select(beams, method='ADVM', N=N_MAIN, kprime=k)[::3]
        ax.plot(col(R, 'fn'), col(R, 'theta_peak'), 'o', ms=2.2, color=STYLE['ADVM']['color'],
                mfc='none', mew=0.7, label='ADVM (paper)' if k == 0 else None)
        ax.annotate(f"k'={k:+d}" if k else "k'=0", (1.005, col(select(beams, method='EXACT', N=N_MAIN, kprime=k, fn=1.0), 'theta_peak')[0]),
                    fontsize=5.5, color=INK, va='center', annotation_clip=False)
    ax.set(xlabel=r'$f/f_{\max}$  (shaded: 5–10 GHz band at $f_{\max}$=10 GHz)',
           ylabel='beam peak angle (deg)', xlim=(0.1, 1.0), ylim=(-92, 92))
    ax.legend(loc='lower left', ncol=3, frameon=False)
    save(fig, 'F1_peak_vs_freq')


def _norm_db(AF):
    return 20 * np.log10(AF / AF.max(axis=1, keepdims=True) + 1e-12)


def f2_paper_fig3():
    fig, axs = plt.subplots(len(PAPER_FNS), 2, figsize=(W, 6.0), sharex=True)
    th = np.degrees(np.arcsin(U_GRID))
    for r, fn in enumerate(PAPER_FNS):
        Ae = array_factor(beam_matrix('EXACT', N_MAIN, fn), fn, U_GRID)
        Aa = array_factor(beam_matrix('ADVM', N_MAIN, fn), fn, U_GRID)
        ax = axs[r, 0]
        ax.plot(th, _norm_db(Ae).T, color=STYLE['EXACT']['color'], lw=0.5)
        ax.plot(th, _norm_db(Aa).T, color=STYLE['ADVM']['color'], lw=0.5, ls='--')
        ax.set(ylim=(-30, 1), ylabel='|AF| (dB)')
        ax.set_title(f'{fn * F_MAX_PAPER / 1e6:.0f} MHz  (f/fmax={fn:g})', loc='left')
        err = np.abs(Aa / Aa.max(1, keepdims=True) - Ae / Ae.max(1, keepdims=True))
        axs[r, 1].plot(th, 20 * np.log10(err.T + 1e-12), color=STYLE['ADVM']['color'], lw=0.4)
        axs[r, 1].set(ylim=(-50, 0), ylabel='error (dB)')
    axs[0, 0].plot([], [], color=STYLE['EXACT']['color'], label='EXACT')
    axs[0, 0].plot([], [], color=STYLE['ADVM']['color'], ls='--', label='ADVM')
    axs[0, 0].legend(loc='lower left', frameon=False, ncol=2)
    for ax in axs[-1]:
        ax.set(xlabel='angle (deg)', xlim=(-90, 90), xticks=(-90, -45, 0, 45, 90))
    axs[0, 1].set_title('|norm. ADVM − norm. EXACT|', loc='left')
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, 'F2_paper_fig3_replication')


def f3_heatmap(k=4):
    fns = np.linspace(0.1, 1.0, 91)
    u = U_GRID[::5]
    idx = k + N_MAIN // 2
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), sharey=True)
    for ax, m in zip(axs, ('PS', 'EXACT')):
        Z = np.array([array_factor(beam_matrix(m, N_MAIN, fn)[idx:idx + 1], fn, u)[0] for fn in fns])
        im = ax.pcolormesh(u, fns, 20 * np.log10(Z / N_MAIN + 1e-12), cmap='Blues', vmin=-30, vmax=0,
                           shading='auto', rasterized=True)
        ax.axvline(2 * k / N_MAIN, color=INK, lw=0.6, ls=':')
        ax.set(title=f"{m}, beam k'={k}", xlabel=r'$\sin\theta$')
        ax.grid(False)
    axs[0].set_ylabel(r'$f/f_{\max}$')
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02)
    cb.set_label('|AF| / N (dB)')
    save(fig, 'F3_heatmap_PS_vs_EXACT')


def f4_bsweep(summ, spec, cplx, bstar):
    S = [s for s in select(summ, band='course_5-10GHz') if s['method'] == 'ADVM-b']
    adv = select(summ, band='course_5-10GHz', method='ADVM')[0]
    bs = col(S, 'b')
    fig, axs = plt.subplots(4, 1, figsize=(W, 5.2), sharex=True)
    kw = dict(color=STYLE['EXACT']['color'], marker='o', ms=3)
    pa = dict(color=STYLE['ADVM']['color'], ls='--', lw=0.9)

    axs[0].plot(bs, col(S, 'max_point_err_deg'), **kw, label='max pointing error')
    axs[0].plot(bs, col(S, 'squint_spread_deg'), color=INK, marker='s', ms=3, ls='-', lw=0.8,
                label='squint spread')
    axs[0].axhline(adv['max_point_err_deg'], **pa, label='paper ADVM (pointing)')
    axs[0].axhline(POINT_ERR_DEG, color=MUTED, lw=0.6, ls=':')
    axs[0].set(ylabel='deg', yscale='symlog', ylim=(0, 5))
    axs[0].set_yscale('symlog', linthresh=0.1)
    axs[0].legend(frameon=False, loc='center right')

    axs[1].plot(bs, col(S, 'worst_sll_delta_db'), **kw, label='worst SLL degradation vs EXACT')
    axs[1].axhline(adv['worst_sll_delta_db'], **pa, label='paper ADVM')
    axs[1].axhline(SLL_DEGRADE_DB, color=MUTED, lw=0.6, ls=':')
    axs[1].set(ylabel='dB')
    axs[1].legend(frameon=False, loc='upper right')

    for b in bs:
        e = col(select(spec, method='ADVM-b', b=b), 'rel_err')
        axs[2].plot(b, max(e.max(), 1e-16), **kw)
    axs[2].axhline(col(select(spec, method='ADVM'), 'rel_err').max(), **pa)
    axs[2].set(ylabel='rel. spectral\nerror (max)', yscale='log', ylim=(1e-16, 2))

    C = select(cplx, method='ADVM-b')
    axs[3].plot(col(C, 'b'), col(C, 'gain_delay'), **kw)
    axs[3].axhline(select(cplx, method='ADVM', N=N_MAIN)[0]['gain_delay'], **pa)
    axs[3].axhline(select(cplx, method='EXACT', N=N_MAIN)[0]['gain_delay'], color=INK, lw=0.6, ls='-.')
    axs[3].text(2, select(cplx, method='EXACT', N=N_MAIN)[0]['gain_delay'] + 3, 'EXACT', fontsize=5.5)
    axs[3].text(2, select(cplx, method='ADVM', N=N_MAIN)[0]['gain_delay'] + 3, 'paper ADVM', fontsize=5.5,
                color=STYLE['ADVM']['color'])
    axs[3].set(ylabel='gain-delay\nblocks', xlabel='b  (twiddles rounded to 2^b-PSK)', xticks=B_SWEEP)
    if bstar != 'none':
        for ax in axs:
            ax.axvline(bstar, color=MUTED, lw=0.6)
        axs[3].text(bstar + 0.1, 75, f'b*={int(bstar)}', fontsize=6)
    fig.tight_layout(h_pad=0.4)
    save(fig, 'F4_bsweep')


def f5_complexity(cplx):
    fig, ax = plt.subplots(figsize=(W, 2.6))
    Ns = np.array(sorted({r['N'] for r in select(cplx, method='DIRECT')}))
    for m, lab in (('DIRECT', 'DIRECT'), ('EXACT', 'EXACT (radix-2 DVM)'), ('ADVM', 'ADVM (paper)')):
        R = select(cplx, method=m)
        ax.loglog(col(R, 'N'), col(R, 'gain_delay'), marker='o', ms=3, color=STYLE[m]['color'],
                  ls=STYLE[m]['ls'] if m != 'ADVM' else '-', label=lab)
    adv0 = select(cplx, method='ADVM', N=16)[0]['gain_delay']
    for ref, lab, scale in ((Ns ** 2, r'$N^2$', (16 - 1) ** 2 / 256), (Ns * np.log2(Ns), r'$N\log_2N$', 130 / 64),
                            (Ns.astype(float), r'$N$', adv0 / 16)):
        ax.loglog(Ns, ref * scale * 0.55, color=MUTED, lw=0.6, ls=':')
        ax.text(Ns[-1] * 1.08, ref[-1] * scale * 0.55, lab, fontsize=6, color=MUTED, va='center')
    ax.set(xlabel='N (beams = elements)', ylabel='gain-delay blocks', xticks=Ns)
    ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.minorticks_off()
    ax.legend(frameon=False, loc='upper left')
    save(fig, 'F5_complexity')


def f6_sll(beams):
    fig, ax = plt.subplots(figsize=(W, 2.2))
    band_shade(ax)
    for m in ('PS', 'EXACT', 'ADVM'):
        R = [r for r in select(beams, method=m, N=N_MAIN) if r['inner']]
        fns = np.unique(col(R, 'fn'))
        worst = [np.nanmax(col(select(R, fn=f), 'sll_db')) for f in fns]
        ax.plot(fns, worst, **STYLE[m], label=m if m != 'ADVM' else 'ADVM (paper)')
    ax.set(xlabel=r'$f/f_{\max}$', ylabel="worst SLL over |k'|≤6 (dB)", xlim=(0.1, 1))
    ax.legend(frameon=False, ncol=3, loc='upper right', bbox_to_anchor=(1, 0.82))
    save(fig, 'F6_sll_vs_freq')


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    beams = read_csv(RES / 'beams.csv')
    summ = read_csv(RES / 'summary_methods.csv')
    spec = read_csv(RES / 'specnorm.csv')
    cplx = read_csv(RES / 'complexity.csv')
    bstar = read_csv(RES / 'bstar.csv')[0]['b_star']
    f1_peak_vs_freq(beams)
    f2_paper_fig3()
    f3_heatmap()
    f4_bsweep(summ, spec, cplx, bstar)
    f5_complexity(cplx)
    f6_sll(beams)


if __name__ == '__main__':
    main()
