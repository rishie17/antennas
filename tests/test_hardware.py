"""Hardware mapping: component counts agree with opcount, delay lengths are the analytic ones,
TEM lines are non-dispersive, waveguide squint follows sin(theta) = (2k'/N) * beta*L_tau / (omega*tau)."""
import numpy as np
import pytest

from dvm.array import array_factor, u_target_dvm
from dvm.hardware import (BUILD, FC_WR90, dispersive_beam_matrix, phase_per_tau, sfg, unit_length)
from dvm.metrics import peaks
from dvm.opcount import GENERIC_PH, count
from dvm.params import C0, F_MAX_COURSE, U_GRID, inner_beams, phi, tau


@pytest.mark.parametrize('N', (16, 32, 64))
@pytest.mark.parametrize('m', sorted(BUILD))
def test_counts_match_opcount(m, N):
    c, L = sfg(m, N)
    o = count(BUILD[m](N, GENERIC_PH))
    assert c['combiners'] + 2 * (c['hyb180'] + c['hyb90']) == o['adders']
    assert (c['delay_lines'], c['gain_const'], c['gain_freqdep']) == (o['delays'], o['gains'], o['anticausal'])
    assert len(L) == c['delay_lines']


@pytest.mark.parametrize('N', (16, 32))
def test_delay_lengths_and_butterflies(N):
    k = np.arange(1, N)
    _, Ld = sfg('DIRECT', N)
    ce, Le = sfg('EXACT', N)
    assert Ld.sum() == (N * (N - 1) // 2) ** 2 and Ld.max() == (N - 1) ** 2      # sum_{k,l} k*l
    assert Le.sum() == (k ** 2).sum() and Le.max() == (N - 1) ** 2 / 2         # 2 x sum k^2/2
    M = 2 * N
    assert ce['hyb180'] == 2 * (M // 2) * int(np.log2(M)) and ce['hyb90'] == 0  # radix-2: M/2 per stage


def test_media():
    assert abs(FC_WR90 - 6.557e9) < 1e6
    assert np.isclose(unit_length('air', 16), C0 * tau(16, F_MAX_COURSE))
    f = np.linspace(5e9, 10e9, 11)
    for m in ('air', 'stripline'):
        assert np.allclose(phase_per_tau(m, 16, f), phi(16, f / F_MAX_COURSE))      # TEM: exactly omega*tau
    assert np.isnan(phase_per_tau('WR-90', 16, 6e9)) and np.isfinite(phase_per_tau('WR-90', 16, 7e9))
    assert np.isclose(phase_per_tau('WG-fc4.5', 16, 7.5e9, design='phase'), phi(16, 0.75))  # phase design: exact at f0


@pytest.mark.parametrize('design', ('group', 'phase'))
def test_waveguide_squint_analytic(design):
    N, f = 16, 9e9
    fn = f / F_MAX_COURSE
    ph_line = float(phase_per_tau('WG-fc4.5', N, f, design=design))
    up, _, _ = peaks(array_factor(dispersive_beam_matrix(N, ph_line), fn, U_GRID), U_GRID)
    ua = u_target_dvm(N) * ph_line / phi(N, fn)
    inner = inner_beams(N) & (np.abs(ua) < 0.95)
    assert np.allclose(up[inner], ua[inner], atol=1e-4)
