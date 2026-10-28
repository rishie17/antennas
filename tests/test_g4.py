"""G4: physics. EXACT beams are squint-free; PS follows the analytic squint law."""
import numpy as np

from dvm.array import array_factor, u_target_dvm, u_target_ps
from dvm.methods import beam_matrix
from dvm.metrics import peaks, u_to_deg
from dvm.params import SWEEP_FNS, U_GRID, inner_beams

N = 16
INNER = inner_beams(N)


def test_exact_is_squint_free():
    for fn in SWEEP_FNS:
        up, _, _ = peaks(array_factor(beam_matrix('EXACT', N, fn), fn, U_GRID), U_GRID)
        err = np.abs(u_to_deg(up) - u_to_deg(u_target_dvm(N)))[INNER]
        assert err.max() < 0.1, (fn, err.max())


def test_ps_follows_analytic_squint():
    for fn in SWEEP_FNS:
        ut = u_target_ps(N, fn)
        ok = INNER & (np.abs(ut) <= 0.95)
        up, _, _ = peaks(array_factor(beam_matrix('PS', N, fn), fn, U_GRID), U_GRID)
        assert np.abs(u_to_deg(up) - u_to_deg(ut))[ok].max() < 0.1, fn


def test_ps_equals_exact_at_fmax():
    assert np.allclose(beam_matrix('PS', N, 1.0), beam_matrix('EXACT', N, 1.0), atol=1e-10)
