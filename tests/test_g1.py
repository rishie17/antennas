"""G1: fast factored DVM (EXACT) == brute force (DIRECT)."""
import numpy as np
import pytest

from dvm.dvm_direct import direct
from dvm.dvm_exact import exact
from dvm.params import SEED


@pytest.mark.parametrize('N', [16, 32, 64])
def test_exact_equals_direct(N):
    rng = np.random.default_rng(SEED)
    for ph in rng.uniform(0, 2 * np.pi, 50):          # 50 random alpha on the unit circle
        A, E = direct(N, ph), exact(N, ph)
        assert np.linalg.norm(E - A) / np.linalg.norm(A) < 1e-10
