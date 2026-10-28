"""G5: ADVM-b (our extension) converges to EXACT as b grows; < 1e-10 for b >= 20."""
import numpy as np
import pytest

from dvm.dvm_approx import advm_b
from dvm.dvm_direct import direct
from dvm.params import SEED


def err(N, ph, b):
    A = direct(N, ph)
    return np.linalg.norm(advm_b(N, ph, b) - A, 2) / np.linalg.norm(A, 2)


@pytest.mark.parametrize('N', [16, 64])
def test_converges(N):
    rng = np.random.default_rng(SEED)
    for ph in rng.uniform(0, 2 * np.pi / N, 5):          # alpha over the physical band
        e = [err(N, ph, b) for b in range(2, 25)]
        assert all(e[i + 1] <= e[i] * (1 + 1e-6) + 1e-13 for i in range(len(e) - 1)), e
        assert max(e[18:]) < 1e-10                          # b >= 20
        # twiddles of an M-point DFT lie exactly on the 2^b grid once 2^b >= M = 2N
        assert e[int(np.log2(2 * N)) - 2] < 1e-12
