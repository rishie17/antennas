"""G3: instrumented gain-delay counts reproduce paper Lemma 3.2 / Table 1."""
import pytest

from dvm.dvm_approx import advm_factors
from dvm.dvm_exact import exact_factors
from dvm.opcount import GENERIC_PH, count

LEMMA_32 = {16: 4 * 16 - 2, 32: 6 * 32 - 6, 64: 8 * 64 - 14, 128: 10 * 128 - 30,
            256: 12 * 256 - 62, 512: 14 * 512 - 126, 1024: 16 * 1024 - 254}
TABLE1_SDVM = {16: 130, 32: 322, 64: 770, 128: 1794, 256: 4098, 512: 9218, 1024: 20482}
TABLE3 = {16: (0, 30, 32), 32: (60, 62, 64), 64: (244, 126, 128), 128: (740, 254, 256)}


@pytest.mark.parametrize('N', sorted(LEMMA_32))
def test_advm_lemma(N):
    assert count(advm_factors(N, GENERIC_PH))['gain_delay'] == LEMMA_32[N]


@pytest.mark.parametrize('N', sorted(TABLE3))
def test_advm_table3_split(N):
    c = count(advm_factors(N, GENERIC_PH))
    assert (c['gains'], c['delays'], c['anticausal']) == TABLE3[N]


@pytest.mark.parametrize('N', sorted(TABLE1_SDVM))
def test_exact_table1(N):
    assert count(exact_factors(N, GENERIC_PH))['gain_delay'] == TABLE1_SDVM[N]
