"""G2: Appendix transcription. S4 S3 S2 S1 S0 == F^_32 (eq. 14) exactly."""
import numpy as np

from dvm.adft import F_HAT_32, F_HAT_32_RULE, S_FACTORS

ADFT_SET = {1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j}


def test_product_equals_eq14():
    P = S_FACTORS[4] @ S_FACTORS[3] @ S_FACTORS[2] @ S_FACTORS[1] @ S_FACTORS[0]
    assert np.array_equal(P, F_HAT_32)


def test_entries_multiplierless():
    assert set(F_HAT_32.ravel().tolist()) <= ADFT_SET
    for S in S_FACTORS:
        assert set(S.ravel().tolist()) <= {0, 1, -1, 1j, -1j}


def test_eq14_follows_kn_mod_32_rule():
    assert np.array_equal(F_HAT_32, F_HAT_32_RULE)


def test_adft_is_close_to_dft():
    F = np.exp(-2j * np.pi * np.outer(np.arange(32), np.arange(32)) / 32)
    # every entry points within 45 degrees of the true twiddle
    assert np.all(np.abs(np.angle(F_HAT_32 / F)) <= np.pi / 4 + 1e-12)
