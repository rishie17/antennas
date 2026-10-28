"""ULA steering vectors and array factor.

Sign convention: a plane wave from angle theta (u = sin theta) reaches element l
EARLIER by l*d*u/c, i.e. x_l = exp(+j*2*pi*f*l*d*u/c). With d = c/(2 f_max) the
per-element phase is pi*fn*l*u. With the pre-delay below, DVM beam k' = k - N/2
then peaks at u = +2k'/N for every frequency (verified in tests/test_g4.py).
"""
import numpy as np


def steering(N, fn, u):
    l = np.arange(N)[:, None]
    return np.exp(1j * np.pi * fn * l * np.asarray(u)[None, :])


def predelay(N, ph):
    """diag(alpha^(-N/2 * l)): centers the beams symmetrically about broadside.
    Anti-causal; realised as a common latency in hardware."""
    return np.exp(1j * ph * (N / 2) * np.arange(N))


def array_factor(B, fn, u):
    """|B @ x(fn, u)| for every beam (rows of B): shape (N_beams, len(u))."""
    return np.abs(B @ steering(B.shape[1], fn, u))


def u_target_dvm(N):
    return 2 * (np.arange(N) - N // 2) / N


def u_target_ps(N, fn):
    return u_target_dvm(N) / fn
