"""ULA steering vectors and array factor.

Sign convention: a plane wave from angle theta (u = sin theta) reaches element l
EARLIER by l*d*u/c, i.e. x_l = exp(+j*2*pi*f*l*d*u/c). With d = c/(2 f_max) the
per-element phase is pi*fn*l*u. With the pre-delay below, DVM beam k' = k - N/2
then peaks at u = +2k'/N for every frequency (verified in tests/test_g4.py).
"""
import numpy as np


def steering(N, fn, u):
    """Element signals for a unit plane wave from each direction u: shape (N, len(u)).
    Phase at element l = 2*pi*f * (l*d*u/c); with d = c/(2 f_max) this is pi*fn*l*u."""
    l = np.arange(N)[:, None]
    return np.exp(1j * np.pi * fn * l * np.asarray(u)[None, :])


def predelay(N, ph):
    """diag(alpha^(-N/2 * l)): centers the beams symmetrically about broadside.
    Anti-causal; realised as a common latency in hardware.

    Why: the raw DVM beam k points at u = 2k/N for k = 0..N-1, i.e. u in [0, 2) -- half of
    that is outside visible space. Multiplying column l by alpha^(-N l/2) turns row k into
    alpha^((k - N/2) l), so beam k' = k - N/2 points at u = 2k'/N in [-1, 1)."""
    return np.exp(1j * ph * (N / 2) * np.arange(N))


def array_factor(B, fn, u):
    """|B @ x(fn, u)| for every beam (rows of B): shape (N_beams, len(u)).
    Row r = beam pattern of beam r: how strongly it responds to a wave from each direction."""
    return np.abs(B @ steering(B.shape[1], fn, u))


def u_target_dvm(N):
    return 2 * (np.arange(N) - N // 2) / N


def u_target_ps(N, fn):
    """Where a phase-shifter beam ends up at frequency fn: a fixed phase 2k'/N*pi*l must equal
    pi*fn*l*u, so u = (2k'/N)/fn -- the beam swings outward as f drops. That's squint."""
    return u_target_dvm(N) / fn
