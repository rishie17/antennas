"""Beam metrics on an array-factor matrix AF (rows = beams, columns = u = sin(theta) grid)."""
import numpy as np


def u_to_deg(u):
    return np.degrees(np.arcsin(np.clip(u, -1, 1)))


def peaks(AF, u):
    """Peak u per beam (parabolic interpolation on the dB pattern) and peak amplitude."""
    i = np.argmax(AF, axis=1)
    r = np.arange(AF.shape[0])
    du = u[1] - u[0]
    inside = (i > 0) & (i < AF.shape[1] - 1)
    ic = np.clip(i, 1, AF.shape[1] - 2)
    ym, y0, yp = (20 * np.log10(AF[r, ic + s] + 1e-300) for s in (-1, 0, 1))
    den = ym - 2 * y0 + yp
    delta = np.where(inside & (den < 0), 0.5 * (ym - yp) / np.where(den == 0, -1, den), 0.0)
    return u[i] + delta * du, AF[r, i], i


def sll_db(AF, ipk):
    """Highest sidelobe outside the main lobe (bounded by the first local minima on either
    side of the peak), relative to the peak, in dB. NaN if the main lobe fills the grid."""
    out = np.full(AF.shape[0], np.nan)
    for r, (a, i) in enumerate(zip(AF, ipk)):
        d = np.diff(a)
        left = np.nonzero(d[:i] < 0)[0]            # last descent going left => first null left
        right = np.nonzero(d[i:] > 0)[0]           # first ascent going right => first null right
        lo = left[-1] + 1 if left.size else 0
        hi = i + right[0] if right.size else len(a) - 1
        side = np.r_[a[:lo], a[hi + 1:]]
        if side.size:
            out[r] = 20 * np.log10(side.max() / a[i])
    return out


def pattern_error_db(AF_a, AF_e):
    """Max over angle of |a/max(a) - e/max(e)| per beam, in dB (paper Fig. 3 right column)."""
    na = AF_a / AF_a.max(axis=1, keepdims=True)
    ne = AF_e / AF_e.max(axis=1, keepdims=True)
    return 20 * np.log10(np.abs(na - ne).max(axis=1) + 1e-300)


def rel_spec_err(A, B):
    """min_s ||A - s B||_2 / ||A||_2 over complex s (convex in s; Nelder-Mead from LS start)."""
    from scipy.optimize import minimize
    s0 = np.vdot(B, A) / np.vdot(B, B)
    f = lambda p: np.linalg.norm(A - (p[0] + 1j * p[1]) * B, 2)
    res = minimize(f, [s0.real, s0.imag], method='Nelder-Mead',
                   options=dict(xatol=1e-12, fatol=1e-14, maxiter=2000))
    return min(res.fun, f([s0.real, s0.imag])) / np.linalg.norm(A, 2)
