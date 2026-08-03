"""Uniform cubic B-splines for trajectories: R^3 position + cumulative SO(3).

Formulation follows Lovegrove/Sommer (the representation used by Kalibr and
Basalt): position is a standard uniform cubic B-spline; rotation is the
cumulative B-spline on SO(3),

    R(u) = C_i * Exp(l1(u) W_i) * Exp(l2(u) W_{i+1}) * Exp(l3(u) W_{i+2}),
    W_k = Log(C_k^-1 C_{k+1}),

with l(u) the cumulative basis. Segment i covers t in [t0 + i dt, t0 + (i+1) dt)
and uses control points i..i+3; valid evaluation range is
[t0, t0 + (K-3) dt) for K control points.

Fitting: position is linear least squares (sparse, with a mild second-
difference regularizer); rotation is scipy least_squares on Log residuals with
control points parameterized as right-multiplied rotation vectors around the
current estimate, exploiting Jacobian sparsity.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

# p(u) = [1 u u^2 u^3] @ M4 @ [c_i, c_i+1, c_i+2, c_i+3]
M4 = np.array([
    [1, 4, 1, 0],
    [-3, 0, 3, 0],
    [3, -6, 3, 0],
    [-1, 3, -3, 1],
]) / 6.0

# cumulative basis: l_j(u) = [1 u u^2 u^3] @ C4[:, j]  (j = 1..3 used)
C4 = np.array([
    [6, 5, 1, 0],
    [0, 3, 3, 0],
    [0, -3, 3, 0],
    [0, 1, -2, 1],
]) / 6.0


def _locate(t, t0, dt, K):
    """Segment index and local coordinate for times t (vectorized)."""
    s = (np.asarray(t) - t0) / dt
    i = np.clip(np.floor(s).astype(int), 0, K - 4)
    u = np.clip(s - i, 0.0, 1.0)
    return i, u


def _u_powers(u, deriv=0):
    u = np.asarray(u)
    if deriv == 0:
        return np.stack([np.ones_like(u), u, u * u, u ** 3], axis=-1)
    if deriv == 1:
        return np.stack([np.zeros_like(u), np.ones_like(u), 2 * u, 3 * u * u],
                        axis=-1)
    if deriv == 2:
        return np.stack([np.zeros_like(u), np.zeros_like(u),
                         2 * np.ones_like(u), 6 * u], axis=-1)
    raise ValueError(deriv)


class PosSpline:
    def __init__(self, ctrl: np.ndarray, t0: float, dt: float):
        self.ctrl = np.asarray(ctrl, float)  # (K, 3)
        self.t0, self.dt = float(t0), float(dt)

    @property
    def K(self):
        return len(self.ctrl)

    def _eval(self, t, deriv=0):
        i, u = _locate(t, self.t0, self.dt, self.K)
        B = _u_powers(u, deriv) @ M4  # (N, 4) basis weights
        idx = i[:, None] + np.arange(4)[None, :]
        out = np.einsum("nj,njk->nk", B, self.ctrl[idx])
        return out / self.dt ** deriv

    def pos(self, t):
        return self._eval(t, 0)

    def vel(self, t):
        return self._eval(t, 1)

    def acc(self, t):
        return self._eval(t, 2)


class So3Spline:
    def __init__(self, ctrl: Rotation, t0: float, dt: float):
        self.ctrl = ctrl  # scipy Rotation, length K
        self.t0, self.dt = float(t0), float(dt)

    @property
    def K(self):
        return len(self.ctrl)

    def rot(self, t) -> Rotation:
        t = np.asarray(t)
        i, u = _locate(t, self.t0, self.dt, self.K)
        lam = _u_powers(u) @ C4  # (N, 4); columns 1..3 are l1..l3
        W = (self.ctrl[:-1].inv() * self.ctrl[1:]).as_rotvec()  # (K-1, 3)
        R = self.ctrl[i]
        for j in range(3):
            R = R * Rotation.from_rotvec(lam[:, j + 1, None] * W[i + j])
        return R

    def angvel_body(self, t, h: float = 1e-4):
        """Body-frame angular velocity via symmetric numeric differentiation."""
        t = np.asarray(t)
        lo = self.t0
        hi = self.t0 + (self.K - 3) * self.dt - 2 * h
        tc = np.clip(t, lo + h, hi)
        Ra = self.rot(tc - h)
        Rb = self.rot(tc + h)
        return (Ra.inv() * Rb).as_rotvec() / (2 * h)


def n_ctrl_for_range(t_min: float, t_max: float, dt: float) -> int:
    return int(np.ceil((t_max - t_min) / dt + 1e-9)) + 3


def fit_pos_spline(t, p, dt, reg: float = 1e-6) -> PosSpline:
    """Linear LS fit of a cubic position spline to samples (t, p)."""
    t = np.asarray(t)
    p = np.asarray(p)
    t0 = float(t.min())
    K = n_ctrl_for_range(t0, float(t.max()), dt)
    i, u = _locate(t, t0, dt, K)
    B = _u_powers(u) @ M4
    rows = np.repeat(np.arange(len(t)), 4)
    cols = (i[:, None] + np.arange(4)[None, :]).ravel()
    A = sparse.csr_matrix((B.ravel(), (rows, cols)), shape=(len(t), K))
    # second-difference regularizer keeps barely-observed ctrl points sane
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(K - 2, K))
    H = (A.T @ A + reg * (D.T @ D)).tocsc()
    ctrl = np.column_stack([spsolve(H, A.T @ p[:, k]) for k in range(3)])
    return PosSpline(ctrl, t0, dt)


def fit_so3_spline(t, R_meas: Rotation, dt, verbose: int = 0) -> So3Spline:
    """Fit a cumulative SO(3) spline to rotation samples via sparse LS."""
    from scipy.optimize import least_squares

    t = np.asarray(t)
    t0 = float(t.min())
    K = n_ctrl_for_range(t0, float(t.max()), dt)

    # init each control point from the sample nearest its greville abscissa
    greville = t0 + (np.arange(K) - 1) * dt
    nearest = np.clip(np.searchsorted(t, greville), 0, len(t) - 1)
    ctrl0 = R_meas[nearest]

    i, _ = _locate(t, t0, dt, K)

    def spline_from(x):
        delta = Rotation.from_rotvec(x.reshape(K, 3))
        return So3Spline(ctrl0 * delta, t0, dt)

    def residual(x):
        R_fit = spline_from(x).rot(t)
        return (R_fit.inv() * R_meas).as_rotvec().ravel()

    S = sparse.lil_matrix((3 * len(t), 3 * K), dtype=bool)
    for j in range(4):
        for a in range(3):
            for b in range(3):
                S[np.arange(len(t)) * 3 + a, (i + j) * 3 + b] = True

    res = least_squares(residual, np.zeros(3 * K), jac_sparsity=S,
                        method="trf", verbose=verbose)
    return spline_from(res.x)
