"""
Slope limiting for DG spaces.

    lim = fd.TVBLimiter(V, M=50.0)       # V a DGSpace, or a ProductSpace with DG fields
    lim(u)                               # limit a Function / ProductFunction in place
    c = lim(c)                           # or an array: returns the limited copy
    stepper = fd.SSPRK(Mform, R, dt, limiter=lim)

The Cockburn-Shu TVB minmod limiter, for any degree p and either DG basis. On each element
the mean ubar and the end values are compared with the neighbors' means:

    right = u(x_{j+1/2}^-) - ubar_j,   left = ubar_j - u(x_{j-1/2}^+)
    dp = ubar_{j+1} - ubar_j,          dm = ubar_j - ubar_{j-1}

An element is troubled when the TVB-modified minmod m(a, dp, dm) changes `right` or `left`,
where m(a, ...) = a if |a| <= M h_j^2 and minmod(a, dp, dm) otherwise. On a troubled element
the linear part is replaced by minmod(slope, dp, dm) (in the Legendre scale, slope = c_1)
and every higher mode is dropped; the mean is untouched, so the limiter conserves mass.
M = 0 is the plain minmod limiter (TVDM), which also flattens smooth extrema and costs an
order of accuracy there; M of the size of |u''| near the extrema keeps full order.
At the ends of a non-periodic space the missing difference is left out of the minmod.

fd.SlopeLimiter(V, limiter, reconstruction) replaces the slope by a finite volume slope of the cell
means instead: the limiters minmod, Van Leer, monotonized central and Van Albada, each with the TVD2
or the UNO2 reconstruction, as in Dutykh, Katsaounis and Mitsotakis (J. Comput. Phys. 230, 2011).
"""
from __future__ import annotations

import numpy as np

__all__ = ["TVBLimiter", "SlopeLimiter", "limited_slope", "VertexLimiter", "minmod"]


def minmod(*args):
    """Elementwise minmod: the argument of smallest magnitude when all have one sign, else 0."""
    a = np.asarray(args[0], dtype=np.float64)
    s = np.sign(a)
    m = np.abs(a)
    for b in args[1:]:
        b = np.asarray(b, dtype=np.float64)
        s = np.where(np.sign(b) == s, s, 0.0)
        m = np.minimum(m, np.abs(b))
    return s * m


class _FieldLimiter:
    """The limiter on one DGSpace: nodal <-> modal maps and element widths."""

    def __init__(self, V, M):
        self.V, self.M = V, float(M)
        self.p, self.ne = V.degree, V.nelem
        self.periodic = V.bc.is_periodic
        grid = np.asarray(V.grid, dtype=np.float64)
        self.h2 = np.diff(grid) ** 2
        if V.basis_name == "legendre":
            self.to_modal = self.to_nodal = None
        else:
            xi = np.asarray(V.reference_points(), dtype=np.float64)
            Vand = np.polynomial.legendre.legvander(xi, self.p)     # nodal = Vand @ modal
            self.to_nodal, self.to_modal = Vand, np.linalg.inv(Vand)

    def limit(self, c):
        """Limited copy of the coefficients c (length dim), and the number of troubled elements."""
        p, ne = self.p, self.ne
        C = np.asarray(c, dtype=np.float64).reshape(ne, p + 1)
        if self.to_modal is not None:
            C = C @ self.to_modal.T
        else:
            C = C.copy()
        if p == 0 or ne < 2:
            return (C @ self.to_nodal.T if self.to_nodal is not None else C).ravel(), 0
        mean = C[:, 0]
        if self.periodic:
            dp, dm = np.roll(mean, -1) - mean, mean - np.roll(mean, 1)
        else:
            dp, dm = np.empty(ne), np.empty(ne)
            dp[:-1] = mean[1:] - mean[:-1]
            dm[1:] = mean[1:] - mean[:-1]
            dp[-1], dm[0] = dm[-1], dp[0]            # one-sided at the ends: the missing difference drops out
        right = C.sum(axis=1) - mean                 # P_l(1) = 1
        left = mean - C @ (-1.0) ** np.arange(p + 1)  # P_l(-1) = (-1)^l
        thr = self.M * self.h2

        def tvb(a):
            return np.where(np.abs(a) <= thr, a, minmod(a, dp, dm))

        bad = (tvb(right) != right) | (tvb(left) != left)
        if bad.any():
            C[bad, 1] = minmod(C[bad, 1], dp[bad], dm[bad])
            C[bad, 2:] = 0.0
        if self.to_nodal is not None:
            C = C @ self.to_nodal.T
        return C.ravel(), int(bad.sum())


class TVBLimiter:
    """The Cockburn-Shu TVB minmod limiter on a DGSpace, or field by field on a ProductSpace.

    V: a DGSpace (either basis, any degree), or a ProductSpace; on a ProductSpace every DG
       field is limited component by component and the other fields are left alone.
    M: the TVB constant. 0 gives the minmod (TVDM) limiter. A constant near max |u''| of the
       smooth parts keeps the limiter off at smooth extrema.
    fields: on a ProductSpace, the indices of the fields to limit (default: every DG field).

    lim(u) limits a Function or ProductFunction in place and returns it; lim(c) returns a
    limited copy of an array. lim.troubled is the number of elements limited by the last call."""

    def __init__(self, V, M: float = 0.0, fields=None):
        from .spaces import ProductSpace
        if M < 0:
            raise ValueError("TVBLimiter: M must be >= 0")
        self.V, self.M, self.troubled = V, float(M), 0
        if isinstance(V, ProductSpace):
            idx = [i for i, f in enumerate(V.fields) if getattr(f, "family", "") == "dg"] if fields is None else list(fields)
            for i in idx:
                if getattr(V.fields[i], "family", "") != "dg":
                    raise TypeError(f"TVBLimiter: field {i} of the ProductSpace is not a DGSpace")
            if not idx:
                raise TypeError("TVBLimiter: the ProductSpace has no DG field to limit")
            self._lims = {i: _FieldLimiter(V.fields[i], M) for i in idx}
        else:
            if getattr(V, "family", "") != "dg":
                raise TypeError(f"TVBLimiter: needs a DGSpace (or a ProductSpace of them), got {type(V).__name__}")
            if fields is not None:
                raise TypeError("TVBLimiter: fields= is for a ProductSpace")
            self._lims = None
            self._lim = _FieldLimiter(V, M)

    def limit(self, c) -> np.ndarray:
        """A limited copy of the coefficient vector c."""
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.size != self.V.dim:
            raise ValueError(f"TVBLimiter: need {self.V.dim} coefficients, got {c.size}")
        if self._lims is None:
            out, self.troubled = self._lim.limit(c)
            return out
        parts = [np.array(q, dtype=np.float64) for q in self.V.split(c)]
        self.troubled = 0
        for i, L in self._lims.items():
            parts[i], k = L.limit(parts[i])
            self.troubled += k
        return np.asarray(self.V.gather(parts), dtype=np.float64)

    def __call__(self, u):
        vec = getattr(u, "vector", None)
        if vec is None:
            return self.limit(u)
        vec[:] = self.limit(vec)
        return u

    def __repr__(self):
        return f"TVBLimiter(M={self.M:g}, on {self.V!r})"


# ============================================================================ 1D: finite volume slope limiters (TVD2, UNO2)

_LIMITERS = {"minmod": "minmod", "mm": "minmod",
             "vanleer": "vanleer", "van leer": "vanleer", "vl": "vanleer",
             "mc": "mc", "monotonized central": "mc",
             "vanalbada": "vanalbada", "van albada": "vanalbada", "va": "vanalbada"}
_RECONSTRUCTIONS = ("tvd2", "uno2")


def _phi_slope(limiter, a, b):
    """phi(a/b) b, the limiter applied to a left difference a and a right difference b, elementwise."""
    if limiter == "minmod":                    # phi(r) = max(0, min(1, r))
        return minmod(a, b)
    if limiter == "mc":                        # phi(r) = max(0, min((1 + r)/2, 2, 2r))
        return minmod(2.0 * a, 0.5 * (a + b), 2.0 * b)
    if limiter == "vanleer":                   # phi(r) = (r + |r|)/(1 + |r|)
        den = np.abs(a) + np.abs(b)
        return np.where(den > 0, (a * np.abs(b) + np.abs(a) * b) / np.where(den > 0, den, 1.0), 0.0)
    den = a * a + b * b                        # vanalbada: phi(r) = (r + r^2)/(1 + r^2), as in DKM (2011)
    return np.where(den > 0, a * b * (a + b) / np.where(den > 0, den, 1.0), 0.0)


def limited_slope(limiter, sm, sp, reconstruction="tvd2", Dm=None, D0=None, Dp=None, dm=None, dp=None):
    """The limited slope of one cell, elementwise, from its left and right slopes sm, sp (divided
    differences of the means). limiter: "minmod", "vanleer", "mc" or "vanalbada".

    reconstruction="tvd2":  S = phi(r) sp with r = sm/sp.
    reconstruction="uno2":  S = phi(r) S+ with r = S-/S+, where S+- = s+- -+ (d+-/2) m(D_j, D_{j+-1})
                            are the slopes corrected by the second divided differences Dm, D0, Dp of
                            the cells j-1, j, j+1, and dm, dp the distances from the center of cell j
                            to its neighbors'. With minmod this is UNO2 of Harten and Osher."""
    key = _LIMITERS[str(limiter).lower()]
    sm, sp = np.asarray(sm, dtype=np.float64), np.asarray(sp, dtype=np.float64)
    if reconstruction == "tvd2":
        return _phi_slope(key, sm, sp)
    if reconstruction != "uno2":
        raise ValueError("limited_slope: reconstruction must be 'tvd2' or 'uno2'")
    Splus = sp - 0.5 * dp * minmod(D0, Dp)
    Sminus = sm + 0.5 * dm * minmod(Dm, D0)
    return _phi_slope(key, Sminus, Splus)


class _SlopeFieldLimiter(_FieldLimiter):
    """A finite volume slope limiter on one DGSpace."""

    def __init__(self, V, limiter, reconstruction, M, troubled):
        super().__init__(V, M)
        self.limiter, self.reconstruction, self.troubled_rule = limiter, reconstruction, troubled
        grid = np.asarray(V.grid, dtype=np.float64)
        self.h = np.diff(grid)
        h = self.h
        if self.periodic:
            self.dp = 0.5 * (h + np.roll(h, -1))     # center of cell j to center of cell j+1
            self.dm = 0.5 * (h + np.roll(h, 1))
        else:
            self.dp, self.dm = np.empty_like(h), np.empty_like(h)
            self.dp[:-1] = self.dm[1:] = 0.5 * (h[:-1] + h[1:])
            self.dp[-1], self.dm[0] = self.dm[-1], self.dp[0]

    def limit(self, c):
        p, ne = self.p, self.ne
        C = np.asarray(c, dtype=np.float64).reshape(ne, p + 1)
        C = C @ self.to_modal.T if self.to_modal is not None else C.copy()
        if p == 0 or ne < 2:
            return (C @ self.to_nodal.T if self.to_nodal is not None else C).ravel(), 0
        mean = C[:, 0]
        if self.periodic:
            Dp_, Dm_ = np.roll(mean, -1) - mean, mean - np.roll(mean, 1)
        else:
            Dp_, Dm_ = np.empty(ne), np.empty(ne)
            Dp_[:-1] = Dm_[1:] = mean[1:] - mean[:-1]
            Dp_[-1], Dm_[0] = Dm_[-1], Dp_[0]        # one-sided at the ends, as in TVBLimiter
        if self.troubled_rule == "all":
            bad = np.ones(ne, dtype=bool)
        else:                                        # the Cockburn-Shu test with the TVB constant M
            right = C.sum(axis=1) - mean
            left = mean - C @ (-1.0) ** np.arange(p + 1)
            thr = self.M * self.h2
            tvb = lambda a: np.where(np.abs(a) <= thr, a, minmod(a, Dp_, Dm_))
            bad = (tvb(right) != right) | (tvb(left) != left)
        if bad.any():
            sp, sm = Dp_ / self.dp, Dm_ / self.dm    # right and left slopes
            s = limited_slope(self.limiter, sm, sp)
            if self.reconstruction == "uno2":
                D = 2.0 * (sp - sm) / (self.dm + self.dp)          # second divided differences
                if self.periodic:
                    s = limited_slope(self.limiter, sm, sp, "uno2", np.roll(D, 1), D, np.roll(D, -1),
                                      self.dm, self.dp)
                elif ne >= 5:                                      # TVD2 in the two cells next to each end
                    j = slice(2, ne - 2)
                    s[j] = limited_slope(self.limiter, sm[j], sp[j], "uno2", D[1:ne - 3], D[j], D[3:ne - 1],
                                         self.dm[j], self.dp[j])
            C[bad, 1] = 0.5 * self.h[bad] * s[bad]                 # the Legendre slope, P_1(1) = 1
            C[bad, 2:] = 0.0
        if self.to_nodal is not None:
            C = C @ self.to_nodal.T
        return C.ravel(), int(bad.sum())


class SlopeLimiter(TVBLimiter):
    """Finite volume slope limiters for DG spaces in 1D: the limiters minmod, Van Leer, monotonized
    central and Van Albada, each with the TVD2 or the UNO2 reconstruction.

        lim = fd.SlopeLimiter(V, "vanleer", "uno2")   # V a DGSpace, or a ProductSpace with DG fields
        rk = fd.SSPRK(M, R, dt, limiter=lim)

    On every element (or only on the troubled ones) the mean is kept, the slope is replaced by the
    limited slope S_j of the cell means and the higher modes are dropped, so the end values are the
    reconstructed interface values of Dutykh, Katsaounis and Mitsotakis (J. Comput. Phys. 230, 2011,
    eqs. 3.10 and 3.11),  u(x_{j+1/2}^-) = W_j + S_j h_j/2,  u(x_{j-1/2}^+) = W_j - S_j h_j/2,  with

        TVD2:  S_j = phi(r_j) s_j^+,  r_j = s_j^- / s_j^+,
        UNO2:  S_j = phi(r_j) S_j^+,  r_j = S_j^- / S_j^+,  S_j^+- = s_j^+- -+ (d_j^+-/2) m(D_j, D_{j+-1}),

        minmod     phi(r) = max(0, min(1, r))
        vanleer    phi(r) = (r + |r|)/(1 + |r|)
        mc         phi(r) = max(0, min((1 + r)/2, 2, 2r))
        vanalbada  phi(r) = (r + r^2)/(1 + r^2)

    where s_j^+- are the divided differences of the means to the right and to the left, d_j^+- the
    distances between the cell centers, D_j the second divided differences and m the minmod
    function. UNO2 with minmod is the reconstruction of Harten and Osher. On a uniform grid these
    are the formulas of the paper. With degree 1 and the default troubled="all", the means evolve
    exactly as in the finite volume scheme of the paper with the same numerical flux: TVD2 is first
    order at smooth extrema, UNO2 second order.

    V:              a DGSpace (either basis, any degree >= 1), or a ProductSpace whose DG fields are
                    limited one by one.
    limiter:        "minmod" ("mm"), "vanleer" ("vl"), "mc" or "vanalbada" ("va").
    reconstruction: "tvd2" (default) or "uno2".
    troubled:       "all" (default) replaces the slope of every element. "tvb" replaces it only on
                    the elements that the Cockburn-Shu test of TVBLimiter flags, and keeps the DG
                    solution elsewhere, which keeps the full order of a degree p >= 2 away from
                    discontinuities. The order at smooth extrema is then set by M.
    M:              the TVB constant of that test (troubled="tvb" only).
    fields:         on a ProductSpace, the indices of the fields to limit (default: every DG field).

    The means never change, so the limiter conserves mass. Van Albada, as written in the paper,
    is not zero where the slopes change sign, so it is not TVD. At the ends of a non-periodic
    space the missing difference is replaced by the other one, and UNO2 falls back to TVD2 in the
    two cells next to each end. lim(u), limit(c) and troubled are as for TVBLimiter."""

    def __init__(self, V, limiter: str = "minmod", reconstruction: str = "tvd2", M: float = 0.0,
                 fields=None, troubled: str = "all"):
        key = str(limiter).lower()
        if key not in _LIMITERS:
            raise ValueError(f"SlopeLimiter: unknown limiter {limiter!r}, use minmod, vanleer, mc or vanalbada")
        rec = str(reconstruction).lower()
        if rec not in _RECONSTRUCTIONS:
            raise ValueError(f"SlopeLimiter: unknown reconstruction {reconstruction!r}, use tvd2 or uno2")
        if troubled not in ("tvb", "all"):
            raise ValueError("SlopeLimiter: troubled must be 'tvb' or 'all'")
        self.limiter, self.reconstruction, self.troubled_rule = _LIMITERS[key], rec, troubled
        super().__init__(V, M, fields)
        make = lambda W: _SlopeFieldLimiter(W, self.limiter, rec, M, troubled)
        if self._lims is None:
            self._lim = make(V)
        else:
            self._lims = {i: make(V.fields[i]) for i in self._lims}

    def __repr__(self):
        return (f"SlopeLimiter({self.limiter!r}, {self.reconstruction!r}, M={self.M:g}, "
                f"troubled={self.troubled_rule!r}, on {self.V!r})")


# ============================================================================ 2D: the vertex-based limiter
class _VertexFieldLimiter:
    """The vertex-based limiter on one DGSpace2D."""

    def __init__(self, V):
        from .elements2d import _dg_tables
        self.V = V
        self.k, self.nc, self.n = V.degree, V.ncells, V.nloc
        T = _dg_tables(V)
        self.mean, self.Pi, self.E = T["mean"], T["Pi"], T["E"]
        rep = np.asarray(getattr(V, "vertex_representatives", np.arange(V.mesh.npoints)))
        self.rv = rep[np.asarray(V.cell_vertices())]                 # (nc, 3) vertex classes (periodic sides joined)
        self.nv = int(self.rv.max()) + 1

    def limit(self, c):
        C = np.asarray(c, dtype=np.float64).reshape(self.nc, self.n)
        if self.k == 0:
            return C.ravel().copy(), 0
        ubar = C @ self.mean                                         # cell means
        Uv = C @ self.Pi.T                                           # vertex values of the P1 part
        umax = np.full(self.nv, -np.inf)
        umin = np.full(self.nv, np.inf)
        for j in range(3):
            np.maximum.at(umax, self.rv[:, j], ubar)
            np.minimum.at(umin, self.rv[:, j], ubar)
        d = Uv - ubar[:, None]
        scale = max(np.abs(ubar).max(), np.abs(d).max(), 1e-300)
        tol = 1e-13 * scale
        up = (umax[self.rv] - ubar[:, None])
        lo = (umin[self.rv] - ubar[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(d > tol, np.minimum(1.0, up / d), np.where(d < -tol, np.minimum(1.0, lo / d), 1.0))
        alpha = np.clip(a.min(axis=1), 0.0, 1.0)
        bad = alpha < 1.0 - 1e-12
        out = C.copy()
        if bad.any():
            ub = ubar[bad][:, None]
            out[bad] = ub + alpha[bad][:, None] * ((Uv[bad] - ub) @ self.E.T)
        self.alpha = alpha
        return out.ravel(), int(bad.sum())


class VertexLimiter:
    """The vertex-based slope limiter (Kuzmin 2010) on a DGSpace2D, or field by field on a 2D
    ProductSpace of DG fields.

        lim = fd.VertexLimiter(V)                    # V = fd.DGSpace(mesh, k)
        rk = fd.SSPRK(M, R, dt, limiter=lim)

    On each triangle the linear part of the solution (its L2 projection onto P1) is scaled about
    the cell mean, by the largest alpha in [0, 1] that keeps its three vertex values between the
    smallest and the largest cell mean of the triangles around each vertex (vertices joined across
    periodic sides count as one).  For P1 this is Kuzmin's limiter.  For k >= 2 a triangle with
    alpha < 1 is replaced by its limited linear part, and the other triangles keep every mode.  The
    mean is never changed, so the limiter conserves mass, and it needs no parameter.  Like every
    such limiter it also clips smooth extrema, where the error is then of first order.

    V:      a DGSpace2D, or a ProductSpace2D whose DG fields are limited one by one.
    fields: on a ProductSpace, the indices of the fields to limit (default: every DG field).

    lim(u) limits a Function or ProductFunction in place and returns it, lim(c) returns a limited
    copy of an array; lim.troubled is the number of triangles limited by the last call."""

    def __init__(self, V, fields=None):
        from .elements2d import DGSpace2D
        self.V, self.troubled = V, 0
        if hasattr(V, "fields") and getattr(V, "tdim", 1) == 2:
            idx = [i for i, f in enumerate(V.fields) if isinstance(f, DGSpace2D)] if fields is None else list(fields)
            for i in idx:
                if not isinstance(V.fields[i], DGSpace2D):
                    raise TypeError(f"VertexLimiter: field {i} of the ProductSpace is not a DGSpace2D")
            if not idx:
                raise TypeError("VertexLimiter: the ProductSpace has no DG field to limit")
            self._lims = {i: _VertexFieldLimiter(V.fields[i]) for i in idx}
        else:
            if not isinstance(V, DGSpace2D):
                raise TypeError(f"VertexLimiter: needs a DGSpace2D (or a 2D ProductSpace of them), got {type(V).__name__}")
            if fields is not None:
                raise TypeError("VertexLimiter: fields= is for a ProductSpace")
            self._lims = None
            self._lim = _VertexFieldLimiter(V)

    def limit(self, c) -> np.ndarray:
        """A limited copy of the coefficient vector c."""
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.size != self.V.dim:
            raise ValueError(f"VertexLimiter: need {self.V.dim} coefficients, got {c.size}")
        if self._lims is None:
            out, self.troubled = self._lim.limit(c)
            return out
        parts = [np.array(q, dtype=np.float64) for q in self.V.split(c)]
        self.troubled = 0
        for i, L in self._lims.items():
            parts[i], k = L.limit(parts[i])
            self.troubled += k
        return np.asarray(self.V.gather(parts), dtype=np.float64)

    def __call__(self, u):
        vec = getattr(u, "vector", None)
        if vec is None:
            return self.limit(u)
        vec[:] = self.limit(vec)
        return u

    def __repr__(self):
        return f"VertexLimiter(on {self.V!r})"
