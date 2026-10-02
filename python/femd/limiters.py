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
"""
from __future__ import annotations

import numpy as np

__all__ = ["TVBLimiter", "VertexLimiter", "minmod"]


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
