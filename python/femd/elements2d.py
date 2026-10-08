"""
femd.elements2d -- vector-valued and broken finite elements on triangles.

    m = fd.rectangle_mesh(0, 0, 1, 1, 16, 16)
    V = fd.FunctionSpace(m, "RT", 2)                 # Raviart-Thomas RT_2, H(div)
    E = fd.FunctionSpace(m, "N1curl", 1)             # Nedelec, first kind, lowest order, H(curl)
    Q = fd.FunctionSpace(m, "DG", 1)                 # broken P_1 (fd.DGSpace(m, 1) is the same)
    W = fd.ProductSpace(V, Q)                        # mixed Poisson / Darcy: RT_k x DG_{k-1}
    (s, u), (t, v) = fd.TrialFunctions(W), fd.TestFunctions(W)
    a = (fd.dot(s, t) + u * fd.div(t) + fd.div(s) * v) * fd.dx

Degree convention (UFL's): RT_k and N1curl_k (k >= 1) lie in P_k^2 and contain P_{k-1}^2, so
RT_1 and N1curl_1 are the lowest-order elements, with one function per edge.  Their test and
trial functions are vectors (two components), on which div(), curl() and dot() act.

Coefficients: on each edge, the normal (RT) or tangential (N1curl) component times the edge
length, at the k Gauss-Legendre points of the edge, for the global orientation of the edge (from
its lower to its higher vertex index; the normal is that direction turned clockwise).  Inside
each triangle, moments against the polynomials of degree k - 2.

Essential conditions: u . n (RT) and u x n (N1curl) are built in by elimination on the sides
named by dirichlet= (or every exterior side with bc="dirichlet"), with data given as a vector
field g(x, y) -> (gx, gy) or, for the normal (tangential) component itself, a scalar g(x, y).

Interpolation is the canonical (commuting) one: edge traces are projected in L^2 onto P_{k-1},
and interior moments taken with a rule exact to degree 2k + 4, so div (curl) of the interpolant
is the L^2 projection of div u (curl u) onto DG_{k-1}, to quadrature accuracy.
"""
from __future__ import annotations

import numpy as np
from . import _femd as _C
from .forms import VectorElementFunction, Function
from .spaces2d import _Lagrange2D, _markers, _points, _arr, _NODES
from ._util import _real_linear

__all__ = ["RTSpace", "N1curlSpace", "DGSpace2D"]


_WHAT = {"value": None, "x": 0, "y": 3, "div": "div", "curl": "curl", "rot": "curl"}


def _vector_values(f, x, y):
    """(n, 2) values of vector data at points: a callable returning a pair (or an (n, 2) / (2, n)
    array), a pair of callables or numbers, or a constant pair."""
    n = x.size
    if callable(f):
        r = f(x, y)
        if isinstance(r, (tuple, list)):
            if len(r) != 2:
                raise ValueError(f"vector data: the callable returned {len(r)} components, not 2")
            return np.column_stack([np.broadcast_to(np.asarray(c, dtype=np.float64), (n,)) for c in r])
        r = np.asarray(r, dtype=np.float64)
        if r.ndim == 2 and r.shape == (2, n):
            return np.ascontiguousarray(r.T)
        if r.ndim == 2 and r.shape == (n, 2):
            return np.ascontiguousarray(r)
        raise ValueError("vector data: the callable must return a pair (fx, fy)")
    if isinstance(f, (tuple, list)) and len(f) == 2:
        cols = []
        for c in f:
            v = c(x, y) if callable(c) else c
            cols.append(np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)))
        return np.column_stack(cols)
    if isinstance(f, (int, float)) and f == 0:
        return np.zeros((n, 2))
    raise TypeError(f"vector data: a callable f(x, y) -> (fx, fy), a pair, or 0; got {f!r}")


def _is_scalar_data(g, xy):
    """Whether boundary data g is scalar (a normal or tangential component) rather than a vector."""
    if isinstance(g, (int, float, np.floating, np.integer)):
        return True
    if isinstance(g, (tuple, list)):
        return False
    if callable(g):
        r = g(xy[:1, 0], xy[:1, 1])
        if isinstance(r, (tuple, list)):
            return False
        r = np.asarray(r)
        return r.ndim <= 1 and r.size <= 1
    raise TypeError(f"boundary data: a number, a pair, or a callable; got {g!r}")


class _VectorElement2D:
    """What RTSpace and N1curlSpace share."""

    tdim = 2
    broken = False
    value_shape = (2,)
    _cname = ""

    def __init__(self, mesh, degree: int, bc="free", dirichlet=None, data=None):
        from .mesh2d import Mesh2D
        if not isinstance(mesh, Mesh2D):
            raise TypeError(f"{type(self).__name__}: mesh must be a triangular Mesh2D, got {type(mesh).__name__} "
                            "(quadrilaterals are not supported yet)")
        if isinstance(dirichlet, dict):
            if data is not None:
                raise TypeError(f"{type(self).__name__}: give the data either in dirichlet={{marker: data}} or in "
                                "data=, not both")
            data, dirichlet = dict(dirichlet), sorted(dirichlet)
        if isinstance(degree, bool) or int(degree) != degree:
            raise TypeError(f"degree must be an integer, got {degree!r}")
        k = int(degree)
        if k < 1:
            raise ValueError(f"{type(self).__name__}: degree must be >= 1 ({self.family}1 is the lowest order), got {k}")
        b = "free" if bc is None else str(bc).lower()
        if b not in ("free", "dirichlet"):
            raise ValueError(f"{type(self).__name__}: bc is 'free' or 'dirichlet' (every exterior side), got {bc!r}")
        marks = sorted(set(_markers(dirichlet, "dirichlet")))
        present = set(np.asarray(mesh.segment_markers).tolist())
        missing = [m for m in marks if m not in present]
        if missing:
            raise ValueError(f"{type(self).__name__}: no side of the mesh carries marker(s) {missing}; "
                             f"the markers present are {sorted(present)}")
        super().__init__(mesh._m, self._family_enum, k, marks, b == "dirichlet")
        self.mesh = mesh
        self._bc = b
        self._pattern = {}
        self._data = None
        if data is not None:
            if len(self.constrained()) == 0:
                raise ValueError(f"{type(self).__name__}: data= gives boundary data, and this space has no "
                                 "essential side; name them with dirichlet=[markers] or bc='dirichlet'")
            self._lift(data)                         # check it now
            self._data = data

    # ---- description ---------------------------------------------------------------------
    @property
    def boundary_data(self):
        """The essential data given with the space (dirichlet={marker: data} or data=), or None."""
        return getattr(self, "_data", None)

    @property
    def bc_description(self) -> str:
        what = self._what
        parts = []
        if self.all_exterior:
            parts.append(f"{what} given on every exterior side")
        if self.dirichlet_markers:
            parts.append(f"{what} given on marker(s) " + ", ".join(map(str, self.dirichlet_markers)))
        return "; ".join(parts) or "free (natural on every side)"

    def __repr__(self):
        return f"{type(self).__name__}(degree={self.degree}, ncells={self.ncells}, bc={self.bc_description}, dim={self.dim})"

    def info(self) -> str:
        k = self.degree
        rows = [
            (type(self).__name__, ""),
            ("mesh", f"{self.mesh.npoints} vertices, {self.ncells} triangles, {self.nedges} edges"),
            ("degree", f"{k}   ({self.family}_{k}: {k} per edge, {k * (k - 1)} inside, {self.nloc} per triangle)"),
            ("conformity", "H(div): normal component continuous" if self.family == "RT"
             else "H(curl): tangential component continuous"),
            ("boundary conditions", self.bc_description),
            ("degrees of freedom", f"{self.raw_dim}"),
            ("dimension", f"{self.dim}   ({self.n_constraints} eliminated by essential conditions)"),
            ("coefficients are", f"{'normal' if self.family == 'RT' else 'tangential'} component x edge length at "
                                 f"{k} Gauss point(s) per edge" + (", interior moments" if k > 1 else "")),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"{a:<{w}}  {v}" if v else a for a, v in rows)

    # ---- numbering ---------------------------------------------------------------------------
    @property
    def unconstrained(self):
        """The same mesh and degree with nothing eliminated (created once); self if nothing is."""
        if self.n_constraints == 0:
            return self
        U = getattr(self, "_unconstrained", None)
        if U is None:
            U = type(self)(self.mesh, self.degree)
            self._unconstrained = U
        return U

    def dof_coordinates(self) -> np.ndarray:
        """(dim, 2): the point of each edge degree of freedom (the centroid for an interior moment)."""
        return self.raw_coordinates()[self.adapted_to_raw()]

    def prolongate(self, coeffs) -> np.ndarray:
        return _real_linear(lambda c: _C.Space2D.prolongate(self, np.ascontiguousarray(_arr(c))), coeffs)

    def restrict(self, raw) -> np.ndarray:
        return _real_linear(lambda r: _C.Space2D.restrict(self, np.ascontiguousarray(_arr(r))), raw)

    def _facet_table(self):
        T = getattr(self, "_ftab", None)
        if T is None:
            cell, local, mark, ext, edge = self.facets()
            cv = self.cell_vertices()
            a = cv[cell, (local + 1) % 3]
            b = cv[cell, (local + 2) % 3]
            T = dict(cell=cell, local=local, marker=mark, exterior=ext.astype(bool), edge=edge,
                     flip=(a > b), a=a, b=b)
            self._ftab = T
        return T

    def boundary_dofs(self, marker=None) -> np.ndarray:
        """Raw indices of the edge degrees of freedom on the sides with this marker (all exterior
        sides for None)."""
        T = self._facet_table()
        sel = T["exterior"] if marker is None else np.isin(T["marker"], _markers(marker, "marker"))
        k = self.degree
        return np.unique((T["edge"][sel][:, None] * k + np.arange(k)[None, :]).ravel()).astype(np.int64)

    # ---- quadrature --------------------------------------------------------------------------
    def cache(self, degree: int | None = None) -> _C.Cache2D:
        return _C.Cache2D(self, int(2 * self.degree if degree is None else degree))

    def boundary_cache(self, degree: int | None = None, marker=None) -> _C.Cache2D:
        d = int(2 * self.degree if degree is None else degree)
        if marker is None:
            return _C.Cache2D(self, d, [], True)
        return _C.Cache2D(self, d, _markers(marker, "marker"), False)

    def pattern(self, trial=None):
        S = self if trial is None else trial
        P = self._pattern.get(id(S))
        if P is None:
            P = _C.cell_pattern(self, S)
            self._pattern[id(S)] = P
        return P

    def at_quad(self, cache, coeffs, deriv=0) -> np.ndarray:
        """Values of derivative code `deriv` (3 c + d) at the cache points."""
        return cache.at_points(np.ascontiguousarray(_arr(coeffs), dtype=np.float64), int(deriv))

    # ---- evaluation ------------------------------------------------------------------------------
    def _combine(self, f, what):
        """f(code) -> values; the quantity `what` built from the codes."""
        w = _WHAT.get(what, what) if isinstance(what, str) else int(what)
        if isinstance(what, str) and what not in _WHAT:
            raise ValueError(f"what is 'value', 'x', 'y', 'div' or 'curl' (or a code 0..5), got {what!r}")
        if w is None:
            return np.stack([f(0), f(3)], axis=-1)
        if w == "div":
            return f(1) + f(5)
        if w == "curl":
            return f(4) - f(2)
        if not 0 <= w < 6:
            raise ValueError(f"derivative code {w} is not in 0..5 (3 component + 0 value, 1 d/dx, 2 d/dy)")
        return f(w)

    def evaluate(self, coeffs, x, y=None, what="value") -> np.ndarray:
        """Values at points: (n, 2) points or x and y arrays.  what = "value" gives (..., 2); "x", "y"
        a component; "div" or "curl" (rot = d u_y / dx - d u_x / dy) a scalar; an integer a raw code.
        NaN outside the mesh.  The normal (RT) or tangential (N1curl) component is continuous across
        edges; on an edge the other one is read from whichever triangle contains the point."""
        P, shape = _points(x, y)
        raw = self.prolongate(coeffs) if self.n_constraints else np.ascontiguousarray(_arr(coeffs), dtype=np.float64)
        out = self._combine(lambda c: np.asarray(self.evaluate_raw(raw, P, c)), what)
        return out.reshape(shape + out.shape[1:])

    def evaluate_cells(self, coeffs, cells, xi, eta, what="value", raw=False) -> np.ndarray:
        """Values at reference points (xi, eta) of the given cells, no point location."""
        r = np.ascontiguousarray(_arr(coeffs), dtype=np.float64) if raw else \
            (self.prolongate(coeffs) if self.n_constraints else np.ascontiguousarray(_arr(coeffs), dtype=np.float64))
        c = np.ascontiguousarray(cells, dtype=np.int32)
        a, b = np.ascontiguousarray(xi, dtype=np.float64), np.ascontiguousarray(eta, dtype=np.float64)
        return self._combine(lambda code: np.asarray(self.evaluate_ref(r, c, a, b, code)), what)

    # ---- data ----------------------------------------------------------------------------------
    def interpolate(self, f) -> np.ndarray:
        """The canonical interpolant of a vector field f(x, y) -> (fx, fy) (or a pair of callables /
        numbers): its edge and interior moments.  Exact on RT_k (N1curl_k), and it commutes with
        div (curl): div of the interpolant is the L^2 projection of div f onto DG_{k-1}."""
        P = np.asarray(self.interpolation_points())
        F = _vector_values(f, P[:, 0], P[:, 1])
        raw = np.asarray(self.interpolate_raw(np.ascontiguousarray(F)))
        return self.restrict(raw) if self.n_constraints else raw

    def mass_matrix(self):
        """int u . v (created once)."""
        M = getattr(self, "_mass", None)
        if M is None:
            from .forms import TrialFunction, TestFunction, dot, dx, form
            M = form(dot(TrialFunction(self), TestFunction(self)) * dx).assemble()
            self._mass = M
        return M

    def project(self, f, degree: int | None = None) -> np.ndarray:
        """L2 projection onto V (zero on the essential sides) of a vector field f(x, y) -> (fx, fy),
        a pair, or a vector expression."""
        from .forms import TestFunction, ListTensor, dot, dx, form
        S = getattr(self, "_mass_solver", None)
        if S is None:
            S = self.mass_matrix().solver()
            self._mass_solver = S
        if isinstance(f, ListTensor):
            v = TestFunction(self)
            b = np.asarray(form(dot(f, v) * (dx(degree) if degree is not None else dx)).assemble())
        else:
            Q = self.cache(degree if degree is not None else 2 * self.degree + 2)
            F = _vector_values(f, np.asarray(Q.x()), np.asarray(Q.y()))
            b = np.asarray(_C.assemble_vector_2d(Q, [0, 3], [np.ascontiguousarray(F[:, 0]), np.ascontiguousarray(F[:, 1])]))
        return np.asarray(S.solve(b).vector)

    def lift(self, g=None):
        """A Function of V.unconstrained carrying the essential data on the eliminated degrees of freedom.

            ug = V.lift(lambda x, y: (x, y))      # a vector field: its normal (tangential) part is used
            ug = V.lift(0.5)                      # u . n = 0.5 (RT) or u . t = 0.5 (N1curl)
            ug = V.lift({1: g1, 3: (0, 1)})       # per marker; sides left out get zero data

        A scalar is the normal component with the outward normal (RT), or the tangential component
        with the boundary's positive orientation, the domain on the left (N1curl): counter-clockwise
        on the outer boundary, clockwise around a hole.  On an interior side, which has no outward
        normal, give a vector field.  The edge values are L^2 projections, so the flux (circulation)
        through every side is exact.  Without arguments it uses the data given with the space, if any."""
        if g is None and self.boundary_data is not None:
            g = self.boundary_data
        return self._lift(g)

    def _lift(self, g=None):
        U = self.unconstrained
        raw = np.zeros(self.raw_dim)
        cons = np.asarray(self.constrained(), dtype=np.int64)
        if g is None or cons.size == 0:
            if g is not None and cons.size == 0 and not (isinstance(g, (int, float)) and g == 0):
                raise ValueError("lift(): this space has no essential condition built in; natural data enter "
                                 "through ds(marker)")
            return VectorElementFunction(U, "lift", raw)
        if isinstance(g, dict):
            allowed = set(self.dirichlet_markers)
            if self.all_exterior:
                allowed |= set(np.asarray(self.mesh.segment_markers).tolist()) | {0}
            for m in sorted(g):
                if int(m) not in allowed:
                    raise ValueError(f"lift(): marker {m} carries no essential condition in this space "
                                     f"(markers {sorted(allowed)}); natural data enter through ds({m})")
                full = self._boundary_values(g[m], int(m))
                d = self.boundary_dofs(int(m))
                d = d[np.isin(d, cons)]
                raw[d] = full[d]
        else:
            full = self._boundary_values(g, None)
            raw[cons] = full[cons]
        return VectorElementFunction(U, "lift", raw)

    def _boundary_values(self, g, marker):
        """Raw edge values of boundary data g on the exterior sides (with this marker)."""
        P = np.asarray(self.interpolation_points())
        k, ne = self.degree, self.nedges
        t, Pj = self.edge_projection()
        nqE = t.size
        Pe = P[:ne * nqE]
        if not _is_scalar_data(g, Pe):
            F = _vector_values(g, P[:, 0], P[:, 1])
            return np.asarray(self.interpolate_raw(np.ascontiguousarray(F)))
        # a scalar: u . n (RT, outward) or u . t (N1curl, the domain on the left) times |e|
        T = self._facet_table()
        sel = T["exterior"] if marker is None else (T["marker"] == marker)
        if not np.all(T["exterior"][sel]):
            raise ValueError("lift(): scalar data give u . n (u . t) with the outward normal, which an interior side "
                             "does not have; give a vector field g(x, y) -> (gx, gy) there")
        raw = np.zeros(self.raw_dim)
        edges, flips = T["edge"][sel], T["flip"][sel]
        xy = Pe[(edges[:, None] * nqE + np.arange(nqE)[None, :]).ravel()]
        val = np.broadcast_to(np.asarray(g(xy[:, 0], xy[:, 1]) if callable(g) else g, dtype=np.float64),
                              (xy.shape[0],)).reshape(edges.size, nqE)
        E = self.edges()[edges]
        V = np.asarray(self.mesh.points)
        length = np.hypot(*(V[E[:, 1]] - V[E[:, 0]]).T)
        s = np.where(flips, -1.0, 1.0) * length
        idx = (edges[:, None] * k + np.arange(k)[None, :])
        raw[idx.ravel()] = ((val @ Pj.T) * s[:, None]).ravel()
        return raw

    # ---- plotting ------------------------------------------------------------------------------
    def triangulation(self, coeffs, refine: int | None = None, what="magnitude"):
        """(matplotlib Triangulation, values) on each cell split into refine^2 triangles (default
        refine: the degree + 1).  what: "magnitude", "x", "y", "div", "curl"."""
        import matplotlib.tri as mtri
        r = max(int(self.degree + 1 if refine is None else refine), 1)
        ij = [(i, j) for j in range(r + 1) for i in range(r + 1 - j)]
        idx = {p: n for n, p in enumerate(ij)}
        sub = []
        for j in range(r):
            for i in range(r - j):
                sub.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
                if i + j < r - 1:
                    sub.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
        L = np.array(ij, dtype=np.float64) / r
        nc, npc = self.ncells, len(ij)
        cells = np.repeat(np.arange(nc, dtype=np.int32), npc)
        xi, eta = np.tile(L[:, 0], nc), np.tile(L[:, 1], nc)
        XY, _ = self.map_points(np.ascontiguousarray(cells), np.ascontiguousarray(xi), np.ascontiguousarray(eta))
        if what == "magnitude":
            vals = np.hypot(*self.evaluate_cells(coeffs, cells, xi, eta, "value").T)
        else:
            vals = self.evaluate_cells(coeffs, cells, xi, eta, what)
        tris = (np.arange(nc)[:, None, None] * npc + np.array(sub)[None, :, :]).reshape(-1, 3)
        return mtri.Triangulation(XY[:, 0], XY[:, 1], tris), vals

    def plot(self, coeffs, ax=None, *, what="magnitude", quiver=True, refine=None, colorbar=True, mesh=False, **kw):
        """The magnitude (or what= "x", "y", "div", "curl") as a colour map, with arrows at the cell
        centroids (quiver=False to leave them out)."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        tri, vals = self.triangulation(coeffs, refine, what)
        kw.setdefault("shading", "gouraud")
        art = ax.tripcolor(tri, vals, **kw)
        if quiver:
            nc = self.ncells
            c = np.arange(nc, dtype=np.int32)
            third = np.full(nc, 1.0 / 3.0)
            XY, _ = self.map_points(c, third, third)
            U = self.evaluate_cells(coeffs, c, third, third, "value")
            ax.quiver(XY[:, 0], XY[:, 1], U[:, 0], U[:, 1], color="w", alpha=0.8)
        if mesh:
            ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.triangles, color="k", lw=0.3, alpha=0.5)
        ax.set_aspect("equal")
        if colorbar:
            plt.colorbar(art, ax=ax, fraction=0.046, pad=0.04)
        return art


class RTSpace(_VectorElement2D, _C.VectorElementSpace2D):
    """Raviart-Thomas elements RT_k (k >= 1) on a triangular Mesh2D: H(div)-conforming vector
    fields whose normal component is continuous across edges.  fd.FunctionSpace(m, "RT", k).

        V = fd.RTSpace(m, 1)                          # lowest order, one flux per edge
        V = fd.RTSpace(m, 2, dirichlet={1: 0.0})      # u . n = 0 built in on side 1

    bc / dirichlet / data: the normal flux u . n is ESSENTIAL here (in mixed Poisson it is the
    Neumann condition of the scalar), so it is eliminated on the sides named; a Dirichlet value of
    the scalar is natural and enters through ds as  g * dot(t, n) * ds(marker)."""
    family = "RT"
    _family_enum = _C.VectorFamily.RT
    _what = "u . n"


class N1curlSpace(_VectorElement2D, _C.VectorElementSpace2D):
    """Nedelec elements of the first kind N1curl_k (k >= 1) on a triangular Mesh2D: H(curl)-
    conforming vector fields whose tangential component is continuous across edges.
    fd.FunctionSpace(m, "N1curl", k).

        E = fd.N1curlSpace(m, 1, bc="dirichlet")      # u x n = 0 on the boundary (a perfect conductor)

    In 2D N1curl_k is RT_k turned by 90 degrees; curl(u) = d u_y / dx - d u_x / dy is a scalar."""
    family = "N1curl"
    _family_enum = _C.VectorFamily.N1curl
    _what = "u x n"


class DGSpace2D(_Lagrange2D, _C.DGSpace2D):
    """Broken (discontinuous) P_k, k >= 0, on a triangular Mesh2D: each triangle carries its own
    (k+1)(k+2)/2 nodal functions, nothing is shared or built in.  fd.DGSpace(m, k) and
    fd.FunctionSpace(m, "DG", k) return it.

    Forms couple the triangles through the interior-facet measure dS, with the restrictions u('-')
    and u('+'), jump(u) = u('-') - u('+'), avg(u) and the normal n('-'), which points out of the '-'
    triangle.  Boundary data enter weakly, through ds.  It is also the pressure (scalar) partner of
    RT_{k+1} and N1curl_{k+1} in mixed methods.

    nodes:    "equispaced" (default) or "lobatto" (warp-and-blend).
    periodic: a pair of side markers (a, b), or a list of pairs: side b is side a translated, and
              its edges become interior facets joined to those of side a (it must match edge for
              edge).  [(4, 2), (1, 3)] makes a rectangle doubly periodic."""
    family = "dg"
    broken = True
    continuity = -1

    def __init__(self, mesh, degree: int, nodes="equispaced", periodic=None):
        from .mesh2d import Mesh2D
        if not isinstance(mesh, Mesh2D):
            raise TypeError(f"DGSpace2D: mesh must be a triangular Mesh2D, got {type(mesh).__name__}")
        if isinstance(degree, bool) or int(degree) != degree:
            raise TypeError(f"degree must be an integer, got {degree!r}")
        k = int(degree)
        if k < 0:
            raise ValueError(f"DGSpace2D: degree must be >= 0, got {k}")
        nd = _NODES.get(str(nodes).lower())
        if nd is None:
            raise ValueError(f"DGSpace2D: nodes is 'equispaced' or 'lobatto', got {nodes!r}")
        _C.DGSpace2D.__init__(self, mesh._m, k, nd == "equispaced")
        self.mesh = mesh
        self._bc = "free"
        self._pattern = {}
        self._ctor = dict(degree=k, nodes=nd)
        from .spaces2d import _periodic_pairs
        self.periodic_pairs, self.periodic_info = _periodic_pairs(periodic), []
        self._data = None
        self.vertex_representatives = np.arange(mesh.npoints)
        if self.periodic_pairs:
            self._join_periodic()

    def _join_periodic(self):
        """Pair the edges of each periodic side pair into interior facets, and the vertices into classes."""
        P = np.asarray(self.mesh.points, dtype=np.float64)
        tol = 1e-8 * float(np.ptp(P, axis=0).max())
        present = set(np.asarray(self.mesh.segment_markers).tolist())
        cell, local, mark, ext, edge = self.facets()
        cv = self.cell_vertices()
        mid = 0.5 * (P[cv[cell, (local + 1) % 3]] + P[cv[cell, (local + 2) % 3]])
        CA, LA, CB, LB, SX, SY = [], [], [], [], [], []
        parent = np.arange(P.shape[0])

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b in self.periodic_pairs:
            for m_ in (a, b):
                if m_ not in present:
                    raise ValueError(f"DGSpace2D: periodic: no side carries marker {m_}; the markers present are "
                                     f"{sorted(present)}")
            ia = np.flatnonzero(ext.astype(bool) & (mark == a))
            ib = np.flatnonzero(ext.astype(bool) & (mark == b))
            if ia.size != ib.size or ia.size == 0:
                raise ValueError(f"DGSpace2D: periodic ({a}, {b}): side {a} has {ia.size} edges and side {b} has "
                                 f"{ib.size}; the mesh must match edge for edge across the two sides")
            shift = mid[ib].mean(axis=0) - mid[ia].mean(axis=0)
            d, j = _C.nearest_points(np.ascontiguousarray(mid[ia]), np.ascontiguousarray(mid[ib] - shift))
            d, j = np.asarray(d), np.asarray(j, dtype=np.int64)
            if np.any(d > tol) or np.unique(j).size != j.size:
                raise ValueError(f"DGSpace2D: periodic ({a}, {b}): side {b} is not side {a} shifted by "
                                 f"({shift[0]:.6g}, {shift[1]:.6g}) edge for edge")
            CA.append(cell[ia[j]]); LA.append(local[ia[j]]); CB.append(cell[ib]); LB.append(local[ib])
            SX.append(np.full(ib.size, shift[0])); SY.append(np.full(ib.size, shift[1]))
            # vertex classes, for the vertex patches of a limiter
            va = np.unique(np.concatenate([cv[cell[ia], (local[ia] + 1) % 3], cv[cell[ia], (local[ia] + 2) % 3]]))
            vb = np.unique(np.concatenate([cv[cell[ib], (local[ib] + 1) % 3], cv[cell[ib], (local[ib] + 2) % 3]]))
            dv, jv = _C.nearest_points(np.ascontiguousarray(P[va]), np.ascontiguousarray(P[vb] - shift))
            jv = np.asarray(jv, dtype=np.int64)
            for s_, m_ in zip(vb, va[jv]):
                ra, rb = find(int(s_)), find(int(m_))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            self.periodic_info.append(dict(pair=(a, b), shift=shift, edges=int(ib.size)))
        cat = lambda L, t: np.ascontiguousarray(np.concatenate(L), dtype=t)          # noqa: E731
        self._set_periodic_facets(cat(CA, np.int32), cat(LA, np.int32), cat(CB, np.int32), cat(LB, np.int32),
                                  cat(SX, np.float64), cat(SY, np.float64))
        self.vertex_representatives = np.array([find(i) for i in range(P.shape[0])])

    @property
    def bc_description(self) -> str:
        if self.periodic_pairs:
            return "; ".join(f"periodic, side {b} joined to side {a}" for a, b in self.periodic_pairs)
        return "none (broken: boundary data enter weakly through ds)"

    def boundary_nodes(self, marker=None) -> np.ndarray:
        raise TypeError("DGSpace2D: a broken space has no boundary nodes of its own")

    def lift(self, g=None):
        if g is not None and not (isinstance(g, (int, float)) and g == 0):
            raise ValueError("lift(): a DG space has no condition built in; boundary data enter through ds")
        return Function(self, "lift", np.zeros(self.dim), _infer=False)

    _lift = lift

    def __repr__(self):
        per = f", periodic={self.periodic_pairs}" if self.periodic_pairs else ""
        return f"DGSpace2D(degree={self.degree}, ncells={self.ncells}{per}, dim={self.dim})"

    def cell_means(self, coeffs) -> np.ndarray:
        """The mean of the Function over every triangle."""
        A = _dg_tables(self)
        return np.asarray(_arr(coeffs), dtype=np.float64).reshape(self.ncells, self.nloc) @ A["mean"]

    def info(self) -> str:
        rows = [
            ("DGSpace2D", ""),
            ("mesh", f"{self.mesh.npoints} vertices, {self.ncells} triangles"),
            ("degree", f"{self.degree}   (broken, {self.nloc} nodes per triangle)"),
            ("dimension", f"{self.dim}"),
            ("interior facets", f"{len(self.interior_facets()[2])}" + (f" ({self.n_periodic_facets} periodic)"
                                                                      if self.n_periodic_facets else "")),
            ("coefficients are", "nodal values per triangle (" + ("equispaced" if self.equispaced else "warp-and-blend")
             + " nodes), two-valued on edges"),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"{a:<{w}}  {v}" if v else a for a, v in rows)


def _dg_tables(V):
    """Reference tables of a DGSpace2D (cached): the mean weights, the L2 projection onto P1 at the
    vertices, and the P1 (barycentric) functions at the nodes."""
    T = getattr(V, "_dgtab", None)
    if T is None:
        mean, Pi, E = (np.asarray(a) for a in _C.dg_vertex_tables(V))     # fe/dg_tables.hpp
        T = dict(mean=mean, Pi=Pi, E=E)
        V._dgtab = T
    return T
