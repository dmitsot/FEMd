"""
femd.spaces2d -- finite element spaces on triangular meshes.

    m = fd.rectangle_mesh(0, 0, 1, 1, 16, 16)
    V = fd.LagrangeSpace(m, 3, dirichlet=[1, 3])        # P3, u = data on sides 1 and 3 (a LagrangeSpace2D)
    V = fd.FunctionSpace(m, "lagrange", 3, bc="dirichlet")   # every exterior side
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    a = fd.dot(fd.grad(u), fd.grad(v)) * fd.dx
    uh = fd.solve(a, f*v*fd.dx + g*v*fd.ds(2), dirichlet={1: 0.0, 3: gD})

Coefficients are nodal values.  A Dirichlet condition is built in by ELIMINATION,
as in 1D: the nodes on the named sides are not degrees of freedom of V, and data on
them is carried by a lift, V.lift(g), a Function of V.unconstrained.
"""
from __future__ import annotations

import numpy as np
from . import _femd as _C
from .forms import Function
from .sparse import SparseMatrix
from ._util import _real_linear

__all__ = ["LagrangeSpace2D", "LagrangeSpaceQ", "ProductSpace2D", "VectorFunctionSpace"]


_NODES = {"lobatto": "lobatto", "gll": "lobatto", "gauss-lobatto": "lobatto", "warp-blend": "lobatto",
          "warp-and-blend": "lobatto", "equispaced": "equispaced", "uniform": "equispaced", "equidistant": "equispaced"}


def _markers(v, what):
    if v is None:
        return []
    if isinstance(v, (int, np.integer)):
        return [int(v)]
    try:
        out = [int(m) for m in v]
    except TypeError:
        raise TypeError(f"{what}: a marker or a list of markers, got {v!r}") from None
    return out


def _points(x, y=None):
    """(n, 2) float64 array from (n, 2) points, or from x and y arrays (any shape, broadcast)."""
    if y is None:
        P = np.asarray(x, dtype=np.float64)
        if P.ndim == 1 and P.size == 2:
            P = P.reshape(1, 2)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("points: an (n, 2) array, or x and y arrays")
        return np.ascontiguousarray(P), P.shape[:1]
    X, Y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return np.ascontiguousarray(np.column_stack([X.ravel(), Y.ravel()])), X.shape


_DERIV = {0: 0, None: 0, "x": 1, "y": 2, 1: 1, 2: 2, (0, 0): 0, (1, 0): 1, (0, 1): 2}


def _deriv_code(d):
    try:
        return _DERIV[d if not isinstance(d, list) else tuple(d)]
    except (KeyError, TypeError):
        raise ValueError(f"deriv must be 0 (value), 'x' or (1, 0) for d/dx, 'y' or (0, 1) for d/dy; got {d!r}") from None


def _lumped_sparse(d, space):
    """A diagonal SparseMatrix with entries d on `space`, refused when an entry is not positive."""
    if not np.all(d > 0):
        i = int(np.argmin(d))
        raise ValueError(f"mass_matrix(lumped=True): the lumped entry is {d[i]:.3g} at dof {i}, not positive, so the "
                         "lumped mass would be singular or indefinite; use Q_k elements with nodes='lobatto' "
                         "(or P_1 on triangles)")
    S = SparseMatrix(_diagonal_csr(d), space)
    S.symmetric = True
    return S


def _diagonal_csr(d):
    """The diagonal matrix of the entries d in the CSR store."""
    d = np.ascontiguousarray(d, dtype=np.float64)
    i = np.arange(d.size, dtype=np.int32)
    return _C.csr_from_triplets(int(d.size), int(d.size), i, i, d)


class _Lagrange2D:
    """What LagrangeSpace2D (P_k on triangles) and LagrangeSpaceQ (Q_k on quadrilaterals) share.

    bc:        "free" (default) or "dirichlet" (a condition on every exterior side).
    dirichlet: a marker or a list of markers whose sides carry a Dirichlet condition.
               The two combine.  Sides not named keep the natural condition: Neumann
               or Robin data enter through ds(marker).  A dict {marker: data} names the
               sides and gives their data at once: dirichlet={1: 0.0, 3: g}.
    data:      the Dirichlet data, anything V.lift takes (a number or a callable g(x, y) for
               every Dirichlet side, or a dict {marker: ...}), for instance with bc="dirichlet".
               fd.solve, fd.newton and fd.IRK use the data of the space when the call gives
               none; a callable g(x, y, t) or g(t) is data in time (fd.IRK).
    nodes:     "equispaced" (default) or "lobatto".  The space is the same either way; the
               nodal basis differs, and with it the conditioning of the matrices and the
               quality of interpolation (interpolate, lift) at high degree.
    periodic:  a pair of markers (a, b), or a list of pairs: side b is identified with side a,
               which it must match node for node after a translation (as rectangle_mesh does).
               [(4, 2), (1, 3)] makes a rectangle doubly periodic.  The nodes of side b are then
               not degrees of freedom of their own: they share those of side a.

    dof_coordinates() gives the nodes; the coefficients of a Function are the values there."""

    family = "lagrange"
    _mesh_type = None             # set by the subclasses
    _cell = "cell"
    tdim = 2
    broken = False
    continuity = 0

    def __init__(self, mesh, degree: int, bc="free", dirichlet=None, nodes="equispaced", periodic=None, data=None):
        M = self._mesh_class()
        if isinstance(dirichlet, dict):
            if data is not None:
                raise TypeError(f"{type(self).__name__}: give the data either in dirichlet={{marker: data}} or in "
                                "data=, not both")
            data, dirichlet = dict(dirichlet), sorted(dirichlet)
        if not isinstance(mesh, M):
            raise TypeError(f"{type(self).__name__}: mesh must be a {M.__name__}, got {type(mesh).__name__}")
        if isinstance(degree, bool) or int(degree) != degree:
            raise TypeError(f"degree must be an integer, got {degree!r}")
        k = int(degree)
        if k < 1:
            raise ValueError(f"{type(self).__name__}: degree must be >= 1, got {k}")
        b = "free" if bc is None else str(bc).lower()
        if b not in ("free", "dirichlet"):
            raise ValueError(f"{type(self).__name__}: bc is 'free' or 'dirichlet' (every exterior side), got {bc!r}; "
                             "name individual sides with dirichlet=[markers]")
        marks = sorted(set(_markers(dirichlet, "dirichlet")))
        present = set(np.asarray(mesh.segment_markers).tolist())
        missing = [m for m in marks if m not in present]
        if missing:
            raise ValueError(f"{type(self).__name__}: no side of the mesh carries marker(s) {missing}; "
                             f"the markers present are {sorted(present)}")
        nd = _NODES.get(str(nodes).lower())
        if nd is None:
            raise ValueError(f"{type(self).__name__}: nodes is 'equispaced' (the default) or 'lobatto' "
                             f"(warp-and-blend on triangles, Gauss-Lobatto tensor on quads), got {nodes!r}")
        pairs = _periodic_pairs(periodic)
        for a_, b_ in pairs:
            for m_ in (a_, b_):
                if m_ not in present:
                    raise ValueError(f"{type(self).__name__}: periodic: no side carries marker {m_}; the markers "
                                     f"present are {sorted(present)}")
                if m_ in marks:
                    raise ValueError(f"{type(self).__name__}: marker {m_} is both Dirichlet and periodic")
        if pairs and b == "dirichlet":
            raise ValueError(f"{type(self).__name__}: bc='dirichlet' puts a Dirichlet condition on every exterior side, "
                             "the periodic ones included; name the Dirichlet sides with dirichlet=[markers]")
        super().__init__(mesh._m, k, marks, b == "dirichlet", nd == "equispaced")
        self.mesh = mesh
        self._bc = b
        self._pattern = {}
        self._ctor = dict(degree=k, nodes=nd)
        self.periodic_pairs, self.periodic_info = pairs, []
        if pairs:
            rep, self.periodic_info = _periodic_classes(self, pairs)
            self._identify(np.ascontiguousarray(rep, dtype=np.int32))
        self._data = None
        if data is not None:
            if len(self.constrained()) == 0:
                raise ValueError(f"{type(self).__name__}: data= gives Dirichlet data, and this space has no Dirichlet "
                                 "side; name them with dirichlet=[markers] or bc='dirichlet'")
            _check_data(self, data)
            self._data = data

    @property
    def boundary_data(self):
        """The Dirichlet data given with the space (dirichlet={marker: data} or data=), or None.
        fd.solve, fd.newton, fd.IRK and V.lift() use it when the call gives no data."""
        return getattr(self, "_data", None)

    @property
    def node_set(self) -> str:
        """'lobatto' or 'equispaced': where the nodes sit (the space is the same, the basis is not)."""
        return "equispaced" if self.equispaced else "lobatto"

    # ---- description -----------------------------------------------------------------
    @property
    def bc_description(self) -> str:
        parts = []
        if self.all_exterior:
            parts.append("Dirichlet on every exterior side")
        if self.dirichlet_markers:
            parts.append("Dirichlet on marker(s) " + ", ".join(map(str, self.dirichlet_markers)))
        for a_, b_ in getattr(self, "periodic_pairs", []):
            parts.append(f"periodic, side {b_} identified with side {a_}")
        return "; ".join(parts) or "free (natural on every side)"

    def __repr__(self):
        return (f"{type(self).__name__}(degree={self.degree}, ncells={self.ncells}, "
                f"bc={self.bc_description}, dim={self.dim})")

    def info(self) -> str:
        rows = [
            (type(self).__name__, ""),
            ("mesh", f"{self.mesh.npoints} vertices, {self.ncells} {self.cell_name}s, {self.nedges} edges"),
            ("degree", f"{self.degree}   (continuity C^0, {self.nloc} nodes per {self.cell_name})"),
            ("boundary conditions", self.bc_description),
            ("nodes", f"{self.raw_dim}"),
            ("dimension", f"{self.dim}   ({self.n_constraints} nodes eliminated by Dirichlet or periodic conditions)"),
            ("coefficients are", "nodal values (" + ("equispaced nodes" if self.equispaced else
                                  ("warp-and-blend nodes" if self.nverts == 3 else "Gauss-Lobatto tensor nodes")) + ")"),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"{k:<{w}}  {v}" if v else k for k, v in rows)

    # ---- numbering ----------------------------------------------------------------------
    @property
    def unconstrained(self):
        """The same mesh and degree with nothing eliminated (created once); self if nothing is."""
        if self.n_constraints == 0:
            return self
        U = getattr(self, "_unconstrained", None)
        if U is None:
            U = type(self)(self.mesh, self.degree, nodes=self.node_set)
            self._unconstrained = U
        return U

    def dof_coordinates(self) -> np.ndarray:
        """(dim, 2) coordinates of the degrees of freedom (the free nodes)."""
        return self.raw_coordinates()[self.adapted_to_raw()]

    def node_coordinates(self) -> np.ndarray:
        """(raw_dim, 2) coordinates of every node, eliminated ones included."""
        return self.raw_coordinates()

    def prolongate(self, coeffs) -> np.ndarray:
        """Adapted -> raw coefficients (zeros on the eliminated nodes): a vector of V.unconstrained.
        A complex vector gives a complex result."""
        return _real_linear(lambda c: _C.Space2D.prolongate(self, np.ascontiguousarray(_arr(c))), coeffs)

    def restrict(self, raw) -> np.ndarray:
        """Raw -> adapted coefficients (drops the eliminated nodes).  A complex vector gives a complex result."""
        return _real_linear(lambda r: _C.Space2D.restrict(self, np.ascontiguousarray(_arr(r))), raw)

    def boundary_nodes(self, marker=None) -> np.ndarray:
        """Raw indices of the nodes on the sides with this marker (all exterior sides for None)."""
        cell, local, mark, ext, edge = self.facets()
        sel = ext.astype(bool) if marker is None else np.isin(mark, _markers(marker, "marker"))
        E = self.edges()[edge[sel]]
        k, nv = self.degree, self.mesh.npoints
        parts = [E.ravel()]
        if k > 1:
            parts.append((nv + edge[sel][:, None] * (k - 1) + np.arange(k - 1)[None, :]).ravel())
        return np.unique(np.concatenate(parts)).astype(np.int64)

    # ---- quadrature ---------------------------------------------------------------------
    def cache(self, degree: int | None = None, rule: str = "gauss", npts: int | None = None) -> _C.Cache2D:
        """Quadrature points on the cells, exact to total degree `degree` (default 2k), or with
        rule="lobatto" (quadrilaterals only) the tensor Gauss-Lobatto rule with npts points per
        direction (default k + 1: the nodes of nodes="lobatto", which make the mass diagonal)."""
        rule = str(rule).lower()
        if rule == "lobatto":
            if self.nverts != 4:
                raise ValueError("cache(rule='lobatto'): the Gauss-Lobatto rule is a tensor rule for quadrilaterals; "
                                 "triangles have none")
            return _C.Cache2D(self, int(self.degree + 1 if npts is None else npts), True)
        if rule != "gauss":
            raise ValueError(f"cache: rule must be 'gauss' or 'lobatto', got {rule!r}")
        return _C.Cache2D(self, int(2 * self.degree if degree is None else degree))

    def boundary_cache(self, degree: int | None = None, marker=None) -> _C.Cache2D:
        """Gauss points on the exterior sides with this marker (all for None), exact to `degree`."""
        d = int(2 * self.degree if degree is None else degree)
        if marker is None:
            return _C.Cache2D(self, d, [], True)
        return _C.Cache2D(self, d, _markers(marker, "marker"), False)

    def pattern(self, trial=None):
        """The CSR pattern of matrices with test space self and trial space `trial` (cached)."""
        S = self if trial is None else trial
        P = self._pattern.get(id(S))
        if P is None:
            P = _C.cell_pattern(self, S)
            self._pattern[id(S)] = P
        return P

    def at_quad(self, cache, coeffs, deriv=0) -> np.ndarray:
        """Values (or d/dx, d/dy) of the Function with these coefficients at the cache points."""
        return cache.at_points(np.ascontiguousarray(_arr(coeffs), dtype=np.float64), _deriv_code(deriv))

    # ---- evaluation ---------------------------------------------------------------------
    def evaluate(self, coeffs, x, y=None, deriv=0) -> np.ndarray:
        """u, du/dx or du/dy at points: (n, 2) points or x and y arrays (then the result has their
        shape).  NaN outside the mesh.  deriv 0, 'x' or (1, 0), 'y' or (0, 1)."""
        P, shape = _points(x, y)
        raw = self.prolongate(coeffs) if self.n_constraints else np.ascontiguousarray(_arr(coeffs), dtype=np.float64)
        out = self.evaluate_raw(raw, P, _deriv_code(deriv))
        return out.reshape(shape)

    def interpolate(self, f) -> np.ndarray:
        """Nodal interpolant: f(x, y) at dof_coordinates() (f vectorized over arrays), an expression in
        fd.x and fd.y, or the values."""
        from .forms import Expr
        pts = self.dof_coordinates()
        if isinstance(f, Expr):                          # an expression in fd.x, fd.y and Constants
            vals = np.array(_values(f, pts), dtype=np.float64)
        elif callable(f):
            vals = np.asarray(f(pts[:, 0], pts[:, 1]), dtype=np.float64)
            vals = np.broadcast_to(vals, (pts.shape[0],)).copy()
        else:
            vals = np.asarray(f, dtype=np.float64).reshape(-1)
            if vals.size == 1:
                vals = np.full(pts.shape[0], float(vals[0]))
        if vals.shape != (self.dim,):
            raise ValueError(f"interpolate: need {self.dim} values (one per degree of freedom), got {vals.size}")
        return vals

    def mass_matrix(self, lumped: bool = False) -> SparseMatrix:
        """int u v.  lumped=True gives a diagonal matrix: the Gauss-Lobatto rule at the nodes of a
        Q_k space with nodes="lobatto" (the entries of u*v*dx(scheme="lobatto")), and the row sums of
        the mass matrix on every other space.  A row sum that is not positive raises ValueError
        (P_k triangles from k = 2: the vertex rows sum to zero or less)."""
        from .forms import TrialFunction, TestFunction, dx, form
        u, v = TrialFunction(self), TestFunction(self)
        if not lumped:
            return form(u * v * dx).assemble()
        if self.nverts == 4 and self.node_set == "lobatto":
            d = np.asarray(form(u * v * dx(scheme="lobatto")).assemble().diagonal(), dtype=np.float64)
        else:
            d = np.asarray(form(u * v * dx).assemble().matvec(np.ones(self.dim)), dtype=np.float64)
        return _lumped_sparse(d, self)

    def project(self, f, degree: int | None = None) -> np.ndarray:
        """L2 projection onto V of a callable f(x, y), a number, or an expression / Function."""
        from .forms import TestFunction, Expr, dx, form, _lift
        S = getattr(self, "_mass_solver", None)
        if S is None:
            S = self.mass_matrix().solver()
            self._mass_solver = S
        v = TestFunction(self)
        if isinstance(f, Expr) or isinstance(f, (int, float)):
            b = form(_lift(f) * v * dx(degree) if degree is not None else _lift(f) * v * dx).assemble()
        elif callable(f):
            Q = self.cache(degree if degree is not None else 2 * self.degree + 2)
            fq = np.asarray(f(Q.x(), Q.y()), dtype=np.float64)
            fq = np.broadcast_to(fq, (Q.nent * Q.nq,)).copy()
            b = _C.assemble_vector_2d(Q, [0], [fq])
        else:
            raise TypeError(f"project: a callable f(x, y), a number or an expression, got {type(f).__name__}")
        return np.asarray(S.solve(b).vector)

    def lift(self, g=None, **by_marker):
        """A Function u_g of V.unconstrained carrying Dirichlet data on the eliminated nodes.

            ug = V.lift(0.5)                         # the same value on every Dirichlet side
            ug = V.lift(lambda x, y: x * y)           # g(x, y), vectorized, on every Dirichlet side
            ug = V.lift({1: 0.0, 3: g3})              # per marker; sides left out get zero data

        The full solution is u = V.prolongate(c) + ug; fd.solve(a, L, dirichlet=...) does it.
        Without arguments it uses the data given with the space, if any.
        Where two sides with different data meet, the corner takes the data of the larger marker."""
        if by_marker:
            raise TypeError("lift(): give per-marker data as a dict, lift({1: g1, 3: g3})")
        if g is None and self.boundary_data is not None:
            g = self.boundary_data
        return self._lift(g)

    def _lift(self, g=None):
        """lift() with exactly the data given: None is zero, the space's own data are not used."""
        U = self.unconstrained
        raw = np.zeros(self.raw_dim)
        nodir = U is self or len(self.constrained()) == 0          # nothing eliminated by a Dirichlet side
        if g is None or nodir:
            if g is not None and nodir and not _is_zero(g):
                raise ValueError("lift(): this space has no Dirichlet condition built in; natural data enter "
                                 "through ds(marker)")
            return Function(U, "lift", raw, _infer=False)
        xy = self.raw_coordinates()
        cons = self.constrained()
        if isinstance(g, dict):
            allowed = set(self.dirichlet_markers)
            if self.all_exterior:
                allowed |= set(np.asarray(self.mesh.segment_markers).tolist()) | {0}
            for m in sorted(g):
                if int(m) not in allowed:
                    raise ValueError(f"lift(): marker {m} carries no Dirichlet condition in this space "
                                     f"(Dirichlet markers {sorted(allowed)}); natural data enter through ds({m})")
                nodes = self.boundary_nodes(int(m))
                nodes = nodes[np.isin(nodes, cons)]
                raw[nodes] = _values(g[m], xy[nodes])
        else:
            raw[cons] = _values(g, xy[cons])
        return Function(U, "lift", raw, _infer=False)

    # ---- plotting -------------------------------------------------------------------------
    def triangulation(self, coeffs, refine: int | None = None, deriv=0):
        """(matplotlib Triangulation, values) of the Function on a sub-triangulation: each cell split
        into refine^2 triangles (two per sub-square on quadrilaterals; default refine: the degree),
        values from the element polynomial."""
        import matplotlib.tri as mtri
        r = max(int(self.degree if refine is None else refine), 1)
        if self.nverts == 3:
            ij = [(i, j) for j in range(r + 1) for i in range(r + 1 - j)]
        else:
            ij = [(i, j) for j in range(r + 1) for i in range(r + 1)]
        idx = {p: n for n, p in enumerate(ij)}
        sub = []
        for j in range(r):
            for i in range(r if self.nverts == 4 else r - j):
                if self.nverts == 3:
                    sub.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
                    if i + j < r - 1:
                        sub.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
                else:
                    sub.append((idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)]))
                    sub.append((idx[(i, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
        L = np.array(ij, dtype=np.float64) / r
        code = _deriv_code(deriv)
        tabs = [self.ref_eval(a, b, 1) for a, b in L]
        T0 = np.array([t[0] for t in tabs])                                                # (np, nloc)
        raw = self.prolongate(coeffs) if self.n_constraints else np.asarray(_arr(coeffs), dtype=np.float64)
        loc = raw[self.cell_raw_dofs()]                                                    # (ncells, nloc)
        npc, nc = len(ij), self.ncells
        cells = np.repeat(np.arange(nc, dtype=np.int32), npc)
        XY, J = self.map_points(np.ascontiguousarray(cells), np.ascontiguousarray(np.tile(L[:, 0], nc)),
                                np.ascontiguousarray(np.tile(L[:, 1], nc)))
        if code == 0:
            vals = (loc @ T0.T).ravel()
        else:
            dxi = (loc @ np.array([t[1] for t in tabs]).T).ravel()
            deta = (loc @ np.array([t[2] for t in tabs]).T).ravel()
            det = J[:, 0] * J[:, 3] - J[:, 1] * J[:, 2]
            if code == 1:                                   # d/dx = Jinv00 d/dxi + Jinv10 d/deta
                vals = (J[:, 3] * dxi - J[:, 2] * deta) / det
            else:                                           # d/dy = Jinv01 d/dxi + Jinv11 d/deta
                vals = (-J[:, 1] * dxi + J[:, 0] * deta) / det
        tris = (np.arange(nc)[:, None, None] * npc + np.array(sub)[None, :, :]).reshape(-1, 3)
        return mtri.Triangulation(XY[:, 0], XY[:, 1], tris), vals

    def plot(self, coeffs, ax=None, *, contour=False, levels=20, refine=None, deriv=0, colorbar=True,
             mesh=False, **kw):
        """Plot a Function of this space: a smooth colour map (tripcolor, Gouraud shading), or filled
        contours with contour=True.  mesh=True overlays the triangles.  Returns the matplotlib artist."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        tri, vals = self.triangulation(coeffs, refine, deriv)
        if contour:
            art = ax.tricontourf(tri, vals, levels=levels, **kw)
        else:
            kw.setdefault("shading", "gouraud")
            art = ax.tripcolor(tri, vals, **kw)
        if mesh:
            if hasattr(self.mesh, "quads"):
                self.mesh.plot(ax=ax, color="k", lw=0.3, alpha=0.5)
            else:
                ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.triangles, color="k", lw=0.3, alpha=0.5)
        ax.set_aspect("equal")
        if colorbar:
            plt.colorbar(art, ax=ax, fraction=0.046, pad=0.04)
        return art


def _arr(c):
    return c.vector if isinstance(c, Function) else np.asarray(c, dtype=np.float64)


def _is_zero(g):
    return isinstance(g, (int, float)) and g == 0


def _values(g, xy):
    from .forms import Expr, Argument, _Context, evaluate, collect_arguments
    if isinstance(g, Expr):                      # an expression in fd.x, fd.y and Constants
        args = []
        collect_arguments(g, args)
        if any(isinstance(a, Argument) for a in args):
            raise TypeError("boundary data: an expression may contain fd.x, fd.y, numbers and Constants, "
                            "not Functions or test and trial functions")
        ctx = _Context(np.ascontiguousarray(xy[:, 0], dtype=np.float64), {})
        ctx.ys = np.ascontiguousarray(xy[:, 1], dtype=np.float64)
        v = np.asarray(evaluate(g, ctx), dtype=np.float64)
        return np.broadcast_to(v, (xy.shape[0],))
    if callable(g):
        v = np.asarray(g(xy[:, 0], xy[:, 1]), dtype=np.float64)
        return np.broadcast_to(v, (xy.shape[0],))
    return float(g)


def _periodic_pairs(periodic):
    """[(a, b), ...] from periodic=: None, one pair of markers, or a list of pairs."""
    if periodic is None:
        return []
    if isinstance(periodic, (tuple, list)) and len(periodic) == 2 and \
            all(isinstance(m, (int, np.integer)) for m in periodic):
        periodic = [periodic]
    pairs = []
    for p in periodic:
        if not (isinstance(p, (tuple, list)) and len(p) == 2 and all(isinstance(m, (int, np.integer)) for m in p)):
            raise ValueError(f"periodic: a pair of markers (a, b) or a list of pairs, got {periodic!r}")
        a, b = int(p[0]), int(p[1])
        if a == b:
            raise ValueError(f"periodic: a side cannot be identified with itself, got ({a}, {b})")
        pairs.append((a, b))
    return pairs


def _periodic_classes(V, pairs):
    """The representative raw node of every raw node when the sides of each pair (a, b) are identified
    by the translation that takes side a to side b, and per pair a dict with the shift and the nodes."""
    xy = np.asarray(V.raw_coordinates(), dtype=np.float64)
    n = xy.shape[0]
    P = np.asarray(V.mesh.points, dtype=np.float64)
    tol = 1e-8 * float(np.ptp(P, axis=0).max())
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    info = []
    for a, b in pairs:
        A, B = V.boundary_nodes(a), V.boundary_nodes(b)
        if A.size != B.size:
            raise ValueError(f"periodic ({a}, {b}): side {a} has {A.size} nodes and side {b} has {B.size}; the mesh "
                             "must match node for node across the two sides")
        shift = xy[B].mean(axis=0) - xy[A].mean(axis=0)
        dist, j = _C.nearest_points(np.ascontiguousarray(xy[A]), np.ascontiguousarray(xy[B] - shift))
        dist, j = np.asarray(dist), np.asarray(j, dtype=np.int64)
        bad = np.flatnonzero(dist > tol)
        if bad.size or np.unique(j).size != j.size:
            q = xy[B[bad[0]]] if bad.size else xy[B[0]]
            raise ValueError(f"periodic ({a}, {b}): side {b} is not side {a} shifted by ({shift[0]:.6g}, {shift[1]:.6g}); "
                             f"the node at ({q[0]:.6g}, {q[1]:.6g}) has no partner.  The mesh must match node for node "
                             "across the two sides (rectangle_mesh and rectangle_quad_mesh do)")
        if np.hypot(*shift) <= tol:
            raise ValueError(f"periodic ({a}, {b}): the two sides coincide")
        for s_, m_ in zip(B, A[j]):
            ra, rb = find(int(s_)), find(int(m_))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        info.append(dict(pair=(a, b), shift=shift, nodes=np.asarray(A[j]), partners=np.asarray(B)))
    rep = np.array([find(i) for i in range(n)], dtype=np.int64)
    return rep, info


class LagrangeSpace2D(_Lagrange2D, _C.LagrangeSpace2D):
    """Continuous P_k Lagrange elements of degree k >= 1 on a triangular Mesh2D.

    fd.LagrangeSpace(mesh, k, ...) with a Mesh2D returns this class; calling it directly is the same.

    bc:        "free" (default) or "dirichlet" (a condition on every exterior side).
    dirichlet: a marker or a list of markers whose sides carry a Dirichlet condition.
               The two combine.  Sides not named keep the natural condition: Neumann
               or Robin data enter through ds(marker).  A dict {marker: data} names the
               sides and gives their data at once: dirichlet={1: 0.0, 3: g}.
    data:      the Dirichlet data, anything V.lift takes (a number or a callable g(x, y) for
               every Dirichlet side, or a dict {marker: ...}), for instance with bc="dirichlet".
               fd.solve, fd.newton and fd.IRK use the data of the space when the call gives
               none; a callable g(x, y, t) or g(t) is data in time (fd.IRK).

    Nodes: the vertices, k-1 per edge and (k-1)(k-2)/2 inside each triangle, at
    the uniform lattice by default, or Warburton's warp-and-blend points with
    nodes="lobatto" (the two coincide for k <= 2)."""

    @staticmethod
    def _mesh_class():
        from .mesh2d import Mesh2D
        return Mesh2D


class LagrangeSpaceQ(_Lagrange2D, _C.LagrangeSpaceQ):
    """Continuous Q_k Lagrange elements of degree k >= 1 on a QuadMesh.

    fd.LagrangeSpace(mesh, k, ...) with a QuadMesh returns this class; the options are those of
    LagrangeSpace2D (bc=, dirichlet=).

    Nodes: the tensor product of the k+1 Gauss-Lobatto points on every quad, (k+1)^2 per quad:
    the vertices, k-1 per edge (the same points as on a triangle edge) and (k-1)^2 inside.
    The geometry is bilinear, so a quad need not be a parallelogram; the mapped space contains
    every polynomial of total degree k, and all of Q_k on parallelograms."""

    @staticmethod
    def _mesh_class():
        from .quadmesh import QuadMesh
        return QuadMesh


def _space_for(mesh):
    """The 2D Lagrange class for a mesh: LagrangeSpace2D for a Mesh2D, LagrangeSpaceQ for a QuadMesh."""
    from .quadmesh import QuadMesh
    return LagrangeSpaceQ if isinstance(mesh, QuadMesh) else LagrangeSpace2D


# =========================================================================== systems on 2D meshes
class ProductSpace2D:
    """Several fields on one triangular mesh, numbered block by block: a system such as
    velocity and pressure.

        V = fd.VectorFunctionSpace(m, 2, dirichlet=[1, 3, 4])      # P2 velocity (a vector block)
        Q = fd.LagrangeSpace(m, 1)                                  # P1 pressure
        W = fd.ProductSpace(V, Q)                                   # Taylor-Hood
        (u, p), (v, q) = fd.TrialFunctions(W), fd.TestFunctions(W)
        a = (fd.inner(fd.grad(u), fd.grad(v)) - p * fd.div(v) + q * fd.div(u)) * fd.dx

    fd.ProductSpace returns this class when its fields are on a 2D mesh.  A block is a scalar
    space, or a vector space (two components); the test and trial functions and the Functions of
    the system unpack by block.  Coefficient vectors are the fields' vectors one after the other,
    offsets[i] the first index of field i."""

    family = "product"
    tdim = 2
    broken = False

    def __init__(self, *blocks):
        if len(blocks) == 1 and isinstance(blocks[0], (list, tuple)):
            blocks = tuple(blocks[0])
        if not blocks:
            raise ValueError("ProductSpace: at least one space")
        fields, self.blocks, self._slip = [], [], []
        self._block_data = []
        for b in blocks:
            if isinstance(b, ProductSpace2D):
                bd = b.boundary_data
                self._block_data.extend([bd] if b.is_vector else (bd if bd is not None else [None] * len(b.blocks)))
                for kind, idx in b.blocks:
                    self.blocks.append((kind, [len(fields) + i for i in idx]))
                for sp_ in b._slip:
                    self._slip.append(dict(sp_, fields=tuple(len(fields) + i for i in sp_["fields"])))
                fields.extend(b.fields)
            elif isinstance(b, _Lagrange2D):
                self._block_data.append(b.boundary_data)
                self.blocks.append(("scalar", [len(fields)]))
                fields.append(b)
            elif getattr(b, "value_shape", None) == (2,) and getattr(b, "tdim", 1) == 2:
                self._block_data.append(b.boundary_data)          # RT / N1curl: one field, a vector argument
                self.blocks.append(("element", [len(fields)]))
                fields.append(b)
            else:
                raise TypeError(f"ProductSpace on a 2D mesh: every entry must be a 2D space, got {type(b).__name__}")
        self._init(fields)

    @classmethod
    def _make(cls, fields, blocks, slip=()):
        P = cls.__new__(cls)
        P._block_data = [None] * len(blocks)
        P.blocks = [(k, list(i)) for k, i in blocks]
        P._slip = [dict(s) for s in slip]
        P._init(list(fields))
        return P

    def _init(self, fields):
        self._fields = tuple(fields)
        m = fields[0].mesh
        if any(f.mesh is not m for f in fields):
            raise ValueError("ProductSpace: every field must be on the same mesh object")
        self.mesh = m
        self.offsets = np.concatenate([[0], np.cumsum([f.dim for f in fields])]).astype(int)
        self._pattern = {}
        self._block_cache = {}
        # the strong slip condition: coefficients x of this space are x_B = Z x of _base, the same
        # system without it; offsets stay those of _base (the fields' own numbering)
        self._Z = self._ZT = self._base = None
        self.slip_info = []
        if getattr(self, "_slip", None):
            from .slip import slip_matrix
            self._base = ProductSpace2D._make(fields, self.blocks)
            self._Z, self.slip_info = slip_matrix(self._base, self._slip)       # CSRMatrix, in C++
            self._ZT = self._Z.transposed()
        else:
            self._slip = []

    # ---- structure ---------------------------------------------------------------------
    @property
    def fields(self): return self._fields
    @property
    def dim(self): return int(self.offsets[-1]) if self._Z is None else int(self._Z.ncols)
    @property
    def n_constraints(self):
        """Nodal values removed: Dirichlet nodes, plus one per slip node and two per slip corner."""
        return sum(f.n_constraints for f in self._fields) + int(self.offsets[-1]) - self.dim
    @property
    def slip_matrix(self):
        """Z as a RectMatrix from this space to the one without the slip condition: x_B = Z x
        (None without one)."""
        if self._Z is None:
            return None
        from .linalg import RectMatrix
        return RectMatrix(self._Z.copy(), self._base, self)

    def _z(self, x):
        """Z x (the coefficients without the slip condition), in C++; a complex x by parts."""
        return _real_linear(lambda v: self._Z.matvec(np.ascontiguousarray(v, dtype=np.float64)), x)

    def _zt(self, x):
        """Z^T x, in C++; a complex x by parts."""
        return _real_linear(lambda v: self._ZT.matvec(np.ascontiguousarray(v, dtype=np.float64)), x)
    @property
    def degree(self): return max(f.degree for f in self._fields)
    @property
    def is_vector(self):
        """True for a single vector block (a VectorFunctionSpace)."""
        return len(self.blocks) == 1 and self.blocks[0][0] == "vector"

    def block_space(self, b):
        """Block b on its own: its LagrangeSpace2D, or a vector space of its components."""
        kind, idx = self.blocks[b]
        if kind in ("scalar", "element"):
            return self._fields[idx[0]]
        S = self._block_cache.get(b)
        if S is None:
            S = ProductSpace2D._make([self._fields[i] for i in idx], [("vector", list(range(len(idx))))])
            self._block_cache[b] = S
        return S

    def split(self, x):
        """Per-field coefficient vectors (with a slip condition: of the fields, after Z)."""
        x = np.asarray(getattr(x, "vector", x), dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"split: {x.size} coefficients, the system has {self.dim}")
        if self._Z is not None:
            x = self._z(x)
        return [x[self.offsets[i]:self.offsets[i + 1]] for i in range(len(self._fields))]

    def gather(self, parts):
        """The inverse of split (with a slip condition: the orthogonal projection Z^T)."""
        parts = [np.asarray(getattr(p, "vector", p), dtype=np.float64).reshape(-1) for p in parts]
        if len(parts) != len(self._fields):
            raise ValueError(f"gather: {len(parts)} parts for {len(self._fields)} fields")
        x = np.concatenate(parts)
        return x if self._Z is None else self._zt(x)

    @property
    def unconstrained(self):
        """The same system with nothing eliminated (created once); self when nothing is."""
        if self.n_constraints == 0:
            return self
        U = self.__dict__.get("_unconstrained")
        if U is None:
            U = ProductSpace2D._make([f.unconstrained for f in self._fields], self.blocks)
            self._unconstrained = U
        return U

    def prolongate(self, coeffs):
        """Adapted -> raw coefficients, field by field: a vector of self.unconstrained.  A complex
        vector gives a complex result."""
        return _real_linear(lambda c: np.concatenate([f.prolongate(p) for f, p in zip(self._fields, self.split(c))]),
                            coeffs)

    def restrict(self, raw):
        """Raw -> adapted: drops the eliminated nodes (and, with a slip condition, applies Z^T)."""
        def real(r):
            U = self.unconstrained
            x = np.concatenate([f.restrict(p) for f, p in zip(self._fields, U.split(r))])
            return x if self._Z is None else self._zt(x)
        return _real_linear(real, raw)

    def pattern(self, trial=None):
        from .forms import _pattern_2d
        return _pattern_2d(self, self if trial is None else trial)

    # ---- data -------------------------------------------------------------------------------
    @property
    def boundary_data(self):
        """The data given with the spaces: for a vector space the data of VectorFunctionSpace(...,
        dirichlet={marker: pair} or data=), for a system a list with one entry per block (None for a
        block without data); None when there are none.  fd.solve, fd.newton, fd.IRK and lift() use it
        when the call gives no data."""
        bd = getattr(self, "_block_data", None) or []
        if all(d is None for d in bd):
            return None
        return bd[0] if self.is_vector else list(bd)

    def _set_vector_data(self, data):
        if data is None:
            return
        if all(len(f.constrained()) == 0 for f in self._fields):
            raise ValueError("VectorFunctionSpace: data= gives Dirichlet data, and this space has no Dirichlet side")
        if not _has_time(data):
            self._lift_exact(data)                # check what can be checked now
        self._block_data = [data]

    def lift(self, g=None):
        """A Function of self.unconstrained carrying Dirichlet data on the eliminated nodes.

            ug = W.lift([g_u, None])          # one entry per block (None: no data)
            ug = V.lift((gx, gy))             # a vector space alone: a pair, one entry per component
            ug = V.lift(lambda x, y: (x, -y)) # or a callable returning the pair
            ug = V.lift({1: (0, 0), 3: gtop}) # or per marker, each a pair or a callable

        A component's entry is anything LagrangeSpace2D.lift takes."""
        if g is None:
            g = self.boundary_data
        return self._lift_exact(g)

    def _lift_exact(self, g):
        """lift() with exactly the data given: None is zero, the spaces' own data are not used."""
        from .forms import _product_function
        U = self.unconstrained
        if g is None:
            return _product_function(U, "lift", np.zeros(U.dim))
        per_block = [g] if self.is_vector else list(g)
        if len(per_block) != len(self.blocks):
            raise ValueError(f"lift(): one entry per block ({len(self.blocks)}), None for a block without data")
        parts = [None] * len(self._fields)
        for (kind, idx), gb in zip(self.blocks, per_block):
            comps = _components(gb, len(idx)) if kind == "vector" else [gb]
            for i, gi in zip(idx, comps):
                parts[i] = self._fields[i]._lift(gi).vector
        return _product_function(U, "lift", np.concatenate(parts))

    def interpolate(self, f):
        """Nodal interpolant, per block as in lift() (a vector block takes a pair or a callable
        returning a pair)."""
        per_block = [f] if self.is_vector else list(f)
        if len(per_block) != len(self.blocks):
            raise ValueError(f"interpolate(): one entry per block ({len(self.blocks)})")
        parts = [None] * len(self._fields)
        for (kind, idx), fb in zip(self.blocks, per_block):
            if kind == "vector" and callable(fb):
                pts = [self._fields[i].dof_coordinates() for i in idx]
                for k, i in enumerate(idx):
                    vals = fb(pts[k][:, 0], pts[k][:, 1])[k]
                    parts[i] = self._fields[i].interpolate(np.broadcast_to(np.asarray(vals, dtype=np.float64),
                                                                            (self._fields[i].dim,)).copy())
                continue
            if kind == "element":
                F = self._fields[idx[0]]
                parts[idx[0]] = np.zeros(F.dim) if fb is None else F.interpolate(fb)
                continue
            comps = _components(fb, len(idx)) if kind == "vector" else [fb]
            for i, fi in zip(idx, comps):
                parts[i] = np.zeros(self._fields[i].dim) if fi is None else self._fields[i].interpolate(fi)
        x = np.concatenate(parts)
        return x if self._Z is None else self._zt(x)         # the tangential part at slip nodes

    def mass_matrix(self, lumped: bool = False):
        """The mass matrix of the system, sum over blocks of (u, v) (created once).  lumped=True
        gives a diagonal matrix as LagrangeSpace.mass_matrix(lumped=True) does, field by field: the
        Gauss-Lobatto rule when every field is a Q_k space with nodes="lobatto", the row sums otherwise."""
        key = "_mass_lumped" if lumped else "_mass"
        M = self.__dict__.get(key)
        if M is None:
            from .forms import TrialFunctions, TestFunctions, dot, dx, form
            terms = [dot(a, b) if kind in ("vector", "element") else a * b
                     for (kind, _), a, b in zip(self.blocks, TrialFunctions(self), TestFunctions(self))]
            if not lumped:
                M = form(sum(terms[1:], terms[0]) * dx).assemble()
            else:
                if any(kind == "element" for kind, _ in self.blocks):
                    raise ValueError("mass_matrix(lumped=True): a Raviart-Thomas or Nedelec block has no lumped mass "
                                     "(its coefficients are moments, not nodal values)")
                lob = all(f.nverts == 4 and f.node_set == "lobatto" for f in self._fields)
                meas = dx(scheme="lobatto") if lob else dx
                A = form(sum(terms[1:], terms[0]) * meas).assemble()
                d = np.asarray(A.diagonal() if lob else A.matvec(np.ones(self.dim)), dtype=np.float64)
                if lob and np.abs(np.asarray(A._K.combine(_diagonal_csr(d), 1.0, -1.0).data)).max(initial=0.0) > \
                        1e-14 * np.abs(d).max():
                    d = np.asarray(A.matvec(np.ones(self.dim)), dtype=np.float64)   # a slip rotation couples components
                M = _lumped_sparse(d, self)
            self.__dict__[key] = M
        return M

    def project(self, f, degree: int | None = None) -> np.ndarray:
        """L2 projection onto the system: the coefficients c with (c, z) = (f, z) for every z in it.

            c = W.project([eta0, (ux0, uy0)])      # one entry per block, as in interpolate()
            c = V.project(u0)                      # a vector space alone: its one entry

        A scalar entry is a callable f(x, y), a number, a form expression or None (zero).  A vector
        entry is a pair of those, a callable returning the pair, or a vector expression
        (fd.as_vector([...]), a VectorFunction).  The projection is onto the space itself, with
        its Dirichlet conditions (zero there) and slip condition: with slip the components are
        projected together, which a component-wise projection would not do."""
        from .forms import Expr, ListTensor, TestFunction, dx, form, _lift
        per_block = [f] if self.is_vector else list(f)
        if len(per_block) != len(self.blocks):
            raise ValueError(f"project(): one entry per block ({len(self.blocks)}), None for zero")
        parts = [None] * len(self._fields)
        for (kind, idx), fb in zip(self.blocks, per_block):
            if kind == "element":
                F = self._fields[idx[0]]
                if fb is None or (isinstance(fb, (int, float)) and fb == 0):
                    parts[idx[0]] = np.zeros(F.dim)
                elif isinstance(fb, ListTensor):
                    from .forms import dot
                    parts[idx[0]] = np.asarray(form(dot(fb, TestFunction(F)) * (dx(degree) if degree is not None else dx)).assemble())
                else:
                    from .elements2d import _vector_values
                    Q = F.cache(degree if degree is not None else 2 * F.degree + 2)
                    V2 = _vector_values(fb, np.asarray(Q.x()), np.asarray(Q.y()))
                    parts[idx[0]] = np.asarray(_C.assemble_vector_2d(Q, [0, 3], [np.ascontiguousarray(V2[:, 0]),
                                                                                np.ascontiguousarray(V2[:, 1])]))
                continue
            if kind == "vector":
                if isinstance(fb, ListTensor):
                    if fb.rank != 1 or fb.shape[0] != len(idx):
                        raise ValueError(f"project(): a vector block needs a vector of {len(idx)} components")
                    comps = [fb[k] for k in range(len(idx))]
                else:
                    comps = _components(fb, len(idx))
            else:
                comps = [fb]
            for i, fi in zip(idx, comps):
                F = self._fields[i]
                if fi is None or (isinstance(fi, (int, float)) and fi == 0):
                    parts[i] = np.zeros(F.dim)
                elif isinstance(fi, (Expr, int, float)):
                    v = TestFunction(F)
                    parts[i] = np.asarray(form(_lift(fi) * v * (dx(degree) if degree is not None else dx)).assemble())
                elif callable(fi):
                    Q = F.cache(degree if degree is not None else 2 * F.degree + 2)
                    fq = np.broadcast_to(np.asarray(fi(Q.x(), Q.y()), dtype=np.float64), (Q.nent * Q.nq,)).copy()
                    parts[i] = np.asarray(_C.assemble_vector_2d(Q, [0], [fq]))
                else:
                    raise TypeError(f"project(): a callable, a number or an expression per component, got {fi!r}")
        b = np.concatenate(parts)
        if self._Z is not None:
            b = self._zt(b)
        S = self.__dict__.get("_mass_solver")
        if S is None:
            S = self.mass_matrix().solver()
            self._mass_solver = S
        return np.asarray(S.solve(b).vector)

    def plot(self, coeffs, ax=None, *, block=0, quiver=True, **kw):
        """Plot one block: a scalar field as LagrangeSpace2D.plot does, a vector field as its magnitude
        with arrows at the vertices (quiver=False to leave them out)."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        parts = self.split(coeffs)
        kind, idx = self.blocks[block]
        if kind in ("scalar", "element"):
            if kind == "element":
                kw.setdefault("quiver", quiver)
            return self._fields[idx[0]].plot(parts[idx[0]], ax=ax, **kw)
        fs = [self._fields[i] for i in idx]
        refine = kw.pop("refine", None)
        tri, vx = fs[0].triangulation(parts[idx[0]], refine)
        _, vy = fs[1].triangulation(parts[idx[1]], refine)
        kw.setdefault("shading", "gouraud")
        colorbar = kw.pop("colorbar", True)
        art = ax.tripcolor(tri, np.hypot(vx, vy), **kw)
        if quiver:
            P = self.mesh.points
            U = [f.evaluate(p, P) for f, p in zip(fs, (parts[i] for i in idx))]
            ax.quiver(P[:, 0], P[:, 1], U[0], U[1], color="w", alpha=0.8)
        ax.set_aspect("equal")
        if colorbar:
            plt.colorbar(art, ax=ax, fraction=0.046, pad=0.04)
        return art

    def info(self) -> str:
        rows = [f"ProductSpace on {self.mesh.ntriangles} triangles, dim {self.dim}"]
        for b, (kind, idx) in enumerate(self.blocks):
            f = self._fields[idx[0]]
            what = f"vector, {len(idx)} components" if kind == "vector" else ("vector element" if kind == "element" else "scalar")
            rows.append(f"  block {b}: {_short(f)} {what}, dofs {sum(self._fields[i].dim for i in idx)}, "
                        f"{f.bc_description}")
        for d in self.slip_info:
            rows.append(f"  slip u.n = 0 on marker(s) {', '.join(map(str, d['markers']))}: {len(d['nodes'])} nodes, "
                        f"{len(d['corners'])} corner(s) fixed")
        return "\n".join(rows)

    def __repr__(self):
        desc = ", ".join((f"{_short(self._fields[i[0]])}^{len(i)}" if k == "vector" else _short(self._fields[i[0]]))
                         for k, i in self.blocks)
        name = "VectorFunctionSpace" if self.is_vector else "ProductSpace"
        sl = "".join(f", slip={d['markers']}" for d in self.slip_info)
        return f"{name}({desc}{sl}, dim={self.dim})"


def _short(f):
    """P2, Q1, DG0, RT1, N1curl2: the family and degree of a field."""
    fam = getattr(f, "family", "lagrange")
    if fam == "lagrange":
        return ("Q" if getattr(f, "nverts", 3) == 4 else "P") + str(f.degree)
    return {"dg": "DG", "RT": "RT", "N1curl": "N1curl"}.get(fam, str(fam)) + str(f.degree)


def _timeless(g):
    """True unless g is a callable of more than two arguments, g(x, y, t), or of one, g(t)."""
    import inspect
    from .forms import Expr
    if not callable(g) or isinstance(g, Expr):          # an expression in fd.x, fd.y is constant in time
        return True
    try:
        return len(inspect.signature(g).parameters) == 2
    except (TypeError, ValueError):
        return True


def _has_time(data):
    """Whether Dirichlet data (a number, a callable, a pair, a dict or a list of those) depend on t."""
    if isinstance(data, dict):
        return any(_has_time(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return any(_has_time(v) for v in data)
    return not _timeless(data)


def _check_data(V, data):
    """Check Dirichlet data given to a scalar space now, where it can be (data in time are checked
    by the time stepper that evaluates them): the markers must carry a Dirichlet condition, and
    numbers and callables g(x, y) must evaluate."""
    if isinstance(data, dict):
        V._lift({m: (gm if _timeless(gm) else 0.0) for m, gm in data.items()})
    elif _timeless(data):
        V._lift(data)


def _components(g, n):
    """Per-component data of a vector block: None, a tuple/list of n entries, a callable returning n
    values (vectorized), or a dict {marker: pair or callable}."""
    if g is None:
        return [None] * n
    if isinstance(g, (tuple, list)):
        if len(g) != n:
            raise ValueError(f"a vector block needs {n} component entries, got {len(g)}")
        return list(g)
    if isinstance(g, dict):
        per = [dict() for _ in range(n)]
        for m, gm in g.items():
            for k, gk in enumerate(_components(gm, n)):
                per[k][m] = 0.0 if gk is None else gk
        return per
    if callable(g):
        return [(lambda x, y, k=k: np.asarray(g(x, y)[k], dtype=np.float64) + 0 * x) for k in range(n)]
    if isinstance(g, (int, float)) and g == 0:
        return [0.0] * n
    raise TypeError(f"vector data must be a pair, a callable returning a pair, or a dict of those; got {g!r}")


def VectorFunctionSpace(mesh, family="lagrange", degree=None, dim: int = 2, *, slip=None, corner_angle=45.0,
                        slip_normal="conservative", **options):
    """A vector-valued space on a 2D mesh: `dim` copies of LagrangeSpace(mesh, degree, **options),
    one per component, with the same Dirichlet conditions.

        V = fd.VectorFunctionSpace(m, 2, dirichlet=[1, 3])          # the family may be left out
        V = fd.VectorFunctionSpace(m, "lagrange", 2, bc="dirichlet")
        V = fd.VectorFunctionSpace(m, 2, dirichlet=4, slip=[1, 3])  # u . n = 0 on sides 1 and 3

    Its test, trial and known functions are vectors: grad() of them is a matrix, div() a scalar.

    slip: markers of sides with the slip condition u . n = 0, imposed strongly: at each node
      there only the tangential component is a degree of freedom (femd.slip).  At a vertex
      between two slip edges the normal is the length-weighted mean of the edge normals, unless
      they differ by more than corner_angle degrees: that corner is fixed, u = 0.  The tangential
      traction is natural (zero unless the form adds it).  V.slip_info lists nodes and normals.
    slip_normal: at the nodes inside an edge, "conservative" (default) uses the edge's normal, which
      makes int u_h . n ds = 0 exact; "smooth" blends the vertex normals along the edge, which
      follows a curved wall better when k >= 3.  The two coincide on straight walls."""
    if not isinstance(family, str):
        family, degree = "lagrange", family
    if str(family).lower() not in ("lagrange", "cg", "p"):
        raise ValueError(f"VectorFunctionSpace: only the 'lagrange' family on 2D meshes, got {family!r}")
    if degree is None:
        raise TypeError("VectorFunctionSpace: give the degree")
    data = options.pop("data", None)
    if isinstance(options.get("dirichlet"), dict):
        if data is not None:
            raise TypeError("VectorFunctionSpace: give the data either in dirichlet={marker: data} or in data=, not both")
        data = dict(options["dirichlet"])
        options["dirichlet"] = sorted(data)
    comp = [_space_for(mesh)(mesh, degree, **options) for _ in range(int(dim))]
    sl = sorted(set(_markers(slip, "slip")))
    if not sl:
        V = ProductSpace2D._make(comp, [("vector", list(range(int(dim))))])
        V._set_vector_data(data)
        return V
    if int(dim) != 2:
        raise ValueError("VectorFunctionSpace: slip= needs dim = 2")
    both = sorted(set(sl) & set(comp[0].dirichlet_markers))
    if both or comp[0]._bc == "dirichlet":
        raise ValueError(f"VectorFunctionSpace: a side cannot carry both a Dirichlet and a slip condition "
                         f"({'bc=dirichlet' if not both else f'marker(s) {both}'})")
    present = set(np.asarray(mesh.segment_markers).tolist())
    missing = [m for m in sl if m not in present]
    if missing:
        raise ValueError(f"VectorFunctionSpace: no side carries slip marker(s) {missing}; the markers present are "
                         f"{sorted(present)}")
    if slip_normal not in ("conservative", "smooth"):
        raise ValueError(f"VectorFunctionSpace: slip_normal is 'conservative' or 'smooth', got {slip_normal!r}")
    if not 0.0 < float(corner_angle) < 180.0:
        raise ValueError(f"VectorFunctionSpace: corner_angle is in degrees, between 0 and 180, got {corner_angle!r}")
    V = ProductSpace2D._make(comp, [("vector", [0, 1])],
                             slip=[dict(fields=(0, 1), markers=sl, corner_angle=float(corner_angle),
                                        normal=slip_normal)])
    V._set_vector_data(data)
    return V
