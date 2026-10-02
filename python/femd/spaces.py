"""
femd.spaces -- the one-dimensional function spaces and the parsing of boundary conditions.

    V = fd.SplineSpace(grid, 3, bc="dirichlet")
    W = fd.LagrangeSpace(grid, 2)
    V = fd.FunctionSpace(grid, "spline", 3)        # the same, by family name

Every space presents the same Python interface (dim, cache, evaluate, at_quad,
basis_matrix, dof_coordinates, project, interpolate, lift, prolongate, restrict,
transfer_matrix, unconstrained) and declares what it is through three attributes
the rest of the package reads instead of testing types:

    family   "spline", "lagrange", ...
    tdim     topological dimension of the mesh (1 here)
    broken   True when the space is discontinuous across element boundaries
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import _femd as _C
from ._femd import Mesh1D, BoundaryCondition, BCSpec, QuadratureCache
from ._util import _vec, _coeff, _real_linear  # noqa: F401
from .forms import Function
from .transfer import Transfer

# --------------------------------------------------------------------------- bc parsing
def Robin(alpha: float, beta: float) -> BoundaryCondition:
    """alpha u + beta u' = 0, built into the basis."""
    return BoundaryCondition.robin(alpha, beta)

def Derivative(m: int) -> BoundaryCondition:
    """u^(m) = 0, built into the basis."""
    return BoundaryCondition.derivative(m)

def _one_bc(x) -> BoundaryCondition:
    if isinstance(x, BoundaryCondition):
        return x
    if x is None:
        return BoundaryCondition.free()
    if isinstance(x, str):
        return BoundaryCondition.from_name(x.lower())
    raise TypeError(f"cannot interpret boundary condition {x!r}")

def _bcspec(bc) -> BCSpec:
    """'free' | 'dirichlet' | 'neumann' | 'clamped' | 'periodic' | BoundaryCondition | (left, right)."""
    if isinstance(bc, BCSpec):
        return bc
    if bc is None:
        return BCSpec()
    if isinstance(bc, str):
        if bc.lower() == "periodic":
            return BCSpec.periodic()
        b = _one_bc(bc)
        return BCSpec(b, b)
    if isinstance(bc, BoundaryCondition):
        return BCSpec(bc, bc)
    if isinstance(bc, (tuple, list)) and len(bc) == 2:
        return BCSpec(_one_bc(bc[0]), _one_bc(bc[1]))
    raise TypeError(f"cannot interpret bc={bc!r}")

def _integer(v, what):
    """v as an int, refusing bools and non-integral numbers instead of truncating them."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer, float, np.floating)) or int(v) != v:
        raise TypeError(f"{what} must be an integer, got {v!r}")
    return int(v)


def _mesh(grid) -> Mesh1D:
    if isinstance(grid, Mesh1D):
        return grid
    return Mesh1D(np.ascontiguousarray(grid, dtype=np.float64))

# --------------------------------------------------------------------------- spaces
class _SpaceMixin:
    """Methods shared by every 1D family; the C++ base does the work.

    The class attributes below are the space protocol: what the rest of the package reads
    to decide what a space can do, instead of testing its type."""

    family = None        # "spline", "lagrange", "dg"
    tdim = 1             # topological dimension of the mesh
    broken = False       # True: discontinuous across elements, derivatives are element-wise

    _data = None         # boundary data given to the constructor (left=, right=), or None

    @property
    def boundary_data(self):
        """The data given with the space, {"left": ..., "right": ...}, or None.  fd.solve,
        fd.newton, fd.IRK and V.lift() use it when the call itself gives no data."""
        return self._data

    def _set_data(self, left, right):
        """Store left=/right= given to the constructor, checked against the built-in conditions
        (a callable of t is checked when a time stepper evaluates it)."""
        if left is None and right is None:
            return
        self._lift(left=None if callable(left) else left, right=None if callable(right) else right)
        self._data = dict(left=left, right=right)

    def cache(self, npts: int | None = None, nder: int = 1, rule: str = "gauss") -> QuadratureCache:
        """Precomputed basis tables at npts points per element (default degree+1), of the
        Gauss-Legendre rule, or with rule="lobatto" of the Gauss-Lobatto rule (at least 2 points)."""
        rule = str(rule).lower()
        if rule not in ("gauss", "lobatto"):
            raise ValueError(f"cache: rule must be 'gauss' or 'lobatto', got {rule!r}")
        if npts is None:
            npts = self.degree + 1 if rule == "gauss" else max(2, self.degree + 1)
        return QuadratureCache(self, int(npts), int(nder), rule == "lobatto")

    def evaluate(self, coeffs, x, k: int = 0) -> np.ndarray:
        """d^k u / dx^k at the points x, for adapted coefficients coeffs (array or Function)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        return _C.FunctionSpace.evaluate(self, _vec(coeffs), np.ascontiguousarray(x), int(k))

    def at_quad(self, cache: QuadratureCache, coeffs, k: int = 0) -> np.ndarray:
        """d^k u / dx^k at every quadrature node of the cache, laid out [e*nq + q]."""
        return _C.FunctionSpace.at_quad(self, cache, _vec(coeffs), int(k))

    def prolongate(self, coeffs) -> np.ndarray:
        """Adapted -> raw coefficients (C^T c).  A complex vector gives a complex result."""
        return _real_linear(lambda c: _C.FunctionSpace.prolongate(self, _vec(c)), coeffs)

    def restrict(self, raw) -> np.ndarray:
        """Raw -> adapted coefficients (C r).  A complex vector gives a complex result."""
        return _real_linear(lambda r: _C.FunctionSpace.restrict(self, _vec(r)), raw)

    def basis_matrix(self, x, k: int = 0) -> sp.csr_matrix:
        """Sparse (len(x) x dim) matrix of d^k N_j(x_i)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        r, c, v = _C.FunctionSpace.basis_at(self, np.ascontiguousarray(x), int(k))
        return sp.csr_matrix((v, (r, c)), shape=(len(x), self.dim))

    def mass_matrix(self, npts: int | None = None, lumped: bool = False) -> "Matrix":
        """M_ij = int N_i N_j, with npts Gauss points per element (default degree+1).

        lumped=True gives the diagonal (lumped) mass. On a Lagrange space with nodes="lobatto" it is
        the Gauss-Lobatto quadrature at the nodes, which is exactly diagonal: the same matrix as
        fd.form(u*v*dx(scheme="lobatto")). On every other space it is the row sum of M,
        d_i = sum_j M_ij (for a DG space with basis="lobatto" this is again the nodal quadrature).
        A row sum that is not positive raises ValueError (equispaced Lagrange of degree 8 or more)."""
        from .linalg import Matrix
        if not lumped:
            return assemble(self.cache(npts, 0), [(0, 0, 1.0)])
        if getattr(self, "family", "") == "lagrange" and getattr(self, "node_set", "") == "lobatto":
            return assemble(self.cache(None if npts is None else npts, 0, "lobatto"), [(0, 0, 1.0)])
        M = assemble(self.cache(npts, 0), [(0, 0, 1.0)])
        d = np.asarray(M.matvec(np.ones(self.dim)), dtype=np.float64)
        if not np.all(d > 0):
            i = int(np.argmin(d))
            raise ValueError(f"mass_matrix(lumped=True): the row sum of the mass matrix is {d[i]:.3g} at dof {i}, "
                             "not positive, so the lumped mass would be singular or indefinite; use a Lagrange "
                             "space with nodes='lobatto'")
        K = _C.AssembledMatrix(self.dim, 0)
        for i, v in enumerate(d):
            K.add_entry(i, i, float(v))
        K.symmetric = True
        return Matrix(K, self.bc.is_periodic, self)

    def project(self, f, npts: int | None = None) -> np.ndarray:
        """L2 projection of a callable f(x), an expression in fd.x (or values at the cache nodes) onto the space.

        A Function of another space is projected exactly, whatever the two grids, through
        transfer_matrix(f.space); npts is then ignored."""
        if isinstance(f, Function):
            return Transfer(self, f.space, "project")(f.vector, _stacklevel=4)
        from .forms import Expr, TestFunction, form, dx
        if isinstance(f, Expr):                      # an expression in fd.x (and Functions): (f, v) by the form language
            v = TestFunction(self)
            b = np.asarray(form(f * v * (dx(npts) if npts is not None else dx)).assemble(), dtype=np.float64)
            M = assemble(self.cache(self.degree + 2, 0), [(0, 0, 1.0)])
            return np.asarray(M.solver().solve(b), dtype=np.float64)
        Q = self.cache(npts if npts is not None else self.degree + 2, 0)
        M = assemble(Q, [(0, 0, 1.0)])
        fq = f(Q.nodes()) if callable(f) else np.asarray(f, dtype=np.float64)
        b = assemble_vector(Q, [(0, fq)], self.dim)
        return np.asarray(M.solver().solve(b), dtype=np.float64)

    @property
    def unconstrained(self):
        """The same grid, degree and continuity with no condition built in, created once and cached.

        Its coefficients are the raw coefficients of this space, so V.prolongate(c) is a
        coefficient vector of V.unconstrained.  A space with nothing built in (free or
        periodic) is its own unconstrained space."""
        if self.bc.is_periodic or self.n_constraints == 0:
            return self
        U = getattr(self, "_unconstrained", None)
        if U is None:
            U = type(self)(self.mesh, bc="free", **self._ctor)
            self._unconstrained = U
        return U

    def lift(self, *g, left=None, right=None):
        """A Function u_g of V.unconstrained satisfying the built-in conditions with data.

            V = fd.SplineSpace(grid, 3, bc=("dirichlet", "neumann"))
            ug = V.lift(left=1.0, right=0.5)        # u(a) = 1, u'(b) = 0.5
            ug = V.lift(left=(1.0, 0.0))            # clamped: u(a) = 1, u'(a) = 0
            ug = V.lift(1.0, 0.5)                   # positional: every functional, left end first

        An end left out gets zero data.  Without arguments it uses the data given with the
        space, SplineSpace(grid, 3, bc="dirichlet", left=1.0), if any.  The full solution of a problem with this data is
        u = V.prolongate(c) + ug, with c the coefficients solved for in V; solve(b, lift=ug)
        and fd.solve(a, L, left=..., right=...) do that for you."""
        if not g and left is None and right is None and self._data is not None:
            left, right = self._data["left"], self._data["right"]
            if callable(left) or callable(right):
                raise TypeError("lift(): the space's boundary data depend on t; give their values, "
                                "V.lift(left=gL(t), right=gR(t))")
        return self._lift(*g, left=left, right=right)

    def _lift(self, *g, left=None, right=None):
        """lift() with exactly the data given: None is zero, the space's own data are not used."""
        from .transfer import _describe
        per = self.bc.is_periodic
        nL = 0 if per else self.bc.left.count
        nR = 0 if per else self.bc.right.count
        if g and (left is not None or right is not None):
            raise TypeError("lift(): pass the data positionally or as left=/right=, not both")
        if g:
            vals = np.asarray(g, dtype=np.float64).ravel()
            if vals.size != nL + nR:
                raise ValueError(f"lift(): the space has {nL + nR} built-in functionals ({nL} left, {nR} right), got {vals.size} values")
        else:
            def side(v, n, where):
                if v is None:
                    return np.zeros(n)
                v = np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
                if per:
                    raise ValueError("lift(): a periodic space has no boundary data")
                bc = getattr(self.bc, where)
                if n == 0:
                    raise ValueError(f"lift(): no condition is built in at the {where} end; natural data "
                                     f"enter the right-hand side through ds('{where}')")
                if v.size != n:
                    conds = ", ".join(_describe(f) for f in bc.functionals)
                    raise ValueError(f"lift(): the {where} end has {n} built-in functional(s) ({conds}), got {v.size} values")
                return v
            vals = np.r_[side(left, nL, "left"), side(right, nR, "right")]
        U = self.unconstrained
        if U is self:
            return Function(self, "lift", np.zeros(self.dim), _infer=False)
        raw = _C.FunctionSpace.lift(self, np.ascontiguousarray(vals))
        return Function(U, "lift", raw, _infer=False)

    def interpolate(self, f) -> np.ndarray:
        """Coefficients of the interpolant of f at the degrees of freedom' coordinates.

        Collocation  sum_j c_j N_j(x_i) = f(x_i)  at x = dof_coordinates(): the nodes for
        Lagrange (where the matrix is the identity) and the Greville abscissae for
        splines (Schoenberg-Whitney guarantees a nonsingular, banded system).  f is a
        callable or an array of values at those points.  The interpolant satisfies the
        built-in boundary conditions, so f should too for the result to be meaningful."""
        pts = self.dof_coordinates()
        vals = np.asarray(f(pts) if callable(f) else f, dtype=np.float64).reshape(-1)
        if vals.shape != pts.shape:
            raise ValueError(f"interpolate: need {pts.size} values (one per degree of freedom), got {vals.size}")
        B = self.basis_matrix(pts)
        if abs(B - sp.identity(self.dim)).max() < 1e-13:          # nodal basis: nothing to solve
            return vals
        return spla.spsolve(B.tocsc(), vals)

    def transfer_matrix(self, source, kind: str = "project") -> "Transfer":
        """The operator taking a Function of `source` to this space, assembled and factored once.

        kind="project": exact L2 projection, M c = B c_source with B the mixed mass matrix
        integrated on the union of both grids.  kind="interpolate": collocation at
        dof_coordinates().  Apply it as T(u) (Function in, Function out) or T @ c (arrays).
        Use it when the same transfer is repeated; u.project_to(W) builds a new one each call."""
        return Transfer(self, source, kind)

    @property
    def grid(self) -> np.ndarray:
        """The vertices the space was built on."""
        return np.asarray(self.mesh.vertices())

    @property
    def a(self) -> float: return self.mesh.a
    @property
    def b(self) -> float: return self.mesh.b

    def info(self) -> str:
        """A multi-line summary of the space; also printed by print(V.info())."""
        m = self.mesh
        kind = type(self).__name__
        cont = getattr(self, "continuity", 0)
        rows = [
            (kind, ""),
            ("interval", f"[{m.a:g}, {m.b:g}]"),
            ("elements", f"{m.nelem}  ({'uniform, h = %g' % m.h(0) if m.is_uniform else 'graded, h in [%g, %g]' % (min(m.h(e) for e in range(m.nelem)), max(m.h(e) for e in range(m.nelem)))})"),
            ("degree", f"{self.degree}   (continuity C^{cont})"),
            ("boundary conditions", self.bc.describe() + ("" if self.bc.is_periodic else f"   ({self.n_constraints} functionals built in)")),
            ("raw basis functions", f"{self.raw_dim}"),
            ("dimension", f"{self.dim}   (degrees of freedom after the boundary conditions)"),
            ("half-bandwidth", f"{self.uband}   (matrices have {2*self.uband+1} nonzeros per row)"),
            ("local functions / element", f"{self.nloc_max}" + ("" if self.nloc_max == self.degree + 1 else f"  (raw {self.degree+1})")),
            ("coefficients are", self._coefficients_are()),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"{k:<{w}}  {v}" if v else k for k, v in rows)


    def _coefficients_are(self) -> str:
        return "B-spline weights (not point values); dof_coordinates() gives their Greville abscissae"


class SplineSpace(_SpaceMixin, _C.SplineSpace):
    """B-splines of `degree` on `grid`, C^{p-1} by default.

    bc: "free" (default) | "dirichlet" | "neumann" | "clamped" | "periodic" |
        a BoundaryCondition | (left, right).  A named condition is BUILT INTO
        the basis; "free" builds in nothing and gives the natural condition.
    continuity: global continuity k in [0, p-1] (default None, meaning p-1).
    continuity_at: {interior node index: k} overrides, k in [0, p-1].
    left, right: the data of the conditions built in at each end (as in V.lift), for instance
      SplineSpace(grid, 3, bc="dirichlet", left=0.0, right=1.0).  fd.solve, fd.newton and fd.IRK
      use them when the call gives none; a callable of t is data in time (fd.IRK).

    A discontinuous space (k = -1) is not available here; use DGSpace, whose forms
    carry the interior facet terms (dS, jump, avg) a DG formulation needs.
    """
    family = "spline"
    def __init__(self, grid, degree: int, bc="free", continuity: int | None = None, continuity_at=None,
                 left=None, right=None):
        p = _integer(degree, "degree")
        if p < 1:
            raise ValueError(f"SplineSpace: degree must be >= 1, got {p}")
        mesh = _mesh(grid)

        def check(k, where):
            k = _integer(k, f"continuity{where}")
            if not 0 <= k <= p - 1:
                dg = ("  A discontinuous space (C^-1) is DGSpace(grid, p), whose forms carry the "
                      "interior facet terms (dS, jump, avg) of a DG formulation."
                      if k < 0 else "")
                raise ValueError(f"SplineSpace: continuity{where} = {k} must lie in [0, p-1] = [0, {p - 1}] "
                                 f"for degree {p}; leave it out for the default C^{p - 1}.{dg}")
            return k

        if continuity is not None:
            continuity = check(continuity, "")
        cat = {}
        for node, k in (continuity_at or {}).items():
            j = _integer(node, "a continuity_at node index")
            if not 1 <= j <= mesh.nelem - 1:
                raise ValueError(f"SplineSpace: continuity_at node {j} is not an interior node "
                                 f"(interior nodes are 1..{mesh.nelem - 1})")
            cat[j] = check(k, f"_at[{j}]")
        super().__init__(mesh, p, _bcspec(bc), -1 if continuity is None else continuity, cat)
        self._grid = np.asarray(self.mesh.vertices())
        self._ctor = dict(degree=p, continuity=continuity, continuity_at=cat)
        k = p - 1 if continuity is None else continuity
        self.continuity = min([k] + list(cat.values()))   # global minimum, used by the form compiler
        self._set_data(left, right)

    def __repr__(self):
        return f"SplineSpace(degree={self.degree}, nelem={self.nelem}, bc={self.bc.describe()}, dim={self.dim})"


class LagrangeSpace(_SpaceMixin, _C.LagrangeSpace):
    """Classical C^0 nodal Lagrange elements of `degree` on `grid`.  Coefficients are nodal values.

    nodes: "equispaced" (default), p+1 equally spaced nodes in each element, or "lobatto",
    the p+1 Gauss-Lobatto points.  The space is the same; the basis differs, and the Lobatto
    one is far better conditioned and interpolates better at high degree (manual, Section 3.2).
    For p <= 2 they coincide.

    Given a Mesh2D instead of a 1D grid, it builds the P_k space on the triangles:
    LagrangeSpace(mesh, k, bc="dirichlet") or LagrangeSpace(mesh, k, dirichlet=[1, 3])
    returns a LagrangeSpace2D (see there for the 2D options)."""
    family = "lagrange"

    def __new__(cls, grid, *args, **kw):
        from .mesh2d import Mesh2D
        from .quadmesh import QuadMesh
        if isinstance(grid, (Mesh2D, QuadMesh)):
            if cls is not LagrangeSpace:
                raise TypeError(f"{cls.__name__} is a 1D space; build the 2D space with fd.LagrangeSpace(mesh, k)")
            from .spaces2d import _space_for
            return _space_for(grid)(grid, *args, **kw)     # not an instance of cls: __init__ is skipped
        return super().__new__(cls)

    def _coefficients_are(self) -> str:
        return "nodal values (" + ("equispaced" if self.equispaced else "Gauss-Lobatto") + " nodes)"

    def __init__(self, grid, degree: int, bc="free", nodes="equispaced", left=None, right=None):
        from .spaces2d import _NODES
        nd = _NODES.get(str(nodes).lower())
        if nd is None:
            raise ValueError(f"LagrangeSpace: nodes is 'equispaced' (the default) or 'lobatto', got {nodes!r}")
        super().__init__(_mesh(grid), int(degree), _bcspec(bc), nd == "equispaced")
        self._ctor = dict(degree=int(degree), nodes=nd)
        self.continuity = 0
        self._set_data(left, right)

    @property
    def node_set(self) -> str:
        """'lobatto' or 'equispaced'."""
        return "equispaced" if self.equispaced else "lobatto"


    def __repr__(self):
        nd = "" if self.equispaced else ", nodes=lobatto"
        return f"LagrangeSpace(degree={self.degree}, nelem={self.nelem}, bc={self.bc.describe()}{nd}, dim={self.dim})"


class ProductSpace(_C.ProductSpace):
    """Several spaces on one mesh, numbered together by position (bounded bandwidth).

    With spaces on a 2D mesh (LagrangeSpace, VectorFunctionSpace) it returns a ProductSpace2D,
    numbered block by block."""
    family = "product"
    tdim = 1

    def __new__(cls, *fields, **kw):
        fs = tuple(fields[0]) if len(fields) == 1 and isinstance(fields[0], (list, tuple)) else fields
        if fs and any(getattr(f, "tdim", 1) == 2 for f in fs):
            if cls is not ProductSpace:
                raise TypeError(f"{cls.__name__} is a 1D product space")
            if kw:
                raise TypeError(f"ProductSpace on a 2D mesh takes no options, got {sorted(kw)}")
            from .spaces2d import ProductSpace2D
            return ProductSpace2D(*fs)                    # not an instance of cls: __init__ is skipped
        return super().__new__(cls)

    @property
    def broken(self):
        return any(f.broken for f in self._fields)
    def __init__(self, *fields, position: bool = True):
        if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
            fields = tuple(fields[0])
        self._fields = fields
        super().__init__(list(fields), position)

    @property
    def fields(self):
        return self._fields

    def split(self, x):
        """Global vector (array or ProductFunction) -> list of per-field arrays."""
        return _C.ProductSpace.split(self, _vec(x))

    def gather(self, parts):
        """Per-field arrays or Functions -> global vector."""
        return _C.ProductSpace.gather(self, [_vec(p) for p in parts])

    @property
    def n_constraints(self) -> int:
        """Functionals built into the fields, all together."""
        return sum(f.n_constraints for f in self._fields)

    @property
    def unconstrained(self):
        """The ProductSpace of the fields' unconstrained spaces (created once).  It is this space
        itself when no field has a condition built in."""
        U = getattr(self, "_unconstrained", None)
        if U is None:
            ufs = [f.unconstrained for f in self._fields]
            U = self if all(a is b for a, b in zip(ufs, self._fields)) else ProductSpace(*ufs)
            self._unconstrained = U
        return U

    def prolongate(self, coeffs) -> np.ndarray:
        """Adapted -> raw coefficients, field by field: a vector of self.unconstrained.  A complex
        vector gives a complex result."""
        def real(c):
            parts = self.split(c)
            return self.unconstrained.gather([f.prolongate(p) for f, p in zip(self._fields, parts)])
        return _real_linear(real, coeffs)

    def _per_field(self, g, where):
        if g is None:
            return [None] * len(self._fields)
        if not isinstance(g, (list, tuple)) or len(g) != len(self._fields):
            raise ValueError(f"{where}: on a ProductSpace give one entry per field (a list of {len(self._fields)}, "
                             "None for a field without data)")
        return list(g)

    @property
    def boundary_data(self):
        """The fields' own boundary data, {"left": [...], "right": [...]} with one entry per field,
        or None when no field has any."""
        ds = [getattr(f, "boundary_data", None) for f in self._fields]
        if all(d is None for d in ds):
            return None
        return dict(left=[d["left"] if d else None for d in ds], right=[d["right"] if d else None for d in ds])

    def lift(self, left=None, right=None):
        """A ProductFunction of self.unconstrained carrying data on the fields' built-in conditions.

            W = fd.ProductSpace(V, Q)
            ug = W.lift(left=[1.0, None], right=[0.0, 2.0])     # one entry per field, as in V.lift

        The full field is W.prolongate(c) + ug.vector."""
        from .forms import ProductFunction
        if left is None and right is None and self.boundary_data is not None:
            left, right = self.boundary_data["left"], self.boundary_data["right"]
        L, R = self._per_field(left, "lift(left=)"), self._per_field(right, "lift(right=)")
        U = self.unconstrained
        parts = [f._lift(left=l, right=r).vector for f, l, r in zip(self._fields, L, R)]
        return ProductFunction(U, "lift", U.gather(parts))

    def __repr__(self):
        return f"ProductSpace({len(self._fields)} fields, dim={self.dim}, uband={self.uband})"


# A cache keeps its space alive (keep_alive in the binding) and remembers it as Q._space, so
# assemble() can find the space of a cache; nothing is held at module level.
_orig_cache = _SpaceMixin.cache
def _cache_remember(self, npts=None, nder=1, rule="gauss"):
    Q = _orig_cache(self, npts, nder, rule)
    Q._space = self
    return Q
_SpaceMixin.cache = _cache_remember

def _space_of(cache):
    V = getattr(cache, "_space", None)
    if V is None:
        raise ValueError("assemble(): pass space=... for a cache not created through V.cache()")
    return V


class DGSpace(_SpaceMixin, _C.DGSpace):
    """Discontinuous piecewise polynomials of `degree` p >= 0 on `grid`.

        V = fd.DGSpace(grid, 2)                          # Legendre basis, diagonal mass matrix
        V = fd.DGSpace(grid, 3, basis="lobatto")         # nodal at the Gauss-Lobatto points
        V = fd.DGSpace(grid, 1, bc="periodic")           # the seam becomes an interior facet

    Every element carries its own p+1 functions, so dim = nelem*(p+1), and a function
    is two-valued at every interior vertex.  Forms couple the elements through the
    interior-facet measure dS with the restrictions u('-') (left element) and u('+')
    (right element), jump(u) = u('-') - u('+') and avg(u).  Derivatives D(u, k) are
    taken element by element, up to k = p.

    basis: "legendre" (default): P_0..P_p on each element, orthogonal, P_l(1) = 1.
           "lobatto": the Lagrange basis on the p+1 Gauss-Lobatto points (p >= 1), so
           coefficients are values and the end nodes are the traces.
    bc:    "free" (default) or "periodic".  Nothing is built into a DG space: boundary
           data enter weakly, through ds.
    """
    family = "dg"
    broken = True

    def __new__(cls, grid, *args, **kw):
        from .mesh2d import Mesh2D
        if isinstance(grid, Mesh2D):
            if cls is not DGSpace:
                raise TypeError(f"{cls.__name__} is a 1D space; build the 2D space with fd.DGSpace(mesh, k)")
            from .elements2d import DGSpace2D
            return DGSpace2D(grid, *args, **kw)              # not an instance of cls: __init__ is skipped
        return super().__new__(cls)

    def __init__(self, grid, degree: int, bc="free", basis: str = "legendre"):
        p = _integer(degree, "degree")
        if p < 0:
            raise ValueError(f"DGSpace: degree must be >= 0, got {p}")
        if bc is None or (isinstance(bc, str) and bc.lower() == "free"):
            periodic = False
        elif isinstance(bc, str) and bc.lower() == "periodic":
            periodic = True
        else:
            raise ValueError(f"DGSpace: bc must be 'free' or 'periodic', got {bc!r}.  A DG space has no built-in "
                             "conditions; impose boundary data weakly through ds")
        b = str(basis).lower()
        if b not in ("legendre", "lobatto"):
            raise ValueError(f"DGSpace: basis must be 'legendre' or 'lobatto', got {basis!r}")
        super().__init__(_mesh(grid), p, periodic, _C.DGBasis.Legendre if b == "legendre" else _C.DGBasis.Lobatto)
        self._ctor = dict(degree=p, basis=b)
        self.basis_name = b
        self.continuity = -1

    def interpolate(self, f) -> np.ndarray:
        """Coefficients of the element-wise interpolant of f: at the Gauss points of each element
        (Legendre) or at its Gauss-Lobatto nodes (Lobatto, where the coefficients are the values).
        f is a callable, or an array of values at dof_coordinates().

        Lobatto samples the element ends, so at a vertex where f jumps both elements receive the
        one value f has there.  Legendre samples inside the elements and reproduces such a jump."""
        pts = self.dof_coordinates()
        vals = np.asarray(f(pts) if callable(f) else f, dtype=np.float64).reshape(-1)
        if vals.shape != pts.shape:
            raise ValueError(f"interpolate: need {pts.size} values (one per degree of freedom), got {vals.size}")
        return np.asarray(self.interpolate_values(np.ascontiguousarray(vals)))

    def _coefficients_are(self) -> str:
        if self.basis_name == "lobatto":
            return "values at the Gauss-Lobatto nodes of each element (two values at every interior vertex)"
        return "Legendre coefficients per element (not point values)"

    def __repr__(self):
        return (f"DGSpace(degree={self.degree}, nelem={self.nelem}, basis={self.basis_name}, "
                f"bc={self.bc.describe()}, dim={self.dim})")


_FAMILIES = {
    "spline": "spline", "b-spline": "spline", "bspline": "spline",
    "lagrange": "lagrange", "cg": "lagrange", "p": "lagrange",
    "dg": "dg", "discontinuous": "dg", "dg-legendre": "dg", "legendre": "dg",
    "dg-lobatto": "dg-lobatto", "lobatto": "dg-lobatto",
}


_FAMILIES_2D = {
    "lagrange": "lagrange", "cg": "lagrange", "p": "lagrange", "q": "lagrange",
    "dg": "dg", "discontinuous": "dg", "discontinuous-lagrange": "dg", "dp": "dg",
    "rt": "rt", "raviart-thomas": "rt", "rtcf": "rt", "rtf": "rt",
    "n1curl": "n1curl", "nedelec": "n1curl", "n1e": "n1curl", "ned": "n1curl", "nédélec": "n1curl",
    "nedelec-1st-kind": "n1curl", "n1-curl": "n1curl",
}


def FunctionSpace(mesh, family: str, degree: int, **options):
    """A function space by family name, on a 1D grid (an array of vertices or a Mesh1D) or a Mesh2D.

        fd.FunctionSpace(grid, "spline", 3, bc="dirichlet")     # = fd.SplineSpace(grid, 3, bc="dirichlet")
        fd.FunctionSpace(grid, "spline", 3, bc="dirichlet", left=0.0, right=1.0)   # with boundary values
        fd.FunctionSpace(grid, "lagrange", 2)                   # = fd.LagrangeSpace(grid, 2)
        fd.FunctionSpace(grid, "dg", 1, bc="periodic")          # = fd.DGSpace(grid, 1, bc="periodic")
        fd.FunctionSpace(grid, "dg-lobatto", 3)                 # = fd.DGSpace(grid, 3, basis="lobatto")

    family (case-insensitive): "spline" (also "bspline"), "lagrange" (also "cg"), "dg" (also
    "discontinuous", "legendre"), "dg-lobatto" (also "lobatto").  The remaining keywords go
    to the family's constructor (bc=, continuity=, continuity_at=, basis=, and the boundary
    values left=, right= of the spline and Lagrange families).
    On a Mesh2D: fd.FunctionSpace(mesh, "lagrange", k, dirichlet=[...]) is fd.LagrangeSpace2D,
    with its options (dirichlet={marker: data}, data=, periodic=, nodes=).  On triangles also
    "DG" (broken P_k, k >= 0: fd.DGSpace2D), "RT" (Raviart-Thomas, also "raviart-thomas":
    fd.RTSpace) and "N1curl" (Nedelec of the first kind, also "nedelec", "N1E": fd.N1curlSpace),
    the last two of degree k >= 1 in UFL's convention (RT1, N1curl1 the lowest order)."""
    from .mesh2d import Mesh2D
    from .quadmesh import QuadMesh
    if isinstance(mesh, (Mesh2D, QuadMesh)):
        fam = str(family).lower().replace("_", "-").replace(" ", "-")
        key2 = _FAMILIES_2D.get(fam)
        if key2 is None:
            raise ValueError(f"FunctionSpace: unknown family {family!r} on a 2D mesh; choose 'lagrange', 'DG', "
                             "'RT' or 'N1curl'")
        if key2 == "lagrange":
            from .spaces2d import _space_for
            return _space_for(mesh)(mesh, degree, **options)
        if isinstance(mesh, QuadMesh):
            raise ValueError(f"FunctionSpace: the {family!r} family exists on triangles only so far")
        from . import elements2d as _E
        return {"dg": _E.DGSpace2D, "rt": _E.RTSpace, "n1curl": _E.N1curlSpace}[key2](mesh, degree, **options)
    key = _FAMILIES.get(str(family).lower())
    if key is None:
        raise ValueError(f"FunctionSpace: unknown family {family!r}; choose one of "
                         "'spline', 'lagrange', 'dg', 'dg-lobatto'")
    if key == "spline":
        return SplineSpace(mesh, degree, **options)
    if key == "lagrange":
        return LagrangeSpace(mesh, degree, **options)
    if "left" in options or "right" in options:
        raise ValueError(f"FunctionSpace: a {family!r} space has no boundary condition built in, so it takes no "
                         "left=/right= values; boundary data of a DG formulation enter weakly, through ds")
    if key == "dg-lobatto":
        options.setdefault("basis", "lobatto")
    return DGSpace(mesh, degree, **options)


from .linalg import assemble, assemble_vector  # noqa: E402  (linalg imports this module)
