"""
Transfer of a Function between two 1D spaces: interpolation and exact L2 projection.

    uW = u.project_to(W)                  # L2 projection, a Function in W
    uW = u.interpolate_to(W)              # collocation at W.dof_coordinates()
    T  = W.transfer_matrix(V)             # the operator, assembled and factored once
    uW = T(u);  cW = T @ u.vector         # Function in, Function out; array in, array out

The projection solves  M_W c_W = B c_V  with  B_ij = int N^W_i N^V_j  over W's
interval.  B is integrated on the union of both grids, where the integrand is a
polynomial of degree p_V + p_W on every piece, with (p_V + p_W)//2 + 1 Gauss points
per piece.  The quadrature is therefore exact whether or not the grids agree, and
the projection is the true L2 projection.  It conserves int u whenever constants
lie in W (every space except one with a condition such as Dirichlet built in).

The interpolation solves  C c_W = E c_V  with  C = N^W(xi),  E = N^V(xi)  at the
Greville abscissae (splines) or nodes (Lagrange) xi of W.  For a nodal target C is
the identity and nothing is solved.

Both maps reproduce u exactly when V is contained in W (a refined grid, a higher
degree, lower continuity).  The target's built-in boundary conditions hold by
construction.  If the source violates one of them, the result is the projection
onto the constrained space, and a warning says so (check=False silences it).

B and E are sparse.  M_W^{-1} B and C^{-1} E are not, so a Transfer keeps the two
factors and applies them, and T.toarray() forms the dense operator on request.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from . import _femd as _C

__all__ = ["Transfer", "mixed_mass"]


def _check_domains(V, W):
    tol = 1e-12 * max(1.0, abs(V.b - V.a))
    if W.a < V.a - tol or W.b > V.b + tol:
        raise ValueError(f"transfer: the target interval [{W.a:g}, {W.b:g}] is not inside "
                         f"the source interval [{V.a:g}, {V.b:g}]")


def _breakpoints(V, W):
    """Union of both grids on W's interval, near-duplicates merged."""
    a, b = W.a, W.b
    g = V.grid
    brk = np.union1d(W.grid, g[(g > a) & (g < b)])
    keep = np.diff(brk) > 1e-13 * (b - a)
    brk = np.r_[brk[0], brk[1:][keep]]
    brk[0], brk[-1] = a, b
    return brk


def _gauss(brk, n):
    xg, wg = (np.asarray(a) for a in _C.gauss_legendre(int(n)))
    h, m = np.diff(brk), 0.5 * (brk[:-1] + brk[1:])
    return (m[:, None] + 0.5 * h[:, None] * xg).ravel(), (0.5 * h[:, None] * wg).ravel()


def _mixed_mass_csr(W, V):
    """B = N_W^T diag(w) N_V in the CSR store (one pass in C++, exact zeros dropped)."""
    _check_domains(V, W)
    x, w = _gauss(_breakpoints(V, W), (W.degree + V.degree) // 2 + 1)
    NW, NV = W._basis_csr(x), V._basis_csr(np.clip(x, V.a, V.b))
    i = np.arange(w.size, dtype=np.int32)
    Dw = _C.csr_from_triplets(int(w.size), int(w.size), i, i, np.ascontiguousarray(w, dtype=np.float64))
    return Dw.triple_product(NW.transposed(), NV)


def _scipy(K):
    """A SciPy copy of a _femd.CSRMatrix, for the functions that return one."""
    return sp.csr_matrix((np.array(K.data), np.array(K.indices), np.array(K.indptr)), shape=(K.nrows, K.ncols))


def mixed_mass(W, V) -> sp.csr_matrix:
    """B_ij = int_{W.a}^{W.b} N^W_i N^V_j, (W.dim x V.dim), exact for any pair of grids (computed in
    C++; returned as a scipy.sparse matrix)."""
    return _scipy(_mixed_mass_csr(W, V))


# --------------------------------------------------------------------------- boundary checks
def _deriv_name(m):
    return "u" + "'" * m if m <= 3 else f"u^({m})"


def _describe(functional):
    parts = []
    for m, al in functional:
        c = "" if al == 1.0 else ("-" if al == -1.0 else f"{al:g} ")
        parts.append(c + _deriv_name(m))
    return " + ".join(parts).replace("+ -", "- ") + " = 0"


def _bc_defects(V, c, W, rtol=1e-8):
    """(where, condition, value) for every condition built into W that the source violates."""
    out = []
    xs = V.grid[(V.grid >= W.a) & (V.grid <= W.b)]
    xs = np.unique(np.r_[W.a, W.b, xs, 0.5 * (xs[:-1] + xs[1:])])
    xs = np.clip(xs, V.a, V.b)
    size = {}
    def scale(m):
        if m not in size:
            size[m] = np.abs(V.evaluate(c, xs, m)).max()
        return size[m]
    def at(x, m):
        return V.evaluate(c, [min(max(x, V.a), V.b)], m)[0]

    if W.bc.is_periodic:
        jump = at(W.b, 0) - at(W.a, 0)
        if abs(jump) > rtol * scale(0) and scale(0) > 0:
            out.append(("at the two ends", "u(a) = u(b)", jump))
        return out
    for where, x, bc in (("left", W.a, W.bc.left), ("right", W.b, W.bc.right)):
        for f in bc.functionals:
            val = sum(al * at(x, m) for m, al in f)
            ref = sum(abs(al) * scale(m) for m, al in f)
            if ref > 0 and abs(val) > rtol * ref:
                out.append((f"at the {where} end", _describe(f), val))
    return out


def _warn_defects(V, c, W, what, stacklevel):
    for where, cond, val in _bc_defects(V, c, W):
        warnings.warn(
            f"{what}: the source does not satisfy the condition {cond} built into the target "
            f"{where} (the source gives {val:.3g}).  The result lies in the constrained space "
            f"and differs from the source near that end.  Pass check=False to silence this.",
            stacklevel=stacklevel)


# --------------------------------------------------------------------------- the operator
class Transfer:
    """The linear map u_V -> u_W from `source` V to `target` W, assembled and factored once.

        T = W.transfer_matrix(V, kind="project")    # or "interpolate"
        T(u)         Function of V  -> Function of W (checks W's boundary conditions)
        T @ c        coefficients   -> coefficients  (plain linear algebra, no checks)
        T.rhs        the sparse factor: B = mixed mass (project) or E = N^V(xi) (interpolate)
        T.lhs        M_W as a Matrix (project), the collocation matrix C, or None if C = I
        T.toarray()  the dense (W.dim x V.dim) operator
    """

    def __init__(self, target, source, kind: str = "project"):
        kind = str(kind).lower()
        if kind not in ("project", "interpolate"):
            raise ValueError(f"Transfer: kind must be 'project' or 'interpolate', not {kind!r}")
        for s, role in ((source, "source"), (target, "target")):
            if not hasattr(s, "basis_matrix"):
                raise TypeError(f"Transfer: the {role} must be a SplineSpace or a LagrangeSpace, "
                                f"not {type(s).__name__}; transfer the fields of a ProductSpace one at a time")
        _check_domains(source, target)
        self.source, self.target, self.kind = source, target, kind
        if kind == "project":
            self._rhsK = _mixed_mass_csr(target, source)
            self.rhs = _scipy(self._rhsK)
            self.lhs = target.mass_matrix()
            solver = self.lhs.solver()
            self._solve = lambda b: np.asarray(solver.solve(b), dtype=np.float64)
        else:
            xi = target.dof_coordinates()
            self._rhsK = source._basis_csr(np.clip(xi, source.a, source.b))
            self.rhs = _scipy(self._rhsK)
            CK = target._basis_csr(xi)
            if CK.identity_defect() < 1e-13:
                self.lhs, self._solve = None, lambda b: b
            else:
                from .linalg import Matrix
                self.lhs = _scipy(CK)
                solver = Matrix.from_csr(CK, periodic=target.bc.is_periodic).solver()
                self._solve = lambda b: np.asarray(solver.solve(b), dtype=np.float64)

    @property
    def shape(self):
        return (self.target.dim, self.source.dim)

    def __matmul__(self, c):
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (self.source.dim,):
            raise ValueError(f"Transfer @ c: need a vector of length {self.source.dim}, got shape {c.shape}")
        return self._solve(self._rhsK.matvec(np.ascontiguousarray(c)))

    def __call__(self, u, check: bool = True, name=None, _stacklevel=3):
        from .forms import Function
        if isinstance(u, Function):
            if u.space is not self.source:
                raise ValueError("Transfer: the Function does not live in this transfer's source space")
            c = u.vector
        else:
            c = np.asarray(u, dtype=np.float64).reshape(-1)
        if check:
            _warn_defects(self.source, c, self.target, f"{self.kind}_to", _stacklevel + 1)
        out = self @ c
        if isinstance(u, Function):
            return Function(self.target, name or u.name, out, _infer=False)
        return out

    def toarray(self) -> np.ndarray:
        R = self.rhs.toarray()
        return np.column_stack([self._solve(R[:, j]) for j in range(R.shape[1])])

    def __repr__(self):
        return f"Transfer({self.kind}, {self.source!r} -> {self.target!r})"
