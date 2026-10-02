"""
femd.sparse -- matrices on unstructured (2D) spaces, their solvers and preconditioners.

    A = fd.form(fd.dot(fd.grad(u), fd.grad(v)) * fd.dx).assemble()   # a SparseMatrix
    S = A.solver()                        # SuperLU, factored once
    uh = S.solve(b)                       # a Function of the space
    P = A.preconditioner("ilu0")          # Jacobi, SSOR or ILU(0), in C++
    uh, info = fd.cg(A, b, M=P)           # preconditioned CG, all in C++

The values live in FEMd's own CSR store (C++), and A.tocsr() is a scipy.sparse
view of the same three arrays, with no copy.  Every solver here, and fd.gmres /
fd.lgmres, reads the store directly.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import _femd as _C
from .forms import Function

__all__ = ["SparseMatrix", "SparseSolver", "Preconditioner"]


def _vector(b):
    v = getattr(b, "vector", b) if not isinstance(b, np.ndarray) else b
    return np.asarray(v)


def _adopt(space, name, x):
    """A Function (or ProductFunction / VectorFunction on a 2D system) holding x."""
    if hasattr(space, "fields"):
        from .forms import _product_function
        return _product_function(space, name, x)
    return Function._adopt(space, name, x)


class SparseMatrix:
    """An assembled matrix in CSR form.

    .shape, .nnz, .symmetric      size, stored entries, whether it was found symmetric
    .space, .row_space            the spaces the columns and the rows index (when known)
    .tocsr()                      scipy.sparse.csr_matrix VIEW of the store (no copy)
    A @ x                         C++ matvec; a Function when the column space is known
    A + B, A - B, a*A, A.T, A @ B arithmetic, sharing the pattern where the patterns agree
    .solver(backend)              SparseSolver: "superlu" (default), "cg", "gmres", "dense"
    .preconditioner(kind)         "jacobi", "ssor" or "ilu0", for fd.cg / fd.gmres
    """

    def __init__(self, K: _C.CSRMatrix, space=None, row_space=None):
        self._K = K
        self.space = space
        self.row_space = row_space if row_space is not None else space
        self._kernel = K.matvec

    @classmethod
    def from_scipy(cls, A, space=None, row_space=None, symmetric=None):
        """Copy a scipy.sparse matrix into the CSR store."""
        A = sp.csr_matrix(A, dtype=np.float64)
        A.sum_duplicates()
        A.sort_indices()
        K = _C.CSRMatrix.from_arrays(A.shape[0], A.shape[1], A.indptr.astype(np.int32), A.indices.astype(np.int32),
                                     np.ascontiguousarray(A.data), False)
        M = cls(K, space, row_space)
        M.symmetric = M.is_symmetric() if symmetric is None else bool(symmetric)
        return M

    # ---- structure -----------------------------------------------------------------
    @property
    def shape(self): return (self._K.nrows, self._K.ncols)
    @property
    def nnz(self): return self._K.nnz
    @property
    def symmetric(self) -> bool: return self._K.symmetric
    @symmetric.setter
    def symmetric(self, v: bool): self._K.symmetric = bool(v)
    @property
    def data(self):
        """Writable view of the stored values."""
        return self._K.data

    def is_symmetric(self, tol: float = 1e-12) -> bool:
        """Measured: max |a_ij - a_ji| <= tol max |a_ij|."""
        return self.shape[0] == self.shape[1] and self._K.asymmetry() <= tol

    def tocsr(self) -> sp.csr_matrix:
        """scipy.sparse.csr_matrix on the same arrays (a view: writes to .data show in both)."""
        A = sp.csr_matrix((self._K.data, self._K.indices, self._K.indptr), shape=self.shape, copy=False)
        A.has_sorted_indices = True
        return A

    def tocoo(self) -> sp.coo_matrix: return self.tocsr().tocoo()
    def toarray(self) -> np.ndarray: return self._K.dense()
    def diagonal(self) -> np.ndarray: return self._K.diagonal()

    # ---- products ------------------------------------------------------------------
    def matvec(self, x, out=None) -> np.ndarray:
        v = np.ascontiguousarray(_vector(x), dtype=np.float64).reshape(-1)
        if out is None:
            return self._kernel(v)
        self._K.matvec_into(v, out)
        return out

    def __matmul__(self, x):
        if isinstance(x, SparseMatrix):
            if self.shape[1] != x.shape[0]:
                raise ValueError(f"A @ B: shapes {self.shape} and {x.shape} do not chain")
            return SparseMatrix(self._K.multiply(x._K), x.space, self.row_space)
        v = _vector(x)
        if np.iscomplexobj(v):
            return self.matvec(v.real) + 1j * self.matvec(v.imag) if v.ndim == 1 else self.tocsr() @ v
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1:
            return self.tocsr() @ v
        y = self._kernel(np.ascontiguousarray(v))
        if self.row_space is not None:
            return _adopt(self.row_space, "product", y)
        return y

    def rmatvec(self, x) -> np.ndarray:
        """A^T x."""
        return self._K.rmatvec(np.ascontiguousarray(_vector(x), dtype=np.float64))

    # ---- arithmetic ----------------------------------------------------------------
    def _check(self, o, what):
        if not isinstance(o, SparseMatrix):
            return NotImplemented
        if o.shape != self.shape:
            raise ValueError(f"{what}: shapes {self.shape} and {o.shape} differ")
        return o

    def _wrap(self, K):
        return SparseMatrix(K, self.space, self.row_space)

    def __add__(self, o):
        o = self._check(o, "A + B")
        return o if o is NotImplemented else self._wrap(self._K.combine(o._K, 1.0, 1.0))

    def __sub__(self, o):
        o = self._check(o, "A - B")
        return o if o is NotImplemented else self._wrap(self._K.combine(o._K, 1.0, -1.0))

    def __neg__(self): return self._wrap(self._K.scaled(-1.0))

    def __mul__(self, a):
        if isinstance(a, (int, float, np.floating, np.integer)):
            return self._wrap(self._K.scaled(float(a)))
        return NotImplemented
    __rmul__ = __mul__

    def __truediv__(self, a):
        if isinstance(a, (int, float, np.floating, np.integer)):
            return self._wrap(self._K.scaled(1.0 / float(a)))
        return NotImplemented

    def __iadd__(self, o):
        o = self._check(o, "A += B")
        if o is NotImplemented:
            return o
        if o._K.pattern is self._K.pattern or (o.nnz == self.nnz and np.array_equal(o._K.indices, self._K.indices)
                                               and np.array_equal(o._K.indptr, self._K.indptr)):
            self._K.data[:] += o._K.data
            self._K.symmetric = self._K.symmetric and o._K.symmetric
            return self
        return self + o

    def __isub__(self, o):
        return self.__iadd__(-o) if isinstance(o, SparseMatrix) else NotImplemented

    @property
    def T(self) -> "SparseMatrix":
        return SparseMatrix(self._K.transposed(), self.row_space, self.space)

    def copy(self) -> "SparseMatrix":
        return self._wrap(self._K.copy())

    # ---- solvers ---------------------------------------------------------------------
    def solver(self, backend="auto", **options) -> "SparseSolver":
        """A factored (or iterative) solver.

        backend: "auto" or "superlu" (SciPy's SuperLU, the default; symmetric mode for a
                 symmetric matrix), "cg" / "gmres" (FEMd's Krylov solvers, preconditioned
                 by options M= ("ilu0" by default), tol=, maxiter=), or "dense" (LU of the
                 dense matrix: small problems and tests)."""
        return SparseSolver(self, backend, **options)

    def preconditioner(self, kind: str = "ilu0", omega: float = 1.0, ordering: str = "rcm") -> "Preconditioner":
        """Jacobi, SSOR(omega) or ILU(0), built in C++ from this matrix.  SSOR and ILU(0) work on the
        matrix reordered by reverse Cuthill-McKee (ordering="rcm", the default) or in the numbering
        of the space (ordering="natural")."""
        return Preconditioner(self, kind, omega, ordering)

    def __repr__(self):
        sym = ", symmetric" if self.symmetric else ""
        return f"SparseMatrix({self.shape[0]} x {self.shape[1]}, nnz={self.nnz}{sym})"


class Preconditioner:
    """z = M^{-1} r in C++, on a SparseMatrix: kind "jacobi", "ssor" (omega) or "ilu0".

    ordering="rcm" (the default) builds SSOR and ILU(0) on P A P^T, P the reverse Cuthill-McKee
    order, and applies P^T M^{-1} P: on unstructured meshes ILU(0) then needs 30 to 45 per cent
    fewer CG iterations.  ordering="natural" keeps the numbering of the space.  Jacobi ignores it.

    Use as M= in fd.cg, fd.gmres and fd.lgmres (the iteration then stays in C++), or call
    it: P(r) or P.solve(r)."""

    def __init__(self, A: SparseMatrix, kind: str = "ilu0", omega: float = 1.0, ordering: str = "rcm"):
        k = str(kind).lower().replace("(", "").replace(")", "")
        o = str(ordering).lower()
        if o not in ("rcm", "natural"):
            raise ValueError(f"preconditioner ordering must be 'rcm' or 'natural', got {ordering!r}")
        if A.shape[0] != A.shape[1]:
            raise ValueError("a preconditioner needs a square matrix")
        if k == "jacobi":
            self._impl = _C.JacobiPreconditioner(A._K)
        elif k in ("ssor", "sgs"):
            self._impl = _C.SSORPreconditioner(A._K, float(omega) if k == "ssor" else 1.0, o)
        elif k in ("ilu0", "ilu", "ic0"):
            self._impl = _C.ILU0Preconditioner(A._K, o)
        else:
            raise ValueError(f"preconditioner kind must be 'jacobi', 'ssor' or 'ilu0', got {kind!r}")
        self.kind = self._impl.name
        self.ordering = self._impl.ordering
        self.space = A.space

    @property
    def size(self): return self._impl.size

    @property
    def min_pivot(self):
        """Smallest ILU(0) pivot (ILU(0) only); a negative one means it is not SPD, not for CG."""
        return getattr(self._impl, "min_pivot", None)

    def solve(self, r):
        return self._impl.apply(np.ascontiguousarray(_vector(r), dtype=np.float64))
    __call__ = solve

    def __repr__(self): return f"Preconditioner({self.kind}, {self.ordering}, n={self.size})"


def _with_lift(space, x, into, lift):
    """u = prolongate(x) + lift, a Function of the space's unconstrained companion."""
    V = space
    if V is None:
        raise TypeError("solve(b, lift=...) needs a matrix whose space is known")
    U = V.unconstrained
    if hasattr(lift, "vector"):
        if getattr(lift, "space", None) is not U:
            raise ValueError("solve(b, lift=...): the lift must be a Function of V.unconstrained, as V.lift() returns")
        lv = lift.vector
    else:
        lv = np.asarray(lift, dtype=np.float64).reshape(-1)
        if lv.shape != (U.dim,):
            raise ValueError(f"solve(b, lift=...): need {U.dim} raw coefficients, got {lv.size}")
    full = (x if U is V else V.prolongate(x)) + lv
    if into is not None:
        if into.space is not U:
            raise ValueError("solve(b, into=u, lift=...): u must be a Function of V.unconstrained")
        into.vector[:] = full
        return into
    return _adopt(U, "solution", full)


class SparseSolver:
    """A solver for a SparseMatrix: solve(b, into=None, lift=None) as LinearSolver does.

    .backend  "cholesky", "ldlt", "superlu", "cg", "gmres" or "dense": the one in use
    .info     the KrylovInfo of the last iterative solve (None for a direct backend)
    .factor   the C++ SparseCholesky (backend "cholesky": nnz_L, flops, min_pivot, ...) or
              SparseLDLT (backend "ldlt": nnz_L, inertia, delayed, two_by_two, ...)

    backend "auto" (the default) takes FEMd's own sparse Cholesky (LDL^T with AMD ordering, in
    C++) for a matrix that is symmetric with a positive diagonal.  A symmetric matrix that the
    Cholesky refuses (a pivot <= 0), or whose diagonal is not positive (a saddle point), goes to
    FEMd's pivoting LDL^T ("ldlt": multifrontal, 1x1 and 2x2 threshold pivots), and every
    other matrix to SuperLU."""

    def __init__(self, A: SparseMatrix, backend="auto", **options):
        b = getattr(backend, "name", backend)
        b = "auto" if b is None else str(b).lower()
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"solver(): the matrix is {A.shape[0]} x {A.shape[1]}, not square")
        if b in ("cholesky", "chol"):
            ordering = options.pop("ordering", "amd")
            pd = options.pop("positive_definite", True)
            if options:
                raise TypeError(f"the cholesky backend takes ordering= and positive_definite=, got {sorted(options)}")
            self.A, self.space, self.backend, self.info = A, A.space, "cholesky", None
            self.factor = _C.SparseCholesky(A._K, ordering, bool(pd))
            return
        if b in ("ldlt", "ldl", "bunch-kaufman", "indefinite"):
            ordering = options.pop("ordering", "amd")
            u = options.pop("threshold", 0.01)
            if options:
                raise TypeError(f"the ldlt backend takes ordering= and threshold=, got {sorted(options)}")
            if not A.symmetric:
                raise ValueError("solver('ldlt'): the matrix is not symmetric (SparseMatrix.symmetric is False); "
                                 "use 'superlu'")
            self.A, self.space, self.backend, self.info = A, A.space, "ldlt", None
            self.factor = _C.SparseLDLT(A._K, ordering, float(u))
            return
        if b in ("auto", "direct"):
            if not options and A.shape[0] > 0 and A.symmetric:
                self.A, self.space, self.info = A, A.space, None
                if bool(np.all(A.diagonal() > 0)):
                    try:
                        self.backend = "cholesky"
                        self.factor = _C.SparseCholesky(A._K, "amd", True)
                        return
                    except RuntimeError:
                        pass                                 # not positive definite: the pivoting LDL^T
                self.backend = "ldlt"
                self.factor = _C.SparseLDLT(A._K, "amd", 0.01)
                return
            b = "superlu"
        if b in ("splu", "superlu", "lu"):
            b = "superlu"
        self.factor = None
        self.A, self.space, self.backend, self.info = A, A.space, b, None
        if b == "superlu":
            if options:
                raise TypeError(f"the superlu backend takes no options, got {sorted(options)}")
            C = A.tocsr().tocsc()
            self._lu = None
            if A.symmetric:
                # symmetric mode (minimum degree on A + A^T, diagonal pivots) is the fast one, but it does
                # not pivot, so it is checked on one solve: a saddle point (zero diagonal block) or another
                # indefinite matrix can lose every digit there, and then the pivoting LU is used instead
                lu = spla.splu(C, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
                x0 = 1.0 + 0.5 * np.sin(1.0 + 0.7 * np.arange(C.shape[0]))
                b0 = C @ x0
                with np.errstate(all="ignore"):
                    x = lu.solve(b0)
                    anorm = abs(C).sum(axis=1).max() if C.nnz else 0.0
                    err = np.abs(C @ x - b0).max() / (anorm * np.abs(x).max() + np.abs(b0).max() + 1e-300)
                if np.all(np.isfinite(x)) and err <= 1e-11:
                    self._lu = lu
            if self._lu is None:
                self._lu = spla.splu(C)
        elif b == "dense":
            if options:
                raise TypeError(f"the dense backend takes no options, got {sorted(options)}")
            import scipy.linalg as sla
            self._lu = sla.lu_factor(A.toarray())
        elif b in ("cg", "gmres", "lgmres"):
            M = options.pop("M", "ilu0")
            if isinstance(M, str):
                M = A.preconditioner(M)
            self._M = M
            self._opts = options
        else:
            raise ValueError(f"unknown backend {backend!r}: 'auto' (default), 'cholesky', 'ldlt', 'superlu', 'cg', "
                             "'gmres', 'lgmres' or 'dense'")

    @property
    def size(self): return self.A.shape[0]

    def _solve_array(self, b):
        if self.backend in ("cholesky", "ldlt"):
            b = np.asarray(b, dtype=np.float64)
            if b.ndim == 1:
                return self.factor.solve(np.ascontiguousarray(b))
            return np.column_stack([self.factor.solve(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])])
        if self.backend == "superlu":
            return self._lu.solve(b)
        if self.backend == "dense":
            import scipy.linalg as sla
            return sla.lu_solve(self._lu, b)
        from . import krylov
        f = {"cg": krylov.cg, "gmres": krylov.gmres, "lgmres": krylov.lgmres}[self.backend]
        x, self.info = f(self.A._K, b, M=self._M, **self._opts)
        return np.asarray(x)

    def solve(self, b, into=None, lift=None):
        bc = np.asarray(_vector(b))
        if np.iscomplexobj(bc):
            if into is not None or lift is not None:
                raise TypeError("solve(): a complex right-hand side gives a complex array; into= and lift= need a real one")
            return self._solve_array(np.ascontiguousarray(bc.real)) + 1j * self._solve_array(np.ascontiguousarray(bc.imag))
        x = np.asarray(self._solve_array(np.ascontiguousarray(bc, dtype=np.float64).reshape(-1)), dtype=np.float64)
        if lift is not None:
            return _with_lift(self.space, x, into, lift)
        if into is not None:
            into.vector[:] = x
            return into
        if self.space is not None:
            return _adopt(self.space, "solution", x)
        return x

    def __repr__(self): return f"SparseSolver({self.backend}, n={self.size})"
