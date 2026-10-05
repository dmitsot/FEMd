"""
femd.linalg -- assembled matrices, factored solvers, block operators and the low-level kernels.

    A = fd.form(...).assemble()      # a Matrix (banded) or a RectMatrix (two spaces)
    S = A.solver()                   # a LinearSolver
    K = fd.block([[A, B], [C, M]])   # one banded Matrix on a ProductSpace
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import _femd as _C
from ._femd import QuadratureCache, Solver, has_fftw
from ._util import _vec, _coeff
from .spaces import ProductSpace, LagrangeSpace, SplineSpace, _space_of
from .forms import Function, ProductFunction, Form, form

# --------------------------------------------------------------------------- assembly
class LinearSolver:
    """A factored operator.  solve(b) returns a Function in the operator's space when that
    space is known (a matrix from a form or from V.cache()), otherwise an ndarray.
    solve(b, into=u) writes into an existing Function.  .vector on the result gives the array."""
    def __init__(self, impl: "_C.LinearSolver", space=None):
        self._impl, self.space = impl, space

    @property
    def backend(self): return self._impl.backend
    @property
    def pivoted(self) -> bool:
        """True when Auto fell back to a banded LU with partial pivoting: the matrix is indefinite
        (Cholesky refused it) or the pivot-free banded LU broke down.  backend is then Band, or
        Cyclic with periodic corners (the folded periodic LU, used when the Cholesky refuses the
        matrix or the corner-corrected solve fails its backward-error check)."""
        return bool(self._impl.pivoted)
    @property
    def size(self): return self._impl.size

    def solve(self, b, into=None, lift=None):
        bc = np.asarray(b.vector if isinstance(b, (Function, ProductFunction)) else b)
        if np.iscomplexobj(bc):
            if into is not None or lift is not None:
                raise TypeError("solve(): a complex right-hand side gives a complex array; into= and lift= need a real one")
            bc = bc.reshape(-1)       # the factorization is real: solve the two parts
            return (self._impl.solve(np.ascontiguousarray(bc.real)) + 1j * self._impl.solve(np.ascontiguousarray(bc.imag)))
        x = self._impl.solve(np.ascontiguousarray(np.asarray(bc, dtype=np.float64).reshape(-1)))
        if lift is not None:
            return self._with_lift(x, into, lift)
        if into is not None:
            into.vector[:] = x
            return into
        if self.space is not None and not isinstance(self.space, ProductSpace):
            return Function._adopt(self.space, "solution", x)
        if isinstance(self.space, ProductSpace):
            return ProductFunction(self.space, "solution", x)
        return x

    def _with_lift(self, x, into, lift):
        """u = prolongate(x) + lift, a Function of the space's unconstrained companion."""
        V = self.space
        if V is None or isinstance(V, ProductSpace):
            raise TypeError("solve(b, lift=...) needs a matrix on one plain space")
        U = V.unconstrained
        if isinstance(lift, Function):
            if lift.space is not U:
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
        return Function._adopt(U, "solution", full)

    def __repr__(self):
        return f"LinearSolver({self.backend}{', pivoted' if self.pivoted else ''}, n={self.size})"


class MatrixInfo:
    """What a matrix actually is, measured rather than declared.

    Every number here is read off the entries.  `Matrix.symmetric` by contrast is
    a declaration the assembler makes from the roles of its terms, and it is
    writable, so the two can disagree; `reserved_uband` is the half-bandwidth the
    space set aside, which a given form need not fill.  `warnings` lists the
    disagreements worth acting on."""

    __slots__ = ("n", "shape", "nnz", "density", "bandwidth", "cyclic_bandwidth", "reserved_uband",
                 "corner_entries", "block_period", "scale", "symmetry", "skew", "circulant",
                 "diagonal_dominance", "periodic", "symmetric_flag", "is_symmetric", "is_skew",
                 "is_circulant", "kind", "backend", "auto_backend", "warnings", "tol")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        w = "".join(f"\n  ! {t}" for t in self.warnings)
        auto = "" if self.backend == self.auto_backend else f"   (Auto would pick {self.auto_backend})"
        band = f"half-bandwidth {self.cyclic_bandwidth if self.corner_entries else self.bandwidth}"
        if self.reserved_uband != (self.cyclic_bandwidth if self.corner_entries else self.bandwidth):
            band += f", space reserves {self.reserved_uband}"
        band += f", {self.corner_entries} wrap entries" if self.corner_entries else ", no wrap"
        period = "-" if not self.block_period else ("1 (circulant)" if self.block_period == 1 else str(self.block_period))
        circ = f"{self.circulant:.1e}" if self.corner_entries else "n/a (not periodic)"
        return (f"Matrix {self.n} x {self.n}, {self.kind}\n"
                f"  storage             band, {band}\n"
                f"  nonzeros            {self.nnz} ({self.density:.1%} of dense)\n"
                f"  symmetry defect     {self.symmetry:<10.1e}  skew defect    {self.skew:.1e}\n"
                f"  circulant defect    {circ:<10s}  block period   {period}\n"
                f"  diagonal dominance  {self.diagonal_dominance:<+10.1e}  scale          {self.scale:.1e}\n"
                f"  solver              {self.backend}{auto}{w}")

class Matrix:
    """An assembled matrix: banded accumulator plus periodic corners.

    .tocsr()  -> scipy.sparse.csr_matrix
    .solver() -> factorized direct solver over the inherited banded/cyclic backends
    .space    -> the (trial) space the columns index, when known
    """
    def __init__(self, K: _C.AssembledMatrix, periodic: bool, space=None):
        self._K = K
        self.periodic = periodic
        self.space = space
        self._kernel = getattr(K, "matvec", None)   # bound once: this sits in time-stepping inner loops
        self._kernel_into = getattr(K, "matvec_into", None)

    @property
    def shape(self):
        return (self._K.n, self._K.n)

    @property
    def symmetric(self) -> bool:
        return self._K.symmetric

    @symmetric.setter
    def symmetric(self, v: bool):
        self._K.symmetric = bool(v)

    def tocoo(self) -> sp.coo_matrix:
        r, c, v = self._K.to_coo()
        return sp.coo_matrix((v, (r, c)), shape=self.shape)

    def tocsr(self) -> sp.csr_matrix:
        return self.tocoo().tocsr()

    def solver(self, backend=Solver.Auto) -> "LinearSolver":
        """Factorized direct solver.  backend: Auto | Dense | Band | SymBand | Cyclic | SymCyclic |
        Circulant | Diagonal (as fd.Solver.X or a case-insensitive string, 'fft' = Circulant).
        Auto measures the matrix (entries below 1e-14 of the largest count as zero): Diagonal when
        only the diagonal remains, else Band/SymBand, or Cyclic/SymCyclic when there are wrap
        entries (Sym* when .symmetric).  Banded backends factor the measured band, not the reserved one.
        Auto factors at once: a symmetric matrix that is not positive definite (Cholesky refuses it),
        or a pivot-free banded LU that breaks down, falls back to the banded LU with partial pivoting
        (.pivoted is then True).  With periodic corners SymCyclic or Cyclic is checked on one solve
        (backward error 1e-11) and falls back to the folded periodic LU with pivoting.  An explicit
        SymBand or SymCyclic raises on such a matrix instead of returning NaN.
        Circulant is the FFT solver: periodic space, uniform grid, constant coefficients, FFTW linked."""
        if isinstance(backend, str):
            names = {n.lower(): getattr(Solver, n) for n in ("Auto", "Dense", "Band", "SymBand", "Cyclic", "SymCyclic", "Circulant", "Diagonal")}
            names.update({"fft": Solver.Circulant, "banded": Solver.Band, "symmetric": Solver.SymBand, "diag": Solver.Diagonal})
            key = backend.strip().lower().replace("_", "").replace("-", "")
            if key not in names:
                raise ValueError(f"unknown solver backend {backend!r}; choose one of "
                                 "Auto, Dense, Band, SymBand, Cyclic, SymCyclic, Circulant, Diagonal (or 'fft')")
            backend = names[key]
        try:
            impl = _C.LinearSolver(self._K, self.periodic, backend)
        except ValueError as e:
            if backend == Solver.Circulant and "not circulant" in str(e):
                raise ValueError(self._why_not_circulant()) from None
            raise
        return LinearSolver(impl, self.space)

    def _why_not_circulant(self) -> str:
        """The FFT backend refused this matrix, which it does by measurement.  Name the cause."""
        st = _C.structure(self._K, 1e-10)
        head = (f"solver('fft'): the matrix is not circulant (relative defect {st.circulant:.1e}, "
                f"tolerance 1e-10), so an FFT solve would invert a different matrix.  ")
        period = f" (block-circulant with period {st.block_period})" if st.block_period > 1 else ""
        V = self.space
        if V is None:
            cause = "The matrix carries no space, so the cause cannot be named; A.classify() shows its structure."
        elif isinstance(V, ProductSpace):
            cause = (f"It couples several fields of a ProductSpace{period}.  Decouple the blocks and "
                     "solve each with the FFT, as examples/rlw_conservative.ipynb does.")
        elif not V.mesh.is_uniform:
            h = np.diff(V.grid)
            cause = (f"The grid is not uniform: element widths range from {h.min():.3g} to {h.max():.3g} "
                     f"(ratio {h.max() / h.min():.3g}).  The FFT solve needs equal elements.")
        elif isinstance(V, LagrangeSpace) and V.degree >= 2:
            cause = (f"Periodic Lagrange elements of degree {V.degree} are not circulant even on a uniform "
                     f"grid: vertex and interior nodes carry different basis functions{period}.  Static "
                     "condensation of the interior nodes leaves a circulant system on the vertices.")
        elif isinstance(V, SplineSpace) and V.continuity < V.degree - 1:
            cat = {k: c for k, c in (V._ctor.get("continuity_at") or {}).items() if c < V.degree - 1}
            if cat:
                cause = (f"continuity_at={cat} lowers the continuity at particular nodes, which breaks "
                         "the shift invariance of the basis.")
            else:
                cause = (f"Splines of degree {V.degree} with continuity C^{V.continuity} repeat a knot at "
                         f"every node, so the basis is shift-invariant only by whole elements{period}.")
        else:
            cause = ("The grid is uniform and the basis is shift-invariant, so a coefficient of the form "
                     "varies in x (a Function or an expression in fd.x), or the matrix was combined from "
                     "such matrices.  The FFT backend needs constant coefficients.")
        tail = (f"  Use '{'SymCyclic' if self.symmetric else 'Cyclic'}' (what Auto picks), "
                "which is O(np) on any periodic grid.")
        return head + cause + tail

    def __matmul__(self, x):
        """`A @ x` with a vector, or `A @ B` with another Matrix.

        A 1-D vector goes through the C++ banded kernel, which reads the assembled
        band directly: O(n(2p+1)) with no index arrays and no sparse conversion, and
        returns a Function when the space is known.  Several columns at once, or an
        older extension without the kernel, fall back to SciPy.  Another Matrix gives
        the product, half-bandwidth p_self + p_B, which is never declared symmetric."""
        if isinstance(x, RectMatrix):
            if self.space is not x.row_space:
                raise ValueError("A @ B: the columns of A and the rows of B index different spaces")
            return RectMatrix(self.tocsr() @ x._A, self.space, x.col_space)
        if isinstance(x, Matrix):
            o = self._pair(x, "A @ B")
            return self._wrap(self._K.multiply(o._K), o)
        v = np.asarray(x)
        if np.iscomplexobj(v):       # the matrix is real: apply it to the two parts, give a complex array
            return self.matvec(v.real) + 1j * self.matvec(v.imag) if v.ndim == 1 else self.tocsr() @ v
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1 and self._kernel is not None:
            if not v.flags.c_contiguous:
                v = np.ascontiguousarray(v)
            y = self._kernel(v)
        else:
            y = self.tocsr() @ v
        if y.ndim == 1 and self.space is not None and not isinstance(self.space, ProductSpace):
            return Function._adopt(self.space, "product", y)
        return y

    def inner(self, x, y=None) -> float:
        """y^T A x, or x^T A x when y is None, straight off the band, without forming A x
        and without BLAS (see fd.ddot).  The rows are added in an ordered chunked sum."""
        xv = x.vector if isinstance(x, Function) else x
        xv = np.ascontiguousarray(np.asarray(xv), dtype=np.float64)
        if y is None:
            yv = xv
        else:
            yv = y.vector if isinstance(y, Function) else y
            yv = np.ascontiguousarray(np.asarray(yv), dtype=np.float64)
        if self._K is None or not hasattr(self._K, "inner"):
            return float(yv @ (self.tocsr() @ xv))
        return self._K.inner(xv, yv)

    def matvec(self, x, out=None) -> np.ndarray:
        """y = A x as a plain array, through the banded kernel.

        out= writes into an existing float64, C-contiguous array of length n and returns it,
        so a time-stepping loop allocates nothing.  out must not share memory with x."""
        v = x.vector if isinstance(x, Function) else x
        if np.iscomplexobj(v):
            if out is not None:
                raise TypeError("matvec(x, out): a complex x needs out=None")
            v = np.asarray(v)
            return self._kernel(np.ascontiguousarray(v.real)) + 1j * self._kernel(np.ascontiguousarray(v.imag))
        v = np.asarray(v, dtype=np.float64)
        if not v.flags.c_contiguous:
            v = np.ascontiguousarray(v)
        if out is None:
            return self._kernel(v)
        if not isinstance(out, np.ndarray) or out.dtype != np.float64 or not out.flags.c_contiguous:
            raise TypeError("matvec(x, out): out must be a C-contiguous float64 ndarray")
        self._kernel_into(v, out)
        return out

    def is_circulant(self, tol: float = 1e-10) -> bool:
        """True when the matrix is generated by its first column (periodic, uniform, constant coefficients)."""
        return self.periodic and _C.circulant_defect(self._K) <= tol

    def symbol(self) -> np.ndarray:
        """Eigenvalues of a circulant matrix, lambda_k = fft(first column): the symbol of the
        discrete operator, ordered by wavenumber index k = 0..n-1 (numpy fft convention).
        Raises if the matrix is not circulant.  Needs no FFTW."""
        if not self.is_circulant():
            raise ValueError("symbol(): matrix is not circulant (needs periodic, uniform grid, constant coefficients)")
        return np.fft.fft(_C.first_column(self._K))

    def classify(self, tol: float = 1e-12) -> "MatrixInfo":
        """Measure what kind of matrix this is: bandwidth, symmetry, skew, circulant
        or block-circulant structure, diagonal dominance, and the solver backend those
        measurements call for.  O(n(2b+1)) with b the measured bandwidth.

        Nothing here is taken from a flag or from the space, so it can disagree with
        `self.symmetric` and with `self._K.uband`.  Any disagreement worth acting on
        shows up in `.warnings`."""
        st = _C.structure(self._K, tol)
        n = st.n
        wrap = st.corner_entries > 0
        sym = st.scale > 0.0 and st.symmetry <= tol
        skw = st.scale > 0.0 and st.skew <= tol
        circ = wrap and st.circulant <= tol

        if st.scale == 0.0:
            kind = "zero"
        else:
            adj = "symmetric " if sym else ("skew-symmetric " if skw else "")
            if st.bandwidth == 0:
                base = "diagonal"
            elif circ:
                base = "circulant"
            elif wrap and st.block_period > 1:
                base = f"block-circulant (period {st.block_period})"
            elif wrap:
                base = "cyclic banded"
            else:
                base = "banded"
            kind = adj + base

        if st.scale > 0.0 and st.bandwidth == 0 and not wrap:
            backend = auto = "Diagonal"
        elif wrap:
            backend = "Circulant" if (circ and has_fftw()) else ("SymCyclic" if sym else "Cyclic")
            auto = "SymCyclic" if self.symmetric else "Cyclic"
        else:
            backend = "SymBand" if sym else "Band"
            auto = "SymBand" if self.symmetric else "Band"

        warn = []
        if sym and not self.symmetric:
            warn.append(f"flagged nonsymmetric but measures symmetric (defect {st.symmetry:.1e}); "
                        f"setting .symmetric = True lets Auto pick {backend}, about half the work")
        if self.symmetric and not sym:
            warn.append(f"flagged symmetric but measures nonsymmetric (defect {st.symmetry:.1e}); "
                        "a symmetric backend would read only one triangle and give wrong answers")
        meas = st.cyclic_bandwidth if wrap else st.bandwidth
        # a narrower band than reserved, or no wrap entries on a periodic space, is handled by
        # solver() (it factors the measured band), so these are notes rather than problems
        if meas < st.declared_uband and backend != "Diagonal":
            warn.append(f"the space reserves uband={st.declared_uband} but this form reaches only {meas}; "
                        f"solver() factors the measured band")
        if backend == "Band" and st.diagonal_dominance <= 0.0:
            warn.append(f"banded LU does not pivot; it is exact for diagonally dominant or SPD systems, and "
                        f"dominance is not established here (min margin {st.diagonal_dominance:.1e})")
        if circ and not has_fftw():
            warn.append("circulant, but this build has no FFTW, so the O(n log n) backend is unavailable")

        return MatrixInfo(
            n=n, shape=(n, n), nnz=st.nnz, density=(st.nnz / float(n * n) if n else 0.0),
            bandwidth=st.bandwidth, cyclic_bandwidth=st.cyclic_bandwidth, reserved_uband=st.declared_uband,
            corner_entries=st.corner_entries, block_period=st.block_period, scale=st.scale,
            symmetry=st.symmetry, skew=st.skew, circulant=st.circulant,
            diagonal_dominance=st.diagonal_dominance, periodic=self.periodic,
            symmetric_flag=self.symmetric, is_symmetric=sym, is_skew=skw, is_circulant=circ,
            kind=kind, backend=backend, auto_backend=auto, warnings=warn, tol=tol)

    # ---- algebra -----------------------------------------------------------
    # All of these work on the band store and return a new Matrix.  Nothing goes
    # through a sparse format.  `symmetric` is propagated conservatively: a sum
    # is symmetric only when both terms are, and a product is never declared
    # symmetric, since A B generally is not even when A and B both are.  Use
    # classify() to measure the result rather than trusting the flag.

    def _pair(self, other, what):
        if not isinstance(other, Matrix):
            return None
        if other.shape != self.shape:
            raise ValueError(f"{what}: dimension mismatch, {self.shape} vs {other.shape}")
        if other.periodic != self.periodic:
            raise ValueError(f"{what}: one operand is periodic and the other is not")
        return other

    def _wrap(self, K, other=None):
        space = self.space if (other is None or other.space is self.space) else None
        return Matrix(K, self.periodic, space)

    def copy(self) -> "Matrix":
        """An independent copy. Take one before an in-place update of a matrix that came
        from a constant form, since such a form caches and returns the same object."""
        return self._wrap(self._K.copy())

    @property
    def T(self) -> "Matrix":
        """The transpose, O(nnz) over the band store. For a skew matrix, `A.T` equals `-A`."""
        return self._wrap(self._K.transposed())

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self                                    # lets sum([...]) work
        o = self._pair(other, "A + B")
        return NotImplemented if o is None else self._wrap(self._K.combine(o._K, 1.0, 1.0), o)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._pair(other, "A - B")
        return NotImplemented if o is None else self._wrap(self._K.combine(o._K, 1.0, -1.0), o)

    def __neg__(self):
        return self._wrap(self._K.scaled(-1.0))

    def __mul__(self, a):
        if not isinstance(a, (int, float)):
            return NotImplemented                          # A * B is not an alias for A @ B
        return self._wrap(self._K.scaled(float(a)))

    __rmul__ = __mul__

    def __truediv__(self, a):
        if not isinstance(a, (int, float)):
            return NotImplemented
        return self._wrap(self._K.scaled(1.0 / float(a)))

    def __iadd__(self, other):
        """In place when the band is wide enough, otherwise a new Matrix. See copy()."""
        o = self._pair(other, "A += B")
        if o is None:
            return NotImplemented
        if self._K.uband >= o._K.uband:
            self._K.axpy(o._K, 1.0)
            return self
        return self._wrap(self._K.combine(o._K, 1.0, 1.0), o)

    def __isub__(self, other):
        o = self._pair(other, "A -= B")
        if o is None:
            return NotImplemented
        if self._K.uband >= o._K.uband:
            self._K.axpy(o._K, -1.0)
            return self
        return self._wrap(self._K.combine(o._K, 1.0, -1.0), o)

    def add(self, cache: QuadratureCache, a: int, b: int, coeff=1.0) -> "Matrix":
        """K += integral coeff (D^a N_i)(D^b N_j)."""
        self._K.add_matrix(cache, int(a), int(b), _coeff(coeff))
        return self

    def add_block(self, P: ProductSpace, I: int, J: int, QI: QuadratureCache, QJ: QuadratureCache, a: int, b: int, coeff=1.0) -> "Matrix":
        self._K.add_block(P, int(I), int(J), QI, QJ, int(a), int(b), _coeff(coeff))
        return self


class RectMatrix:
    """An operator from a trial space V to a test space W, two different spaces on one grid:
    B_ij = a(S_j, T_i) with T_i the basis of W (rows) and S_j that of V (columns).

    It comes out of a form whose test and trial functions live in different spaces,

        u, q = fd.TrialFunction(V), fd.TestFunction(W)
        B = fd.form(D(u) * q * dx).assemble()        # W.dim x V.dim

    and is the off-diagonal block of a coupled system, assembled into one banded
    Matrix with fd.block([[A, B], [C, M]]).  It is stored as a SciPy CSR matrix,
    since a rectangular block has no diagonal to band around and is never factored.

        B @ v        Function of W for a vector (or Function of V), RectMatrix for a Matrix
        B.T          the transposed operator, from W to V
        B + C, a*B   algebra between blocks with the same two spaces
        B.tocsr()    a SciPy copy
    """
    def __init__(self, A, row_space, col_space):
        A = sp.csr_matrix(A)
        A.sum_duplicates()
        if A.shape != (row_space.dim, col_space.dim):
            raise ValueError(f"RectMatrix: shape {A.shape} does not match ({row_space.dim}, {col_space.dim})")
        self._A, self.row_space, self.col_space = A, row_space, col_space

    @property
    def shape(self):
        return self._A.shape

    @property
    def space(self):
        """The row (test) space, which is where B @ x lives."""
        return self.row_space

    def tocsr(self) -> sp.csr_matrix:
        return self._A.copy()

    def tocoo(self) -> sp.coo_matrix:
        return self._A.tocoo()

    def toarray(self) -> np.ndarray:
        return self._A.toarray()

    def copy(self) -> "RectMatrix":
        return RectMatrix(self._A.copy(), self.row_space, self.col_space)

    @property
    def T(self) -> "RectMatrix":
        return RectMatrix(self._A.T.tocsr(), self.col_space, self.row_space)

    def solver(self, *args, **kw):
        raise TypeError(f"a rectangular {self.shape[0]} x {self.shape[1]} block has no solver; assemble the "
                        "coupled system with fd.block([[A, B], [C, D]]) and solve that")

    def matvec(self, x, out=None) -> np.ndarray:
        """B x as a plain array; out= copies the result into an existing array."""
        v = x.vector if isinstance(x, Function) else np.asarray(x, dtype=np.float64)
        if v.shape != (self.shape[1],):
            raise ValueError(f"RectMatrix.matvec: need a vector of length {self.shape[1]}, got shape {v.shape}")
        y = self._A @ v
        if out is None:
            return y
        out[:] = y
        return out

    def __matmul__(self, x):
        if isinstance(x, RectMatrix):
            if self.col_space is not x.row_space:
                raise ValueError("B @ C: the columns of B and the rows of C index different spaces")
            return RectMatrix(self._A @ x._A, self.row_space, x.col_space)
        if isinstance(x, Matrix):
            if self.col_space is not x.space:
                raise ValueError("B @ A: the columns of B and the rows of A index different spaces")
            return RectMatrix(self._A @ x.tocsr(), self.row_space, x.space)
        if isinstance(x, Function):
            if x.space is not self.col_space:
                raise ValueError("B @ u: u does not live in the trial space of B")
            x = x.vector
        v = np.asarray(x, dtype=np.float64)
        y = self._A @ v
        if y.ndim == 1:
            return Function._adopt(self.row_space, "product", np.ascontiguousarray(y))
        return y

    def _same(self, o, what):
        if not isinstance(o, RectMatrix):
            return NotImplemented
        if o.row_space is not self.row_space or o.col_space is not self.col_space:
            raise ValueError(f"{what}: the two blocks map between different spaces")
        return o

    def __add__(self, o):
        o = self._same(o, "B + C")
        return o if o is NotImplemented else RectMatrix(self._A + o._A, self.row_space, self.col_space)

    def __sub__(self, o):
        o = self._same(o, "B - C")
        return o if o is NotImplemented else RectMatrix(self._A - o._A, self.row_space, self.col_space)

    def __neg__(self):
        return RectMatrix(-self._A, self.row_space, self.col_space)

    def __mul__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return RectMatrix(float(a) * self._A, self.row_space, self.col_space)

    __rmul__ = __mul__

    def __truediv__(self, a):
        if not np.isscalar(a):
            return NotImplemented
        return RectMatrix(self._A / float(a), self.row_space, self.col_space)

    def __repr__(self):
        return (f"RectMatrix({self.shape[0]} x {self.shape[1]}, {self._A.nnz} nonzeros, "
                f"{self.row_space!r} <- {self.col_space!r})")


def _block_spaces(B):
    if isinstance(B, RectMatrix):
        return B.row_space, B.col_space
    if isinstance(B, Matrix):
        if B.space is None or isinstance(B.space, ProductSpace):
            raise TypeError("block(): a diagonal block must be a Matrix on one plain space")
        return B.space, B.space
    raise TypeError(f"block(): entries must be Matrix, RectMatrix or None, got {type(B).__name__}")


def block(blocks, space=None, symmetric=None) -> Matrix:
    """Assemble a block operator into one banded Matrix on a ProductSpace.

        A = fd.form(... u*v ...).assemble()       # V x V
        B = fd.form(... p*v ...).assemble()       # V x W  (trial p in W, test v in V)
        C = fd.form(... u*q ...).assemble()       # W x V
        M = fd.form(... p*q ...).assemble()       # W x W
        K = fd.block([[A, B], [C, M]])            # on ProductSpace(V, W), position-ordered
        uv = K.solver().solve(P.gather([f, g]))   # a ProductFunction; uv.split() gives the fields

    `blocks` is a square list of lists, None (or 0) for a zero block.  Field I is the space
    of row I, and must equal the space of column I.  The fields are numbered together by
    position, so the result keeps the bandwidth of one element and the banded and cyclic
    solvers apply.  Pass `space=` to reuse a ProductSpace you already have.  The symmetric
    flag is measured on the result, so Auto picks Cholesky exactly when it is safe; pass
    symmetric=True or False to set it without the measurement (an O(n w) pass)."""
    rows = [list(r) for r in blocks]
    nf = len(rows)
    if nf == 0 or any(len(r) != nf for r in rows):
        raise ValueError("block(): need a square list of lists of blocks")
    rows = [[None if (b is None or (np.isscalar(b) and b == 0)) else b for b in r] for r in rows]
    fields = [None] * nf

    def settle(k, S, where):
        if fields[k] is None:
            fields[k] = S
        elif fields[k] is not S:
            raise ValueError(f"block(): {where} {k} is indexed by two different spaces; field {k} must be one space "
                             "for its row and its column")

    for I, r in enumerate(rows):
        for J, B in enumerate(r):
            if B is None:
                continue
            rs, cs = _block_spaces(B)
            settle(I, rs, "row")
            settle(J, cs, "column")
    missing = [k for k in range(nf) if fields[k] is None]
    if missing:
        raise ValueError(f"block(): field(s) {missing} have no nonzero block to say which space they are")
    if space is None:
        space = ProductSpace(*fields)
    elif not isinstance(space, ProductSpace) or len(space.fields) != nf or \
            any(a is not b for a, b in zip(space.fields, fields)):
        raise ValueError("block(space=P): P.fields must be exactly the spaces of the block rows, in order")
    K = _C.AssembledMatrix(space.dim, space.uband)
    for I, r in enumerate(rows):
        for J, B in enumerate(r):
            if B is None:
                continue
            if isinstance(B, Matrix) and hasattr(B._K, "add_scaled_block"):
                K.add_scaled_block(space, I, J, B._K, 1.0)          # straight off the band, in C++
                continue
            if isinstance(B, Matrix):
                rr, cc, vv = B._K.to_coo()
            else:
                C = B._A.tocoo()
                rr, cc, vv = C.row, C.col, C.data
            K.add_coo(space, I, J, np.ascontiguousarray(rr, dtype=np.int32),
                      np.ascontiguousarray(cc, dtype=np.int32), np.ascontiguousarray(vv, dtype=np.float64))
    if symmetric is None:
        st = _C.structure(K, 1e-12)
        K.symmetric = bool(st.scale > 0.0 and st.symmetry <= 1e-12)
    else:
        K.symmetric = bool(symmetric)
    return Matrix(K, space.periodic, space)


def _space_data(V, left, right, dirichlet):
    """The boundary data of a call: the call's own when it gives any, otherwise the data given with
    the space V (boundary_data), or none."""
    if left is not None or right is not None or dirichlet is not None:
        return left, right, dirichlet
    d = getattr(V, "boundary_data", None)
    if d is None:
        return None, None, None
    if getattr(V, "tdim", 1) == 2:
        return None, None, d
    return d["left"], d["right"], None


def solve(a, L, left=None, right=None, backend=Solver.Auto, dirichlet=None):
    """Solve a(u, v) = L(v) for all v in V, with data on the conditions built into V.

        V = fd.SplineSpace(grid, 3, bc=("dirichlet", "free"))
        u, v = fd.TrialFunction(V), fd.TestFunction(V)
        a = D(u)*D(v)*dx
        L = f*v*dx + g*v*ds("right")             # natural data: u'(b) = g
        uh = fd.solve(a, L, left=1.0)            # strong data:  u(a) = 1

    a and L are forms or the expressions that make them.  left= and right= give the
    values of the functionals built in at each end, as in V.lift().  With data, the
    solution is u = u_0 + u_g, where u_g = V.lift(...) and u_0 in V solves
    a(u_0, v) = L(v) - a(u_g, v), and it is returned as a Function of V.unconstrained.
    Without data it is a Function of V.  backend is passed to Matrix.solver().

    On a 2D mesh the data is dirichlet= (a number, a callable g(x, y), or a dict
    {marker: number or callable}, as in V.lift()), and backend is "auto" / "superlu",
    "cg", "gmres" or "dense" (SparseMatrix.solver()).

    When the call gives no data, the data given with the space are used:
    SplineSpace(grid, 3, bc="dirichlet", left=0.0, right=1.0), or in 2D
    LagrangeSpace(m, k, dirichlet={1: g1, 4: g4}).  Data in the call replace them."""
    a = a if isinstance(a, Form) else form(a)
    L = L if isinstance(L, Form) else form(L)
    if a.rank != 2 or L.rank != 1:
        raise TypeError(f"solve(a, L): a must be bilinear (rank 2) and L linear (rank 1), got ranks {a.rank} and {L.rank}")
    V = a.test_space
    if L.test_space is not V:
        raise ValueError("solve(a, L): a and L must have the same test space")
    left, right, dirichlet = _space_data(V, left, right, dirichlet)
    if getattr(V, "tdim", 1) == 2:
        if left is not None or right is not None:
            raise TypeError("solve(): left= and right= are the ends of a 1D mesh; on a 2D mesh give dirichlet=")
        if a.trial_space is not V:
            raise TypeError("solve(a, L): the form must be square (trial space = test space)")
        A = a.assemble()
        b = np.array(L.assemble(), dtype=np.float64)
        S = A.solver("auto" if backend is Solver.Auto else backend)
        if dirichlet is None or V.n_constraints == 0:
            if dirichlet is not None and not (isinstance(dirichlet, (int, float)) and dirichlet == 0):
                raise ValueError("solve(dirichlet=...): the space has no Dirichlet condition built in; create it "
                                 "with dirichlet=[markers] or bc='dirichlet'")
            return S.solve(b)
        ug = V.lift(dirichlet)
        b -= a.action(ug).assemble()
        return S.solve(b, lift=ug)
    if dirichlet is not None:
        raise TypeError("solve(dirichlet=...) is for 2D meshes; in 1D give left= and right=")
    A = a.assemble()
    b = np.array(L.assemble(), dtype=np.float64)
    S = A.solver(backend)
    if left is None and right is None:
        return S.solve(b)
    if isinstance(V, ProductSpace) or a.trial_space is not V:
        raise TypeError("solve(a, L, left=..., right=...): boundary data needs a square form on one plain space")
    ug = V.lift(left=left, right=right)
    b -= a.action(ug).assemble()
    return S.solve(b, lift=ug)


def assemble(cache: QuadratureCache, terms, space=None) -> Matrix:
    """Assemble sum of terms (a, b, coeff): integral coeff (D^a u)(D^b v), test index a, trial index b.

    coeff is a number or an array of values at the cache nodes (cache.nodes()).
    `space` defaults to the cache's space; pass it for a ProductSpace with block terms
    given as (I, J, QI, QJ, a, b, coeff).
    """
    if isinstance(space, ProductSpace):
        M = Matrix(_C.AssembledMatrix(space.dim, space.uband), space.periodic, space)
        for t in terms:
            M.add_block(space, *t)
        return M
    V = space if space is not None else _space_of(cache)
    M = Matrix(_C.AssembledMatrix(V.dim, V.uband), V.bc.is_periodic, V)
    for (a, b, c) in terms:
        M.add(cache, a, b, c)
    return M


def assemble_vector(cache: QuadratureCache, terms, n: int) -> np.ndarray:
    """f_i = sum over terms (a, coeff) of integral coeff (D^a N_i)."""
    f = np.zeros(n)
    for (a, c) in terms:
        f += _C.assemble_vector(cache, int(a), _coeff(c), int(n))
    return f


def assemble_block_vector(P: ProductSpace, terms) -> np.ndarray:
    """Terms (I, QI, a, coeff) into the product numbering."""
    f = np.zeros(P.dim)
    for (I, QI, a, c) in terms:
        f += _C.assemble_block_vector(P, int(I), QI, int(a), _coeff(c))
    return f


def assemble_scalar(cache: QuadratureCache, coeff) -> float:
    return _C.assemble_scalar(cache, _coeff(coeff))
