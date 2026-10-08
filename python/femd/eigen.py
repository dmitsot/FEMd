"""
femd.eigen -- sparse eigenproblems  A x = lambda M x  on any space.

    lam, modes = fd.eigs(a, m, k=6, sigma=0.0)          # forms or assembled matrices
    r = fd.eigs(A, M, k=12, sigma=5.5)                   # an EigenResult: r.values, r.vectors, r.functions

Shift-invert Arnoldi in C++ (solve/eigen.hpp, with restarts and locking; Lanczos in the M inner
product for symmetric problems), with the inner solve (A - sigma M)^{-1} done by FEMd's own
factorization of the shifted matrix: banded or cyclic in 1D, sparse Cholesky / pivoting LDL^T /
sparse LU in 2D, so the k eigenvalues nearest sigma come
out after one factorization.  sigma must not be an eigenvalue (A - sigma M must be invertible):
for a curl-curl operator, whose kernel is every gradient, put sigma inside the spectrum of
interest, away from 0.
"""
from __future__ import annotations

import numpy as np

from ._util import _norm2

__all__ = ["eigs", "EigenResult"]


class EigenResult:
    """The result of fd.eigs: values (sorted by distance to sigma, then by value), the coefficient
    vectors as the columns of `vectors`, and `functions`, one Function per eigenvector when the
    matrix knows its space.  residuals[i] = |A x - lambda M x| / ((|A|_1 + |lambda| |M|_1) |x|), the
    backward error of each pair.  Unpacks as
    lam, X = result."""

    def __init__(self, values, vectors, space, sigma, residuals, symmetric):
        self.values, self.vectors, self.space, self.sigma = values, vectors, space, sigma
        self.residuals, self.symmetric = residuals, symmetric

    @property
    def functions(self):
        if self.space is None or self.vectors is None:
            return []
        from .sparse import _adopt
        out = []
        for i in range(self.vectors.shape[1]):
            v = self.vectors[:, i]
            if np.iscomplexobj(v):
                raise TypeError("functions: the eigenvectors are complex (a non-symmetric problem); use .vectors")
            out.append(_adopt(self.space, f"mode{i}", np.ascontiguousarray(v, dtype=np.float64)))
        return out

    def __iter__(self):
        return iter((self.values, self.vectors))

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        vals = ", ".join(f"{v:.6g}" for v in np.real_if_close(self.values)[:8])
        more = ", ..." if len(self.values) > 8 else ""
        return (f"EigenResult({len(self.values)} eigenvalues near sigma = {self.sigma:g}: [{vals}{more}], "
                f"max residual {np.max(self.residuals) if len(self.residuals) else 0:.1e})")


def _matrix(x, what):
    from .forms import Form, form, FormExpr, Integral
    if isinstance(x, (Form,)):
        return x.assemble()
    if isinstance(x, (FormExpr, Integral)):
        return form(x).assemble()
    if hasattr(x, "solver") and hasattr(x, "matvec"):
        return x
    raise TypeError(f"eigs(): {what} must be a bilinear form or an assembled Matrix / SparseMatrix, got {type(x).__name__}")


def _declared_symmetric(A):
    try:
        return bool(A.symmetric)
    except Exception:
        return False


def _measured_symmetric(A):
    try:
        C = A.tocsr()
        scale = max(abs(C).max(), 1e-300)
        return abs(C - C.T).max() <= 1e-12 * scale
    except Exception:
        return _declared_symmetric(A)


def eigs(a, m=None, k: int = 6, sigma: float = 0.0, symmetric=None, which: str = "nearest", tol: float = 0.0,
         maxiter=None, ncv=None, vectors: bool = True, backend="auto", v0=None) -> EigenResult:
    """The k eigenpairs of A x = lambda M x nearest sigma.

    a, m:      bilinear forms (or their expressions) or assembled matrices on one space; m = None is
               the standard problem A x = lambda x.  For a Galerkin problem m is the mass form u*v*dx.
    k:         how many eigenpairs.
    sigma:     the shift; the result is the k eigenvalues nearest it.  It must not be an eigenvalue.
    symmetric: None measures A and M; True uses the M inner product (Lanczos; M must then be positive
               definite), False plain Arnoldi (complex results possible).
    which:     "nearest" (default, shift-invert about sigma), "largest" magnitude (M^{-1} A, no shifted
               factorization) or "smallest" (the shift 0).
    tol, maxiter, ncv: the relative residual tolerance of a Ritz pair (1e-10), the restarts (300) and the
               Arnoldi vectors per restart (max(2k + 1, 20)).  vectors=False skips the eigenvectors.
    v0:        the starting vector.  The default is a fixed pseudo-random one, so a call gives the same
               result every time.
    backend:   the solver of the shifted matrix (A - sigma M).solver(backend).

    Returns an EigenResult; lam, X = fd.eigs(...) unpacks it, and .functions gives Functions."""
    A = _matrix(a, "a")
    M = _matrix(m, "m") if m is not None else None
    n = A.shape[0]
    if A.shape != (n, n) or (M is not None and M.shape != (n, n)):
        raise ValueError("eigs(): A and M must be square and of one size")
    k = int(k)
    if not 1 <= k < n - 1:
        raise ValueError(f"eigs(): k must be between 1 and n - 2 = {n - 2}, got {k}")
    if symmetric is None:
        symmetric = _measured_symmetric(A) and (M is None or _measured_symmetric(M))
    space = getattr(A, "space", None)
    from . import _femd as _C

    def matvec_of(K):
        return lambda x: np.ascontiguousarray(K.matvec(np.ascontiguousarray(x, dtype=np.float64)), dtype=np.float64)

    if v0 is None:
        v0 = np.random.default_rng(12345).standard_normal(n)
    v0 = np.ascontiguousarray(getattr(v0, "vector", v0), dtype=np.float64)
    w = str(which).lower()
    if w not in ("nearest", "largest", "smallest"):
        raise ValueError(f"eigs(): which is 'nearest', 'largest' or 'smallest', got {which!r}")
    Amv = matvec_of(A)
    Mmv = matvec_of(M) if M is not None else None
    if w in ("nearest", "smallest"):
        # shift-invert: OP = (A - sigma M)^{-1} M, theta = 1 / (lambda - sigma); "smallest" is the shift 0
        shift = float(sigma) if w == "nearest" else 0.0
        if M is not None:
            C = A - shift * M if shift != 0 else A.copy()
        else:
            C = _shifted(A, shift)
        try:
            S = C.solver(backend) if backend != "auto" else C.solver()
        except Exception as e:
            raise ValueError(f"eigs(): the shifted matrix A - {shift:g} M could not be factored ({e}); sigma may be "
                             "an eigenvalue (a curl-curl or pure Neumann operator has 0 in its spectrum): move it") from None
        op = (lambda x: _solve(S, Mmv(x))) if M is not None else (lambda x: _solve(S, x))
        back = lambda theta: shift + 1.0 / theta                                               # noqa: E731
    else:
        # the largest in magnitude of OP = M^{-1} A directly, theta = lambda
        Ms = M.solver() if M is not None else None
        op = (lambda x: _solve(Ms, Amv(x))) if M is not None else Amv
        back = lambda theta: theta                                                             # noqa: E731
    B = Mmv if (symmetric and M is not None) else None
    theta, Xs, _, restarts, _, converged, message = _C.arnoldi_largest(
        n, op, B, int(k), int(ncv or 0), float(tol), int(maxiter or 0), bool(symmetric), v0)
    if not converged:
        raise ValueError(f"eigs(): Arnoldi did not converge after {restarts} restarts ({message}); raise maxiter or ncv, "
                         "or move sigma")
    lam = back(np.asarray(theta))
    X = np.ascontiguousarray(np.asarray(Xs).reshape(len(lam), n).T)
    if symmetric:                                              # Lanczos in the M inner product: everything real
        lam, X = lam.real, np.ascontiguousarray(X.real)
    res = (lam, X) if vectors else lam
    Aop = Amv
    Mop = Mmv
    lam, X = (res if vectors else (res, None))
    lam = np.asarray(lam)
    if not symmetric and np.allclose(np.imag(lam), 0.0, atol=1e-12 * max(1.0, np.abs(lam).max())):
        lam = lam.real
        if X is not None and np.allclose(np.imag(X), 0.0, atol=1e-10 * max(1e-300, np.abs(X).max())):
            X = X.real
    key = np.lexsort((np.real(lam), np.abs(lam - sigma))) if w == "nearest" else np.argsort(np.real(lam))
    lam = lam[key]
    if X is not None:
        X = X[:, key]
    resid = np.zeros(len(lam))
    if X is not None:
        nA, nM = _norm1(A), (_norm1(M) if M is not None else 1.0)
        def apply(f, x):
            if np.iscomplexobj(x):
                return f(np.ascontiguousarray(x.real)) + 1j * f(np.ascontiguousarray(x.imag))
            return f(np.ascontiguousarray(x))

        for i in range(len(lam)):
            ax = apply(Aop, X[:, i])
            mx = apply(Mop, X[:, i]) if Mop is not None else X[:, i]
            scale = (nA + abs(lam[i]) * nM) * _norm2(X[:, i])
            resid[i] = _norm2(ax - lam[i] * mx) / max(scale, 1e-300)
    return EigenResult(lam, X, space, float(sigma), resid, bool(symmetric))


def _norm1(A):
    """The 1-norm of a matrix (largest column sum of |a_ij|), the scale of the residuals, in C++."""
    K = getattr(A, "_K", None)
    if K is not None and hasattr(K, "norm1"):
        return float(K.norm1())
    from .timestep import _as_sparse
    return float(_as_sparse(A)._K.norm1())


def _solve(S, b):
    out = S.solve(np.ascontiguousarray(b, dtype=np.float64))
    return np.asarray(getattr(out, "vector", out), dtype=np.float64)


def _shifted(A, shift):
    """A - shift I for a SparseMatrix (the standard problem), in C++, exact zeros dropped."""
    from .sparse import SparseMatrix
    if isinstance(A, SparseMatrix):
        C = SparseMatrix(A._K.shifted(float(shift)), space=A.space)
        C.symmetric = C.is_symmetric()
        return C
    raise TypeError("eigs(): the standard problem (m=None) needs a 2D SparseMatrix; in 1D pass a mass matrix "
                    "(or the identity) as m")
