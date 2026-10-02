"""
femd.eigen -- sparse eigenproblems  A x = lambda M x  on any space.

    lam, modes = fd.eigs(a, m, k=6, sigma=0.0)          # forms or assembled matrices
    r = fd.eigs(A, M, k=12, sigma=5.5)                   # an EigenResult: r.values, r.vectors, r.functions

Shift-invert Lanczos (symmetric problems, SciPy's eigsh) or Arnoldi (eigs), with the inner solve
(A - sigma M)^{-1} done by FEMd's own factorization of the shifted matrix: banded or cyclic in
1D, sparse Cholesky / pivoting LDL^T / SuperLU in 2D, so the k eigenvalues nearest sigma come
out after one factorization.  sigma must not be an eigenvalue (A - sigma M must be invertible):
for a curl-curl operator, whose kernel is every gradient, put sigma inside the spectrum of
interest, away from 0.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

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
    symmetric: None measures A and M; True uses Lanczos (eigsh, M must then be positive definite),
               False Arnoldi (eigs, complex results possible).
    which:     "nearest" (default, shift-invert about sigma), or "largest" / "smallest" magnitude
               without a shift (no factorization; slow for the smallest).
    tol, maxiter, ncv: passed to ARPACK.  vectors=False skips the eigenvectors.
    v0:        ARPACK's starting vector.  The default is a fixed pseudo-random one, so a call gives
               the same result every time (ARPACK's own default is random).
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
    Aop = spla.LinearOperator((n, n), matvec=lambda x: np.asarray(A.matvec(np.ascontiguousarray(np.real(x), dtype=np.float64)))
                              + (1j * np.asarray(A.matvec(np.ascontiguousarray(np.imag(x), dtype=np.float64)))
                                 if np.iscomplexobj(x) else 0.0), dtype=np.float64)
    Mop = None
    if M is not None:
        Mop = spla.LinearOperator((n, n), matvec=lambda x: np.asarray(M.matvec(np.ascontiguousarray(np.real(x), dtype=np.float64)))
                                  + (1j * np.asarray(M.matvec(np.ascontiguousarray(np.imag(x), dtype=np.float64)))
                                     if np.iscomplexobj(x) else 0.0), dtype=np.float64)
    if v0 is None:
        v0 = np.random.default_rng(12345).standard_normal(n)
    opts = dict(k=k, tol=tol, maxiter=maxiter, ncv=ncv, return_eigenvectors=bool(vectors),
                v0=np.asarray(getattr(v0, "vector", v0), dtype=np.float64))
    w = str(which).lower()
    if w == "nearest":
        if M is not None:
            C = A - float(sigma) * M if sigma != 0 else A.copy()
        else:
            import scipy.sparse as sp
            Cs = A.tocsr() - float(sigma) * sp.identity(n, format="csr")
            C = _like(A, Cs)
        try:
            S = C.solver(backend) if backend != "auto" else C.solver()
        except Exception as e:
            raise ValueError(f"eigs(): the shifted matrix A - {sigma:g} M could not be factored ({e}); sigma may be "
                             "an eigenvalue (a curl-curl or pure Neumann operator has 0 in its spectrum): move it") from None

        def inv(x):
            x = np.asarray(x)
            if np.iscomplexobj(x):
                return _solve(S, x.real) + 1j * _solve(S, x.imag)
            return _solve(S, x)
        OPinv = spla.LinearOperator((n, n), matvec=inv, dtype=np.float64)
        if symmetric:
            res = spla.eigsh(Aop, M=Mop, sigma=float(sigma), OPinv=OPinv, which="LM", **opts)
        else:
            res = spla.eigs(Aop, M=Mop, sigma=float(sigma), OPinv=OPinv, which="LM", **opts)
    elif w in ("largest", "smallest"):
        if M is not None:
            Ms = M.solver()
            Minv = spla.LinearOperator((n, n), matvec=lambda x: _solve(Ms, np.real(x)), dtype=np.float64)
            opts["Minv"] = Minv
        which_ = "LM" if w == "largest" else "SM"
        res = (spla.eigsh if symmetric else spla.eigs)(Aop, M=Mop, which=which_, **opts)
    else:
        raise ValueError(f"eigs(): which is 'nearest', 'largest' or 'smallest', got {which!r}")
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
        for i in range(len(lam)):
            ax = Aop @ X[:, i]
            mx = Mop @ X[:, i] if Mop is not None else X[:, i]
            scale = (nA + abs(lam[i]) * nM) * np.linalg.norm(X[:, i])
            resid[i] = np.linalg.norm(ax - lam[i] * mx) / max(scale, 1e-300)
    return EigenResult(lam, X, space, float(sigma), resid, bool(symmetric))


def _norm1(A):
    """The 1-norm of a matrix (largest column sum of |a_ij|), the scale of the residuals."""
    C = abs(A.tocsr())
    return float(np.asarray(C.sum(axis=0)).max()) if C.nnz else 1.0


def _solve(S, b):
    out = S.solve(np.ascontiguousarray(b, dtype=np.float64))
    return np.asarray(getattr(out, "vector", out), dtype=np.float64)


def _like(A, Cs):
    """A matrix of A's kind holding the scipy matrix Cs (the standard problem's shift)."""
    from .sparse import SparseMatrix
    if isinstance(A, SparseMatrix):
        return SparseMatrix.from_scipy(Cs.tocsr(), space=A.space, symmetric=None)
    raise TypeError("eigs(): the standard problem (m=None) needs a 2D SparseMatrix; in 1D pass a mass matrix "
                    "(or the identity) as m")
