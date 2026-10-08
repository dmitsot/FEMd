"""
Krylov solvers and complex-step Newton from the vendored csnewton (D. Mitsotakis).

    x, info = fd.gmres(A, b, M=None)            # A: Matrix, fd.block system, callable, anything with matvec
    x, info = fd.lgmres(A, b, M=A0.solver())    # M: LinearSolver, Matrix (factored once), callable, None
    x, info = fd.cg(A, b, M=A.preconditioner("ilu0"))    # SPD systems, 2D SparseMatrix or 1D Matrix
    x, info = fd.csnewton(F, x0)                # F(x) = 0, F accepting real and complex arrays

When A is a Matrix or a SparseMatrix and M is None, a LinearSolver or a sparse
Preconditioner, the whole iteration runs in C++ (the kernel and the factored or
incomplete solve).  Anything else calls back into
Python once per matvec or preconditioner application.

The preconditioner is applied on the left, and `tol` bounds
||M^{-1} r|| / ||M^{-1} b||, the standard left-preconditioned relative residual,
whatever the scaling of M.  info.residual is the TRUE relative residual
||b - A x|| / ||b||, recomputed at the end.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import _femd as _C

__all__ = ["gmres", "lgmres", "cg", "csnewton", "KrylovInfo", "CSNewtonInfo"]


@dataclass
class KrylovInfo:
    converged: bool
    iterations: int
    residual: float            # ||b - A x|| / ||b||
    precond_residual: float    # ||M^{-1}(b - A x)|| / ||M^{-1} b||, what tol bounds
    method: str

    def __repr__(self):
        state = "converged" if self.converged else "NOT converged"
        return (f"KrylovInfo({self.method} {state} in {self.iterations} iterations, residual {self.residual:.2e}, "
                f"preconditioned {self.precond_residual:.2e})")


@dataclass
class CSNewtonInfo:
    converged: bool
    newton_iterations: int
    krylov_iterations: int
    residual: float            # ||F(x)||_2


def _vector(b):
    from .forms import Function, ProductFunction
    if isinstance(b, (Function, ProductFunction)):
        b = b.vector
    v = np.asarray(b)
    if np.iscomplexobj(v):
        raise TypeError("the Krylov solvers are real; solve the real and imaginary parts separately")
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)


def _callable(f, n, what):
    def g(x):
        y = np.ascontiguousarray(np.asarray(f(x), dtype=np.float64).reshape(-1))
        if y.shape != (n,):
            raise ValueError(f"{what} returned {y.size} values, expected {n}")
        return y
    return g


def _operator(A, n):
    """(C++ operator or Python callable, space of the result)."""
    from . import Matrix, RectMatrix, ProductSpace
    from .sparse import SparseMatrix
    if isinstance(A, _C.CSRMatrix):
        if (A.nrows, A.ncols) != (n, n):
            raise ValueError(f"krylov: A is {A.nrows} x {A.ncols}, b has length {n}")
        return A, None
    if isinstance(A, SparseMatrix):
        if A.shape != (n, n):
            raise ValueError(f"krylov: A is {A.shape[0]} x {A.shape[1]}, b has length {n}")
        return A._K, A.row_space
    if isinstance(A, Matrix):
        if A.shape != (n, n):
            raise ValueError(f"gmres: A is {A.shape[0]} x {A.shape[1]}, b has length {n}")
        return A._K, A.space
    if isinstance(A, RectMatrix):
        if A.shape != (n, n):
            raise ValueError(f"gmres: A is {A.shape[0]} x {A.shape[1]}, b has length {n}")
        return A._K, A.row_space                              # the CSR store: the matvec in C++
    if hasattr(A, "matvec"):
        return _callable(A.matvec, n, "A.matvec"), getattr(A, "space", None)
    if hasattr(A, "__matmul__") and not callable(A):
        return _callable(lambda x: A @ x, n, "A @ x"), None
    if callable(A):
        return _callable(A, n, "A"), None
    raise TypeError(f"gmres: cannot use {type(A).__name__} as an operator")


def _preconditioner(M, n):
    from . import Matrix, LinearSolver
    from .sparse import SparseMatrix, Preconditioner, SparseSolver
    if M is None:
        return None
    if isinstance(M, _C.SparsePreconditioner):
        M_impl = M
    elif isinstance(M, Preconditioner):
        M_impl = M._impl
    else:
        M_impl = None
    if M_impl is not None:
        if M_impl.size != n:
            raise ValueError(f"krylov: the preconditioner has size {M_impl.size}, b has length {n}")
        return M_impl
    if isinstance(M, SparseMatrix):
        M = M.solver()
    if isinstance(M, SparseSolver):
        if M.backend in ("cholesky", "ldlt"):
            return M.factor                                  # applied in C++
        return _callable(lambda r: M._solve_array(r), n, "M.solve")
    if isinstance(M, Matrix):
        M = M.solver()
    if isinstance(M, LinearSolver):
        if M.size != n:
            raise ValueError(f"gmres: the preconditioner has size {M.size}, b has length {n}")
        return M._impl
    if hasattr(M, "solve"):
        return _callable(lambda r: np.asarray(M.solve(r)), n, "M.solve")
    if callable(M):
        return _callable(M, n, "M")
    raise TypeError(f"gmres: cannot use {type(M).__name__} as a preconditioner")


def _wrap(x, space, name="solution"):
    from . import ProductSpace
    from .forms import Function, ProductFunction
    if space is None:
        return x
    if isinstance(space, ProductSpace):
        return ProductFunction(space, name, x)
    if hasattr(space, "fields"):                         # a system on a 2D mesh
        from .forms import _product_function
        return _product_function(space, name, x)
    return Function._adopt(space, name, x)


def _krylov(A, b, M, x0, restart, maxiter, tol, lgmres, k_aug, warn):
    bv = _vector(b)
    n = bv.size
    Aop, space = _operator(A, n)
    Mop = _preconditioner(M, n)
    x0v = np.zeros(n) if x0 is None else _vector(x0)
    if x0v.shape != (n,):
        raise ValueError(f"gmres: x0 has length {x0v.size}, b has length {n}")
    restart = int(min(restart, max(n, 1)))
    maxiter = int(maxiter if maxiter is not None else max(1000, 10 * restart))
    x, conv, its, res, pres = _C.krylov_solve(Aop, Mop, bv, x0v, bool(lgmres), restart, maxiter, float(tol), int(k_aug))
    info = KrylovInfo(bool(conv), int(its), float(res), float(pres), "lgmres" if lgmres else "gmres")
    if warn and not info.converged:
        warnings.warn(f"{info.method} did not converge in {info.iterations} iterations "
                      f"(preconditioned residual {info.precond_residual:.2e}, tol {tol:.1e}); "
                      "raise maxiter or restart, or improve the preconditioner", RuntimeWarning, stacklevel=3)
    return _wrap(x, space), info


def gmres(A, b, M=None, x0=None, *, restart=30, maxiter=None, tol=1e-10, warn=True):
    """Solve A x = b by GMRES(restart), left-preconditioned by M.  Returns (x, KrylovInfo).

    A: a Matrix (kernel in C++), a square RectMatrix, anything with .matvec, or a callable.
    M: None, a LinearSolver (in C++), a Matrix (factored with Auto), anything with .solve,
       or a callable r -> M^{-1} r.
    x is a Function (ProductFunction) when A's space is known, an array otherwise.
    tol bounds ||M^{-1} r|| / ||M^{-1} b||.  A RuntimeWarning is issued if it is not met."""
    return _krylov(A, b, M, x0, restart, maxiter, tol, False, 0, warn)


def cg(A, b, M=None, x0=None, *, maxiter=None, tol=1e-10, warn=True):
    """Solve A x = b by preconditioned conjugate gradients.  Returns (x, KrylovInfo).

    A must be symmetric positive definite: a SparseMatrix or a Matrix (kernel in C++),
    anything with .matvec, or a callable.  M (symmetric positive definite too): None, a
    Preconditioner (A.preconditioner("ilu0"), "ssor", "jacobi"), a LinearSolver or
    SparseSolver, or a callable r -> M^{-1} r.  With a native A and M the iteration runs in C++
    with the GIL released.  tol bounds the TRUE relative residual ||b - A x|| / ||b||."""
    bv = _vector(b)
    n = bv.size
    Aop, space = _operator(A, n)
    Mop = _preconditioner(M, n)
    x0v = np.zeros(n) if x0 is None else _vector(x0)
    if x0v.shape != (n,):
        raise ValueError(f"cg: x0 has length {x0v.size}, b has length {n}")
    maxiter = int(maxiter if maxiter is not None else max(1000, 10 * n))
    x, conv, its, res, breakdown = _C.cg_solve(Aop, Mop, bv, x0v, maxiter, float(tol))
    info = KrylovInfo(bool(conv), int(its), float(res), float(res), "cg")
    if warn and breakdown:
        warnings.warn("cg broke down (p^T A p <= 0 or r^T M^{-1} r <= 0): A or the preconditioner is not "
                      "symmetric positive definite; use gmres", RuntimeWarning, stacklevel=2)
    elif warn and not info.converged:
        warnings.warn(f"cg did not converge in {info.iterations} iterations (residual {info.residual:.2e}, "
                      f"tol {tol:.1e}); raise maxiter or improve the preconditioner", RuntimeWarning, stacklevel=2)
    return _wrap(x, space), info


def lgmres(A, b, M=None, x0=None, *, restart=20, k_aug=2, maxiter=None, tol=1e-10, warn=True):
    """LGMRES(restart, k_aug): GMRES that keeps k_aug error approximations across restarts,
    which helps where GMRES(m) stagnates (Baker, Jessup and Manteuffel 2005).  As gmres."""
    return _krylov(A, b, M, x0, restart, maxiter, tol, True, k_aug, warn)


def csnewton(F, x0, M=None, *, tol=1e-10, maxiter=50, h=1e-20, restart=20, krylov_maxiter=200,
             krylov_tol=1e-6, method="gmres", k_aug=2, warn=True):
    """Complex-step Newton-Krylov for F(x) = 0, the vendored csnewton.csnewton.

    Each Newton step solves J(x) s = F(x) by GMRES (or LGMRES), with the Jacobian-vector
    product formed by the complex step  J(x) v = Im F(x + i h v) / h,  exact to round-off
    for h ~ 1e-20.  F must therefore accept complex arrays and be analytic: write it with
    NumPy operations (np.sin, **, @ with real matrices ...), not abs, max or comparisons.
    M is a left preconditioner for J, as in gmres.  Returns (x, CSNewtonInfo).

    For a residual FORM use fd.newton, which has the exact Jacobian and the boundary data."""
    x0v = _vector(x0)
    n = x0v.size

    def Fw(z):
        r = np.asarray(F(z))
        if np.iscomplexobj(z):
            r = np.ascontiguousarray(r, dtype=np.complex128).reshape(-1)
        else:
            if np.iscomplexobj(r):
                r = r.real
            r = np.ascontiguousarray(r, dtype=np.float64).reshape(-1)
        if r.shape != (n,):
            raise ValueError(f"csnewton: F returned {r.size} values for {n} unknowns")
        return r

    Mop = _preconditioner(M, n)
    x, conv, nit, git, res = _C.csnewton(Fw, Mop, x0v, int(maxiter), float(tol), float(h), int(restart),
                                         int(krylov_maxiter), float(krylov_tol), method == "lgmres", int(k_aug))
    info = CSNewtonInfo(bool(conv), int(nit), int(git), float(res))
    if warn and not info.converged:
        warnings.warn(f"csnewton did not converge in {nit} Newton steps (||F|| = {res:.2e})", RuntimeWarning, stacklevel=2)
    return x, info
