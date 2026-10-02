"""
Newton's method for residual forms, with direct or Krylov linear solves.

    u = fd.Function(V)                                  # the unknown, an initial guess
    R = fd.form((D(u)*D(v) + u**3*v - f*v) * dx)        # the residual, linear in the test function v
    info = fd.newton(R, u)                              # u is updated in place

The Jacobian is the symbolic derivative of the form, R.derivative(u), so every step
is an exact Newton step.  By default it is assembled and solved by the banded or
cyclic direct solver, O(np) per step.  linear="gmres" (or "lgmres") solves the
same systems by Krylov iteration instead, with the Jacobian-vector product

    jv="assembled"      J assembled each step, product through the banded kernel
    jv="exact"          matrix-free, the action of R.derivative(u) on the direction
    jv="complex-step"   Im R(u + i h v) / h, exact to round-off (D. Mitsotakis,
                        Numer. Math. 157, 2025), needs only the residual
    jv="fd"             a one-sided difference, the fallback

and a left preconditioner

    precond="frozen"    J at the first iterate, factored once and refactored when the
                        iteration count grows (the default)
    precond="linear"    the constant-coefficient part of J, factored once
    precond="jacobian"  J at every step (one iteration: for testing)
    precond=None        none
    precond=<LinearSolver | Matrix | callable>

Boundary data: a field with Dirichlet-type data lives in V.unconstrained,
u = u_g + u_0 with u_g = V.lift(left=..., right=...) and u_0 in V, and Newton moves
u_0 only, so the data hold exactly at every iterate:

    u = fd.Function(V.unconstrained)
    fd.newton(R, u, left=1.0)

and on a ProductSpace W the unknown is a ProductFunction of W.unconstrained, with one entry
per field (None for a field without data), or the data given with the fields:

    w = fd.Functions(W.unconstrained)
    fd.newton(R, w, left=[1.0, None], right=[0.0, 2.0])
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ["newton", "NewtonInfo"]


@dataclass
class NewtonInfo:
    converged: bool
    iterations: int
    residuals: list = field(default_factory=list)      # ||R|| at every iterate, the first is the initial one
    steps: list = field(default_factory=list)          # line-search step length taken at each iteration
    linear_iterations: list = field(default_factory=list)
    message: str = ""

    @property
    def residual(self):
        return self.residuals[-1] if self.residuals else float("nan")

    def __repr__(self):
        state = "converged" if self.converged else "NOT converged"
        lin = f", {sum(self.linear_iterations)} Krylov iterations" if any(self.linear_iterations) else ""
        return (f"NewtonInfo({state} in {self.iterations} iterations, ||R|| {self.residuals[0]:.2e} -> "
                f"{self.residual:.2e}{lin}{', ' + self.message if self.message else ''})")


def _setup_lifted_2d(V, u, dirichlet):
    """u in V.unconstrained on a 2D mesh (a scalar space or a system): Newton moves the free
    coefficients c in V, u = u_g + P c, and u_g is either V.lift(dirichlet) or the boundary values
    u already holds."""
    if dirichlet is not None:
        ug = np.asarray(V.lift(dirichlet).vector, dtype=np.float64)
    else:
        ug = u.vector - V.prolongate(V.restrict(u.vector))
    P = V.prolongate
    state = [np.asarray(V.restrict(u.vector), dtype=np.float64)]

    def set_c(c):
        state[0] = np.array(c, dtype=np.float64)
        u.vector[:] = ug + P(state[0])
    set_c(state[0])
    return V, (lambda: state[0].copy()), set_c, ug, P


def _setup_lifted_product(W, u, left, right):
    """u a ProductFunction of W.unconstrained (1D): Newton moves the free coefficients c in W,
    u = u_g + P c, with u_g = W.lift(left, right) (one entry per field, None for none) and
    P = W.prolongate.  The initial guess is u minus the data, interpolated field by field."""
    from . import Transfer
    ug = np.asarray(W.lift(left=left, right=right).vector, dtype=np.float64)
    U = W.unconstrained
    parts = []
    for f, du in zip(W.fields, U.split(u.vector - ug)):
        fu = f.unconstrained
        parts.append(np.asarray(du, dtype=np.float64) if fu is f else
                     np.asarray(Transfer(f, fu, "interpolate")(np.asarray(du), check=False), dtype=np.float64))
    state = [np.asarray(W.gather(parts), dtype=np.float64)]
    P = W.prolongate

    def set_c(c):
        state[0] = np.array(c, dtype=np.float64)
        u.vector[:] = ug + P(state[0])
    set_c(state[0])
    return W, (lambda: state[0].copy()), set_c, ug, P


def _setup(R, u, left, right, dirichlet=None):
    """(V, get_c, set_c, lift_vector, prolongate) for the unknown u of the residual R."""
    from . import Transfer
    from .forms import Function, ProductFunction, VectorElementFunction
    from .linalg import _space_data
    V = R.test_space
    two_d = getattr(V, "tdim", 1) == 2
    from . import ProductSpace as _PS1
    is_prod = isinstance(V, _PS1) or (two_d and hasattr(V, "fields"))
    explicit = left is not None or right is not None or dirichlet is not None
    if not explicit and getattr(V, "boundary_data", None) is not None:
        U = getattr(V, "unconstrained", V)
        if isinstance(u, ProductFunction) and u.product is V and U is not V:
            raise ValueError("newton: the fields carry boundary data, so the unknown must be a ProductFunction of "
                             "W.unconstrained, w = fd.Functions(W.unconstrained), which can hold the data")
        if getattr(u, "space", None) is V and U is not V:
            raise ValueError("newton: the space carries boundary data, so the unknown must be a Function of "
                             "V.unconstrained, u = fd.Function(V.unconstrained), which can hold the data")
        left, right, dirichlet = _space_data(V, None, None, None)
    if dirichlet is not None and not two_d:
        raise ValueError("newton: dirichlet= is for 2D meshes; in 1D give left= and right=")
    if two_d and (left is not None or right is not None):
        raise ValueError("newton: left= and right= are the ends of a 1D mesh; on a 2D mesh give dirichlet=")
    if two_d and getattr(u, "space", None) is not V and getattr(u, "space", None) is getattr(V, "unconstrained", None):
        return _setup_lifted_2d(V, u, dirichlet)
    if two_d and dirichlet is not None:
        raise ValueError("newton: with dirichlet= the unknown must be a Function of V.unconstrained, so that it can "
                         "carry the data")
    if isinstance(u, ProductFunction):
        if not is_prod:
            raise ValueError("newton: the unknown is a ProductFunction but the residual is not on a ProductSpace")
        if u.product is V:
            if left is not None or right is not None:
                raise ValueError("newton: boundary data need the unknown in W.unconstrained, "
                                 "w = fd.Functions(W.unconstrained), so that it can carry the data")
            return V, (lambda: u.vector.copy()), (lambda c: u.vector.__setitem__(slice(None), c)), None, None
        if u.product is not V.unconstrained:
            raise ValueError("newton: the unknown must be a ProductFunction of the residual's ProductSpace W, "
                             "or of W.unconstrained")
        return _setup_lifted_product(V, u, left, right)
    if not isinstance(u, (Function, VectorElementFunction)):
        raise TypeError("newton: the unknown must be a Function or a ProductFunction")
    if is_prod:
        raise ValueError("newton: the residual is on a ProductSpace but the unknown is a plain Function")
    if u.space is V:
        if left is not None or right is not None:
            raise ValueError("newton: boundary data need the unknown in V.unconstrained, "
                             "u = fd.Function(V.unconstrained), so that it can carry the data")
        return V, (lambda: u.vector.copy()), (lambda c: u.vector.__setitem__(slice(None), c)), None, None
    if u.space is not V.unconstrained:
        raise ValueError("newton: the unknown must live in the test space V of the residual, or in V.unconstrained")
    ug = V.lift(left=left, right=right).vector
    state = [Transfer(V, u.space, "interpolate")(u.vector - ug, check=False)]   # the guess, minus the data, in V
    P = V.prolongate

    def set_c(c):
        state[0] = np.array(c, dtype=np.float64)
        u.vector[:] = ug + P(state[0])
    set_c(state[0])
    return V, (lambda: state[0].copy()), set_c, ug, P


def residual_function(R, u, left=None, right=None, *, dirichlet=None):
    """The residual as a function of the free coefficients, for solvers that work on plain arrays
    (fd.csnewton, scipy.optimize.root, a Newton loop of your own).

        F, c0, set_u = fd.residual_function(R, u)
        c, info = fd.csnewton(F, c0)
        set_u(c)                                  # u = u_g + P c, the boundary data included

    R and u are as for fd.newton, and the boundary data come from the same places: the space, or
    left=/right= (1D) and dirichlet= (2D) given here, which replace the space's.  F(c) is R at the
    field u = u_g + P c, where the lift u_g carries the boundary data and P = V.prolongate maps the
    coefficients of V to those of V.unconstrained (for u in V itself, u = c).  F accepts complex c,
    as the complex step needs, and leaves u alone.  c0 is the current u as free coefficients, the
    initial guess, and set_u(c) writes the field of c into u.  Like fd.newton, this first gives u
    the boundary data."""
    from . import form, Form
    if not isinstance(R, Form):
        R = form(R)
    if R.rank != 1:
        raise TypeError(f"residual_function: the residual must be a rank-1 form, got rank {R.rank}")
    V, get_c, set_c, ug, P = _setup(R, u, left, right, dirichlet)
    name = u.name
    if name not in R.functions:
        raise ValueError(f"residual_function: the form does not read the unknown '{name}' "
                         f"(its Functions are {R.functions})")

    def F(c):
        c = np.asarray(c).reshape(-1)
        full = c if P is None else ug + P(c)
        return np.asarray(R.assemble(**{name: full}))

    return F, get_c(), set_c


def newton(R, u, left=None, right=None, *, dirichlet=None, tol=1e-10, rtol=0.0, maxiter=50, line_search=True,
           linear="direct", jv="assembled", precond="frozen", backend="Auto",
           krylov_tol=1e-8, restart=30, krylov_maxiter=1000, cs_h=1e-20, verbose=False, warn=True):
    """Solve R(u; v) = 0 for all test functions v by Newton's method.  Returns a NewtonInfo.

    R: a rank-1 form (or the expression that makes one) depending on the Function u.
    u: the unknown, updated in place: a Function of the test space V, of V.unconstrained
       when boundary data left=/right= are given (as in V.lift), or a ProductFunction of the
       test ProductSpace W, or of W.unconstrained for data on the fields (left=/right= one entry
       per field, as in W.lift, or the data given with the fields).
       On a 2D mesh (a scalar space or a system): u in V.unconstrained carries Dirichlet data,
       dirichlet= (as in V.lift) or, without it, the boundary values u holds already.
    Stops when ||R|| <= max(tol, rtol * ||R(u_0)||).  With line_search, each step is
    halved until ||R|| decreases by the Armijo factor (1 - 1e-4 s).
    linear: "direct", "gmres" or "lgmres".  jv, precond: see the module docstring.
    backend: the direct solver backend (as in Matrix.solver), also used for precond."""
    from . import form, Form, Matrix, LinearSolver, gmres, lgmres, ProductSpace
    from .forms import Function, ProductFunction
    if not isinstance(R, Form):
        R = form(R)
    if R.rank != 1:
        raise TypeError(f"newton: the residual must be a rank-1 form, got rank {R.rank}")
    if linear not in ("direct", "gmres", "lgmres"):
        raise ValueError("newton: linear must be 'direct', 'gmres' or 'lgmres'")
    if jv not in ("assembled", "exact", "complex-step", "fd"):
        raise ValueError("newton: jv must be 'assembled', 'exact', 'complex-step' or 'fd'")
    if linear == "direct" and jv != "assembled":
        raise ValueError("newton: jv= chooses the Krylov matvec; use it with linear='gmres' or 'lgmres'")

    V, get_c, set_c, ug, P = _setup(R, u, left, right, dirichlet)
    plain = P is not None or isinstance(u, Function)
    J = R.derivative(u, space=V if plain else None)

    def residual():
        return np.array(R.assemble(), dtype=np.float64)

    # ---- the Jacobian-vector product for the Krylov modes
    Jmat = [None]
    if jv == "exact":
        dv = (ProductFunction(V, "_newton_direction") if isinstance(V, ProductSpace) and getattr(V, "tdim", 1) == 1
              else Function(V, "_newton_direction", _infer=False))
        Jaction = J.action(dv)

        def matvec(d):
            dv.vector[:] = d
            return np.asarray(Jaction.assemble(), dtype=np.float64)
    elif jv == "complex-step":
        def matvec(d):
            c = get_c()
            dz = cs_h * (P(d) if P is not None else d)
            z = (ug + P(c) if P is not None else c) + 1j * dz
            return np.asarray(R._assemble_owner({id(u): z})).imag / cs_h
    elif jv == "fd":
        def matvec(d):
            c = get_c()
            nd = np.linalg.norm(d)
            if nd == 0.0:
                return np.zeros_like(d)
            eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(c)) / nd
            r0 = residual()
            set_c(c + eps * d)
            r1 = residual()
            set_c(c)
            return (r1 - r0) / eps
    else:
        matvec = None

    # ---- the preconditioner for the Krylov modes
    frozen = [None]

    def factor(M):
        return M.solver(backend)

    def preconditioner():
        if precond is None:
            return None
        if isinstance(precond, (LinearSolver, Matrix)) or callable(precond) and not isinstance(precond, str):
            return precond
        if precond == "linear":
            if frozen[0] is None:
                frozen[0] = factor(R.derivative(u, space=V if plain else None, part="constant").assemble())
            return frozen[0]
        if precond == "jacobian":
            return factor(J.assemble())
        if precond == "frozen":
            if frozen[0] is None:
                frozen[0] = factor(J.assemble())
            return frozen[0]
        raise ValueError("newton: precond must be 'frozen', 'linear', 'jacobian', None, a LinearSolver, a Matrix or a callable")

    info = NewtonInfo(False, 0)
    r = residual()
    nr = float(np.linalg.norm(r))
    info.residuals.append(nr)
    target = max(tol, rtol * nr)
    if verbose:
        print(f"newton  0: ||R|| = {nr:.3e}")
    for it in range(1, maxiter + 1):
        if nr <= target:
            info.converged = True
            break
        c = get_c()
        # ---- the Newton direction: J d = -R
        if linear == "direct":
            try:
                d = np.asarray(factor(J.assemble()).solve(-r), dtype=np.float64)
            except (ValueError, RuntimeError) as e:
                info.message = f"the Jacobian could not be factored ({e})"
                break
            info.linear_iterations.append(0)
        else:
            A = J.assemble() if jv == "assembled" else matvec
            solve = gmres if linear == "gmres" else lgmres
            M = preconditioner()
            dsol, kinfo = solve(A, -r, M=M, restart=restart, maxiter=krylov_maxiter, tol=krylov_tol, warn=False)
            d = np.asarray(dsol, dtype=np.float64)
            info.linear_iterations.append(kinfo.iterations)
            if precond == "frozen" and kinfo.iterations > max(5, restart // 2):
                frozen[0] = None                             # it has gone stale: refactor next time
            if not kinfo.converged and verbose:
                print(f"          {kinfo}")
        if not np.all(np.isfinite(d)):
            info.message = "the Newton direction is not finite; the Jacobian is singular or nearly so"
            break
        # ---- line search on ||R||
        s = 1.0
        while True:
            set_c(c + s * d)
            r_new = residual()
            n_new = float(np.linalg.norm(r_new))
            ok = np.isfinite(n_new) and n_new <= (1.0 - 1e-4 * s) * nr
            if ok or (not line_search and np.isfinite(n_new)) or s < 2.0**-30:
                break
            s *= 0.5
        if not np.isfinite(n_new) or (line_search and not ok):
            set_c(c)
            info.message = ("the residual is not finite along the Newton direction" if not np.isfinite(n_new) else
                            "the line search could not reduce ||R||; the Jacobian may be singular or the guess too far")
            break
        r, nr = r_new, n_new
        info.iterations, info.steps = it, info.steps + [s]
        info.residuals.append(nr)
        if verbose:
            lin = f", {info.linear_iterations[-1]} Krylov iterations" if linear != "direct" else ""
            print(f"newton {it:2d}: ||R|| = {nr:.3e}, step {s:g}{lin}")
    else:
        info.converged = nr <= target
    if not info.converged and nr <= target:
        info.converged = True
    if warn and not info.converged:
        warnings.warn(f"newton did not converge: ||R|| = {nr:.2e} after {info.iterations} iterations"
                      + (f" ({info.message})" if info.message else ""), RuntimeWarning, stacklevel=2)
    return info
