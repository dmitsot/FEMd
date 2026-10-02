"""
Newton for algebraic systems, and implicit Runge-Kutta time stepping on top of it.

    x, info = fd.newton_system(F, J, x0)          # F(x) = 0, J(x) its Jacobian

    irk = fd.IRK(M, [S, (-1.0, Nf)], dt)          # M u' = S u - N(u), Gauss-Legendre, 2 stages (order 4)
    info = irk.step(u)                            # one step, u updated in place
    rk = fd.ERK(M, f, dt, "rk4")                  # classical explicit RK4 for non-stiff problems

`IRK` builds the stage equations, their exact Jacobian as one banded block matrix
(`fd.block`), and solves them by Newton's method with direct solves, so a time
loop never has to spell out the stages or the Newton iteration.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .newton import NewtonInfo

__all__ = ["newton_system", "IRK", "SSPRK", "ERK", "butcher"]


# --------------------------------------------------------------------------- Newton for F(x) = 0
def _linear_solve(Jx, r, backend):
    from . import Matrix, RectMatrix, LinearSolver
    if isinstance(Jx, Matrix):
        return np.asarray(Jx.solver(backend).solve(r), dtype=np.float64)
    if isinstance(Jx, RectMatrix):
        Jx = Jx.tocsr()
    if isinstance(Jx, LinearSolver) or (hasattr(Jx, "solve") and not sp.issparse(Jx) and not isinstance(Jx, np.ndarray)):
        return np.asarray(Jx.solve(r), dtype=np.float64)
    if sp.issparse(Jx):
        return np.asarray(spla.spsolve(sp.csc_matrix(Jx), r), dtype=np.float64)
    if isinstance(Jx, np.ndarray):
        return np.linalg.solve(Jx, r)
    raise TypeError(f"newton_system: J(x) must return a Matrix, a LinearSolver, a SciPy sparse matrix or an ndarray, "
                    f"got {type(Jx).__name__}")


def newton_system(F, J, x0, *, tol=1e-10, rtol=0.0, xtol=0.0, maxiter=50, line_search=True, backend="Auto",
                  verbose=False, warn=True):
    """Solve F(x) = 0 by Newton's method.  Returns (x, NewtonInfo).

    F: x -> residual vector.  J: x -> the Jacobian at x, as a Matrix (factored with
    `backend`), a LinearSolver (already factored, reused as it is), a SciPy sparse
    matrix, or a dense ndarray.  Each iteration solves J(x) d = -F(x); with line_search
    the step is halved until ||F|| decreases by the Armijo factor.  Stops when
    ||F|| <= max(tol, rtol ||F(x0)||).  A simplified Newton method is J returning the
    same LinearSolver every time.

    xtol > 0 also stops when the error left after a step is estimated below xtol ||x||, with
    the contraction rate theta = ||d_k|| / ||d_{k-1}||:  theta / (1 - theta) ||d_k|| <= xtol ||x||
    (as in Hairer and Wanner's IRK codes), or, on the first iteration and once the iteration no
    longer contracts, when ||d_k|| <= xtol ||x|| itself.  It also stops when a full step leaves
    ||F|| where it was (not 10% smaller) while ||d_k|| <= sqrt(xtol) ||x||: then ||F|| is at its
    floor and the steps are round-off noise.  This ends the iteration once the step is at
    round-off, when ||F|| sits at its floor, which grows with the size of the problem and can
    lie above an absolute tol."""
    x = np.array(x0, dtype=np.float64).reshape(-1)
    r = np.asarray(F(x), dtype=np.float64)
    nr = float(np.linalg.norm(r))
    info = NewtonInfo(False, 0, residuals=[nr])
    target = max(tol, rtol * nr)
    nd_prev = 0.0
    if verbose:
        print(f"newton  0: ||F|| = {nr:.3e}")
    for it in range(1, maxiter + 1):
        if nr <= target:
            break
        try:
            d = _linear_solve(J(x), -r, backend)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            info.message = f"the Jacobian could not be factored ({e})"
            break
        if not np.all(np.isfinite(d)):
            info.message = "the Newton direction is not finite; the Jacobian is singular or nearly so"
            break
        nd = float(np.linalg.norm(d))
        theta = nd / nd_prev if nd_prev > 0.0 else None
        nd_prev = nd
        bound = xtol * np.linalg.norm(x)
        if theta is not None and theta < 1.0:
            done = theta / (1.0 - theta) * nd <= bound          # contracting: estimated error left
        else:
            done = nd <= bound       # first step, or steps of pure round-off that no longer contract
        if xtol > 0.0 and done:
            x = x + d
            r = np.asarray(F(x), dtype=np.float64)
            info.iterations = it
            info.steps.append(1.0)
            info.linear_iterations.append(0)
            info.residuals.append(float(np.linalg.norm(r)))
            info.converged = True
            info.message = "converged on the step: the error left is below xtol, ||F|| is at its round-off floor"
            if verbose:
                print(f"newton {it:2d}: ||F|| = {info.residuals[-1]:.3e}, step {nd:.1e}, stopped on the step")
            return x, info
        s = 1.0
        while True:
            x_new = x + s * d
            r_new = np.asarray(F(x_new), dtype=np.float64)
            n_new = float(np.linalg.norm(r_new))
            ok = np.isfinite(n_new) and n_new <= (1.0 - 1e-4 * s) * nr
            if ok or (not line_search and np.isfinite(n_new)) or s < 2.0**-30:
                break
            s *= 0.5
        if not np.isfinite(n_new) or (line_search and not ok):
            info.message = ("the residual is not finite along the Newton direction" if not np.isfinite(n_new) else
                            "the line search could not reduce ||F||")
            break
        stalled = xtol > 0.0 and n_new >= 0.9 * nr and nd <= np.sqrt(xtol) * np.linalg.norm(x)
        if stalled and n_new > nr:
            x_new, r_new, n_new = x, r, nr            # keep the better of the two
        x, r, nr = x_new, r_new, n_new
        info.iterations = it
        info.steps.append(s)
        info.linear_iterations.append(0)
        info.residuals.append(nr)
        if verbose:
            print(f"newton {it:2d}: ||F|| = {nr:.3e}, step {s:g}")
        if stalled:
            info.converged = True
            info.message = "converged: ||F|| stalls at its round-off floor while the step is tiny"
            return x, info
    info.converged = nr <= target
    if warn and not info.converged:
        warnings.warn(f"newton_system did not converge: ||F|| = {nr:.2e} after {info.iterations} iterations"
                      + (f" ({info.message})" if info.message else ""), RuntimeWarning, stacklevel=2)
    return x, info


# --------------------------------------------------------------------------- Butcher tableaus
def butcher(method="gauss", stages=2):
    """(A, b, c) of a Runge-Kutta method.

    "gauss": Gauss-Legendre, order 2s, A-stable, symplectic, conserves every quadratic invariant.
    "radau": Radau IIA, order 2s-1, L-stable, stiffly accurate (the last stage is the new value).
    Explicit (stages is then ignored): "euler", "midpoint" (also "rk2"), "heun", "ralston",
    "rk3", "rk4", "3/8", the tableaux of fd.ERK."""
    r3, r6, r15 = np.sqrt(3.0), np.sqrt(6.0), np.sqrt(15.0)
    m = str(method).lower()
    if m in _EXPLICIT_NAMES:
        from . import _femd as _C
        A, b, c, _, _ = _C.explicit_tableau(m)
        s_ = len(b)
        return np.asarray(A, dtype=np.float64).reshape(s_, s_), np.asarray(b), np.asarray(c)
    if m in ("gauss", "gauss-legendre", "gl"):
        if stages == 1:
            A, b = [[0.5]], [1.0]
        elif stages == 2:
            A, b = [[1 / 4, 1 / 4 - r3 / 6], [1 / 4 + r3 / 6, 1 / 4]], [1 / 2, 1 / 2]
        elif stages == 3:
            A = [[5 / 36, 2 / 9 - r15 / 15, 5 / 36 - r15 / 30],
                 [5 / 36 + r15 / 24, 2 / 9, 5 / 36 - r15 / 24],
                 [5 / 36 + r15 / 30, 2 / 9 + r15 / 15, 5 / 36]]
            b = [5 / 18, 4 / 9, 5 / 18]
        else:
            raise ValueError("butcher: Gauss-Legendre with 1, 2 or 3 stages")
    elif m in ("radau", "radau iia", "radauiia", "radau-iia"):
        if stages == 1:
            A, b = [[1.0]], [1.0]
        elif stages == 2:
            A, b = [[5 / 12, -1 / 12], [3 / 4, 1 / 4]], [3 / 4, 1 / 4]
        elif stages == 3:
            A = [[(88 - 7 * r6) / 360, (296 - 169 * r6) / 1800, (-2 + 3 * r6) / 225],
                 [(296 + 169 * r6) / 1800, (88 + 7 * r6) / 360, (-2 - 3 * r6) / 225],
                 [(16 - r6) / 36, (16 + r6) / 36, 1 / 9]]
            b = [(16 - r6) / 36, (16 + r6) / 36, 1 / 9]
        else:
            raise ValueError("butcher: Radau IIA with 1, 2 or 3 stages")
    else:
        raise ValueError(f"butcher: unknown method {method!r}; use 'gauss', 'radau', or pass (A, b)")
    A, b = np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)
    return A, b, A.sum(axis=1)


_EXPLICIT_NAMES = {"euler", "forward euler", "rk1", "midpoint", "rk2", "explicit midpoint", "heun", "trapezoid",
                   "explicit trapezoid", "ralston", "rk3", "kutta3", "rk4", "classical", "classical rk4", "3/8",
                   "rk38", "3/8 rule"}


# --------------------------------------------------------------------------- the right-hand side
def _arity(fn):
    """Number of required positional parameters of a callable, None when it cannot be told."""
    import inspect
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_POSITIONAL for p in ps):
        return None
    return sum(1 for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)


class _Rhs:
    """f(t, U) and f'(t, U) from a list of terms, or from a callable.

    Terms: a Matrix (a constant linear operator), a rank-1 Form in the unknown, a rank-2 Form,
    each optionally (coefficient, term).  A rank-2 Form that depends on nothing that changes is
    assembled once; one that contains a Function or a Constant is re-assembled at each call.
    `time` is the Constant set to t before every evaluation.  With `lifted`, U is the full field
    in V.unconstrained and every Jacobian is taken with respect to the coefficients in V."""

    def __init__(self, rhs, jacobian, unknown, V, time, lifted, need_jacobian=True):
        from . import Form, ProductSpace
        from . import Matrix as _BandMatrix
        from .sparse import SparseMatrix
        from .forms import Constant, Function, ProductFunction
        Matrix = (_BandMatrix, SparseMatrix)
        if time is not None and not isinstance(time, Constant):
            raise TypeError("IRK: time= must be an fd.Constant (the one the right-hand side's forms contain)")
        self.time, self.lifted, self.V = time, lifted, V
        self.callable = callable(rhs) and not isinstance(rhs, (Matrix, Form, list, tuple))
        if self.callable:
            if lifted:
                raise ValueError("IRK: boundary data need the right-hand side as Forms, so that it can act "
                                 "on the lifted field; a callable cannot")
            if jacobian is None and need_jacobian:
                raise ValueError("IRK: a callable right-hand side needs jacobian=, a callable returning a Matrix")
            self._f, self._J = rhs, jacobian
            self._ft = _arity(rhs) == 2          # f(t, U) rather than f(U)
            self._Jt = jacobian is not None and _arity(jacobian) == 2
            return
        if jacobian is not None:
            raise ValueError("IRK: jacobian= goes with a callable right-hand side; for Matrices and Forms it is derived")
        terms = rhs if isinstance(rhs, list) else [rhs]
        self.linear, self.forms, self.bilinear = None, [], []
        U = V.unconstrained
        for t in terms:
            coef, T = (float(t[0]), t[1]) if isinstance(t, tuple) else (1.0, t)
            if isinstance(T, Form) and T.rank == 2 and T.constant and not lifted:
                T = T.assemble()
            if isinstance(T, Matrix):
                if lifted:
                    raise ValueError("IRK: with boundary data a linear term must be a rank-2 Form, so that it can act "
                                     "on the lifted field; pass the form instead of the assembled Matrix")
                if T.shape != (V.dim, V.dim):
                    raise ValueError(f"IRK: a matrix term is {T.shape[0]} x {T.shape[1]}, the space has dim {V.dim}")
                self.linear = coef * T if self.linear is None else self.linear + coef * T
            elif isinstance(T, Form) and T.rank == 1:
                self.forms.append((coef, T, self._unknown(T, unknown)))
            elif isinstance(T, Form) and T.rank == 2:
                S = U if lifted else V
                holder = ProductFunction(S, "_irk_state") if isinstance(S, ProductSpace) else \
                    Function(S, "_irk_state", _infer=False)
                self.bilinear.append((coef, T, T.action(holder), holder))
            else:
                raise TypeError(f"IRK: a right-hand-side term must be a Matrix, a SparseMatrix or a Form, got {type(T).__name__}")
        if lifted:
            for _, T, name in self.forms:
                if name is None:
                    continue
                a = next(a for a in T._funcs.values() if a.name == name)
                if (a.product if a.product is not None else a.space) is not U:
                    raise ValueError(f"IRK: with boundary data the unknown '{name}' must be a Function of "
                                     "V.unconstrained, so that it can carry the data")

    @staticmethod
    def _unknown(T, unknown):
        from .forms import Function, ProductFunction
        if not T.functions:
            return None                          # a source term: it does not depend on the unknown
        if unknown is not None:
            name = unknown.name if isinstance(unknown, (Function, ProductFunction)) else str(unknown)
            if name not in T.functions:
                raise ValueError(f"IRK: the form {T} does not contain the unknown '{name}'")
            return name
        if len(T.functions) != 1:
            raise ValueError(f"IRK: the form has Functions {T.functions}; say which is the unknown with unknown=")
        return T.functions[0]

    def _at(self, t):
        if self.time is not None:
            self.time.assign(t)

    def f(self, t, U):
        self._at(t)
        if self.callable:
            return np.asarray(self._f(t, U) if self._ft else self._f(U), dtype=np.float64)
        out = self.linear.matvec(U) if self.linear is not None else 0.0
        for coef, T, name in self.forms:
            out = out + coef * np.asarray(T.assemble(**({} if name is None else {name: U})), dtype=np.float64)
        for coef, _, act, holder in self.bilinear:
            holder.vector[:] = U
            out = out + coef * np.asarray(act.assemble(), dtype=np.float64)
        return np.asarray(out, dtype=np.float64)

    def J(self, t, U):
        self._at(t)
        if self.callable:
            return self._J(t, U) if self._Jt else self._J(U)
        space = self.V if self.lifted else None
        out = self.linear
        for coef, T, name in self.forms:
            if name is None:
                continue                         # a source term has no Jacobian
            Jt = coef * T.jacobian(name, space=space, **{name: U})
            out = Jt if out is None else out + Jt
        for coef, T, _, _ in self.bilinear:
            Jt = coef * T.assemble()
            out = Jt if out is None else out + Jt
        return out


class _Data:
    """Data on the conditions built into V as functions of time: the lift u_g(t) and its derivative.

    Each end takes a number, a tuple (one value per functional built in there), or a callable of t
    returning either.  The derivative is taken by the complex step Im g(t + ih)/h, exact to round-off
    for data written with NumPy operations, and by a fourth-order central difference for a callable
    that does not accept a complex t."""

    def __init__(self, V, left, right):
        self.V, self.left, self.right = V, left, right
        self.fields = hasattr(V, "fields") and not hasattr(V, "degree")     # a ProductSpace: one entry per field
        if self.fields:
            self.left, self.right = V._per_field(left, "IRK(left=)"), V._per_field(right, "IRK(right=)")

    @staticmethod
    def _value(g, t):
        return g(t) if callable(g) else g

    @staticmethod
    def _rate(g, t):
        if not callable(g):
            return None if g is None else np.zeros(np.size(g))
        h = 1e-20
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")                 # e.g. math.cos casting a complex to real
                z = np.asarray(g(complex(float(t), h)))
            if np.iscomplexobj(z) and np.all(np.isfinite(z)) and np.any(z.imag != 0.0):
                return np.atleast_1d(z.imag / h)
            if np.iscomplexobj(z) and np.all(np.isfinite(z)):
                # imaginary part exactly 0: constant data, or a function that dropped it; tell them apart
                pass
        except (TypeError, ValueError):
            pass
        e = 1e-3 * max(1.0, abs(t))
        vals = [np.atleast_1d(np.asarray(g(t + k * e), dtype=np.float64)) for k in (-2, -1, 1, 2)]
        return (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * e)

    def _each(self, fn, g, t):
        return [fn(x, t) for x in g] if self.fields else fn(g, t)

    def lift(self, t):
        return self.V.lift(left=self._each(self._value, self.left, t), right=self._each(self._value, self.right, t)).vector

    def lift_rate(self, t):
        return self.V.lift(left=self._each(self._rate, self.left, t), right=self._each(self._rate, self.right, t)).vector


class IRK:
    """An implicit Runge-Kutta stepper for  M u' = f(t, u).

        irk = fd.IRK(M, rhs, dt, method="gauss", stages=2)
        for n in range(nsteps):
            info = irk.step(u)                 # u: a Function (or array) of M's space, updated in place

    M: the mass Matrix (or rank-2 Form), on a plain space V.
    rhs: f, as one term or a list of terms, each a Matrix (a linear operator), a rank-1
      Form in a Function of V (the nonlinear part, evaluated at the stage value), or a
      rank-2 Form, optionally weighted as (coefficient, term).  The Jacobian f'(U) is then
      exact: the Matrices themselves plus Form.jacobian of each Form.  Prefer rank-1 Forms,
      linear terms included: an assembled matrix applied in the residual carries O(eps/h^2)
      rounding in its entries, which makes conserved quantities drift (manual, Section 7.7).
      Alternatively rhs is a callable U -> f(U) or (t, U) -> f(t, U), with jacobian= a callable
      U -> Matrix or (t, U) -> Matrix.
    time: an fd.Constant contained in the forms of rhs.  Before every evaluation of f and of
      its Jacobian it is set to the time of the stage, t_n + c_i dt, so f may depend on t.
    t0: the initial time.  irk.t counts from it.
    left, right: data on the conditions built into V, each a number, a tuple, or a callable of
      t.  The field is then u = u_g(t) + P c, with u_g = V.lift(left=g_L(t), right=g_R(t)), and
      the stages solve for c in V:  M c' = f(t, u_g + P c) - M u_g'(t).  u is a Function of
      V.unconstrained, M must be a rank-2 Form (it acts on u_g'), rhs must be Forms, with the
      unknown a Function of V.unconstrained, and u_g' comes from the data by the complex step.
      The data hold exactly at every stage and at every step.
    method, stages: "gauss" (Gauss-Legendre, order 2s, conserves quadratic invariants) or
      "radau" (Radau IIA, order 2s-1, L-stable), s = 1, 2, 3; or method=(A, b).
    newton: "exact" re-evaluates the stage Jacobian at every iteration (classical Newton,
      quadratic convergence); "simplified" factors it once per step at (t_n, u^n) and reuses it.
    tol, xtol: Newton stops when ||F|| <= tol, or when the error left in the stages is estimated
      below xtol ||K|| from the contraction rate of the steps, which catches the round-off floor
      of ||F|| on large problems, where it can exceed tol (3e-10 for KdV at n = 8000).
    unknown: which Function of the Forms is the unknown, when they have more than one.

    On a 2D mesh (M a SparseMatrix, or a rank-2 Form, of a 2D space or system) the same call works:
      dirichlet: the data on the Dirichlet sides built into V, anything V.lift takes, where a
        callable g(x, y, t) or g(t) makes them depend on time (g(x, y) and numbers do not); u is
        then a Function of V.unconstrained.  Time-dependent data need M as a rank-2 Form.
      newton: "simplified" (the default in 2D): f' frozen at (t_n, u^n), the stage matrix split by
        the eigenvalues of A into s systems of size n (complex for a conjugate pair, one per pair),
        factored once per step; "frozen": that factorization kept from step to step and renewed
        when Newton takes more than irk.refresh (6) iterations or fails; "exact": the s n x s n
        stage Jacobian in the CSR store, factored at every iteration (fewest iterations, costliest).
      M may be singular, as for Stokes and Navier-Stokes (no pressure block): use Radau IIA.
      irk.factorizations counts the sparse factorizations.

    The stage equations  M K_i = F(t_n + c_i dt, u^n + dt sum_j a_ij K_j)  are solved together,
    their Jacobian  delta_ij M - dt a_ij F'(U_i)  assembled by fd.block into one banded matrix on
    ProductSpace(V, ..., V), so every Newton iteration is one direct O(n s^2 p) solve.  The
    stages of the last step start the next Newton iteration.  With time-dependent boundary data,
    stiff problems show the usual order reduction of Runge-Kutta methods towards the stage order.
    """

    def __new__(cls, M=None, *args, **kwargs):
        if cls is IRK and _is_2d_mass(M):
            return super().__new__(_IRK2D)                  # a mass matrix on a 2D mesh
        return super().__new__(cls)

    def __init__(self, M, rhs, dt, method="gauss", stages=2, *, jacobian=None, unknown=None, time=None, t0=0.0,
                 left=None, right=None, dirichlet=None, newton="exact", tol=1e-12, rtol=0.0, xtol=1e-12, maxiter=20,
                 line_search=False, backend="Auto"):
        from . import Matrix, Form, ProductSpace, Transfer
        from .forms import Function
        if dirichlet is not None:
            raise ValueError("IRK: dirichlet= is for 2D meshes; in 1D give left= and right=")
        Mform = None
        if isinstance(M, Form):
            if M.rank != 2:
                raise TypeError("IRK: M must be a rank-2 Form or a Matrix")
            Mform, M = M, M.assemble()
        if not isinstance(M, Matrix) or M.space is None:
            raise TypeError("IRK: M must be a Matrix (or rank-2 Form) of a space or a ProductSpace")
        if isinstance(method, (tuple, list)) and len(method) == 2:
            A, b = (np.array(m, dtype=np.float64) for m in method)
            if A.shape != (b.size, b.size):
                raise ValueError("IRK: method=(A, b) needs a square A with len(b) rows")
            self.A, self.b, self.c = A, b, A.sum(axis=1)
            self.method = "custom"
        else:
            self.A, self.b, self.c = butcher(method, stages)
            self.method = f"{method}, {self.b.size} stages"
        if newton not in ("exact", "simplified"):
            raise ValueError("IRK: newton must be 'exact' or 'simplified'")
        self.M, self.V, self.dt = M, M.space, float(dt)
        self.s, self.n = self.b.size, self.V.dim
        from .linalg import _space_data
        left, right, _ = _space_data(self.V, left, right, None)          # the space's data, if the call has none
        self.lifted = left is not None or right is not None
        self.rhs = _Rhs(rhs, jacobian, unknown, self.V, time, self.lifted)
        self.time = time
        if time is not None:
            time.assign(t0)
        # the stages on one ProductSpace: every field of V, once per stage, numbered by position
        self.product = isinstance(self.V, ProductSpace)
        self._fields = list(self.V.fields) if self.product else [self.V]
        self.nf = len(self._fields)
        self.P = ProductSpace(*(self._fields * self.s))
        if self.product:
            parts = self.V.split(np.arange(self.V.dim, dtype=np.float64))
            fld, loc = np.empty(self.V.dim, dtype=np.int64), np.empty(self.V.dim, dtype=np.int64)
            for f, p in enumerate(parts):
                g = np.rint(np.asarray(p)).astype(np.int64)
                fld[g], loc[g] = f, np.arange(g.size)
            self._fld, self._loc = fld, loc
        self.newton, self.tol, self.rtol, self.maxiter = newton, tol, rtol, maxiter
        self.xtol = xtol
        self.line_search, self.backend = line_search, backend
        self.K = None                       # the stages of the last step, in the product numbering
        self.t, self.history = float(t0), []
        if self.lifted:
            periodic = self.V.periodic if self.product else self.V.bc.is_periodic
            if periodic or self.V.n_constraints == 0:
                raise ValueError("IRK: left=/right= give data on built-in conditions, and this space has none")
            if Mform is None:
                raise TypeError("IRK: with boundary data, pass M as a rank-2 Form (it acts on the lift's time "
                                "derivative), not as an assembled Matrix")
            self.U = self.V.unconstrained
            self.data = _Data(self.V, left, right)
            if self.product:
                from .forms import ProductFunction
                self._hold = ProductFunction(self.U, "_irk_lift_rate")
                maps = [None if f.unconstrained is f else Transfer(f, f.unconstrained, "interpolate") for f in self._fields]
                U, V = self.U, self.V
                self._to_V = lambda u: V.gather([p if T is None else T @ p for T, p in zip(maps, U.split(u))])  # noqa: E731
            else:
                self._hold = Function(self.U, "_irk_lift_rate", _infer=False)
                T = Transfer(self.V, self.U, "interpolate")
                self._to_V = lambda u: T @ u                                                             # noqa: E731
            self._Mact = Mform.action(self._hold)
            self._prolong = self.V.prolongate

    # ---- the semi-discrete right-hand side in the coefficients c of V --------------------------
    def _full(self, t, c):
        return self.data.lift(t) + self._prolong(c) if self.lifted else c

    def _F(self, t, c):
        r = self.rhs.f(t, self._full(t, c))
        if self.lifted:
            self._hold.vector[:] = self.data.lift_rate(t)
            r = r - np.asarray(self._Mact.assemble(), dtype=np.float64)
        return r

    def _FJ(self, t, c):
        return self.rhs.J(t, self._full(t, c))

    def times(self):
        """The stage times t_n + c_i dt of the next step."""
        return self.t + self.dt * self.c

    def _split(self, Kp):
        """Stage vectors in V's numbering from one vector of the stage ProductSpace."""
        parts = self.P.split(Kp)
        if not self.product:
            return parts
        nf = self.nf
        return [self.V.gather(parts[i * nf:(i + 1) * nf]) for i in range(self.s)]

    def _gather(self, vecs):
        """One vector of the stage ProductSpace from the stage vectors in V's numbering."""
        if not self.product:
            return self.P.gather(vecs)
        return self.P.gather([p for v in vecs for p in self.V.split(v)])

    def stages(self, cn, Kp):
        """The stage values U_i = c^n + dt sum_j a_ij K_j, from K in the product numbering."""
        Ks = self._split(Kp)
        dt, A = self.dt, self.A
        return Ks, [cn + dt * sum(A[i, j] * Ks[j] for j in range(self.s)) for i in range(self.s)]

    def residual(self, cn, Kp):
        Ks, Us = self.stages(cn, Kp)
        ts = self.times()
        return self._gather([np.asarray(self.M.matvec(Ks[i]), dtype=np.float64) - self._F(ts[i], Us[i])
                             for i in range(self.s)])

    def jacobian(self, Us, ts=None):
        from . import block
        dt, A = self.dt, self.A
        ts = self.times() if ts is None else ts
        Js = [self._FJ(ts[i], U) for i, U in enumerate(Us)]
        blocks = [[(self.M if i == j else None) for j in range(self.s)] for i in range(self.s)]
        for i in range(self.s):
            for j in range(self.s):
                if A[i, j] != 0.0:
                    term = (-dt * A[i, j]) * Js[i]
                    blocks[i][j] = term if blocks[i][j] is None else blocks[i][j] + term
        if not self.product:
            return block(blocks, space=self.P, symmetric=False)   # a stage Jacobian is not symmetric
        return self._block_fields(blocks)

    def _block_fields(self, blocks):
        """block() for blocks that are Matrices on the ProductSpace V: each is scattered field
        pair by field pair into the stage ProductSpace."""
        from . import Matrix
        from . import _femd as _C
        K = _C.AssembledMatrix(self.P.dim, self.P.uband)
        nf = self.nf
        for i, row in enumerate(blocks):
            for j, B in enumerate(row):
                if B is None:
                    continue
                r, c, v = (np.asarray(x) for x in B._K.to_coo())
                fr, fc = self._fld[r], self._fld[c]
                for a in range(nf):
                    for b in range(nf):
                        m = (fr == a) & (fc == b)
                        if m.any():
                            K.add_coo(self.P, i * nf + a, j * nf + b, np.ascontiguousarray(self._loc[r[m]], dtype=np.int32),
                                      np.ascontiguousarray(self._loc[c[m]], dtype=np.int32),
                                      np.ascontiguousarray(v[m], dtype=np.float64))
        K.symmetric = False
        return Matrix(K, self.P.periodic, self.P)

    def step(self, u, dt=None):
        """Advance u by one step, in place, and irk.t by dt.  Returns the NewtonInfo.

        u is a Function (or array) of M's space, or of V.unconstrained when boundary data are given."""
        from .forms import Function, ProductFunction
        if dt is not None and float(dt) != self.dt:
            self.dt, self.K = float(dt), None
        vec = u.vector if hasattr(u, "vector") else u
        if self.lifted:
            if (isinstance(u, Function) and u.space is not self.U) or (isinstance(u, ProductFunction) and u.product is not self.U):
                raise ValueError("IRK: with boundary data, u must be a Function (ProductFunction) of V.unconstrained")
            if np.size(vec) != self.U.dim:
                raise ValueError(f"IRK: with boundary data, u needs {self.U.dim} raw coefficients, got {np.size(vec)}")
            cn = np.asarray(self._to_V(np.asarray(vec, dtype=np.float64) - self.data.lift(self.t)), dtype=np.float64)
        else:
            cn = np.array(vec, dtype=np.float64)
        if self.K is None:
            k0 = np.asarray(self.M.solver().solve(self._F(self.t, cn)), dtype=np.float64)
            self.K = self._gather([k0] * self.s)
        if self.newton == "exact":
            J = lambda Kp: self.jacobian(self.stages(cn, Kp)[1])          # noqa: E731
        else:
            frozen = self.jacobian([cn] * self.s, [self.t] * self.s).solver(self.backend)
            J = lambda Kp: frozen                                          # noqa: E731
        K, info = newton_system(lambda Kp: self.residual(cn, Kp), J, self.K, tol=self.tol, rtol=self.rtol, xtol=self.xtol,
                                maxiter=self.maxiter, line_search=self.line_search, backend=self.backend, warn=False)
        if not info.converged:
            raise RuntimeError(f"IRK: Newton did not converge at t = {self.t + self.dt:g}: ||F|| = {info.residual:.2e}"
                               f" after {info.iterations} iterations{' (' + info.message + ')' if info.message else ''}; "
                               "reduce dt, raise maxiter, or loosen tol")
        self.K = K
        Ks = self._split(K)
        c_new = cn + self.dt * sum(self.b[i] * Ks[i] for i in range(self.s))
        self.t += self.dt
        vec[:] = self._full(self.t, c_new)
        if self.time is not None:
            self.time.assign(self.t)
        self.history.append(info.iterations)
        return info

    def __repr__(self):
        extra = ", boundary data" if self.lifted else ""
        return f"IRK({self.method}, dt={self.dt:g}, t={self.t:g}, newton={self.newton}{extra}, on {self.V!r})"


# --------------------------------------------------------------------------- IRK on 2D meshes
def _is_2d_mass(M):
    """True for a mass matrix (or rank-2 form) on a 2D space: IRK then dispatches to _IRK2D."""
    from . import Form
    from .sparse import SparseMatrix
    if isinstance(M, SparseMatrix):
        return getattr(M.space, "tdim", 1) == 2
    if isinstance(M, Form) and M.rank == 2:
        return getattr(M.test_space, "tdim", 1) == 2
    return False


def _tableau(method, stages):
    """(A, b, c, description) from a method name and stage count, or from method=(A, b)."""
    if isinstance(method, (tuple, list)) and len(method) == 2:
        A, b = (np.array(m, dtype=np.float64) for m in method)
        if A.shape != (b.size, b.size):
            raise ValueError("IRK: method=(A, b) needs a square A with len(b) rows")
        return A, b, A.sum(axis=1), "custom"
    A, b, c = butcher(method, stages)
    return A, b, c, f"{method}, {b.size} stages"


def _time_rate(fn, t):
    """d/dt of fn(t) (an array or a number): the complex step Im fn(t + ih)/h, exact to round-off for
    NumPy code, and a fourth-order central difference when fn does not carry a complex t through."""
    h = 1e-20
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.asarray(fn(complex(float(t), h)))
        if np.iscomplexobj(z) and np.all(np.isfinite(z)):
            return z.imag / h
    except (TypeError, ValueError):
        pass
    e = 1e-3 * max(1.0, abs(t))
    v = [np.asarray(fn(t + k * e), dtype=np.float64) for k in (-2, -1, 1, 2)]
    return (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * e)


def _numbers(v):
    """A number, or a tuple of numbers (a vector block's value), from what a data callable g(t) returned."""
    a = np.asarray(v, dtype=np.float64)
    return float(a) if a.ndim == 0 else tuple(float(e) for e in a.ravel())


class _Data2D:
    """Dirichlet data on a 2D space as functions of time.

    The data are anything V.lift takes (a number, a callable, a pair for a vector block, a dict per
    marker, a list per block), where a callable g(x, y, t) or g(t) depends on time and a callable
    g(x, y) or a number does not.  g(t) returns a number, or a pair for a vector block.  The lift is
    u_g(t) = V.lift(data at t), and its derivative V.lift(d data / dt) by the complex step in t."""

    def __init__(self, V, g):
        self.V, self.g = V, g
        self.timedep = self._timedep(g)
        self._static = None if self.timedep else np.asarray(V.lift(g).vector, dtype=np.float64)

    @classmethod
    def _timedep(cls, g):
        if isinstance(g, dict):
            return any(cls._timedep(v) for v in g.values())
        if isinstance(g, (list, tuple)):
            return any(cls._timedep(v) for v in g)
        return callable(g) and _arity(g) in (1, 3)

    @classmethod
    def _at(cls, g, t, rate):
        """The data at time t (rate: their time derivative) in the form V.lift takes."""
        if isinstance(g, dict):
            return {k: cls._at(v, t, rate) for k, v in g.items()}
        if isinstance(g, (list, tuple)):
            return type(g)(cls._at(v, t, rate) for v in g)
        if g is None:
            return None
        if not callable(g):
            return 0.0 if rate else g
        n = _arity(g)
        if n == 3:
            if rate:
                return lambda x, y: _time_rate(lambda s: g(x, y, s), t) + 0.0 * x
            return lambda x, y: g(x, y, t)
        if n == 1:
            return _numbers(_time_rate(g, t) if rate else g(t))
        return 0.0 if rate else g

    def lift(self, t):
        if self._static is not None:
            return self._static
        return np.asarray(self.V.lift(self._at(self.g, t, False)).vector, dtype=np.float64)

    def lift_rate(self, t):
        return np.asarray(self.V.lift(self._at(self.g, t, True)).vector, dtype=np.float64)


def _csr(J):
    """A Jacobian from _Rhs (SparseMatrix, SciPy sparse, dense array or None) as a SciPy CSR matrix."""
    from .sparse import SparseMatrix
    if J is None:
        return None
    if isinstance(J, SparseMatrix):
        return J.tocsr()
    if sp.issparse(J):
        return sp.csr_matrix(J)
    if isinstance(J, np.ndarray):
        return sp.csr_matrix(J)
    raise TypeError(f"IRK: on a 2D mesh the Jacobian must be a SparseMatrix, a SciPy sparse matrix or an ndarray, "
                    f"got {type(J).__name__}")


class _StageSolver:
    """The simplified-Newton matrix  I (x) M - dt A (x) J  solved through A = T diag(lam) T^{-1}:
    one system M - dt lam_k J of size n per real eigenvalue and one complex system per conjugate
    pair, instead of one real system of size s n (Butcher's transformation, as in Hairer and
    Wanner's RADAU5).  A 2D factorization of the s n system costs about s^3 times one of size n."""

    def __init__(self, M, J, A, dt, backend="auto"):
        from .sparse import SparseMatrix
        s = A.shape[0]
        lam, T = np.linalg.eig(A)
        T = T.astype(np.complex128)
        partner = [None] * s
        for k in range(s):
            if partner[k] is not None or abs(lam[k].imag) <= 1e-12 * abs(lam[k]):
                continue
            j = next((j for j in range(k + 1, s) if partner[j] is None and abs(lam[j] - np.conj(lam[k])) <= 1e-10 * abs(lam[k])),
                     None)
            if j is None:
                raise ValueError("the eigenvalues of A do not come in conjugate pairs")
            partner[j] = k
            T[:, j] = np.conj(T[:, k])
        if np.linalg.cond(T) > 1e10:
            raise ValueError("A is not (well) diagonalizable")
        self.s, self.n, self.T, self.Ti, self.partner = s, M.shape[0], T, np.linalg.inv(T), partner
        Mc, Jc = M.tocsr(), _csr(J)
        self.solves = [None] * s
        self.kinds = []
        for k in range(s):
            if partner[k] is not None:
                continue
            l = lam[k]
            if abs(l.imag) <= 1e-12 * abs(l):
                B = SparseMatrix.from_scipy(Mc if Jc is None else Mc - (dt * l.real) * Jc)
                S = B.solver(backend)
                self.solves[k] = lambda z, S=S: (S._solve_array(np.ascontiguousarray(z.real))
                                                 + 1j * S._solve_array(np.ascontiguousarray(z.imag)))
                self.kinds.append(S.backend)
            else:
                C = Mc.astype(np.complex128) if Jc is None else Mc - (dt * l) * Jc
                lu = spla.splu(sp.csc_matrix(C, dtype=np.complex128))
                self.solves[k] = lu.solve
                self.kinds.append("superlu (complex)")

    def solve(self, r):
        Z = self.Ti @ np.asarray(r, dtype=np.float64).reshape(self.s, self.n)
        W = np.empty_like(Z)
        for k in range(self.s):
            if self.partner[k] is None:
                W[k] = self.solves[k](np.ascontiguousarray(Z[k]))
        for k in range(self.s):
            if self.partner[k] is not None:
                W[k] = np.conj(W[self.partner[k]])
        return np.ascontiguousarray((self.T @ W).real.ravel())


class _IRK2D(IRK):
    """IRK on a 2D space: fd.IRK dispatches here when M is a SparseMatrix, or a rank-2 Form, on a 2D mesh.

    The stages are stacked one after the other, K = [K_1; ...; K_s].  newton="simplified" (the
    default here) freezes f' at (t_n, u^n) and splits I (x) M - dt A (x) f' by the eigenvalues of A
    into systems of size n (_StageSolver); "frozen" keeps that factorization from step to step and
    refactors when Newton slows down (one factorization for a whole run of a linear problem);
    "exact" assembles delta_ij M - dt a_ij f'(U_i) as one s n x s n matrix in the CSR store at every
    iteration.  Boundary data: dirichlet= as in V.lift, with callables g(x, y, t) or g(t) for data
    that change in time, and u a Function of V.unconstrained.  M may be singular (a
    differential-algebraic system such as Stokes or Navier-Stokes, where the pressure has no time
    derivative); Radau IIA is then the method of choice."""

    def __init__(self, M, rhs, dt, method="gauss", stages=2, *, jacobian=None, unknown=None, time=None, t0=0.0,
                 dirichlet=None, left=None, right=None, newton="simplified", tol=1e-12, rtol=0.0, xtol=1e-12, maxiter=20,
                 line_search=False, backend="Auto"):
        from . import Form
        from .sparse import SparseMatrix
        from .forms import Function
        if left is not None or right is not None:
            raise ValueError("IRK: left= and right= are the ends of a 1D mesh; on a 2D mesh give dirichlet=")
        Mform = None
        if isinstance(M, Form):
            Mform, M = M, M.assemble()
        if not isinstance(M, SparseMatrix) or M.space is None or M.shape[0] != M.shape[1]:
            raise TypeError("IRK: on a 2D mesh M must be a square SparseMatrix of a space (from a form) or a rank-2 Form")
        self.A, self.b, self.c, self.method = _tableau(method, stages)
        if newton not in ("exact", "simplified", "frozen"):
            raise ValueError("IRK: newton must be 'exact', 'simplified' or 'frozen'")
        self.M, self.V, self.dt = M, M.space, float(dt)
        self.s, self.n = self.b.size, self.V.dim
        self._frozen, self.refresh, self.factorizations = None, 6, 0
        if M.shape[0] != self.n:
            raise ValueError(f"IRK: M is {M.shape[0]} x {M.shape[1]}, its space has dim {self.n}")
        from .linalg import _space_data
        _, _, dirichlet = _space_data(self.V, None, None, dirichlet)      # the space's data, if the call has none
        self.lifted = dirichlet is not None
        self.rhs = _Rhs(rhs, jacobian, unknown, self.V, time, self.lifted)
        self.time = time
        if time is not None:
            time.assign(t0)
        self.product = hasattr(self.V, "fields")
        self.newton, self.tol, self.rtol, self.maxiter = newton, tol, rtol, maxiter
        self.xtol = xtol
        self.line_search = line_search
        b = getattr(backend, "name", backend)
        self.backend = "auto" if b is None or str(b).lower() == "auto" else str(b).lower()
        self.K = None
        self.t, self.history = float(t0), []
        self._Msolver = None
        self.data = None
        if self.lifted:
            fields = self.V.fields if self.product else [self.V]
            if sum(f.n_constraints for f in fields) == 0:
                raise ValueError("IRK: dirichlet= gives data on Dirichlet sides built into the space, and this space "
                                 "has none")
            self.U = self.V.unconstrained
            self.data = _Data2D(self.V, dirichlet)
            if self.data.timedep:
                if Mform is None:
                    raise TypeError("IRK: with time-dependent boundary data, pass M as a rank-2 Form (it acts on the "
                                    "lift's time derivative), not as an assembled SparseMatrix")
                self._hold = Function(self.U, "_irk_lift_rate", _infer=False)
                self._Mact = Mform.action(self._hold)

    # ---- the semi-discrete right-hand side in the coefficients c of V --------------------------
    def _full(self, t, c):
        return self.data.lift(t) + self.V.prolongate(c) if self.lifted else c

    def _F(self, t, c):
        r = np.asarray(self.rhs.f(t, self._full(t, c)), dtype=np.float64)
        if self.lifted and self.data.timedep:
            self._hold.vector[:] = self.data.lift_rate(t)
            r = r - np.asarray(self._Mact.assemble(), dtype=np.float64)
        return r

    def _split(self, K):
        return list(np.asarray(K, dtype=np.float64).reshape(self.s, self.n))

    def _gather(self, vecs):
        return np.concatenate([np.asarray(v, dtype=np.float64) for v in vecs])

    def stages(self, cn, K):
        """The stage values U_i = c^n + dt sum_j a_ij K_j, with K = [K_1; ...; K_s]."""
        Ks = np.asarray(K, dtype=np.float64).reshape(self.s, self.n)
        Us = cn[None, :] + self.dt * (self.A @ Ks)
        return list(Ks), list(Us)

    def residual(self, cn, K):
        Ks, Us = self.stages(cn, K)
        ts = self.times()
        return np.concatenate([self.M.matvec(Ks[i]) - self._F(ts[i], Us[i]) for i in range(self.s)])

    def jacobian(self, Us, ts=None):
        """The stage Jacobian delta_ij M - dt a_ij f'(t_i, U_i), one SparseMatrix of size s n."""
        from .sparse import SparseMatrix
        ts = self.times() if ts is None else ts
        Mc = self.M.tocsr()
        rows = []
        for i in range(self.s):
            Ji = _csr(self.rhs.J(ts[i], self._full(ts[i], Us[i])))
            row = []
            for j in range(self.s):
                blk = Mc if i == j else None
                if self.A[i, j] != 0.0 and Ji is not None:
                    term = (-self.dt * self.A[i, j]) * Ji
                    blk = term if blk is None else blk + term
                row.append(blk)
            rows.append(row)
        B = sp.bmat(rows, format="csr")
        return SparseMatrix.from_scipy(B, symmetric=False)

    def _factor(self, B):
        return B.solver(self.backend)

    def stage_solver(self, cn):
        """The simplified-Newton solver, with f' frozen at (t_n, c^n): decoupled by the eigenvalues of A
        (_StageSolver), or the s n block matrix when A is not diagonalizable."""
        J = self.rhs.J(self.t, self._full(self.t, cn))
        try:
            S = _StageSolver(self.M, J, self.A, self.dt, self.backend)
            self.factorizations += len(S.kinds)
            return S
        except ValueError:
            self.factorizations += 1
            return self._factor(self.jacobian([cn] * self.s, [self.t] * self.s))

    def _k0(self, cn):
        """The first Newton guess for the stages: K_i = M^{-1} f(t_n, u^n), or zero when M is singular."""
        F = self._F(self.t, cn)
        if self._Msolver is None:
            try:
                self._Msolver = self.M.solver()
            except (RuntimeError, ValueError):
                self._Msolver = False                    # singular: a differential-algebraic system
        if self._Msolver is False:
            return np.zeros(self.n)
        k = np.asarray(self._Msolver._solve_array(np.ascontiguousarray(F)), dtype=np.float64)
        return k if np.all(np.isfinite(k)) else np.zeros(self.n)

    def step(self, u, dt=None):
        """Advance u by one step, in place, and irk.t by dt.  Returns the NewtonInfo.

        u is a Function (ProductFunction, VectorFunction) or array of M's space, or of V.unconstrained
        when dirichlet= is given."""
        if dt is not None and float(dt) != self.dt:
            self.dt, self.K = float(dt), None
        vec = u.vector if hasattr(u, "vector") else u
        if not isinstance(vec, np.ndarray):
            raise TypeError("IRK.step: u must be a Function, a ProductFunction or a float64 array")
        if self.lifted:
            if vec.size != self.U.dim:
                raise ValueError(f"IRK: with dirichlet=, u must be a Function of V.unconstrained ({self.U.dim} "
                                 f"coefficients), got {vec.size}")
            cn = np.asarray(self.V.restrict(np.asarray(vec, dtype=np.float64) - self.data.lift(self.t)),
                            dtype=np.float64)
        else:
            if vec.size != self.n:
                raise ValueError(f"IRK: u has {vec.size} coefficients, M's space has {self.n}")
            cn = np.array(vec, dtype=np.float64)
        if self.K is None:
            self.K = np.tile(self._k0(cn), self.s)
        solve = lambda J: newton_system(lambda K: self.residual(cn, K), J, self.K, tol=self.tol, rtol=self.rtol,  # noqa: E731
                                        xtol=self.xtol, maxiter=self.maxiter, line_search=self.line_search, warn=False)
        if self.newton == "exact":
            def J(K):
                self.factorizations += 1
                return self._factor(self.jacobian(self.stages(cn, K)[1]))
            K, info = solve(J)
        else:
            fresh = self.newton == "simplified" or self._frozen is None or self._frozen[0] != self.dt
            if fresh:
                self._frozen = (self.dt, self.stage_solver(cn))
            K, info = solve(lambda K: self._frozen[1])
            if self.newton == "frozen" and not fresh and not info.converged:
                self._frozen = (self.dt, self.stage_solver(cn))        # gone stale: refactor at (t_n, u^n), retry
                K, info = solve(lambda K: self._frozen[1])
            elif self.newton == "frozen" and info.iterations > self.refresh:
                self._frozen = None                                    # slow: refactor at the next step's start
        if not info.converged:
            raise RuntimeError(f"IRK: Newton did not converge at t = {self.t + self.dt:g}: ||F|| = {info.residual:.2e}"
                               f" after {info.iterations} iterations{' (' + info.message + ')' if info.message else ''}; "
                               "reduce dt, raise maxiter, or loosen tol")
        self.K = K
        Ks = np.asarray(K).reshape(self.s, self.n)
        c_new = cn + self.dt * (self.b @ Ks)
        self.t += self.dt
        vec[:] = self._full(self.t, c_new)
        if self.time is not None:
            self.time.assign(self.t)
        self.history.append(info.iterations)
        return info

    def __repr__(self):
        extra = ", boundary data" if self.lifted else ""
        return f"IRK({self.method}, dt={self.dt:g}, t={self.t:g}, newton={self.newton}{extra}, on {self.V!r})"


# --------------------------------------------------------------------------- explicit SSP Runge-Kutta
_SSP = {(1, 1): "forward Euler", (2, 2): "Heun (SSPRK(2,2))", (3, 3): "Shu-Osher (SSPRK(3,3))",
        (4, 3): "SSPRK(4,3)", (10, 4): "Ketcheson SSPRK(10,4)"}
_SSP_CFL = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0, (4, 3): 2.0, (10, 4): 6.0}


class SSPRK:
    """An explicit strong-stability-preserving Runge-Kutta stepper for  M u' = f(t, u).

        lim = fd.TVBLimiter(V, M=50.0)                      # optional
        rk = fd.SSPRK(M, R, dt, order=3, limiter=lim)
        for n in range(nsteps):
            rk.step(u)                                      # u: a Function or ProductFunction, in place

    M: the mass Matrix, SparseMatrix (2D) or rank-2 Form, of a space or a ProductSpace. It is
      factored once (a DG Legendre mass is diagonal, so every solve is a division; a DGSpace2D mass
      is block diagonal, and its sparse Cholesky factor has no fill).
    rhs: f, as in IRK: one term or a list of terms, each a Matrix, a rank-1 Form in the unknown,
      or a rank-2 Form, optionally (coefficient, term); or a callable U -> f(U) or (t, U) -> f(t, U).
      No Jacobian is needed.
    order, stages: (1, 1) forward Euler, (2, 2) Heun, (3, 3) Shu-Osher (the default), (4, 3)
      with SSP coefficient 2, (10, 4) Ketcheson's fourth-order method with SSP coefficient 6.
      Each is a convex combination of forward Euler steps, so any property forward Euler has
      under dt <= dt_FE (TVD in the means, a maximum principle, positivity) holds for
      dt <= cfl * dt_FE, with cfl = rk.cfl.
    limiter: a callable applied to every stage (and the result), array in, array out, such as
      fd.TVBLimiter.
    time: an fd.Constant in the forms, set to the time of each stage; t0 the initial time.
    unknown: which Function of the Forms is the unknown, when they have more than one.
    backend: the solver backend for M (Auto measures it)."""

    def __init__(self, M, rhs, dt, order: int = 3, stages=None, *, limiter=None, unknown=None, time=None,
                 t0=0.0, backend="Auto"):
        from . import Matrix, Form
        from .sparse import SparseMatrix
        if isinstance(M, Form):
            if M.rank != 2:
                raise TypeError("SSPRK: M must be a rank-2 Form, a Matrix or a SparseMatrix")
            M = M.assemble()
        if not isinstance(M, (Matrix, SparseMatrix)) or M.space is None:
            raise TypeError("SSPRK: M must be a Matrix, a SparseMatrix (2D) or a rank-2 Form, of a space or a ProductSpace")
        order = int(order)
        stages = {1: 1, 2: 2, 3: 3, 4: 10}.get(order) if stages is None else int(stages)
        if (stages, order) not in _SSP:
            raise ValueError(f"SSPRK: (stages, order) must be one of {sorted(_SSP, key=lambda k: (k[1], k[0]))}, "
                             f"got ({stages}, {order})")
        if limiter is not None and not callable(limiter):
            raise TypeError("SSPRK: limiter must be callable, array in, array out")
        self.M, self.V, self.dt = M, M.space, float(dt)
        self.order, self.stages = order, stages
        self.method, self.cfl = _SSP[(stages, order)], _SSP_CFL[(stages, order)]
        self.limiter = limiter
        self.rhs = _Rhs(rhs, None, unknown, self.V, time, False, need_jacobian=False)
        self._solver = M.solver(backend)
        self.time, self.t = time, float(t0)
        if time is not None:
            time.assign(t0)

    @property
    def solver(self):
        """The factored solver of the mass matrix M: a SparseSolver (2D; .backend, and .factor for the
        sparse Cholesky: nnz_L, min_pivot, ...) or a LinearSolver (1D).  None when M = I."""
        return self._solver

    def rate(self, t, u):
        """u' = M^{-1} f(t, u), the forward Euler direction."""
        return np.asarray(self._solver.solve(self.rhs.f(t, u)), dtype=np.float64)

    def _lim(self, u):
        return u if self.limiter is None else np.asarray(self.limiter(u), dtype=np.float64)

    def _advance(self, u0):
        dt, t, L, lim = self.dt, self.t, self.rate, self._lim
        s = (self.stages, self.order)
        if s == (1, 1):
            return lim(u0 + dt * L(t, u0))
        if s == (2, 2):
            u1 = lim(u0 + dt * L(t, u0))
            return lim(0.5 * u0 + 0.5 * (u1 + dt * L(t + dt, u1)))
        if s == (3, 3):
            u1 = lim(u0 + dt * L(t, u0))
            u2 = lim(0.75 * u0 + 0.25 * (u1 + dt * L(t + dt, u1)))
            return lim(u0 / 3.0 + 2.0 / 3.0 * (u2 + dt * L(t + 0.5 * dt, u2)))
        if s == (4, 3):
            h = 0.5 * dt
            u1 = lim(u0 + h * L(t, u0))
            u2 = lim(u1 + h * L(t + h, u1))
            u3 = lim(2.0 / 3.0 * u0 + (u2 + h * L(t + dt, u2)) / 3.0)
            return lim(u3 + h * L(t + h, u3))
        # Ketcheson (2008), SSPRK(10,4), low storage
        h = dt / 6.0
        q1, q2, tq = u0.copy(), u0.copy(), t
        for _ in range(5):
            q1 = lim(q1 + h * L(tq, q1))
            tq += h
        q2 = q2 / 25.0 + 9.0 / 25.0 * q1
        q1 = lim(15.0 * q2 - 5.0 * q1)               # = 3/5 u0 + 2/5 u5, a convex combination
        tq = t + dt / 3.0
        for _ in range(4):
            q1 = lim(q1 + h * L(tq, q1))
            tq += h
        return lim(q2 + 0.6 * q1 + 0.1 * dt * L(t + dt, q1))

    def step(self, u, dt=None):
        """Advance u by one step, in place, and rk.t by dt.  u is a Function, a ProductFunction or
        an array of M's space."""
        if dt is not None:
            self.dt = float(dt)
        vec = u.vector if hasattr(u, "vector") else u
        if np.size(vec) != self.V.dim:
            raise ValueError(f"SSPRK: u has {np.size(vec)} coefficients, M's space has {self.V.dim}")
        vec[:] = self._advance(np.array(vec, dtype=np.float64).reshape(-1))
        self.t += self.dt
        if self.time is not None:
            self.time.assign(self.t)
        return u

    def __repr__(self):
        lim = f", limiter={self.limiter!r}" if self.limiter is not None else ""
        return f"SSPRK({self.method}, dt={self.dt:g}, t={self.t:g}{lim}, on {self.V!r})"


class _KrylovMass:
    """M^{-1} r by a preconditioned Krylov method: the mass solve of ERK without a factorization.

    method "cg" (M symmetric positive definite, the usual mass matrix), "gmres" or "lgmres".
    precond: "jacobi" (default: a mass matrix is spectrally equivalent to its diagonal, so the
      iteration count does not grow with refinement), "ssor", "ilu0", None, or anything fd.cg /
      fd.gmres take as M=.  tol: the relative residual of every solve (default 1e-12, well below
      the time error).  warm_start: start each solve from the last solution, which is the previous
      stage's rate and differs from this one by O(dt).  The solves run in C++ (CSR kernel and
      preconditioner); only the call per stage passes through Python."""

    def __init__(self, M, method, precond="jacobi", tol=1e-12, maxiter=None, restart=30, k_aug=2,
                 warm_start=True):
        from .sparse import SparseMatrix
        from . import Matrix
        if isinstance(M, SparseMatrix):
            A = M
        elif isinstance(M, Matrix):
            A = SparseMatrix.from_scipy(M.tocsr(), symmetric=None)
        else:
            raise TypeError(f"ERK: a Krylov backend needs M as a Matrix or SparseMatrix, got {type(M).__name__}")
        if A.shape[0] != A.shape[1]:
            raise ValueError("ERK: M must be square")
        self.A, self.backend = A, method
        if isinstance(precond, str):
            self.precond = None if precond.lower() == "none" else A.preconditioner(precond)
        else:
            self.precond = precond
        self.tol, self.maxiter, self.restart, self.k_aug = float(tol), maxiter, int(restart), int(k_aug)
        self.warm_start = bool(warm_start)
        self.iterations, self.info, self._x = [], None, None

    @property
    def size(self):
        return self.A.shape[0]

    def reset(self):
        """Forget the warm start (the next solve starts from zero)."""
        self._x = None

    def __call__(self, r):
        from . import krylov
        r = np.ascontiguousarray(r, dtype=np.float64)
        x0 = self._x if self.warm_start and self._x is not None else None
        K = self.A._K
        if self.backend == "cg":
            x, info = krylov.cg(K, r, M=self.precond, x0=x0, tol=self.tol, maxiter=self.maxiter, warn=False)
        elif self.backend == "gmres":
            x, info = krylov.gmres(K, r, M=self.precond, x0=x0, tol=self.tol, maxiter=self.maxiter,
                                   restart=self.restart, warn=False)
        else:
            x, info = krylov.lgmres(K, r, M=self.precond, x0=x0, tol=self.tol, maxiter=self.maxiter,
                                    restart=self.restart, k_aug=self.k_aug, warn=False)
        self.info = info
        self.iterations.append(info.iterations)
        if not info.converged:
            raise RuntimeError(f"ERK: the {self.backend} mass solve did not converge ({info}); raise maxiter, "
                               "loosen tol, or change the preconditioner in solver_options")
        x = np.ascontiguousarray(x, dtype=np.float64)
        self._x = x.copy()
        return x

    solve = __call__

    def __repr__(self):
        its = f", {sum(self.iterations)} iterations in {len(self.iterations)} solves" if self.iterations else ""
        pk = getattr(self.precond, "kind", None if self.precond is None else type(self.precond).__name__)
        return f"KrylovMassSolver({self.backend}, precond={pk}, tol={self.tol:g}{its})"


_KRYLOV = ("cg", "gmres", "lgmres")


class ERK:
    """An explicit Runge-Kutta stepper for non-stiff systems  M u' = f(t, u), the stage loop in C++.

        rk = fd.ERK(M, f, dt, "rk4")                        # the classical fourth-order method
        for n in range(nsteps):
            rk.step(u)                                      # u: a Function or ProductFunction, in place

    M: the mass Matrix, SparseMatrix (2D) or rank-2 Form, of a space or a ProductSpace, factored
      once.  A 1D Matrix is applied in C++, and so is a SparseMatrix factored by FEMd's sparse
      Cholesky (the default for a symmetric positive definite one); otherwise SciPy's SuperLU.  M=None with
      a callable rhs integrates the plain ODE u' = f(t, u) (M = I).
    rhs: f, exactly as in IRK: one term or a list of terms, each a Matrix, a rank-1 Form in the
      unknown, or a rank-2 Form, optionally (coefficient, term); or a callable U -> f(U) or
      (t, U) -> f(t, U).  No Jacobian is needed.
    method: "rk4" (default, order 4), "3/8" (order 4), "rk3" (order 3), "midpoint" or "rk2",
      "heun", "ralston" (order 2), "euler" (order 1), or an explicit tableau (A, b) or (A, b, c)
      with A strictly lower triangular.  fd.butcher(name) returns the named tableaux.
    time: an fd.Constant in the forms, set to t + c_i dt before each stage; t0 the initial time.
    unknown: which Function of the Forms is the unknown, when they have more than one.
    backend: the solver for M.  A direct one (the default "Auto": banded in 1D, sparse Cholesky or
      SuperLU in 2D, factored once), or a Krylov method, "cg", "gmres" or "lgmres", with no
      factorization at all: each stage solves M k = f to a relative residual `tol`, preconditioned
      and warm-started from the previous stage (_KrylovMass), in 1D as in 2D.
    solver_options: for a Krylov backend, a dict with precond ("jacobi" by default, "ssor",
      "ilu0", None, or a preconditioner object), tol (1e-12), maxiter, restart and k_aug (GMRES,
      LGMRES) and warm_start (True).  rk.solver.iterations lists the iterations of every solve.

    An explicit method is stable only for dt below a limit set by the fastest mode of M^{-1} f
    (a CFL condition for hyperbolic problems, dt ~ h^2 for diffusion).  For stiff problems use
    IRK (Radau IIA), and SSPRK when a limiter or a strong-stability property is needed."""

    def __init__(self, M, rhs, dt, method="rk4", *, unknown=None, time=None, t0=0.0, backend="Auto",
                 solver_options=None):
        from . import _femd as _C
        from . import Matrix, Form
        from .sparse import SparseMatrix
        if isinstance(M, Form):
            if M.rank != 2:
                raise TypeError("ERK: M must be a rank-2 Form, a Matrix or a SparseMatrix")
            M = M.assemble()
        bname = str(getattr(backend, "name", backend)).lower()
        if solver_options is not None and bname not in _KRYLOV:
            raise TypeError(f"ERK: solver_options go with a Krylov backend ({', '.join(_KRYLOV)}), not {backend!r}")
        if bname in _KRYLOV:
            if not isinstance(M, (Matrix, SparseMatrix)) or M.space is None:
                raise TypeError("ERK: a Krylov backend needs M as a Matrix, SparseMatrix or rank-2 Form of a space")
            opts = dict(solver_options or {})
            unknown_opts = set(opts) - {"precond", "tol", "maxiter", "restart", "k_aug", "warm_start"}
            if unknown_opts:
                raise TypeError(f"ERK: unknown solver_options {sorted(unknown_opts)}; they are precond, tol, maxiter, "
                                "restart, k_aug, warm_start")
            self._solver = _KrylovMass(M, bname, **opts)
            self.V, self._Minv = M.space, self._solver
        elif M is None:
            if not (callable(rhs) and not isinstance(rhs, (Form, list, tuple))):
                raise TypeError("ERK: M=None (u' = f(t, u)) needs a callable right-hand side")
            self.V, self._Minv, self._solver = None, None, None
        elif isinstance(M, Matrix):
            if M.space is None:
                raise TypeError("ERK: M must be a Matrix of a space or a ProductSpace (from a form or V.cache())")
            self._solver = M.solver(backend)
            self.V, self._Minv = M.space, self._solver._impl          # the factored solve runs in C++
        elif isinstance(M, SparseMatrix):
            if M.space is None:
                raise TypeError("ERK: M must be a SparseMatrix of a space (from a form)")
            self._solver = M.solver("auto" if getattr(backend, "name", backend) in ("Auto", "auto") else backend)
            self.V = M.space
            S = self._solver
            if S.backend in ("cholesky", "ldlt"):
                self._Minv = S.factor                        # the stage loop solves in C++
            else:
                self._Minv = lambda r: np.asarray(S._solve_array(np.asarray(r)), dtype=np.float64)
        else:
            raise TypeError(f"ERK: M must be a Matrix, a SparseMatrix, a rank-2 Form or None, got {type(M).__name__}")
        if isinstance(method, str):
            self._rk = _C.ExplicitRK(method)
        else:
            A, b = np.asarray(method[0], dtype=np.float64), np.asarray(method[1], dtype=np.float64).ravel()
            s_ = b.size
            if A.shape != (s_, s_):
                raise ValueError(f"ERK: A must be {s_} x {s_} for {s_} weights, got {A.shape}")
            c = np.asarray(method[2], dtype=np.float64).ravel() if len(method) > 2 else A.sum(axis=1)
            self._rk = _C.ExplicitRK(np.ascontiguousarray(A.ravel()), np.ascontiguousarray(b), np.ascontiguousarray(c), 0)
        self.method, self.order, self.stages = self._rk.name, self._rk.order, self._rk.stages
        self.dt = float(dt)
        if self.V is None:
            ft = _arity(rhs) == 2
            self._f = (lambda t, U: np.asarray(rhs(t, U), dtype=np.float64)) if ft else \
                (lambda t, U: np.asarray(rhs(U), dtype=np.float64))
            if time is not None:
                raise TypeError("ERK: time= sets a Constant in forms; a callable f(t, U) receives t itself")
        else:
            self.rhs = _Rhs(rhs, None, unknown, self.V, time, False, need_jacobian=False)
            self._f = lambda t, U: np.ascontiguousarray(self.rhs.f(t, U), dtype=np.float64)
        self.time, self.t = time, float(t0)
        if time is not None:
            time.assign(t0)

    @property
    def solver(self):
        """The solver of the mass matrix M: a SparseSolver (2D; .backend, and .factor for the sparse
        Cholesky: nnz_L, min_pivot, ...), a LinearSolver (1D), or with a Krylov backend a
        _KrylovMass (.backend, .iterations per solve, .info, .reset()).  None when M = I."""
        return self._solver

    def rate(self, t, u):
        """u' = M^{-1} f(t, u)."""
        r = self._f(t, np.ascontiguousarray(getattr(u, "vector", u), dtype=np.float64))
        if self._Minv is None:
            return r
        if callable(self._Minv):
            return self._Minv(r)
        return np.asarray(self._Minv.solve(r), dtype=np.float64)

    def step(self, u, dt=None):
        """Advance u by one step, in place, and rk.t by dt.  u is a Function, a ProductFunction or a
        float64 array (of M's space).  Returns u."""
        if dt is not None:
            self.dt = float(dt)
        vec = u.vector if hasattr(u, "vector") else u
        if not (isinstance(vec, np.ndarray) and vec.dtype == np.float64 and vec.ndim == 1 and vec.flags.c_contiguous):
            raise TypeError("ERK.step: u must be a Function, a ProductFunction or a contiguous float64 array")
        if self.V is not None and vec.size != self.V.dim:
            raise ValueError(f"ERK: u has {vec.size} coefficients, M's space has {self.V.dim}")
        self._rk.step(vec, self.t, self.dt, self._f, self._Minv)
        self.t += self.dt
        if self.time is not None:
            self.time.assign(self.t)
        return u

    def __repr__(self):
        on = f" on {self.V!r}" if self.V is not None else ""
        return f"ERK({self.method}, order {self.order}, {self.stages} stages, dt={self.dt:g}, t={self.t:g}{on})"
