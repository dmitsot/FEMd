# Nonlinear solvers and time stepping

[Manual](../manual.md) · [API reference](index.md)

## G. Nonlinear solvers and time stepping

| name | kind | purpose |
|---|---|---|
| `newton(R, u, ...)` | function | Newton's method for a residual form, with exact Jacobian and boundary data |
| `NewtonInfo` | class | the convergence report of `newton`, `newton_system` and `IRK.step` |
| `newton_system(F, J, x0, ...)` | function | Newton's method for any algebraic system $F(x) = 0$ |
| `residual_function(R, u, ...)` | function | a residual form as a function of the free coefficients, for `csnewton` and other array solvers |
| `csnewton(F, x0, ...)` | function | complex-step Newton-Krylov on a NumPy function |
| `CSNewtonInfo` | class | the convergence report of `csnewton` |
| `IRK` | class | implicit Runge-Kutta stepper for $M u' = f(t, u)$, with time-dependent boundary data |
| `butcher(method, stages)` | function | the tableau of a Gauss-Legendre, Radau IIA or named explicit method |
| `SSPRK` | class | explicit SSP Runge-Kutta stepper for $M u' = f(t, u)$, with an optional limiter |
| `VertexLimiter` | class | the vertex-based slope limiter for DG on triangles |
| `ERK` | class | classical explicit Runge-Kutta stepper (RK2, RK4, ...) for non-stiff $M u' = f(t, u)$ |
| `TVBLimiter` | class | the Cockburn-Shu TVB minmod limiter for DG spaces |
| `SlopeLimiter` | class | the minmod, Van Leer, MC and Van Albada slope limiters with the TVD2 or UNO2 reconstruction, for DG spaces in 1D |
| `limited_slope(limiter, sm, sp, reconstruction, ...)` | function | the slope of one of these limiters |
| `minmod(*args)` | function | elementwise minmod |

### G.1 `newton(R, u, left=None, right=None, *, tol=1e-10, rtol=0, maxiter=50, line_search=True, linear="direct", jv="assembled", precond="frozen", backend="Auto", krylov_tol=1e-8, restart=30, krylov_maxiter=1000, cs_h=1e-20, verbose=False, warn=True)`

Solves $R(u; v) = 0$ for every test function $v$ (manual, Sections [7.2](../manual.md#72-newtons-method) to [7.5](../manual.md#75-newton-krylov)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `R` | rank-1 `Form` or its expression | required | The residual, linear in the test function and depending on `u`. |
| `u` | `Function` or `ProductFunction` | required | The unknown, holding the initial guess and updated in place. A Function of the test space $V$, of `V.unconstrained` when `left`/`right` are given, or a `ProductFunction` of the test `ProductSpace` $W$, of `W.unconstrained` when its fields have data. |
| `left`, `right` | number, tuple or `None` | `None` | Data on the conditions built into $V$, as in `V.lift`. On a `ProductSpace`, a list with one entry per field (`None` for a field without data), as in `W.lift`. Without them (and without `dirichlet=` in 2D), the data given with the space are used, if any. |
| `tol` | float | `1e-10` | Absolute stopping tolerance on $\lVert R \rVert_2$. |
| `rtol` | float | `0` | Relative tolerance. Stops when $\lVert R \rVert \le \max(\texttt{tol}, \texttt{rtol}\,\lVert R(u_0) \rVert)$. |
| `maxiter` | int | `50` | Newton iterations. |
| `line_search` | bool | `True` | Armijo backtracking by halving. `False` takes full steps. |
| `linear` | str | `"direct"` | `"direct"` (banded or cyclic factorization), `"gmres"` or `"lgmres"`. |
| `jv` | str | `"assembled"` | Jacobian-vector product for the Krylov modes: `"assembled"`, `"exact"` (matrix-free action of `R.derivative(u)`), `"complex-step"` ($J v = \mathrm{Im}\, R(u + i h v)/h$, one complex residual per product, exact to rounding) or `"fd"`. Must be `"assembled"` with `linear="direct"`. `linear="gmres"`, `jv="complex-step"` and `precond=None` together are the complex-step Newton-Krylov method, with no Jacobian matrix formed. |
| `precond` | str, `None`, `LinearSolver`, `Matrix` or callable | `"frozen"` | Left preconditioner for the Krylov modes: `"frozen"` (Jacobian at the first iterate, refactored when the iteration count grows), `"linear"` (the constant-coefficient part, factored once), `"jacobian"` (every step), `None`, or an operator. `"frozen"`, `"linear"` and `"jacobian"` assemble (part of) the Jacobian; `None` does not. |
| `backend` | `Solver` or name | `"Auto"` | Direct solver backend, for the Newton systems and the preconditioner. |
| `krylov_tol` | float | `1e-8` | Preconditioned relative tolerance of each Krylov solve. |
| `restart` | int | `30` | Krylov restart length. |
| `krylov_maxiter` | int | `1000` | Iteration limit of each Krylov solve. |
| `cs_h` | float | `1e-20` | Step $h$ of the complex-step product. |
| `verbose` | bool | `False` | Print $\lVert R \rVert$ at each iteration. |
| `warn` | bool | `True` | Issue a `RuntimeWarning` when not converged. |

**Outputs**

| output | type | meaning |
|---|---|---|
| return value | `NewtonInfo` | The convergence report. |
| `u` | updated in place | The solution, or the last good iterate when the iteration stops early. |

```python
info = fd.newton(R, u)                                        # exact Jacobian, direct banded solve
info = fd.newton(R, u, linear="gmres", jv="complex-step",     # complex-step Newton-Krylov,
                 precond=None)                                # no Jacobian formed
```

### G.1a `residual_function(R, u, left=None, right=None, *, dirichlet=None)`

The residual form as a function of the free coefficients, for solvers that work on plain arrays
(`csnewton`, `scipy.optimize.root`, a Newton loop of your own). The arguments are those of `newton`,
and the boundary data come from the same places.

| output | type | meaning |
|---|---|---|
| `F` | callable | $c \mapsto R(u_g + P c)$, a vector of length `V.dim`. $u_g$ is the lift of the boundary data and $P$ = `V.prolongate` (for `u` in $V$ itself, $u = c$). Accepts complex $c$, as the complex step needs, and leaves `u` unchanged. |
| `c0` | array, length `V.dim` | The current `u` as free coefficients: the initial guess. |
| `set_u` | callable | `set_u(c)` writes the field $u_g + P c$ into `u`. |

As `newton`, it first gives `u` the boundary data. **Raises** `TypeError` for a form that is not
rank 1, and `ValueError` when the form does not read `u`.

```python
F, c0, set_u = fd.residual_function(R, u)
c, info = fd.csnewton(F, c0)
set_u(c)
```

### G.2 `NewtonInfo`

| attribute | type | meaning |
|---|---|---|
| `converged` | bool | Whether the stopping test was met. |
| `iterations` | int | Newton iterations taken. |
| `residuals` | list of float | $\lVert R \rVert$ at every iterate, the first one initial. |
| `residual` | float | The last entry of `residuals`. |
| `steps` | list of float | The line-search step length of each iteration. |
| `linear_iterations` | list of int | Krylov iterations per Newton step, 0 for direct solves. |
| `message` | str | Why the iteration stopped early, or how it converged, empty otherwise. |

### G.3 `newton_system(F, J, x0, *, tol=1e-10, rtol=0, xtol=0, maxiter=50, line_search=True, backend="Auto", verbose=False, warn=True)`

Newton's method for $F(x) = 0$, with the same line search and stopping test as
`newton` ([manual, Section 7.6](../manual.md#76-implicit-runge-kutta-time-stepping-and-algebraic-systems)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `F` | callable $x \mapsto F(x)$ | required | The residual, an array of the length of `x`. |
| `J` | callable $x \mapsto J(x)$ | required | The Jacobian at `x`, as a `Matrix` (factored with `backend`), a `LinearSolver` (used as it is, so returning the same one gives simplified Newton), a SciPy sparse matrix, or a dense array. |
| `x0` | array_like | required | Initial guess. Not modified. |
| `tol`, `rtol` | float | `1e-10`, `0` | Stop when $\lVert F \rVert \le \max(\texttt{tol}, \texttt{rtol}\,\lVert F(x_0) \rVert)$. |
| `xtol` | float | `0` | When positive, also stop when the error left after a step is estimated below $\texttt{xtol}\,\lVert x \rVert$ from the contraction rate, or when $\lVert F \rVert$ stalls at its round-off floor while the step is tiny. |
| `maxiter` | int | `50` | Newton iterations. |
| `line_search` | bool | `True` | Armijo backtracking. |
| `backend` | `Solver` or name | `"Auto"` | Backend for a `Matrix` Jacobian. |
| `verbose`, `warn` | bool | `False`, `True` | Print each iteration, warn when not converged. |

**Outputs**

| output | type | meaning |
|---|---|---|
| `x` | array | The solution, or the last good iterate. |
| `info` | `NewtonInfo` | The convergence report. |

### G.4 `csnewton(F, x0, M=None, *, tol=1e-10, maxiter=50, h=1e-20, restart=20, krylov_maxiter=200, krylov_tol=1e-6, method="gmres", k_aug=2, warn=True)`

The vendored complex-step Newton-Krylov solver. Each step solves $J s = F$ by
GMRES or LGMRES with $J v = \mathrm{Im}\, F(x + ihv)/h$. For a residual form
use `newton`.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `F` | callable | required | $x \mapsto F(x)$, accepting real and complex arrays and analytic in $x$ (NumPy arithmetic and functions, no `abs`, `max` or comparisons). |
| `x0` | array_like | required | Initial guess. |
| `M` | as in `gmres` | `None` | Left preconditioner for $J$. |
| `tol` | float | `1e-10` | Stopping tolerance on $\lVert F \rVert_2$. |
| `maxiter` | int | `50` | Newton iterations. |
| `h` | float | `1e-20` | Complex step. |
| `restart` | int | `20` | Krylov restart length. |
| `krylov_maxiter` | int | `200` | Iteration limit of each Krylov solve. |
| `krylov_tol` | float | `1e-6` | Relative tolerance of each Krylov solve. |
| `method` | str | `"gmres"` | `"gmres"` or `"lgmres"`. |
| `k_aug` | int | `2` | LGMRES augmentation vectors. |
| `warn` | bool | `True` | Warn when not converged. |

**Outputs**

| output | type | meaning |
|---|---|---|
| `x` | array | The solution. |
| `info` | `CSNewtonInfo` | The convergence report. |

### G.5 `CSNewtonInfo`

| attribute | type | meaning |
|---|---|---|
| `converged` | bool | Whether `tol` was met. |
| `newton_iterations` | int | Newton steps taken. |
| `krylov_iterations` | int | Krylov iterations over all steps. |
| `residual` | float | $\lVert F(x) \rVert_2$ at the end. |

### G.6 `IRK(M, rhs, dt, method="gauss", stages=2, *, jacobian=None, unknown=None, time=None, t0=0.0, left=None, right=None, dirichlet=None, newton="exact", tol=1e-12, rtol=0, xtol=1e-12, maxiter=20, line_search=False, backend="Auto")`

Implicit Runge-Kutta stepper for $M u' = f(t, u)$, with optional time-dependent
data on the built-in boundary conditions (manual, Sections [7.6](../manual.md#76-implicit-runge-kutta-time-stepping-and-algebraic-systems) and [7.7](../manual.md#77-tolerances-round-off-and-conservation)). On a 2D mesh see [L.11](elements-2d.md#l11-irk-on-2d-meshes)
(`dirichlet=` and the Newton modes there).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `M` | `Matrix` or rank-2 `Form` | required | The mass matrix, on a space or a `ProductSpace` $V$. Must be a `Form` when `left` or `right` is given. |
| `rhs` | `Form`, `Matrix`, list, or callable | required | $f$. Usually one rank-1 `Form` in a Function of $V$, evaluated with that Function replaced by the stage value. Or a list of terms, each a rank-1 `Form`, a `Matrix` or a rank-2 `Form`, optionally weighted as `(coefficient, term)`, so `[A, (-1.0, Nf)]` is $AU - \mathcal N(U)$. A rank-2 `Form` that contains a Function or a Constant is re-assembled at every evaluation, one that does not is assembled once. Or a callable $U \mapsto f(U)$ or $(t, U) \mapsto f(t, U)$ with `jacobian`. With boundary data, forms only, in a Function of `V.unconstrained`. |
| `dt` | float | required | Time step. |
| `method` | str or `(A, b)` | `"gauss"` | `"gauss"` (Gauss-Legendre, order $2s$) or `"radau"` (Radau IIA, order $2s-1$), or a custom tableau `(A, b)` with square `A`. |
| `stages` | int | `2` | $s$ = 1, 2 or 3 for the built-in methods. Ignored with a custom tableau. |
| `jacobian` | callable or `None` | `None` | $U \mapsto f'(U)$ or $(t, U) \mapsto f'(t, U)$ as a `Matrix`, required with a callable `rhs` and refused otherwise. |
| `unknown` | Function or str | `None` | Which Function of the forms is the state, when they contain more than one. |
| `time` | `Constant` or `None` | `None` | The Constant that stands for $t$ in the forms of `rhs`. Set to the stage time before each evaluation of $f$ and $f'$, and to the new time after each step. |
| `t0` | float | `0.0` | The initial time. |
| `left`, `right` | number, tuple, callable of $t$, or `None` | `None` | Data on the conditions built into $V$ at each end, one value per built-in functional. On a `ProductSpace`, a list with one such entry per field (`None` for none). The state is then a Function of `V.unconstrained` carrying the data, and the stages solve $M c' = f(t, u_g + Pc) - M u_g'$ for the coefficients in $V$. The derivative of callable data is taken by the complex step, or by a central difference when the callable refuses a complex $t$. Without them, the data given with the space are used, if any. |
| `newton` | str | `"exact"` | `"exact"` re-evaluates the stage Jacobian at every iteration. `"simplified"` factors it once per step at $u^n$. |
| `tol`, `rtol` | float | `1e-12`, `0` | Absolute and relative tolerances on the stage residual. |
| `xtol` | float | `1e-12` | Step-based stopping test of `newton_system`, which ends the iteration at the round-off floor. `0` switches it off. |
| `maxiter` | int | `20` | Newton iterations per step. |
| `line_search` | bool | `False` | Armijo backtracking in the stage solve. |
| `backend` | `Solver` or name | `"Auto"` | Backend for the stage Jacobian. |

**Output.** An `IRK` object.

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `step(u, dt=None)` | `u` a Function, `ProductFunction` or array of $V$, or of `V.unconstrained` with boundary data, updated in place. `dt` float, a new step size from now on, `None` keeps the current one. | `NewtonInfo` | Advances one step and `t` by `dt`. Raises `RuntimeError` when Newton fails, with the reason. |
| `t` | attribute | float | The current time, starting at `t0`. |
| `times()` | none | array, length $s$ | The stage times $t + c_i \Delta t$ of the next step. |
| `dt` | attribute | float | Current step. |
| `history` | attribute | list of int | Newton iterations of each step. |
| `A`, `b`, `c` | attribute | arrays | The Butcher tableau. |
| `s` | attribute | int | Number of stages. |
| `M`, `V`, `P` | attribute | | The mass matrix, its space, and the `ProductSpace` of the stages, every field of $V$ once per stage. |
| `K` | attribute | array or `None` | The stage derivatives of the last step, in the numbering of `P`. |
| `stages(cn, K)` | `cn` array, the coefficients of $u^n$. `K` array in the numbering of `P`. | `(Ks, Us)` | The per-stage slopes $K_i$ and stage values $U_i = u^n + \Delta t \sum_j a_{ij} K_j$, two lists of arrays. |
| `residual(cn, K)` | as `stages` | array, length `P.dim` | The stage residual $M K_i - F(t_i, U_i)$, gathered, with $F = f$ or, with boundary data, $F(t, c) = f(t, u_g + Pc) - M u_g'$. |
| `jacobian(Us, ts=None)` | `Us` list of $s$ stage values. `ts` their times, default `times()`. | `Matrix` on `P` | $\delta_{ij} M - \Delta t\, a_{ij} F'(t_i, U_i)$, assembled with `block`. |

### G.7 `butcher(method="gauss", stages=2)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `method` | str | `"gauss"` | `"gauss"` (also `"gauss-legendre"`, `"gl"`) or `"radau"` (also `"radau iia"`, `"radau-iia"`), or an explicit method of `ERK` (`"rk4"`, `"midpoint"`, ...). |
| `stages` | int | `2` | 1, 2 or 3. Ignored for an explicit method. |

**Outputs**

| output | type | meaning |
|---|---|---|
| `A` | array $s \times s$ | Stage coefficients $a_{ij}$. |
| `b` | array, length $s$ | Weights. |
| `c` | array, length $s$ | Nodes, the row sums of `A`. |

### G.8 `SSPRK(M, rhs, dt, order=3, stages=None, *, limiter=None, unknown=None, time=None, t0=0.0, backend="Auto")`

Explicit strong-stability-preserving Runge-Kutta stepper for $M u' = f(t, u)$
([manual, Section 7.8](../manual.md#78-explicit-ssp-runge-kutta-and-limiting)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `M` | `Matrix`, `SparseMatrix` or rank-2 `Form` | required | The mass matrix, on a space or a `ProductSpace` (1D or 2D). Factored once. |
| `rhs` | `Form`, `Matrix`, list, or callable | required | $f$, as for `IRK`, with no Jacobian needed. A callable is $U \mapsto f(U)$ or $(t, U) \mapsto f(t, U)$. |
| `dt` | float | required | Time step. |
| `order` | int | `3` | 1, 2, 3 or 4. |
| `stages` | int or `None` | `None` | `None` picks 1, 2, 3 and 10 stages for orders 1 to 4. `(stages, order) = (4, 3)` gives the four-stage third-order method. |
| `limiter` | callable or `None` | `None` | Applied to every stage and to the result, array in and array out, such as a `TVBLimiter` (1D) or a `VertexLimiter` (triangles). |
| `unknown` | Function or str | `None` | Which Function of the forms is the state, when they contain more than one. |
| `time`, `t0` | `Constant`, float | `None`, `0.0` | As for `IRK`. The Constant is set to each stage's time. |
| `backend` | `Solver` or name | `"Auto"` | Backend for `M`. |

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `step(u, dt=None)` | `u` a Function, `ProductFunction` or array of $V$, updated in place. `dt` a new step from now on. | `u` | Advances one step and `t` by `dt`. |
| `rate(t, u)` | `t` float, `u` array | array | $M^{-1} f(t, u)$, the forward Euler direction. |
| `t`, `dt` | attribute | float | Current time and step. |
| `cfl` | attribute | float | The SSP coefficient: the method keeps a forward Euler property for $\Delta t \le$ `cfl` $\Delta t_{FE}$. |
| `method`, `order`, `stages` | attribute | str, int, int | The method. |
| `M`, `V`, `limiter` | attribute | | The mass matrix, its space and the limiter. |
| `solver` | attribute | solver | The factored solver of $M$, as for `ERK`. |

### G.9 `TVBLimiter(V, M=0.0, fields=None)`

The Cockburn-Shu TVB minmod limiter ([manual, Section 7.8](../manual.md#78-explicit-ssp-runge-kutta-and-limiting)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `V` | `DGSpace` or `ProductSpace` | required | A DG space of any degree and either basis, or a `ProductSpace` whose DG fields are limited component by component. |
| `M` | float | `0.0` | TVB constant. Elements whose end jumps are below $M h^2$ are left alone. `0` is plain minmod. |
| `fields` | list of int or `None` | `None` | On a `ProductSpace`, the fields to limit. Default every DG field. |

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `lim(u)` | `u` a Function or `ProductFunction` (limited in place), or an array | `u`, or a limited copy of the array | Limits every troubled element: mean kept, slope minmodded, higher modes dropped. |
| `limit(c)` | `c` array, length `V.dim` | array | A limited copy. |
| `troubled` | attribute | int | Elements limited by the last call. |

`minmod(a, b, ...)` returns, elementwise, the argument of smallest magnitude when
all have one sign, and 0 otherwise.

### G.9b `SlopeLimiter(V, limiter="minmod", reconstruction="tvd2", M=0.0, fields=None, troubled="all")`

The limiters minmod, Van Leer, monotonized central and Van Albada, each with the TVD2 or the UNO2 reconstruction, as in Dutykh, Katsaounis and Mitsotakis, J. Comput. Phys. 230 (2011) ([manual, Section 7.8](../manual.md#78-explicit-ssp-runge-kutta-and-limiting)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `V` | `DGSpace` or `ProductSpace` | required | A DG space of degree $p \ge 1$ and either basis, or a `ProductSpace` whose DG fields are limited one by one. |
| `limiter` | str | `"minmod"` | `"minmod"` (`"mm"`), `"vanleer"` (`"vl"`), `"mc"` or `"vanalbada"` (`"va"`). |
| `reconstruction` | str | `"tvd2"` | `"tvd2"`, the limiter applied to the differences of the means, or `"uno2"`, applied to the differences corrected by the second differences. |
| `M` | float | `0.0` | The TVB constant of the troubled-element test, used with `troubled="tvb"`. |
| `fields` | list of int or `None` | `None` | On a `ProductSpace`, the fields to limit. Default every DG field. |
| `troubled` | str | `"all"` | `"all"` replaces the slope of every element. `"tvb"` only that of the elements flagged by the Cockburn-Shu test of `TVBLimiter`. |

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `lim(u)` | `u` a Function or `ProductFunction` (limited in place), or an array | `u`, or a limited copy of the array | On every chosen element the mean is kept, the slope becomes the limited slope $S_j$ of the means and the higher modes are dropped. |
| `limit(c)` | `c` array, length `V.dim` | array | A limited copy. |
| `troubled` | attribute | int | Elements changed by the last call. |
| `limiter`, `reconstruction` | attribute | str | The choices. |

`limited_slope(limiter, sm, sp, reconstruction="tvd2", Dm=None, D0=None, Dp=None, dm=None, dp=None)`
returns, elementwise, the limited slope of a cell from its left and right divided differences `sm`, `sp`.
UNO2 also takes the second divided differences of the cells $j-1$, $j$, $j+1$ and the distances
`dm`, `dp` from the center of the cell to its neighbors' centers.

### G.9a `VertexLimiter(V, fields=None)`

The vertex-based limiter of Kuzmin (2010) on triangles ([manual, Section 12.9](../manual.md#129-discontinuous-galerkin-on-triangles)).

| argument | type | default | meaning |
|---|---|---|---|
| `V` | `DGSpace2D` or 2D `ProductSpace` | required | A broken $P_k$ space, or a system whose DG fields are limited one by one. |
| `fields` | list of int or `None` | `None` | On a `ProductSpace`, the fields to limit. Default every `DGSpace2D` field. |

The linear part ($L^2$ projection onto $P_1$) of each triangle is scaled about its mean by the largest
$\alpha \le 1$ that keeps the vertex values within the means of the triangles around each vertex
(joined across periodic sides). For $k \ge 2$ a triangle with $\alpha < 1$ becomes its limited linear
part. Members as for `TVBLimiter`: `lim(u)` (in place), `limit(c)` (a copy), `troubled`, and
`alpha`, the factors of the last call.

### G.10 `ERK(M, rhs, dt, method="rk4", *, unknown=None, time=None, t0=0.0, backend="Auto", solver_options=None)`

Explicit Runge-Kutta stepper for $M u' = f(t, u)$ ([manual, Section 7.9](../manual.md#79-classical-explicit-runge-kutta)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `M` | `Matrix`, `SparseMatrix`, rank-2 `Form` or `None` | required | The mass matrix, factored once. `None` with a callable `rhs`: $M = I$. |
| `rhs` | as in `IRK` | required | $f$: a term or a list of terms (Matrix, SparseMatrix, rank-1 Form in the unknown, rank-2 Form, optionally `(c, T)`), or a callable `f(U)` / `f(t, U)`. |
| `dt` | float | required | Step size. |
| `method` | str or tuple | `"rk4"` | `"euler"`, `"midpoint"` (`"rk2"`), `"heun"`, `"ralston"`, `"rk3"`, `"rk4"`, `"3/8"`, or an explicit tableau `(A, b)` / `(A, b, c)`. |
| `unknown` | Function or name | `None` | The unknown, when the forms contain several Functions. |
| `time` | `Constant` | `None` | Set to the stage time before each evaluation of $f$, and to $t_{n+1}$ after the step. |
| `t0` | float | `0.0` | Initial time. |
| `backend` | solver backend | `"Auto"` | For `M`: a direct backend, factored once (1D `Solver` names, 2D `SparseSolver` backends), or `"cg"`, `"gmres"`, `"lgmres"`, an iterative solve at every stage with no factorization ([manual, Section 7.9](../manual.md#79-classical-explicit-runge-kutta)). |
| `solver_options` | dict or `None` | `None` | Krylov backends only: `precond` (`"jacobi"`, `"ssor"`, `"ilu0"`, `None`, or a preconditioner), `tol` (`1e-12`), `maxiter`, `restart` (30), `k_aug` (2), `warm_start` (`True`). |

**Raises** `ValueError` for an unknown method name or a tableau that is not
strictly lower triangular, or whose weights do not sum to 1, and `TypeError`
for an `M` that is not a matrix of a known space.

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `step(u, dt=None)` | Function, ProductFunction or float64 array | `u` | One step in place. `dt` changes the step size from here on. |
| `rate(t, u)` | time, state | array | $M^{-1} f(t, u)$. |
| `t`, `dt` | attribute | float | Current time and step size. |
| `method`, `order`, `stages` | attribute | str, int, int | The tableau (`order` is 0 for a user tableau). |
| `solver` | attribute | `SparseSolver`, `LinearSolver` or `None` | The factored solver of $M$ (`None` for $M = I$). On a 2D space `rk.solver.backend` is `"cholesky"` or `"superlu"`, and `rk.solver.factor` holds the Cholesky's `nnz_L`, `min_pivot`, `max_pivot`, `ordering`. With a Krylov backend a `KrylovMassSolver`: `backend`, `iterations` (per solve), `info` (the last `KrylovInfo`), `precond`, `tol`, `reset()` (drop the warm start). |

The C++ class is `femd::ExplicitRK` in `timestep/explicit_rk.hpp`, with
`ExplicitTableau::named(name)`. Its `step(u, t, dt, rate)` takes any callable
`rate(t, U, k)` that sets $k = M^{-1} f(t, U)$.
