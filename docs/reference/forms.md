# Form language and low-level assembly

[Manual](../manual.md) · [API reference](index.md)

## D. Form language

Weak forms are written with symbols and measures and compiled by `form`
([manual, Section 6](../manual.md#6-forms)).

| name | kind | purpose |
|---|---|---|
| `TrialFunction(V)`, `TestFunction(V)` | function | the unknown and the test function on one space |
| `TrialFunctions(P)`, `TestFunctions(P)` | function | per-field symbols on a `ProductSpace` |
| `D(e, k=1)` | function | derivative of an expression |
| `x` | symbol | the coordinate |
| `Constant(value, name=None)` | class | a number read at assembly, for time and parameters |
| `sin`, `cos`, `exp`, `log`, `tanh`, `sqrt` | function | pointwise functions of known fields and `x` |
| `dx` | measure | integral over the interval |
| `set_quadrature_degree(n)`, `quadrature_degree()` | function | one rule for every form of a code, or the inferred rules |
| `set_compiled_integrands(on=True)`, `compiled_integrands()` | function | assemble forms in C++ with the `Integrand` programs (default) or with the Python driver and NumPy |
| `ds` | measure | evaluation at the ends |
| `dS` | measure | the interior facets (vertices between elements, and the periodic seam) |
| `e('-')`, `e('+')` | restriction | the value from the left or the right element on a facet |
| `jump(e)`, `avg(e)` | function | $e^- - e^+$ and $\frac12(e^- + e^+)$ |
| `form(expr, space=None)` | function | compiles an expression into a `Form` |
| `Form` | class | a compiled form, assembled to a number, vector or matrix |

`Function` and `Functions` ([Section C](functions.md#c-functions-and-transfer-between-spaces)) are the known fields that forms read.

### D.1 Symbols

| call | inputs | returns | meaning |
|---|---|---|---|
| `TrialFunction(V)` | `V` a space | symbol | The unknown. It becomes the column index of a matrix. |
| `TestFunction(V)` | `V` a space | symbol | The test function. It becomes the row index. |
| `TrialFunctions(P)` | `P` a `ProductSpace` | tuple of symbols | One trial symbol per field. |
| `TestFunctions(P)` | `P` a `ProductSpace` | tuple of symbols | One test symbol per field. |
| `x` | none | symbol | The coordinate. Data enter forms as expressions in `x`, as Functions, as Constants or as numbers. |

### D.1a `Constant(value=0.0, name=None)`

A number in a form whose value is read at assembly ([manual, Section 6.9](../manual.md#69-constants)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `value` | float | `0.0` | The initial value. |
| `name` | str or `None` | `None` | The name used by overrides in `Form.assemble`. `None` reads it from the variable, falling back to `"c"`. |

**Output.** A `Constant`, usable in expressions wherever a number is.

| member | inputs | returns | meaning |
|---|---|---|---|
| `value` | attribute | float | The current value. |
| `name` | attribute | str | The name. |
| `assign(value)` | `value` a number or a Constant | `self` | Sets the value. |
| `float(c)` | none | float | The value. |

A form containing a Constant is not cached (`Form.constant` is false), and
`Form.constants` lists the names of its Constants. `Form.assemble(name=value)` and
`Form.jacobian(..., name=value)` override a Constant for that call.

### D.2 `D(e, k=1)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `e` | expression | required | A symbol, a Function or any expression built from them. |
| `k` | int | `1` | Derivative order. On a test or trial symbol, $k$ may be at most the continuity plus one ([manual, Section 6.5](../manual.md#65-higher-derivatives)). |

**Output.** An expression. `D` is linear and applies the product and chain
rules, so `D(w*w)` expands to `D(w)*w + w*D(w)` and `D(fd.sin(w))` to
`fd.cos(w)*D(w)`.

### D.3 Pointwise functions

`sin(e)`, `cos(e)`, `exp(e)`, `log(e)`, `sqrt(e)`, `sinh(e)`, `cosh(e)`, `tanh(e)`, `sech(e)`,
`abs(e)` (as `fd.abs`, outside `__all__` so that a star import keeps the builtin), `sign(e)`,
and `max_value(a, b)`, `min_value(a, b)`, written as $(a + b \pm |a - b|)/2$.

Each is differentiated symbolically (for `D`, `Dx`, `grad` and the Jacobians), with
$\mathrm{sech}' = -\mathrm{sech}\,\tanh$ and $|z|' = \mathrm{sign}\, z$ (so numerical
fluxes with `abs` and `max_value` have Jacobians). `sech` is evaluated as
$2e^{-|z|}/(1 + e^{-2|z|})$, which does not overflow for large arguments, so a
solitary wave such as `A * fd.sech(k * (fd.x - x0))**2` can be written directly
on a long domain.

| argument | type | meaning |
|---|---|---|
| `e` | expression or number | The argument. It may contain Functions and `x`, but not test or trial symbols. |

**Output.** An expression. Powers are written `e**n`. A test or trial symbol
inside `e` raises `TypeError`.

### D.4 Measures `dx`, `ds` and `dS`

| call | inputs | returns | meaning |
|---|---|---|---|
| `dx` | none | measure | $\int_a^b \cdot\,dx$ with the quadrature rule inferred from the polynomial degree of each term. |
| `dx(n)`, `dx(quad_degree=n)` | `n` int, Gauss points per element | measure | The same with the rule fixed. |
| `dx(scheme="lobatto")`, `dx(n, scheme="lobatto")` | `n` int $\ge 2$, Gauss-Lobatto points per element, default $p+1$ for the highest degree $p$ of the term's test and trial spaces | measure | The Gauss-Lobatto rule, exact to degree $2n-3$. On a `LagrangeSpace` with `nodes="lobatto"` of degree $n-1$, `u*v*dx(scheme="lobatto")` is exactly diagonal, the lumped mass ([manual, Section 6.7](../manual.md#67-quadrature)). On a 2D mesh the tensor rule on quadrilaterals, diagonal on $Q_k$ with `nodes="lobatto"`. Refused on triangles. |
| `ds` | none | measure | The sum of the values at both ends. |
| `ds(where)` | `where` `"left"`, `"right"` or `"both"` | measure | The value at the named end. |
| `dS` | none | measure | The sum over the interior facets: in 1D the vertices $x_1, \dots, x_{n_e-1}$, and the seam of a periodic space. On triangles every edge shared by two triangles, and the periodic edge pairs of a `DGSpace2D(..., periodic=)`, with a Gauss rule inferred as for `ds`. |

`expr * dx`, `expr * ds(...)` and `expr * dS` give integrals, and sums of integrals
give a form expression. `ds` has no meaning on a periodic space.

**Restrictions on facets** ([manual, Section 6.10](../manual.md#610-interior-facets-ds-jump-avg))

| call | inputs | returns | meaning |
|---|---|---|---|
| `e('-')`, `e('+')` | `e` an expression, a symbol or a Function | expression | `e` from the left (`'-'`) or right (`'+'`) element. Constants and `x` pass through. |
| `jump(e)` | `e` an expression or vector | expression | `e('-') - e('+')`. In 1D the jump times the outward normal of the left element. |
| `jump(e, n)` | `e` scalar or vector, `n` the facet normal | vector or scalar | UFL's `e('-') n('-') + e('+') n('+')`, or `dot(e('-'), n('-')) + dot(e('+'), n('+'))` for a vector `e`. |
| `n('-')`, `n('+')` | `n = FacetNormal()` | vector | On a 2D interior facet, the unit normal out of the `'-'` triangle, and its opposite. In a `dS` term the normal must be restricted. |
| `avg(e)` | `e` an expression | expression | `(e('-') + e('+'))/2`. |

Restrictions are allowed only in `dS` terms. There the test and trial functions
must be restricted, and so must a Function of a broken space (DG, RT, N1curl) or any derivative.
A tensor restricts componentwise (`grad(u)('-')`, `as_vector([...])('+')`). A
restriction to one side of an expression already restricted to the other raises
`ValueError`.

### D.4a `set_quadrature_degree(degree=None)` and `quadrature_degree()`

One quadrature rule for every form created afterwards, instead of a rule inferred per form.

| argument | type | default | meaning |
|---|---|---|---|
| `degree` | positive int or `None` | `None` | What `dx(degree)` would set on each measure of a form: on a 1D mesh the number of Gauss points per element (exact to degree $2n-1$), on a 2D mesh the degree of exactness on the cells, the sides and the interior facets. `None` returns to the inferred rules. |

A measure with its own `quad_degree=` keeps it, `dx(scheme="lobatto")` is not affected, and forms made before the call keep their rule. **Raises** `ValueError` for a degree that is not a positive integer. `quadrature_degree()` returns the current setting, `None` when the rules are inferred ([manual, Section 6.7](../manual.md#67-quadrature)).

### D.4b `set_compiled_integrands(on=True)` and `compiled_integrands()`

The coefficient of each term of a form (known fields and their derivatives, numbers and `Constant`s, `x`, `y`, the facet normal, sums, products, powers and the elementary functions) is compiled once into a register program, `_femd.Integrand`, and evaluated at the quadrature points in C++, in parallel, with the same bits for any number of threads. `set_compiled_integrands(False)` evaluates the expression tree with NumPy instead. Complex-valued fields and Constants (the complex-step derivative, rank 0 and 1) run the same programs in complex arithmetic with NumPy's rules (`Integrand.evaluate_complex`), the fields and kernels acting on the real and imaginary parts separately. The two agree to rounding ([manual, Section 6.2](../manual.md#62-compiling-and-assembling)). `compiled_integrands()` returns the switch. With the switch on, the assembly itself is one C++ call as well (`_femd.FormAssembler2D` and `_femd.FormAssembler1D`, built once per form by `Form._compiled_2d` and `Form._compiled_1d`; a rectangular 1D form comes out as a `RectMatrix` on the CSR store): the fields at the quadrature points, the coefficients and the kernels, without the GIL, so threads may assemble one form concurrently; the Python driver that calls the kernels term by term serves the switch off.

### D.5 `form(expr, space=None)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `expr` | `expr*dx`, `expr*ds`, or a sum of those | required | The weak form. It must be linear in the test and trial symbols. |
| `space` | space | `None` | Needed only when `expr` contains no symbol and no Function, such as `form(fd.x**2 * dx, space=V)`. |

**Output.** A `Form` of rank 2 (test and trial symbols), 1 (test only) or 0
(neither). **Raises** `TypeError` for terms of different rank, a nonlinear use
of a test or trial symbol, or a trial symbol without a test symbol. Compiling may warn when a built-in derivative
condition is natural for the operator ([manual, Section 6.8](../manual.md#68-the-strong-versus-natural-warning)).

### D.6 `Form`

**Attributes**

| attribute | type | meaning |
|---|---|---|
| `rank` | int | 0, 1 or 2. |
| `functions` | list of str | Names of the known fields the form reads. |
| `constants` | list of str | Names of the Constants the form reads. |
| `constant` | bool | True when there are no known fields and no Constants. The result is then assembled once and cached. |
| `quad_degree` | int | Gauss points per element, inferred or set by `dx(n)`. |
| `test_space`, `trial_space` | space or `None` | The row and column spaces. |

**Methods**

| method | inputs | returns | meaning |
|---|---|---|---|
| `assemble(**overrides)` | `name=value` per known field or Constant to replace for this call, `value` an array or a Function for a field and a number for a Constant. A complex array gives a complex result (rank 0 and 1). | float (rank 0), array (rank 1), `Matrix` or `RectMatrix` (rank 2) | Evaluates the form with the current coefficients of its Functions. A rank-2 form with test and trial symbols on different spaces gives a `RectMatrix`. |
| `action(w)` | `w` a `Function` of any space on the interval, or a `ProductFunction` of the form's `ProductSpace` or of its `unconstrained` companion | rank-1 `Form` | $a(w, v)$, the trial symbol replaced by `w` (field by field on a `ProductSpace`). Rank-2 forms only. |
| `derivative(wrt, space=None, part="all")` | `wrt` a Function, a `ProductFunction` or a field name. `space` the trial space of the result, `V` when `wrt` lives in `V.unconstrained` (for a `ProductFunction` of `W.unconstrained`, `W`). `part` `"all"` or `"constant"` (only the terms whose coefficient contains no Function). | rank-2 `Form` | The exact Jacobian $\partial R / \partial w$ of a rank-1 form, not yet assembled. It keeps the quadrature rule of the residual, so its `assemble()` is `jacobian(wrt, space)`. |
| `jacobian(wrt, space=None, **overrides)` | as `derivative`, plus overrides as in `assemble` | `Matrix` or `RectMatrix` | `derivative(wrt, space).assemble(**overrides)`, flagged nonsymmetric. |

`assemble` raises `KeyError` for an override that names no field of the form.
`derivative` and `jacobian` raise `TypeError` on a form that is not rank 1 and
`ValueError` when the form does not depend on `wrt`.

---

## H. Low-level assembly

The three kernels under the form language ([manual, Section 9](../manual.md#9-low-level-assembly)). A term names
derivative orders and a coefficient, which is a number or an array of values at
the cache nodes, `cache.nodes()`.

| name | kind | purpose |
|---|---|---|
| `QuadratureCache` | class | basis tables at Gauss points |
| `assemble(cache, terms, space=None)` | function | rank 2, a `Matrix` |
| `assemble_vector(cache, terms, n)` | function | rank 1, a vector |
| `assemble_block_vector(P, terms)` | function | rank 1 on a `ProductSpace` |
| `assemble_scalar(cache, coeff)` | function | rank 0, a number |

### H.1 `QuadratureCache`

Built with `V.cache(npts=None, nder=1, rule="gauss")` ([A.3](spaces.md#a3-members-common-to-splinespace-and-lagrangespace)) or directly as
`QuadratureCache(space, npts, nder, lobatto=False)`. Only a cache made with `V.cache` lets
`assemble` find its space without `space=`.

| member | inputs | returns | meaning |
|---|---|---|---|
| `nelem` | attribute | int | Elements. |
| `nq` | attribute | int | Quadrature points per element. |
| `lobatto` | attribute | bool | Whether the rule is Gauss-Lobatto rather than Gauss-Legendre. |
| `nder` | attribute | int | Highest derivative tabulated. |
| `nodes()` | none | array, length `nelem*nq` | Quadrature nodes, `[e*nq + q]`. |
| `weights()` | none | array, length `nelem*nq` | Quadrature weights, including the element Jacobian. |

### H.2 `assemble(cache, terms, space=None)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `cache` | `QuadratureCache` | required | The tables. |
| `terms` | list | required | Tuples `(a, b, coeff)`, each adding $\int c\,(D^a N_i)(D^b N_j)$ with `a` the derivative on the test function (row) and `b` on the trial function (column). With a `ProductSpace` `space`, tuples `(I, J, QI, QJ, a, b, coeff)` for block $(I, J)$. |
| `space` | space or `ProductSpace` | `None` | The space of the matrix, taken from the cache when `None`. |

**Output.** A `Matrix`. It is flagged symmetric unless a term has $a \ne b$.

### H.3 `assemble_vector(cache, terms, n)`

| argument | type | default | meaning |
|---|---|---|---|
| `cache` | `QuadratureCache` | required | The tables. |
| `terms` | list | required | Tuples `(a, coeff)`, each adding $\int c\,D^a N_i$. |
| `n` | int | required | Length of the result, `V.dim`. |

**Output.** An array of length `n`.

### H.4 `assemble_block_vector(P, terms)`

| argument | type | default | meaning |
|---|---|---|---|
| `P` | `ProductSpace` | required | The product space. |
| `terms` | list | required | Tuples `(I, QI, a, coeff)`, each adding $\int c\,D^a N_i$ over the test functions of field `I`, with `QI` a cache of that field. |

**Output.** An array of length `P.dim`, in product numbering.

### H.5 `assemble_scalar(cache, coeff)`

| argument | type | default | meaning |
|---|---|---|---|
| `cache` | `QuadratureCache` | required | The tables. |
| `coeff` | number or array at the cache nodes | required | The integrand $c$. |

**Output.** The float $\int_a^b c\,dx$.
