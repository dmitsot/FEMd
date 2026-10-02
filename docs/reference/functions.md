# Functions and transfer between spaces

[Manual](../manual.md) · [API reference](index.md)

## C. Functions and transfer between spaces

A `Function` is a field that carries its coefficients. In a form it is read
live at assembly ([manual, Section 5](../manual.md#5-functions)).

| name | kind | purpose |
|---|---|---|
| `Function` | class | a known field in one space |
| `ProductFunction` | class | a known field on a `ProductSpace`, one global vector |
| `Functions(P)` | function | builds a `ProductFunction` |
| `Transfer` | class | the operator taking a Function of one space to another |
| `mixed_mass(W, V)` | function | the mass matrix between two spaces, exact for any two grids |

### C.1 `Function(V, name=None, vector=None)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `V` | `SplineSpace` or `LagrangeSpace` | required | The space. |
| `name` | str or `None` | `None` | The name used by `repr` and by keyword overrides in `Form.assemble`. `None` reads it from the variable on the source line, `u = fd.Function(V)` giving `"u"`, and falls back to `"w"` with a warning where no source is available. |
| `vector` | array_like of length `V.dim`, or a linear combination of Functions of `V` | `None` | Initial coefficients, copied. `None` gives zero. An array or expression passed as the second positional argument is taken as `vector`, so `fd.Function(V, u + v)` works. |

**Output.** A `Function`. **Raises** `ValueError` when the vector length is
not `V.dim`, or when an expression is not linear in Functions of `V`.

**Attributes**

| attribute | type | meaning |
|---|---|---|
| `vector` | array, length `V.dim` | The coefficients $c_j$ of $u = \sum_j c_j N_j$. Writable in place. |
| `space` | space | `V`. |
| `name` | str | The name. |
| `shape` | tuple | `vector.shape`. |

**Methods**

| method | inputs | returns | meaning |
|---|---|---|---|
| `u(x, k=0)` | `x` scalar or array_like of points. `k` int, derivative order. | array | $d^k u/dx^k$ at the points. |
| `derivative(x, k=1)` | as `u(x, k)` | array | The same, with `k` defaulting to 1. |
| `at_quad(cache, k=0)` | `cache` a `QuadratureCache` of `V`. `k` int. | array, length `nelem*nq` | Values at the quadrature nodes, `[e*nq + q]`. |
| `project(f, npts=None)` | `f` a callable, an expression in `fd.x`, values at the cache nodes, or a `Function` of another space. `npts` int, as in `V.project`. | `self` | Sets the coefficients to the $L^2$ projection. |
| `interpolate(f)` | `f` a callable or values at `V.dof_coordinates()`. | `self` | Sets the coefficients to the interpolant. |
| `project_to(W, check=True)` | `W` the target space. `check` bool, warn when `u` violates a condition built into `W`. | new `Function` of `W`, same name | Exact $L^2$ projection ([manual, Section 5.4](../manual.md#54-moving-a-function-to-another-space)). Builds a new operator on every call. |
| `interpolate_to(W, check=True)` | as `project_to` | new `Function` of `W` | Interpolation at `W.dof_coordinates()`. |
| `assign(other)` | `other` a `Function` of `V`, an array of length `V.dim`, or a linear combination such as `u0 + 0.5*dt*k1`. | `self` | Overwrites the coefficients. A nonlinear expression raises `ValueError`. |
| `copy(name=None)` | `name` str, default the current name. | new `Function` | An independent copy. |

**Operators**

| expression | result |
|---|---|
| `u + v`, `u - v`, `a*u`, `-u`, `u*v`, `u**n`, `u/a` | A symbolic expression, for forms, `assign` or `Function(V, expr)`. Never a new Function. |
| `u + arr`, `u * arr` with a NumPy array | A NumPy array (the vector combined with the array). |
| `u += w`, `u -= w` | In place on the vector. `w` is a Function, an array or a linear combination. |
| `u *= a`, `u /= a` | In place scaling by a number. |
| `np.asarray(u)`, `len(u)`, `u[i]`, `u[i] = v` | The vector, its length, and element access. |

### C.2 `ProductFunction` and `Functions(P, name=None, vector=None)`

A known field on a `ProductSpace`. Build it with `Functions`.

**Inputs of `Functions`**

| argument | type | default | meaning |
|---|---|---|---|
| `P` | `ProductSpace` | required | The space. |
| `name` | str or `None` | `None` | The name, read from the variable when `None`, falling back to `"w"`. |
| `vector` | array_like of length `P.dim` | `None` | Initial global coefficients, zero when `None`. |

**Output.** A `ProductFunction`.

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `vector` | attribute | array, length `P.dim` | The global coefficients, in product numbering. |
| `product` | attribute | `ProductSpace` | `P`. |
| `name` | attribute | str | The name. |
| `components` | attribute | tuple | One symbol per field, for use in forms. `a, b = U` unpacks them. |
| `U[i]` | `i` int | symbol | Component `i`. |
| `split()` | none | tuple of `Function` | One Function per field, copies of the current coefficients, named `name[i]`. |
| `assign(other)` | `other` an array or `ProductFunction` of length `P.dim` | `self` | Overwrites the global vector. |
| `project(f, degree=None)`, `interpolate(f)` | `f` a list with one entry per field, each an expression in `fd.x`, a callable or `None` (zero). `degree` the number of Gauss points | `self` | Sets each field by $L^2$ projection or interpolation in its own space. |

### C.3 `mixed_mass(W, V)`

The matrix $B_{ij} = \int_{W.a}^{W.b} N^W_i N^V_j$, integrated on the union of
the two grids, so it is exact for any pair of grids.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `W` | space | required | The row (test) space. Its interval must lie inside that of `V`. |
| `V` | space | required | The column space. |

**Output.** `scipy.sparse.csr_matrix` of shape `(W.dim, V.dim)`.

### C.4 `Transfer(target, source, kind="project")`

The linear map from Functions of `source` to Functions of `target`, assembled and
factored once. Usually built as `target.transfer_matrix(source, kind)`.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `target` | `SplineSpace` or `LagrangeSpace` | required | The space to transfer to, $W$. Its interval must lie inside that of `source`. |
| `source` | `SplineSpace` or `LagrangeSpace` | required | The space to transfer from, $V$. |
| `kind` | str | `"project"` | `"project"` solves $M_W c_W = B c_V$, the exact $L^2$ projection. `"interpolate"` solves $C c_W = E c_V$, collocation at `target.dof_coordinates()`. |

**Output.** A `Transfer`. **Raises** `ValueError` for another `kind` or an
interval that is not contained, `TypeError` for a `ProductSpace`.

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `T(u, check=True, name=None)` | `u` a `Function` of `source`, or an array of its coefficients. `check` bool, warn when `u` violates a condition built into `target`. `name` str, the name of the result, default that of `u`. | `Function` of `target`, or an array for an array | The transferred field. |
| `T @ c` | `c` array of length `source.dim` | array of length `target.dim` | The same map on coefficients, without the boundary check. |
| `shape` | attribute | tuple | `(target.dim, source.dim)`. |
| `source`, `target`, `kind` | attribute | | The constructor arguments. |
| `rhs` | attribute | sparse matrix | $B$ (project) or $E = N^V(\xi)$ (interpolate). |
| `lhs` | attribute | `Matrix`, sparse matrix or `None` | $M_W$ (project), the collocation matrix $C$ (interpolate), or `None` when $C$ is the identity. |
| `toarray()` | none | dense array, `shape` | The full operator. Dense, since $M_W^{-1} B$ is not sparse. |
