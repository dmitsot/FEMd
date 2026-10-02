# Spaces and boundary conditions

[Manual](../manual.md) · [API reference](index.md)

## A. Spaces and 1D meshes

A space is a grid, a polynomial degree and a set of boundary conditions built
into the basis ([manual, Section 3](../manual.md#3-spaces-in-one-dimension)).

| name | kind | purpose |
|---|---|---|
| `SplineSpace` | class | B-splines of any degree, $C^{p-1}$ by default |
| `LagrangeSpace` | class | classical $C^0$ nodal elements |
| `DGSpace` | class | discontinuous piecewise polynomials of degree $p \ge 0$ |
| `FunctionSpace(mesh, family, degree)` | function | a space by family name |
| `ProductSpace` | class | several spaces on one mesh, numbered together for coupled systems |
| `Mesh1D` | class | the element vertices of a 1D grid |

### A.1 `SplineSpace(grid, degree, bc="free", continuity=None, continuity_at=None, left=None, right=None)`

B-splines of degree $p$ on `grid`. The coefficients of a spline are weights,
not point values ([manual, Section 5.1](../manual.md#51-what-uvector-is)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `grid` | array_like or `Mesh1D` | required | Strictly increasing element vertices $x_0 < \dots < x_{n_e}$. Need not be uniform. |
| `degree` | int | required | Polynomial degree $p \ge 1$. |
| `bc` | see [Section B](#b-boundary-conditions) | `"free"` | The boundary conditions built into the basis. |
| `continuity` | int or `None` | `None` | Global continuity $C^k$ across interior vertices, $0 \le k \le p-1$. `None` means $p-1$. `continuity=0` spans the Lagrange space of the same degree. |
| `continuity_at` | dict `{node: k}` or `None` | `None` | Lower continuity at chosen interior vertices only. `node` is an index in `1..nelem-1`, `k` lies in $[0, p-1]$. |
| `left`, `right` | number, tuple or `None` | `None` | Data of the conditions built in at each end, one value per built-in functional, as in `lift`. `fd.solve`, `fd.newton`, `fd.IRK` and `lift()` use them when a call gives no data. A callable of $t$ is data in time (`fd.IRK`). Checked against `bc` when the space is made. |

**Output.** A `SplineSpace`. **Raises** `TypeError` for a non-integer degree
or continuity, `ValueError` for $p < 1$, a continuity outside $[0, p-1]$
(including $-1$, which a discontinuous space would need) or a node that is not
interior.

**Members specific to splines**

| member | inputs | returns | meaning |
|---|---|---|---|
| `greville()` | none | array, length `dim` | The Greville abscissa of each coefficient, the same as `dof_coordinates()`. |
| `knots()` | none | array | The full knot vector, with its end and interior multiplicities. |
| `continuity` | attribute | int | The lowest continuity anywhere, global or from `continuity_at`. |

All members common to both families are in [A.3](#a3-members-common-to-splinespace-and-lagrangespace).

### A.2 `LagrangeSpace(grid, degree, bc="free", nodes="equispaced", left=None, right=None)`

Classical $C^0$ nodal elements with equispaced (default) or Gauss-Lobatto nodes inside each
element ([manual, Section 3.2](../manual.md#32-lagrangespacegrid-degree-bcfree-nodesequispaced-leftnone-rightnone)). The coefficients are the values at the nodes.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `grid` | array_like or `Mesh1D` | required | Strictly increasing element vertices. |
| `degree` | int | required | Polynomial degree $p \ge 1$. |
| `bc` | see [Section B](#b-boundary-conditions) | `"free"` | The boundary conditions built into the basis. |
| `nodes` | str | `"equispaced"` | `"equispaced"` ($p+1$ equally spaced points in each element) or `"lobatto"` (the $p+1$ Gauss-Lobatto points). Same space, different basis. |
| `left`, `right` | number, tuple or `None` | `None` | Data of the conditions built in at each end, one value per built-in functional, as in `lift`. `fd.solve`, `fd.newton`, `fd.IRK` and `lift()` use them when a call gives no data. A callable of $t$ is data in time (`fd.IRK`). Checked against `bc` when the space is made. |

**Output.** A `LagrangeSpace`.

**Members specific to Lagrange**

| member | inputs | returns | meaning |
|---|---|---|---|
| `nodes()` | none | array, length `dim` | The nodes, where the coefficients are values. The same as `dof_coordinates()`. |
| `node_set` | attribute | str | `"lobatto"` or `"equispaced"`. |
| `continuity` | attribute | int | Always 0. |

### A.3 Members common to `SplineSpace` and `LagrangeSpace`

**Attributes**

| attribute | type | meaning |
|---|---|---|
| `dim` | int | Degrees of freedom after the built-in conditions. Not the number of grid points ([manual, Section 3.3](../manual.md#33-dimension)). |
| `degree` | int | $p$. |
| `nelem` | int | $n_e$. |
| `grid` | array | The vertices the space was built on. |
| `a`, `b` | float | The ends of the interval. |
| `mesh` | `Mesh1D` | The underlying mesh. |
| `bc` | `BCSpec` | The built-in conditions at both ends. |
| `n_constraints` | int | The number of built-in functionals, both ends together. |
| `raw_dim` | int | Basis size before the conditions are built in, `unconstrained.dim`. |
| `nloc_max` | int | The most basis functions with support on one element. |
| `uband` | int | Half-bandwidth of a matrix on this space. A row has at most `2*uband+1` nonzeros. |
| `unconstrained` | space | The same grid, degree and continuity with nothing built in, created once and cached. It is the space itself when nothing is built in (free or periodic). |
| `boundary_data` | dict or `None` | The data given with the space, `{"left": ..., "right": ...}`, or `None`. |
| `family` | str | `"spline"`, `"lagrange"` or `"dg"`. |
| `tdim` | int | Topological dimension of the mesh, 1. |
| `broken` | bool | Whether the space is discontinuous across elements. |

**Methods**

| method | inputs | returns | meaning |
|---|---|---|---|
| `dof_coordinates()` | none | array, length `dim` | One coordinate per degree of freedom: the node for Lagrange, the Greville abscissa for splines. |
| `element_dofs(e)` | `e` int, an element index in `0..nelem-1` | int array | Global indices of the basis functions with support on element `e`. |
| `evaluate(coeffs, x, k=0)` | `coeffs` array of length `dim` or a `Function`. `x` scalar or array_like of points in $[a, b]$. `k` int, derivative order. | array, `len(x)` | $d^k u/dx^k$ at the points, with $u = \sum_j c_j N_j$. |
| `basis_matrix(x, k=0)` | `x` array_like of points. `k` int, derivative order. | `scipy.sparse.csr_matrix`, `len(x)` $\times$ `dim` | Entry $(i, j)$ is $d^k N_j(x_i)$. `basis_matrix(x) @ c` equals `evaluate(c, x)`. |
| `cache(npts=None, nder=1, rule="gauss")` | `npts` int, points per element, default $p+1$ (at least 2 for Lobatto). `nder` int, highest derivative tabulated. `rule` `"gauss"` or `"lobatto"`. | `QuadratureCache` | Basis tables at the Gauss-Legendre or Gauss-Lobatto points, for the low-level kernels ([Section H](forms.md#h-low-level-assembly)). |
| `at_quad(cache, coeffs, k=0)` | `cache` a `QuadratureCache` of this space. `coeffs` array or `Function`. `k` int, at most `cache.nder`. | array, length `nelem*nq` | $d^k u/dx^k$ at every quadrature node, laid out as `[e*nq + q]`. |
| `mass_matrix(npts=None, lumped=False)` | `npts` int, Gauss points per element, default $p+1$. `lumped` bool. | `Matrix` | $M_{ij} = \int N_i N_j$. With `lumped=True` a diagonal matrix: the Gauss-Lobatto rule at the nodes of a `LagrangeSpace` with `nodes="lobatto"` (the same matrix as `u*v*dx(scheme="lobatto")`), and the row sums $\sum_j M_{ij}$ on every other space. Raises `ValueError` when a row sum is not positive ([manual, Section 6.7](../manual.md#67-quadrature)). |
| `project(f, npts=None)` | `f` a callable $f(x)$ on arrays, an expression in `fd.x` (such as `fd.sech(fd.x)**2`), an array of values at the nodes of `cache(npts, 0)`, or a `Function` of another space. `npts` int, Gauss points per element, default $p+2$, ignored for a `Function`. | array, length `dim` | Coefficients of the $L^2$ projection. A `Function` is projected exactly whatever the two grids ([manual, Section 5.4](../manual.md#54-moving-a-function-to-another-space)). |
| `interpolate(f)` | `f` a callable, or an array of length `dim` with values at `dof_coordinates()`. | array, length `dim` | Coefficients of the interpolant at `dof_coordinates()`. Nothing is solved for Lagrange, a banded collocation system for splines. `f` should satisfy the built-in conditions. |
| `lift(*g, left=None, right=None)` | Either positional `g`, one value per built-in functional with the left end first, or `left` and `right`, each a number or a tuple with one value per functional built in at that end. An end left out gets zero. | `Function` of `unconstrained` | A function $u_g$ satisfying the built-in functionals with the given data ([manual, Section 4.1](../manual.md#41-lifting-step-by-step)). Without arguments it uses `boundary_data`. |
| `prolongate(c)` | `c` adapted coefficients, length `dim` (array or `Function`). | array, length `raw_dim` | $C^{\mathsf T} c$, the same function in the raw basis. A complex `c` gives a complex result. |
| `restrict(r)` | `r` raw coefficients, length `raw_dim`. | array, length `dim` | $C r$, the transpose of `prolongate`. It takes a raw load vector to the adapted one. A complex `r` gives a complex result. |
| `transfer_matrix(source, kind="project")` | `source` the space to transfer from. `kind` `"project"` or `"interpolate"`. | `Transfer` | The operator from `source` to this space, assembled and factored once ([C.4](functions.md#c4-transfertarget-source-kindproject)). |
| `info()` | none | str | A multi-line summary: interval, elements, degree, conditions, dimensions, bandwidth. |

`lift` raises `TypeError` when data are given both positionally and by keyword,
and `ValueError` when the number of values does not match the built-in
functionals, when data are given at an end with nothing built in (natural data
go through `ds`), or on a periodic space.

### A.4 `ProductSpace(*fields, position=True)`

Several spaces on one mesh, each with its own degree and conditions, numbered
together so that a coupled matrix stays banded (manual, Sections [3.5](../manual.md#35-productspacefields) and [6.6](../manual.md#66-coupled-systems)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `*fields` | spaces, or one list or tuple of spaces | required | The component spaces, on one mesh. All periodic or none. |
| `position` | bool | `True` | `True` numbers all degrees of freedom by coordinate, which keeps the bandwidth of one element. `False` numbers field after field, which gives a bandwidth of order `dim` and is meant for testing. |

**Output.** A `ProductSpace`.

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `dim` | attribute | int | Total degrees of freedom. |
| `uband` | attribute | int | Half-bandwidth of the coupled matrix. |
| `nfields` | attribute | int | Number of fields. |
| `fields` | attribute | tuple of spaces | The component spaces, in order. |
| `periodic` | attribute | bool | Whether the fields are periodic. |
| `global_index(field, j)` | `field` int, field number. `j` int, local index in that field. | int | The global index. |
| `local_index(g)` | `g` int, a global index. | `(field, j)` | The field and local index of `g`. |
| `split(x)` | `x` global array of length `dim`, or a `ProductFunction`. | list of arrays | One coefficient array per field. |
| `gather(parts)` | `parts` a list of per-field arrays or `Function`s. | array, length `dim` | The global vector. |
| `n_constraints` | attribute | int | Functionals built into the fields, all together. |
| `unconstrained` | attribute | `ProductSpace` | The product of the fields' unconstrained spaces, itself when no field has a condition built in. |
| `prolongate(c)` | `c` array, length `dim` | array | The raw coefficients, field by field, a vector of `unconstrained`. |
| `lift(left=None, right=None)` | `left`, `right` a list with one entry per field, each as for `V.lift` (`None` for no data). | `ProductFunction` of `unconstrained` | The fields' lifts gathered. The full field is `prolongate(c) + lift(...).vector`. |

A `ProductSpace` with a discontinuous field reserves the coupling of neighboring
elements through the facets in `uband`, so `dS` terms stay inside the band.

### A.4a `DGSpace(grid, degree, bc="free", basis="legendre")`

Discontinuous piecewise polynomials ([manual, Section 3.6](../manual.md#36-dgspacegrid-degree-bcfree-basislegendre)). Its members are those of
[A.3](#a3-members-common-to-splinespace-and-lagrangespace), with nothing built in.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `grid` | array_like or `Mesh1D` | required | Strictly increasing element vertices. |
| `degree` | int | required | $p \ge 0$. |
| `bc` | str | `"free"` | `"free"` or `"periodic"`. Periodic adds the seam as an interior facet. |
| `basis` | str | `"legendre"` | `"legendre"`, orthogonal with a diagonal mass matrix, or `"lobatto"`, nodal at the Gauss-Lobatto points ($p \ge 1$). |

**Output.** A `DGSpace` with `dim` $= n_e(p+1)$, `uband` $= 2p+1$ ($1$ for
$p = 0$), `continuity` $= -1$ and `broken` true. **Raises** `ValueError` for
$p < 0$, another `bc`, another basis, or the Lobatto basis with $p = 0$.

**Members specific to DG**

| member | inputs | returns | meaning |
|---|---|---|---|
| `basis_name` | attribute | str | `"legendre"` or `"lobatto"`. |
| `reference_points()` | none | array, length $p+1$ | The Gauss points (Legendre) or Gauss-Lobatto nodes (Lobatto) on $[-1, 1]$. |
| `interpolate(f)` | `f` a callable or values at `dof_coordinates()` | array, length `dim` | Element-wise interpolation at those points. |

### A.4b `FunctionSpace(mesh, family, degree, **options)`

| argument | type | default | meaning |
|---|---|---|---|
| `mesh` | array_like, `Mesh1D`, `Mesh2D` or `QuadMesh` | required | The grid, or a 2D mesh (then only `"lagrange"`). |
| `family` | str | required | `"spline"` (also `"bspline"`), `"lagrange"` (also `"cg"`), `"dg"` (also `"discontinuous"`, `"legendre"`), `"dg-lobatto"` (also `"lobatto"`), case-insensitive. |
| `degree` | int | required | $p$. |
| `**options` | | | Passed to the family's constructor: `bc`, `continuity`, `continuity_at`, `basis`, and the boundary values `left`, `right` of the spline and Lagrange families. On a 2D mesh the options of `LagrangeSpace` there (`dirichlet`, `data`, `periodic`, `nodes`). |

**Output.** A `SplineSpace`, `LagrangeSpace` or `DGSpace`, or on a 2D mesh a `LagrangeSpace2D` or `LagrangeSpaceQ`. **Raises**
`ValueError` for an unknown family, and for `left=`/`right=` on a DG space, which has no condition built in.

### A.5 `Mesh1D(vertices)`

The vertices of a 1D grid. A space builds one from its `grid` argument, and
`V.mesh` returns it.

**Constructors**

| constructor | inputs | returns |
|---|---|---|
| `Mesh1D(vertices)` | `vertices` array_like, strictly increasing. | `Mesh1D` |
| `Mesh1D.uniform(a, b, nelem)` | `a`, `b` float, the interval. `nelem` int, the number of elements. | `Mesh1D` with equal elements |

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `nelem` | attribute | int | Number of elements. |
| `a`, `b` | attribute | float | The ends. |
| `is_uniform` | attribute | bool | Whether all elements have the same width, detected from the vertices. |
| `vertices()` | none | array, length `nelem+1` | The vertices. |
| `h(e)` | `e` int, element index. | float | The width of element `e`. |
| `element_of(x)` | `x` float, a point in $[a, b]$. | int | The element containing `x`. `x = b` gives the last element. A point outside $[a, b]$ by more than $10^{-12}(b-a)$ raises. |

---

## B. Boundary conditions

What is built into a basis is homogeneous. Data are added through `ds` (natural
conditions) or through `left=`/`right=` in `solve`, `lift` and `newton`
(built-in conditions). See [manual, Section 4](../manual.md#4-boundary-conditions).

| name | kind | purpose |
|---|---|---|
| `bc=` | argument | how a space is told its conditions |
| `BoundaryCondition` | class | the condition at one end, as a list of functionals |
| `BCSpec` | class | the conditions at both ends, or periodic |
| `Robin(alpha, beta)` | function | shortcut for $\alpha u + \beta u' = 0$ |
| `Derivative(m)` | function | shortcut for $u^{(m)} = 0$ |

### B.1 The `bc=` argument

Accepted by `SplineSpace` and `LagrangeSpace`.

| value | meaning |
|---|---|
| `"free"` or `None` | Nothing built in. The weak form gives the natural condition. |
| `"dirichlet"` | $u = 0$ at both ends. |
| `"neumann"` | $u' = 0$ at both ends. |
| `"clamped"` | $u = u' = 0$ at both ends. |
| `"periodic"` | The two ends identified. |
| a `BoundaryCondition` | That condition at both ends. |
| `(left, right)` | One condition per end, each a name, a `BoundaryCondition` or `None`. |
| a `BCSpec` | Used as it is. |

Names are case-insensitive. `"periodic"` cannot appear inside a pair.

### B.2 `BoundaryCondition`

The condition at one end, a list of linear functionals
$\lambda(u) = \sum_m \alpha_m u^{(m)}(x_{\text{end}})$, each set to zero.
Built with a static constructor, not called directly.

**Constructors**

| constructor | inputs | built in |
|---|---|---|
| `BoundaryCondition.free()` | none | nothing |
| `BoundaryCondition.dirichlet()` | none | $u = 0$ |
| `BoundaryCondition.neumann()` | none | $u' = 0$ |
| `BoundaryCondition.clamped()` | none | $u = 0$ and $u' = 0$ |
| `BoundaryCondition.robin(alpha, beta)` | `alpha`, `beta` float | $\alpha u + \beta u' = 0$ |
| `BoundaryCondition.derivative(m)` | `m` int, derivative order | $u^{(m)} = 0$ |
| `BoundaryCondition.custom(functionals)` | `functionals` a list with one entry per functional, each a list of `(m, alpha_m)` pairs | $\sum_m \alpha_m u^{(m)} = 0$ for each entry |
| `BoundaryCondition.from_name(name)` | `name` one of `"free"`, `"dirichlet"`, `"neumann"`, `"clamped"` | as named |

For example `custom([[(0, 1.0)], [(2, 1.0)]])` builds in $u = 0$ and $u'' = 0$,
the simply supported beam.

**Attributes**

| attribute | type | meaning |
|---|---|---|
| `name` | str | `"free"`, `"dirichlet"` and so on. |
| `count` | int | Number of functionals built in. |
| `max_order` | int | Highest derivative order involved, $-1$ for free. |
| `functionals` | list | One list of `(order, weight)` pairs per functional. |

### B.3 `BCSpec`

The conditions of a space at both ends. `V.bc` returns one, and the `bc=`
argument is converted to one.

**Constructors**

| constructor | inputs | returns |
|---|---|---|
| `BCSpec()` | none | free at both ends |
| `BCSpec(left, right)` | `left`, `right` `BoundaryCondition` | one condition per end |
| `BCSpec.periodic()` | none | periodic |
| `BCSpec.from_name(name)` | `name` a condition name or `"periodic"` | that condition at both ends |

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `left`, `right` | attribute | `BoundaryCondition` | The condition at each end. |
| `is_periodic` | attribute | bool | Whether the ends are identified. |
| `describe()` | none | str | `"periodic"` or `"(left, right)"` with the names. |

### B.4 `Robin(alpha, beta)` and `Derivative(m)`

| function | inputs | returns |
|---|---|---|
| `Robin(alpha, beta)` | `alpha`, `beta` float | `BoundaryCondition.robin(alpha, beta)`, building in $\alpha u + \beta u' = 0$ |
| `Derivative(m)` | `m` int | `BoundaryCondition.derivative(m)`, building in $u^{(m)} = 0$ |
