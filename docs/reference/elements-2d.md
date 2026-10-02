# Finite elements on triangles and quadrilaterals

[Manual](../manual.md) · [API reference](index.md)

## L. Finite elements on triangles

| name | kind | purpose |
|---|---|---|
| `LagrangeSpace(mesh, k)`, `LagrangeSpace2D` | class | continuous $P_k$ on a `Mesh2D` |
| `y`, `Dx`, `grad`, `div`, `dot`, `inner`, `as_vector`, `FacetNormal` | form language | 2D symbols and operators |
| `ds(marker)` | measure | boundary integrals by side marker |
| `SparseMatrix` | class | the CSR store a 2D bilinear form assembles into |
| `SparseSolver` | class | SuperLU, CG, GMRES or dense LU behind `solve()` |
| `Preconditioner` | class | Jacobi, SSOR or ILU(0), in C++ |
| `cg` | function | preconditioned conjugate gradients |
| `solve(a, L, dirichlet=...)` | function | a 2D linear problem with Dirichlet data |
| `RTSpace`, `N1curlSpace`, `DGSpace2D` | class | Raviart-Thomas, Nédélec and broken $P_k$ on triangles (L.13) |
| `curl`, `rot` | form language | the 2D curl |

### L.1 `LagrangeSpace(mesh, degree, bc="free", dirichlet=None, nodes="equispaced", periodic=None, data=None)` on a `Mesh2D`

Continuous piecewise polynomials of degree `degree` on the triangles of `mesh`
([manual, Section 11.1](../manual.md#111-p_k-elements-on-triangles)). `LagrangeSpace` sees the `Mesh2D` and returns a
`LagrangeSpace2D`, which may also be called directly with the same arguments.
A 2D space is therefore not an instance of the 1D class `LagrangeSpace`. Test
`V.tdim == 2` and `V.family == "lagrange"` instead, as the rest of the package
does.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `mesh` | `Mesh2D` | required | The mesh. It must outlive the space, which the space ensures by holding it. |
| `degree` | int | required | $k \ge 1$. |
| `bc` | str | `"free"` | `"dirichlet"` builds a Dirichlet condition into every exterior side. |
| `dirichlet` | int, list of int, dict or `None` | `None` | Markers of the sides with a Dirichlet condition. Combines with `bc`. A dict `{marker: data}` names the sides and gives their data. |
| `data` | anything `V.lift` takes, or `None` | `None` | The Dirichlet data, for instance with `bc="dirichlet"`. `fd.solve`, `fd.newton`, `fd.IRK` and `V.lift()` use the data of the space when a call gives none (`V.boundary_data`). A callable $g(x, y, t)$ or $g(t)$ is data in time (`fd.IRK`). |
| `nodes` | str | `"equispaced"` | `"equispaced"`: the uniform lattice. `"lobatto"`: warp-and-blend points on triangles, the Gauss-Lobatto tensor grid on quads. Same space, different basis ([manual, Section 11.1](../manual.md#111-p_k-elements-on-triangles)). |
| `periodic` | pair of int, list of pairs, or `None` | `None` | `(a, b)`: side `b` identified with side `a`, which it must match node for node after a translation. `[(4, 2), (1, 3)]`: a doubly periodic rectangle ([manual, Section 12.1](../manual.md#121-boundary-conditions)). |

**Raises** `ValueError` for a degree below 1, an unknown `bc`, or a marker no side
carries, and `TypeError` for a mesh that is not a `Mesh2D`. With `periodic`, `ValueError` for a
side that is also Dirichlet (or `bc="dirichlet"`), a side paired with itself, or sides that do not
match node for node.

**Members**

| member | inputs | returns | meaning |
|---|---|---|---|
| `dim` | attribute | int | Degrees of freedom (free nodes). |
| `raw_dim`, `n_constraints` | attribute | int | All nodes, and the eliminated ones. |
| `degree`, `ncells`, `nedges`, `nloc` | attribute | int | Degree, triangles, edges, nodes per triangle. |
| `mesh` | attribute | `Mesh2D` | The mesh. |
| `dirichlet_markers`, `all_exterior` | attribute | list, bool | The condition built in. |
| `node_set` | attribute | str | `"lobatto"` or `"equispaced"`. |
| `unconstrained` | attribute | `LagrangeSpace2D` | The same space with nothing eliminated (itself when nothing is). |
| `is_periodic`, `periodic_pairs` | attribute | bool, list | Whether nodes are identified, and the pairs `(a, b)`. |
| `periodic_info` | attribute | list of dict | Per pair: `pair`, `shift` (the translation from side `a` to side `b`), `nodes` and `partners` (matched raw nodes of the two sides). |
| `representatives()` | none | int array, `raw_dim` | The raw node each raw node is identified with (itself when none). `prolongate` copies a value to every node of its class, `restrict` reads the representative. |
| `dof_coordinates()` | none | `(dim, 2)` array | Coordinates of the degrees of freedom. |
| `node_coordinates()` | none | `(raw_dim, 2)` array | Coordinates of every node. |
| `boundary_nodes(marker=None)` | marker(s) | int array | Raw indices of the nodes on those sides (every exterior side for `None`). |
| `prolongate(c)`, `restrict(raw)` | array or Function | array | Adapted to raw (zeros on eliminated nodes) and back. |
| `interpolate(f)` | callable `f(x, y)` or values | array | Nodal interpolant. |
| `project(f, degree=None)` | callable, number or expression | array | $L^2$ projection onto `V`. |
| `lift(g)` | number, callable or `{marker: data}` | Function of `unconstrained` | Dirichlet data on the eliminated nodes. |
| `evaluate(c, x, y=None, deriv=0)` | coefficients, points | array | Values or `deriv="x"` / `"y"` derivatives, NaN outside. |
| `mass_matrix(lumped=False)` | bool | `SparseMatrix` | $\int u v$. With `lumped=True` a diagonal matrix: the tensor Gauss-Lobatto rule at the nodes of $Q_k$ with `nodes="lobatto"`, the row sums otherwise. Raises `ValueError` when a row sum is not positive ($P_k$ triangles from $k = 2$). |
| `cache(degree=None, rule="gauss", npts=None)`, `boundary_cache(degree=None, marker=None)` | int, marker(s) | `Cache2D` | Quadrature points on the cells or the sides. `rule="lobatto"` (quadrilaterals only) gives the tensor Gauss-Lobatto rule with `npts` points per direction, default $k+1$. |
| `at_quad(cache, c, deriv=0)` | cache, coefficients | array | Values at the cache points. |
| `pattern(trial=None)` | space | `SparsityPattern` | The CSR pattern with this test space (cached). |
| `triangulation(c, refine=None, deriv=0)` | coefficients | `(Triangulation, values)` | For plotting, `refine`$^2$ sub-triangles per cell (default $k$). |
| `plot(c, ax=None, *, contour=False, levels=20, refine=None, deriv=0, colorbar=True, mesh=False, **kw)` | coefficients | matplotlib artist | Gouraud colour map, or filled contours. |
| `info()` | none | str | A summary. |

A Function `u` of this space adds `u.at(points)`, `u.at(x, y, deriv=0)` and
`u.plot(**kw)`. Calling `u(x)` is refused on a 2D space.

### L.2 Form language additions

| name | meaning |
|---|---|
| `y` | The coordinate $y$ (`x` is $x$). |
| `Dx(e, i)` | $\partial e / \partial x_i$, `i` 0 or 1, with the product and chain rules on coefficients. |
| `grad(e)` | $(\partial_x e, \partial_y e)$. On a 1D space it is `D(e)`. |
| `div(w)` | $\partial_x w_0 + \partial_y w_1$. |
| `dot(a, b)`, `inner(a, b)` | $\sum_i a_i b_i$, or $ab$ for scalars. |
| `as_vector([a, b])` | A vector expression. Supports `+`, `-`, scalar `*` and `/`, indexing. |
| `FacetNormal()` | $(n_x, n_y)$, the outward unit normal, in `ds` terms only. |
| `ds`, `ds(m)`, `ds((m1, m2))` | Every exterior side, the sides with marker `m`, or with any of the markers. `quad_degree=` overrides the inferred degree. |

The measures `dS` and `ds("left")` belong to 1D meshes and are refused on a 2D
one. Second derivatives are refused on the $C^0$ space.

### L.3 `SparseMatrix`

| member | inputs | returns | meaning |
|---|---|---|---|
| `shape`, `nnz` | attribute | tuple, int | Size and stored entries. |
| `symmetric` | attribute, settable | bool | Measured after assembly for a square form (asymmetry $\le 10^{-13}$). |
| `space`, `row_space` | attribute | space | What the columns and the rows index. |
| `data` | attribute | array | Writable view of the values. |
| `tocsr()` | none | `scipy.sparse.csr_matrix` | A view on the same arrays. |
| `tocoo()`, `toarray()`, `diagonal()` | none | | Conversions. |
| `A @ x` | vector or Function | Function or array | C++ matvec. A `SparseMatrix` operand gives the product. |
| `matvec(x, out=None)`, `rmatvec(x)` | vector | array | $Ax$ and $A^T x$. |
| `A + B`, `A - B`, `a * A`, `A / a`, `-A`, `A.T`, `A += B` | | `SparseMatrix` | Arithmetic, on the shared pattern when there is one. |
| `is_symmetric(tol=1e-12)` | float | bool | Measured symmetry. |
| `copy()` | none | `SparseMatrix` | A copy of the values (the pattern stays shared). |
| `ordering(kind="rcm")` | `"rcm"`, `"amd"` or `"natural"` | int array `p` | A renumbering from the pattern (symmetrized), computed in C++. `p[k]` is the unknown that becomes number `k`. Reverse Cuthill-McKee gives a narrow band, approximate minimum degree little fill in a Cholesky factor. |
| `bandwidth(p=None)` | permutation | int | The half-bandwidth $\max \lvert i - j \rvert$ over the stored entries, in the order `p` (natural when `None`). |
| `permuted(p)` | permutation | `SparseMatrix` | $P A P^T$, row and column `k` being row and column `p[k]` of $A$. A new matrix with `space = None`. |
| `solver(backend="auto", **options)` | str | `SparseSolver` | See [L.4](#l4-sparsesolver). |
| `preconditioner(kind="ilu0", omega=1.0, ordering="rcm")` | str, float, str | `Preconditioner` | See [L.5](#l5-preconditionera-kindilu0-omega10-orderingrcm). |
| `SparseMatrix.from_scipy(A, space=None, row_space=None, symmetric=None)` | scipy matrix | `SparseMatrix` | A copy into the store. |

### L.4 `SparseSolver`

Made by `A.solver(backend, **options)`.

| backend | options | meaning |
|---|---|---|
| `"auto"` | none | `"cholesky"` for a symmetric matrix with a positive diagonal, falling back to `"ldlt"` when it meets a pivot $\le 0$. `"ldlt"` for any other symmetric matrix (a saddle point), and `"superlu"` for a nonsymmetric one. `S.backend` says which. |
| `"cholesky"` (also `"chol"`) | `ordering="amd"` or `"natural"`, `positive_definite=True` | FEMd's sparse $LDL^T$ in C++ ([manual, Section 12.3](../manual.md#123-matrices-and-solvers)). Raises `RuntimeError` on a pivot $\le 0$, or with `positive_definite=False` on a zero pivot. `S.factor` is the C++ `SparseCholesky`. |
| `"ldlt"` (also `"ldl"`, `"indefinite"`) | `ordering="amd"` or `"natural"`, `threshold=0.01` | FEMd's pivoting sparse $LDL^T$ for symmetric matrices, definite or not: multifrontal, with $1\times1$ and $2\times2$ pivots accepted by a threshold test in $(0, 0.5]$ ([manual, Section 12.3](../manual.md#123-matrices-and-solvers)). `S.factor` is the C++ `SparseLDLT`, with `inertia` (positive, negative, zero eigenvalues), `nnz_L`, `delayed`, `two_by_two`, `supernodes`, `max_front`, `min_pivot` and `refactor(K)`. Raises `ValueError` for a nonsymmetric matrix and `RuntimeError` for a singular one. |
| `"superlu"` | none | SciPy's SuperLU. For a symmetric matrix it tries symmetric mode (minimum degree on $A + A^T$, no pivoting), checks one solve and refactors with partial pivoting when the backward error exceeds $10^{-11}$. |
| `"cg"`, `"gmres"`, `"lgmres"` | `M="ilu0"` (a kind, a `Preconditioner` or anything `fd.cg` takes), `tol`, `maxiter`, ... | FEMd's Krylov solvers. |
| `"dense"` | none | LU of the dense matrix. |

`solve(b, into=None, lift=None)` returns a Function of the matrix's space, or
of its unconstrained companion with `lift=`, as `LinearSolver.solve` does
([Section F.1](linalg.md#f1-linearsolver)). `info` holds the `KrylovInfo` of the last iterative solve.

### L.5 `Preconditioner(A, kind="ilu0", omega=1.0, ordering="rcm")`

| kind | $M$ |
|---|---|
| `"jacobi"` | $\mathrm{diag}(A)$ |
| `"ssor"` | $\frac{\omega}{2-\omega}(D/\omega + L)(D/\omega)^{-1}(D/\omega + U)$, $0 < \omega < 2$ |
| `"ilu0"` | $LU$ on the pattern of $A$ with no fill |

`ordering="rcm"` (the default) builds SSOR and ILU(0) on $P A P^T$, $P$ the reverse Cuthill-McKee order, and applies $P^T M^{-1} P$. `ordering="natural"` keeps the numbering of the space. Jacobi has no ordering. `P.ordering` says which.

Pass it as `M=` to `cg`, `gmres` or `lgmres`, or apply it with `P(r)` or
`P.solve(r)`. `min_pivot` is the smallest ILU(0) pivot.

### L.6 `cg(A, b, M=None, x0=None, *, maxiter=None, tol=1e-10, warn=True)`

Preconditioned conjugate gradients for a symmetric positive definite `A`.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `A` | `SparseMatrix`, `Matrix`, object with `.matvec`, or callable | required | The operator. A `SparseMatrix` or `Matrix` runs in C++. |
| `b` | array or Function | required | Right-hand side. |
| `M` | `None`, `Preconditioner`, `LinearSolver`, `SparseSolver`, `SparseMatrix`, object with `.solve`, or callable | `None` | Symmetric positive definite preconditioner. A `Preconditioner` or `LinearSolver` runs in C++. |
| `x0` | array or Function | `None` | Initial guess. |
| `maxiter` | int or `None` | `None` | `max(1000, 10 n)` when `None`. |
| `tol` | float | `1e-10` | Bound on the true relative residual $\lVert b - Ax \rVert / \lVert b \rVert$. |
| `warn` | bool | `True` | A `RuntimeWarning` on breakdown ($p^T A p \le 0$) or when `tol` is not met. |

**Outputs.** `(x, info)`, `x` a Function when `A` knows its space, `info` a
`KrylovInfo` with `method="cg"` and `precond_residual` equal to `residual`.

`gmres` and `lgmres` ([F.4](linalg.md#f4-gmresa-b-mnone-x0none--restart30-maxiternone-tol1e-10-warntrue), [F.5](linalg.md#f5-lgmresa-b-mnone-x0none--restart20-k_aug2-maxiternone-tol1e-10-warntrue)) accept a `SparseMatrix` for `A` and a
`Preconditioner` or `SparseSolver` for `M` in the same way.

### L.7 `solve(a, L, left=None, right=None, backend=Solver.Auto, dirichlet=None)`

On a 2D space, `dirichlet=` gives the data (as `V.lift` takes it) and `backend`
is a `SparseMatrix.solver` backend. With data the result is a Function of
`V.unconstrained`, without it a Function of `V`. `left=` and `right=` are
refused on a 2D space and `dirichlet=` on a 1D one.

### L.8 Low level

| name | meaning |
|---|---|
| `_femd.Cache2D(V, degree)` | Points and weights on the triangles, exact to total degree `degree`. |
| `_femd.Cache2D(V, degree, markers, all)` | Gauss points on the exterior sides with those markers (all when `all`), with outward normals. |
| `_femd.cell_pattern(T, S)` | The CSR pattern coupling test space `T` and trial space `S`. |
| `_femd.assemble_matrix_2d(QT, QS, a, b, coeffs, K)` | $K \mathrel{+}= \sum_t \int c_t\, (D^{a_t} T_i)(D^{b_t} S_j)$, codes 0 value, 1 $\partial_x$, 2 $\partial_y$. |
| `_femd.assemble_vector_2d(Q, a, coeffs)` | $f_i = \sum_t \int c_t\, D^{a_t} N_i$. |
| `_femd.assemble_scalar_2d(Q, coeff)` | $\int c$. |
| `_femd.triangle_rule(d)` | `(xi, eta, w)` on the reference triangle, exact to degree `d`. |
| `_femd.ReferenceTriangle(k)` | Nodes and `eval(xi, eta, nder)` of the reference element. |

C++ headers: `sparse/csr.hpp` (`SparsityPattern`, `CSRMatrix` and its algebra),
`sparse/precond.hpp`, `sparse/amd.hpp` (`ordering::amd_order`, `ordering::rcm_order`, `ordering::natural_order`),
`sparse/cholesky.hpp` (`SparseCholesky`), `sparse/ldlt.hpp` (`SparseLDLT`), `krylov/cg.hpp`, `quadrature/triangle.hpp`,
`fe/simplex_basis.hpp` (Jacobi and Dubiner polynomials, warp-and-blend nodes,
`ReferenceTriangle`), `fe/lagrange_space_2d.hpp`, `fe/cache_2d.hpp` and
`forms/assembly_2d.hpp`.

### L.9 Systems on 2D meshes

**`VectorFunctionSpace(mesh, family="lagrange", degree=None, dim=2, *, slip=None, corner_angle=45.0,
slip_normal="conservative", **options)`** returns a `ProductSpace2D` with one vector block of `dim`
components, each `LagrangeSpace(mesh, degree, **options)`. The family may be left out,
`VectorFunctionSpace(m, 2, dirichlet=[1])`. `dirichlet={marker: pair}` or `data=` gives the Dirichlet data with the space, and a `ProductSpace2D` collects the data of its blocks (`W.boundary_data`, one entry per block).

| argument | type | default | meaning |
|---|---|---|---|
| `slip` | int, list of int or `None` | `None` | Markers of the sides with $u\cdot n = 0$, imposed strongly ([manual, Section 12.5](../manual.md#125-systems-vector-fields-and-several-unknowns)). Needs `dim=2`, and no marker may also be a Dirichlet side. |
| `corner_angle` | float | `45.0` | Degrees. A slip vertex whose two edge normals differ by more is a corner and gets $u = 0$. |
| `slip_normal` | str | `"conservative"` | At the nodes inside an edge, the edge's normal (exact $\int u_h\cdot n\, ds = 0$), or `"smooth"`, the vertex normals blended along the edge (second order on curved walls for $k \ge 3$). |

**Raises** `ValueError` for a slip marker no side carries, a marker that is both Dirichlet and slip
(or `bc="dirichlet"` with `slip=`), an unknown `slip_normal`, or `corner_angle` outside $(0, 180)$.

**`ProductSpace(*spaces)`** with spaces on a 2D mesh returns a **`ProductSpace2D`**. Entries are
`LagrangeSpace` (scalar blocks) and `VectorFunctionSpace` (vector blocks) on the same mesh object.

| member | inputs | returns | meaning |
|---|---|---|---|
| `fields` | attribute | tuple | The scalar component spaces, in numbering order. |
| `blocks` | attribute | list | `("vector", [i, j])` or `("scalar", [i])` per block. |
| `offsets` | attribute | int array | First index of each field; `offsets[-1]` is `dim` (with a slip condition, the size before it). |
| `dim`, `n_constraints`, `mesh`, `is_vector` | attribute | | Size, eliminated nodes, the mesh, whether it is a single vector block. |
| `block_space(b)` | int | space | Block `b` on its own. |
| `unconstrained` | attribute | `ProductSpace2D` | The same system with nothing eliminated. |
| `split(x)`, `gather(parts)` | vector, list | list, vector | Per-field views and their concatenation (with a slip condition, the field values $Zx$, and $Z^T$ of the concatenation). |
| `slip_matrix` | attribute | `scipy.sparse` or `None` | $Z$: the coefficients without the slip condition are $Z x$. |
| `slip_info` | attribute | list of dict | Per slip condition: `markers`, `nodes` (raw indices), `points`, `normals`, `corners`, `corner_points`, `fields`. |
| `prolongate(c)`, `restrict(raw)` | vector | vector | Field by field, as for `LagrangeSpace2D`. |
| `lift(g)`, `interpolate(f)` | per-block data | ProductFunction, vector | Data as in [manual, Section 12.5](../manual.md#125-systems-vector-fields-and-several-unknowns). |
| `project(f, degree=None)` | per-block data | vector | $L^2$ projection onto the system, with its Dirichlet (zero) and slip conditions. A scalar entry is a callable, a number, an expression or `None`. A vector entry is a pair of those, a callable returning the pair, or a vector expression. The mass matrix is factored once. |
| `mass_matrix(lumped=False)` | bool | `SparseMatrix` | $\sum_\text{blocks} (u, v)$, created once. `lumped=True` gives the diagonal lumped mass, field by field (Gauss-Lobatto when every field is $Q_k$ with Lobatto nodes, row sums otherwise), also created once. |
| `pattern(trial=None)` | space | `SparsityPattern` | The CSR pattern (cached). |
| `plot(c, ax=None, *, block=0, quiver=True, **kw)` | coefficients | artist | A scalar block as `LagrangeSpace2D.plot`, a vector block as its magnitude with arrows. |
| `info()` | none | str | A summary per block. |

**Functions.** `TestFunctions(W)`, `TrialFunctions(W)` and a `ProductFunction` unpack by block (a
vector block as a vector expression). `TestFunction(V)` and `TrialFunction(V)` on a vector space
are vectors, and refuse a system of several blocks. `Function(W)` gives a `ProductFunction`, or a
**`VectorFunction`** on a vector space, which adds `at(x, y=None)` (values with a last axis of
length 2), `split()` into component Functions, `plot()`, and every vector operation.

**Tensor functions.** `as_matrix`, `outer`, `transpose`, `sym`, `skew`, `tr`, `Identity(d)`,
`nabla_grad`, and `grad`, `div`, `dot`, `inner` extended to matrices, as tabulated in [manual,
Section 12.5](../manual.md#125-systems-vector-fields-and-several-unknowns). `ListTensor.rank`, `.shape` and `.T` describe and transpose an expression.

**`newton(R, u, ..., dirichlet=None)`.** On a 2D mesh `u` may be a Function of
`V.unconstrained` (scalar or system). Newton then moves the free coefficients, and the boundary
values are `V.lift(dirichlet)` or, without `dirichlet=`, those `u` holds.

**`Mesh2D.remark(rule)`.** `rule(x, y, marker)` receives the side midpoints and their markers as
arrays and returns the new markers (positive). The result is a new `Mesh2D` with the same points
and triangles.

C++: `product_pattern(T, toff, nrows, S, soff, ncols)` and the `row_offset`, `col_offset`
arguments of `assemble_matrix_2d` in `forms/assembly_2d.hpp` assemble a block into the matrix of
the whole system.

### L.10 Quadrilateral meshes and $Q_k$

**`QuadMesh`** (from the functions below).

| member | inputs | returns | meaning |
|---|---|---|---|
| `points`, `quads`, `neighbours`, `segments`, `segment_markers`, `point_markers` | attribute | array views | As for `Mesh2D`. `neighbours[q, i]` is across the edge from vertex $i$ to vertex $i+1$. |
| `npoints`, `nquads` (`ncells`), `nsegments` | attribute | int | Counts. |
| `areas()`, `min_angles()`, `max_angles()`, `aspects()` | none | array | Per quad (angles in degrees, aspect = longest over shortest edge). |
| `locate(points)` | `(n, 2)` array | int array | The quad of each point, -1 outside (walk, then a bucket grid). |
| `boundary_edges(marker=None)` | int | `(k, 2)` array | Boundary edges, by marker. |
| `smooth(passes=1)` | int | `QuadMesh` | A smoothed copy (convexity kept). |
| `remark(rule)` | `rule(x, y, marker)` | `QuadMesh` | A copy with new side markers. |
| `triangulation()` | none | `Triangulation` | Each quad split along its 0-2 diagonal, for plotting. |
| `plot(ax=None, *, color="k", lw=0.5, markers=False)` | | axes | Draw the quads, the sides coloured by marker with `markers=True`. |
| `validate()` | none | str | `""` when sound. |

**`rectangle_quad_mesh`**, **`mapped_quad_mesh`**, **`quadrangulate`**, **`quad_mesh_from_triangles`**
and **`quad_mesh_from_arrays`** as tabulated in [manual, Section 10.5](../manual.md#105-quadrilateral-meshes). `quadrangulate` passes its
other keywords to `triangulate` and records `report["triangles"]`, `report["min_angle"]`,
`report["max_angle"]` and the triangulate report.

**`LagrangeSpaceQ(mesh, degree, bc="free", dirichlet=None)`**, what `LagrangeSpace` and
`FunctionSpace(..., "lagrange", ...)` return for a `QuadMesh`. Its members are those of
`LagrangeSpace2D` ([L.1](#l1-lagrangespacemesh-degree-bcfree-dirichletnone-nodesequispaced-periodicnone-datanone-on-a-mesh2d)). `nverts` is 4, `cell_name` is `"quadrilateral"` and `affine` is true only
when every quad is a parallelogram. `VectorFunctionSpace` and `ProductSpace` accept it.

**Form language.** `sinh`, `cosh` and `sech` join the pointwise functions ([Section D.3](forms.md#d3-pointwise-functions); they work on 1D meshes too).

C++: `mesh/quad_mesh.hpp` (`QuadMesh`, `from_triangles`, `smooth`, `locate`),
`fe/space_2d.hpp` (`Space2D`, the base of both 2D families: numbering, elimination, facets,
evaluation), `fe/lagrange_space_quad.hpp` (`ReferenceQuad`, `LagrangeSpaceQ`), and `Cache2D` with
per-point Jacobians.

### L.11 `IRK` on 2D meshes

`fd.IRK(M, rhs, dt, method, stages, *, jacobian=None, unknown=None, time=None, t0=0.0, dirichlet=None, newton="simplified", tol=1e-12, rtol=0, xtol=1e-12, maxiter=20, line_search=False, backend="auto")`
is chosen when `M` is a `SparseMatrix` or rank-2 `Form` on a 2D space or system ([manual, Section 12.6](../manual.md#126-implicit-time-stepping)).
The inputs are those of [G.6](time-stepping.md#g6-irkm-rhs-dt-methodgauss-stages2--jacobiannone-unknownnone-timenone-t000-leftnone-rightnone-dirichletnone-newtonexact-tol1e-12-rtol0-xtol1e-12-maxiter20-line_searchfalse-backendauto), with these differences.

| argument | meaning |
|---|---|
| `M` | `SparseMatrix` or rank-2 `Form`. It may be singular (no pressure block). A `Form` when the data depend on time. |
| `jacobian` | With a callable `rhs`: returns a `SparseMatrix`, a SciPy sparse matrix or an array. |
| `dirichlet` | Anything `V.lift` takes, with callables $g(x, y, t)$ or $g(t)$ for data in time. The state is then a Function of `V.unconstrained`. `left=`, `right=` are refused. Without it, the data given with the space are used, if any. |
| `newton` | `"simplified"` (default), `"frozen"` or `"exact"`, see the table in [manual, Section 12.6](../manual.md#126-implicit-time-stepping). |
| `backend` | `SparseSolver` backend for the real stage systems (`"auto"`: Cholesky or SuperLU). |

| member | meaning |
|---|---|
| `step(u, dt=None)` | As in [G.6](time-stepping.md#g6-irkm-rhs-dt-methodgauss-stages2--jacobiannone-unknownnone-timenone-t000-leftnone-rightnone-dirichletnone-newtonexact-tol1e-12-rtol0-xtol1e-12-maxiter20-line_searchfalse-backendauto). `u` a Function, `ProductFunction`, `VectorFunction` or array of $V$, or of `V.unconstrained` with `dirichlet=`. A new `dt` drops the frozen factorization. |
| `K` | The stages of the last step, stacked: $[K_1; \dots; K_s]$, length $sn$. |
| `stages(cn, K)`, `residual(cn, K)` | As in [G.6](time-stepping.md#g6-irkm-rhs-dt-methodgauss-stages2--jacobiannone-unknownnone-timenone-t000-leftnone-rightnone-dirichletnone-newtonexact-tol1e-12-rtol0-xtol1e-12-maxiter20-line_searchfalse-backendauto), in the stacked numbering. |
| `jacobian(Us, ts=None)` | The $sn \times sn$ stage Jacobian as a `SparseMatrix`. |
| `stage_solver(cn)` | The simplified-Newton solver at $(t, c^n)$, with `.solve(r)` and `.kinds` (the backend of each factored system). |
| `factorizations` | Sparse factorizations so far. |
| `refresh` | Iteration count above which `"frozen"` renews its factorization at the next step (6). |
| `data` | The `_Data2D` of `dirichlet=`: `lift(t)`, `lift_rate(t)`, `timedep`. |

A tableau that is not diagonalizable (a custom `(A, b)`) falls back to the $sn$ block matrix in the
simplified modes.

### L.12 Files

**`write_vtk(filename, *fields, mesh=None, degree=None, cell_data=None, segments=None, binary=True)`**
writes a `.vtu` file ([manual, Section 12.7](../manual.md#127-files-vtk-output-and-gmsh-import)) and returns its name (`.vtu` is appended when missing).

| argument | type | default | meaning |
|---|---|---|---|
| `filename` | str or path | required | The file to write. |
| `*fields` | `Function`, `VectorFunction`, `ProductFunction`, or `(name, field)` | none | Fields on one 2D mesh. A system is written one array per block, `name_0`, `name_1`, ... |
| `mesh` | `Mesh2D` or `QuadMesh` | `None` | The mesh, required when no field is given. |
| `degree` | int | highest field degree (1 for a mesh) | Order of the output cells: 1 gives linear triangles and quads, $k > 1$ VTK Lagrange cells of order $k$. |
| `cell_data` | dict of arrays | `None` | One value (or row) per cell. |
| `segments` | bool or `None` | `None` | Write the sides as line cells with their markers (`marker` cell data). `None`: only for the mesh alone. |
| `binary` | bool | `True` | Base64 binary arrays, or ASCII. |

**Raises** `TypeError` for a field that is not on a 2D space or when nothing is given, `ValueError`
for fields on different meshes, repeated names, or `cell_data` of the wrong length.

**`VTKSeries(filename, **options)`**, a `.pvd` collection. `write(t, *fields, **options)` writes the
next `stem_000000.vtu` beside the `.pvd` with `write_vtk` (options merged with those of the
constructor), adds it at time `t`, rewrites the `.pvd` and returns the file's path. `len(series)` is
the number of files, `series.entries` the `(t, file)` pairs.

**`read_gmsh(filename, *, marker=0, tol=1e-9)`** reads a Gmsh `.msh` file, ASCII format 4.1 or 2.2.

| argument | type | default | meaning |
|---|---|---|---|
| `filename` | str or path | required | The file. |
| `marker` | int | `0` | Marker of the boundary edges that no line element covers. |
| `tol` | float | `1e-9` | Relative spread in $z$ above which the mesh counts as not planar. |

**Output.** A `Mesh2D` (triangles) or `QuadMesh` (quadrilaterals) with these `report` keys.

| key | meaning |
|---|---|
| `physical_names` | `{name: tag}` of the 1D physical groups, the side markers |
| `region_names` | `{name: tag}` of the 2D physical groups |
| `cell_markers` | int array, one per cell: its physical tag, or its surface tag |
| `gmsh_version` | `"4.1"` or `"2.2"` |
| `order` | order of the elements in the file (1 for linear) |

**Raises** `ValueError` for a binary file, another format version, a file without nodes and
elements, 3D elements or other unsupported types, a mix of triangles and quadrilaterals, no 2D
cells, or a mesh that is not planar. A `RuntimeWarning` when higher-order nodes are dropped from
curved sides.

### L.13 Raviart-Thomas, Nédélec and broken $P_k$ on triangles

**`RTSpace(mesh, degree, bc="free", dirichlet=None, data=None)`**, **`N1curlSpace(mesh, degree, bc="free", dirichlet=None, data=None)`**
(`fd.FunctionSpace(mesh, "RT", k)`, `"N1curl"`, also `"raviart-thomas"`, `"nedelec"`, `"N1E"`).
$H(\mathrm{div})$ and $H(\mathrm{curl})$ vector elements of degree $k \ge 1$ in UFL's convention, on a
`Mesh2D` (manual, [Section 11.4](../manual.md#114-raviart-thomas-nédélec-and-broken-p_k)).

| argument | type | default | meaning |
|---|---|---|---|
| `mesh` | `Mesh2D` | required | Triangles only. A `QuadMesh` raises `TypeError`. |
| `degree` | int | required | $k \ge 1$. `RT1`, `N1curl1` are the lowest order. |
| `bc` | str | `"free"` | `"dirichlet"` builds the essential condition ($u\cdot n$ or $u\times n$) into every exterior side. |
| `dirichlet` | marker, list or dict | `None` | Sides with the essential condition. A dict `{marker: data}` gives the data too. |
| `data` | as `lift` | `None` | The essential data, used by `solve`, `newton` and `lift()` when the call gives none. |

| member | meaning |
|---|---|
| `dim`, `raw_dim`, `nloc`, `ncells`, `nedges`, `degree`, `family` | Sizes. `nloc` $= k(k+2)$. `family` is `"RT"` or `"N1curl"`. |
| `ncomp`, `piola`, `value_shape` | 2, 1 (contravariant) or 2 (covariant), `(2,)`. |
| `interpolate(f)` | Canonical interpolant of `f(x, y) -> (fx, fy)` (or a pair of callables or numbers): edge traces projected onto $P_{k-1}$, interior moments. Commutes with div (curl). Adapted coefficients. |
| `project(f, degree=None)` | $L^2$ projection of the same, or of a vector expression. |
| `mass_matrix()` | $\int u \cdot v$, cached. |
| `lift(g=None)` | A `VectorElementFunction` of `V.unconstrained` carrying the essential data: a vector field, a scalar $u\cdot n$ (outward) or $u\cdot t$ (the domain on the left, for N1curl) on exterior sides, or `{marker: ...}`. |
| `evaluate(c, x, y=None, what="value")` | `"value"` gives `(..., 2)`, `"x"`, `"y"`, `"div"`, `"curl"` one number per point. NaN outside. |
| `evaluate_cells(c, cells, xi, eta, what="value")` | The same at reference points of given cells. |
| `boundary_dofs(marker=None)` | Raw indices of the edge degrees of freedom on those sides. |
| `interpolation_points()`, `interpolate_raw(F)` | Sample points of the interpolant, and raw coefficients from `(n, 2)` values there. |
| `edge_projection()` | `(t, P)`: the edge rule on $[0, 1]$ and the matrix with `dof_j = sum_q P[j, q] f(t_q)`. |
| `cache`, `boundary_cache`, `pattern`, `at_quad`, `prolongate`, `restrict`, `unconstrained`, `dof_coordinates`, `cell_signs()` | As for `LagrangeSpace2D`. `at_quad(Q, c, code)` takes the code $3c + d$. |
| `plot(c, what="magnitude", quiver=True)`, `triangulation(c, refine, what)`, `info()` | Plotting and a summary. |

The derivative codes of a vector element are $3c + d$, component $c$ and $d = 0$ (value), 1
($\partial_x$), 2 ($\partial_y$). `div` is codes 1 + 5 and `curl` codes 4 − 2.

**`VectorElementFunction`**, what `fd.Function(V)` returns on these spaces. A vector expression
(`div(w)`, `curl(w)`, `dot(w, v)`, `w[0]`) that carries `vector`, `space`, `name`, with
`interpolate`, `project`, `assign`, `copy`, `at(x, y=None, what="value")` and `plot`.

**`curl(e)`**, `rot(e)`. Of a 2D vector, the scalar $\partial_x e_y - \partial_y e_x$. Of a scalar, the
vector $(\partial_y e, -\partial_x e)$.

**`DGSpace2D(mesh, degree, nodes="equispaced", periodic=None)`** (`fd.DGSpace(mesh2d, k)`, `fd.FunctionSpace(mesh, "DG", k)`).
Broken $P_k$, $k \ge 0$: the nodal basis of `LagrangeSpace2D` on every triangle, nothing shared or
built in, `dim` $= n_c (k+1)(k+2)/2$. The members of `LagrangeSpace2D` apply (`interpolate`,
`project`, `mass_matrix`, `evaluate`, `plot`, `cache`, ...). Forms couple the triangles through
`dS` (manual, [Section 12.9](../manual.md#129-discontinuous-galerkin-on-triangles)).
`periodic=[(a, b), ...]` joins side $b$ to side $a$, edge for edge, as interior facets.

| member | meaning |
|---|---|
| `interior_facets()` | `(cells (nf, 2), local edges (nf, 2), edge, periodic, shift (nf, 2))`, the `'-'` side first. Every 2D space has it. |
| `n_periodic_facets`, `periodic_pairs`, `periodic_info`, `vertex_representatives` | The periodic facets, the pairs, their shifts, and the vertex classes joined across them. |
| `cell_means(c)` | The mean over every triangle. |

**`InteriorFacetCache2D(space, degree)`** (in `femd._femd`), the points of the `dS` measure: `nf`,
`nq`, `x()`, `y()`, `weights()`, `normal_x()`, `normal_y()` (out of the `'-'` cell), `cells()` and
`at_points(c, side, code)`, side 0 for `'-'` and 1 for `'+'`. `facet_pattern(T, toff, nrows, S, soff,
ncols)` is the pattern of a form with `dS` terms, and `assemble_facet_{scalar,vector,matrix}_2d`
the kernels, each term with its sides.

**In systems.** `ProductSpace(V, Q)` takes these spaces as blocks. An RT or N1curl block unpacks as a
vector, `w.split()` gives a `VectorElementFunction` for it, and `lift`, `interpolate`, `project`,
`mass_matrix`, `plot` and `write_vtk` handle it.

**Raises** `TypeError` for a mesh that is not a `Mesh2D`, `ValueError` for a degree below 1 (0 for DG), an
unknown marker, or `data=` on a space with no essential side.
