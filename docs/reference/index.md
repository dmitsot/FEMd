# FEMd API reference

[Manual](../manual.md) · [README](../../README.md)

All names are in `femd`, imported as `fd`. The reference is organized by
category, one or two categories per page (listed at the end). Each category opens with a catalog of what it contains, followed by
one entry per function or object. An entry gives the signature, what the call
does, a table of its inputs and a table of its outputs. A class entry lists its
constructor arguments and then its members, with the inputs and outputs of each
method.

**Conventions.**

- Arrays are NumPy `float64` unless stated otherwise, and indices are 0-based.
- "array_like" means anything `np.asarray` accepts, such as a list, a tuple or an
  array.
- A default of "required" means the argument has no default.
- Arguments after `*` in a signature are keyword-only.
- $V$ is a one-dimensional space, `V.dim` its number of degrees of freedom, $p$
  its degree and $n_e$ its number of elements, except in [Section L](elements-2d.md#l-finite-elements-on-triangles), where it is
  a `LagrangeSpace2D`.
- "Adapted" coefficients are those of `V`, with the built-in boundary conditions
  eliminated. "Raw" coefficients are those of `V.unconstrained`, the same basis
  with nothing built in.

**Categories.**

| | category | contents |
|---|---|---|
| A | [Spaces and 1D meshes](spaces.md#a-spaces-and-1d-meshes) | `SplineSpace`, `LagrangeSpace`, `ProductSpace`, `Mesh1D` |
| B | [Boundary conditions](spaces.md#b-boundary-conditions) | the `bc=` argument, `BoundaryCondition`, `BCSpec`, `Robin`, `Derivative` |
| C | [Functions and transfer between spaces](functions.md#c-functions-and-transfer-between-spaces) | `Function`, `ProductFunction`, `Functions`, `Transfer`, `mixed_mass` |
| D | [Form language](forms.md#d-form-language) | `TrialFunction(s)`, `TestFunction(s)`, `D`, `x`, pointwise functions, `dx`, `ds`, `form`, `Form` |
| E | [Matrices](linalg.md#e-matrices) | `Matrix`, `RectMatrix`, `block`, `MatrixInfo` |
| F | [Linear solvers](linalg.md#f-linear-solvers) | `LinearSolver`, `Solver`, `solve`, `gmres`, `lgmres`, `KrylovInfo`, `eigs`, `EigenResult` |
| G | [Nonlinear solvers and time stepping](time-stepping.md#g-nonlinear-solvers-and-time-stepping) | `newton`, `NewtonInfo`, `newton_system`, `csnewton`, `CSNewtonInfo`, `IRK`, `butcher` |
| H | [Low-level assembly](forms.md#h-low-level-assembly) | `QuadratureCache`, `assemble`, `assemble_vector`, `assemble_block_vector`, `assemble_scalar` |
| I | [Build and threading](build-cpp.md#i-build-and-threading) | `__version__`, `has_fftw`, `has_openmp`, `set_num_threads`, `get_num_threads`, `omp_threshold` |
| J | [2D meshing](meshing-2d.md#j-2d-meshing) | `triangulate`, `circle`, `Mesh2D`, `smooth`, `restore_delaunay`, `split_triangles`, `bisect_triangles`, `refine_triangles`, `Domain`, `orient2d`, `incircle` |
| K | [C++ core](build-cpp.md#k-c-core) | the header-only library underneath |
| L | [Finite elements on triangles and quadrilaterals](elements-2d.md#l-finite-elements-on-triangles) | `LagrangeSpace` on a `Mesh2D` or `QuadMesh`, `LagrangeSpace2D`, `LagrangeSpaceQ`, `QuadMesh`, `quadrangulate`, `VectorFunctionSpace`, `ProductSpace2D`, `grad`, `div`, `dot`, `Dx`, `FacetNormal`, `ds(marker)`, `SparseMatrix`, `SparseSolver`, `Preconditioner`, `cg`, `write_vtk`, `VTKSeries`, `read_gmsh`, `RTSpace`, `N1curlSpace`, `DGSpace2D`, `VectorElementFunction`, `curl` |

**Pages.**

- [Spaces and boundary conditions](spaces.md) (A, B)
- [Functions and transfer between spaces](functions.md) (C)
- [Form language and low-level assembly](forms.md) (D, H)
- [Matrices and linear solvers](linalg.md) (E, F)
- [Nonlinear solvers and time stepping](time-stepping.md) (G)
- [2D meshing](meshing-2d.md) (J)
- [Finite elements on triangles and quadrilaterals](elements-2d.md) (L)
- [Build, threading and the C++ core](build-cpp.md) (I, K)
