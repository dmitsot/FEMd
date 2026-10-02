# Build, threading and the C++ core

[Manual](../manual.md) · [API reference](index.md)

## I. Build and threading

| function | inputs | returns | meaning |
|---|---|---|---|
| `has_fftw()` | none | bool | Whether the build links FFTW, which `Solver.Circulant` needs. |
| `has_openmp()` | none | bool | Whether the build uses OpenMP ([manual, Section 1.1](../manual.md#11-openmp)). |
| `set_num_threads(k)` | `k` int $\ge 1$ | `None` | Threads for the OpenMP loops. A no-op without OpenMP. |
| `get_num_threads()` | none | int | Threads the OpenMP loops will use, 1 without OpenMP. |
| `omp_threshold()` | none | int | Loops go parallel above this size (elements times quadrature points, points, or rows). |

---

## K. C++ core

The Python layer is thin over `include/femd/`, header-only C++17. The
corresponding types are `Mesh1D`, `KnotVector`, `SplineSpace`, `LagrangeSpace`
(both `FunctionSpace`), `BoundaryCondition`, `BCSpec`, `ConstraintOperator`,
`QuadratureCache`, `Field<Scalar>`, `ProductSpace`, `AssembledMatrix`, the
kernels `assemble_matrix` / `assemble_vector` / `assemble_scalar`, and
`LinearSolver` with the `Solver` enum. For the mesher: `Point`, `Domain`,
`Mesh2D`, `MeshOptions`, `MeshReport` and `triangulate` in `mesh/` and
`triangulate/`, with `RefineOptions` / `refine`, `laplacian_smooth` and
`restore_delaunay` underneath, and the predicates in `pred::`. `Mesh2D` also
exposes the topology primitives an adaptive loop needs -- `insert_point`,
`split_edge_at`, `split_segment_at`, `flip_edge`, `legalize_around`,
`one_ring`, `edge_triangles`, plus `split_triangle` and `split_edge_midpoint`,
and `triangulate/local_refine.hpp` for the three schemes above.
`fe/dg_space.hpp` holds `DGSpace` (a `FunctionSpace` family whose raw basis is the
adapted one, degree 0 allowed through the `min_degree` argument of the base
constructor, the facet coupling added with `widen_uband`), and
`fe/facet_cache.hpp` holds `FacetCache` (the traces of a space on both sides of
every interior facet) with the kernels `assemble_facet_matrix`,
`facet_triplets`, `assemble_facet_vector` and `facet_values`.
`FunctionSpace::broken()` is true for `DGSpace`, and `ProductSpace` then adds the
facet coupling of neighboring elements to its half-bandwidth.
`FunctionSpace::raw_eval_ref(e, xi, ...)` evaluates at a reference coordinate
(the quadrature and facet caches use it, so $\xi$ is never rounded through $x$).
On the Python side the package is split into `spaces`, `linalg`, `forms`,
`mesh2d`, `transfer`, `krylov`, `newton`, `timestep` (with `IRK` and `SSPRK`) and
`limiters`, all re-exported from `femd`.
`mesh/structured.hpp` holds `mesh_from_arrays`, `rectangle_mesh`, `mapped_mesh`
and `structured_grid_mesh` (the triangulation of a grid you computed yourself),
with the `Diagonal` enum. `triangulate/coarsen.hpp` holds `remove_vertex` and
`remove_vertices` with `RemoveOptions` and `RemoveReport`. They work in place and
leave the removed points unreferenced until `compact()`, whose return value is
the vertex map. `Mesh2D::merge_segments_at(v)` joins the two segments at a
boundary vertex. Point location: `Mesh2D::walk(p, hint)` is the straight walk
alone (it returns $-1$ when inconclusive), `Mesh2D::locate(p, hint)` adds a scan
when the walk fails, and `PointLocator` in `mesh/point_locator.hpp` adds the
bucket grid for many queries. The C++ versions
refine **in place**; only the Python wrappers copy. `Mesh2D::compact(false)`
squeezes out dead triangle slots while leaving point numbering alone. New fields
go at the END of `MeshOptions`:
callers brace-initialize it positionally, so inserting one in the middle
silently changes what every existing call means. The solvers in `linalg/`, `fft/` and
`iterative/` were inherited from Poseidon and are owned here. `femd.hpp` is
the umbrella header.
