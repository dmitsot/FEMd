# 2D meshing

[Manual](../manual.md) · [API reference](index.md)

## J. 2D meshing

Separate from the 1D library. It meshes polygons, builds structured meshes,
and refines, coarsens and smooths the result ([manual, Section 10](../manual.md#10-two-dimensional-meshes)).

| name | kind | purpose |
|---|---|---|
| `triangulate(boundary, holes=None, ...)` | function | quality-guaranteed Delaunay triangulation of a polygonal region |
| `circle(cx, cy, r, n=32, clockwise=False)` | function | points on a circle, for holes |
| `rectangle_mesh(x0, y0, x1, y1, nx, ny, ...)` | function | structured mesh of a rectangle |
| `mapped_mesh(bottom, right, top, left, ...)` | function | structured mesh of a curved four-sided region |
| `mesh_from_arrays(points, triangles, ...)` | function | a `Mesh2D` from any point and triangle arrays |
| `Mesh2D` | class | a triangular mesh, arrays and topology |
| `smooth(mesh, passes=1, ...)` | function | Laplacian smoothing of an existing mesh |
| `restore_delaunay(mesh, max_sweeps=64)` | function | flip edges back to constrained Delaunay |
| `split_triangles(mesh, triangles)` | function | barycentric $1 \to 3$ split |
| `bisect_triangles(mesh, triangles, levels=1)` | function | Rivara longest-edge bisection |
| `refine_triangles(mesh, triangles, ...)` | function | circumcenter insertion with the quality guarantee |
| `remove_vertices(mesh, vertices, ...)` | function | coarsening: take vertices out and refill the holes |
| `Domain` | class | the input region, for geometric queries |
| `orient2d`, `incircle` | function | exact geometric predicates |

Every function that takes a mesh returns a **new** `Mesh2D` and leaves the one
passed in untouched, since the arrays of a mesh are borrowed views.

### J.1 `triangulate(boundary, holes=None, *, min_angle=20.0, max_area=None, max_edge=None, smooth=0, marker=1, hole_markers=None, vertex_markers=None, interior_vertex=False, max_points=500_000, check=False)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `boundary` | `(n, 2)` array_like, $n \ge 3$ | required | The outer polygon, one row per vertex, **first vertex not repeated at the end**. Either winding, re-oriented counter-clockwise internally. |
| `holes` | sequence of `(n, 2)` array_like | `None` | Regions to cut out, either winding. They must not touch or cross each other or the outer boundary. |
| `min_angle` | float, degrees | `20.0` | Smallest interior angle met by every triangle. `0` turns the shape criterion off. See the ceilings below. |
| `max_area` | float or `None` | `None` | Largest triangle area. `None` is unbounded. |
| `max_edge` | float or `None` | `None` | Longest edge. `None` is unbounded. Independent of `max_area`. |
| `smooth` | int | `0` | Laplacian smoothing passes after refinement, as `smooth()` with `restore_delaunay=True`, with the moves also kept within `max_area` and `max_edge` and a final refinement pass, so every bound still holds ([manual, Section 10.1](../manual.md#101-smoothing-a-mesh-you-already-have)). |
| `marker` | int | `1` | Marker of the outer boundary's vertices and segments. `0` is reserved for interior vertices. |
| `hole_markers` | sequence of int | `None` | One marker per hole, default `marker+1, marker+2, ...`. |
| `vertex_markers` | `(n,)` array_like of int | `None` | One marker per outer-boundary vertex, overriding `marker`. Vertices inserted on a segment inherit the marker of the segment's ring. |
| `interior_vertex` | bool | `False` | Require every triangle to have a vertex off the boundary, so no element has all three vertices constrained. Halves the reachable `min_angle`. |
| `max_points` | int | `500_000` | Vertex budget. Refinement stops with `RuntimeError` past it. |
| `check` | bool | `False` | Run `Mesh2D.validate()` before returning and raise if it reports a fault. |

**Output.** A `Mesh2D`, with `report` filled in ([J.3](#j3-mesh2d)).

**Ceilings on `min_angle`**, checked before any work is done.

- $34^\circ$ or more raises `ValueError`. Ruppert's proof covers $20.7^\circ$ and
  the algorithm reaches about $33^\circ$ in practice.
- More than the sharpest corner of the input raises `ValueError` naming that
  corner, since a triangle filling a corner of angle $\alpha$ cannot have a
  larger smallest angle.
- With `interior_vertex=True`, more than **half** the sharpest corner raises
  `ValueError`, since two triangles must then share that corner.

**Raises.** `ValueError` for a ring with fewer than 3 vertices, a wrong array
shape, a mismatched marker count or an unreachable `min_angle`. `RuntimeError`
when the point budget runs out, with a message naming the criteria still
failing, and when refinement would split a boundary segment below the feature
floor ($10^{-9}$ of the domain diagonal).

### J.2 `circle(cx, cy, r, n=32, clockwise=False)`

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `cx`, `cy` | float | required | The center. |
| `r` | float | required | The radius. |
| `n` | int | `32` | Number of points. |
| `clockwise` | bool | `False` | Direction. Irrelevant to `triangulate`, which re-orients every ring. |

**Output.** An `(n, 2)` array of points evenly spaced on the circle, starting
at `(cx + r, cy)`. The result is a polygon, so the mesh is exact for an $n$-gon
of slightly smaller area, and its $n$ edges put a floor under the element size
near the hole. A 40-gon of radius $0.32$ has area $0.32038$ against the disc's
$0.32170$ and edges about $0.05$ long.

### J.3 `Mesh2D`

Returned by `triangulate`, the structured generators and the functions that take a mesh.
Not constructed directly.
The array attributes are **views borrowed from the C++ mesh**. They cost
nothing to take and must not be written to, so `.copy()` one to edit it.

**Arrays and sizes**

| attribute | type | meaning |
|---|---|---|
| `points` | `(npoints, 2)` float64 | Vertex coordinates. |
| `triangles` | `(ntriangles, 3)` int32 | Vertex indices, every triangle counter-clockwise. |
| `neighbours` | `(ntriangles, 3)` int32 | `[t, i]` is the triangle across the edge opposite local vertex `i` of `t`, or $-1$ on the boundary. |
| `segments` | `(nsegments, 2)` int32 | The constrained boundary edges. |
| `point_markers` | `(npoints,)` int32 | 0 for interior vertices, otherwise the marker of the boundary ring. |
| `segment_markers` | `(nsegments,)` int32 | The marker of each segment. |
| `npoints`, `ntriangles`, `nsegments` | int | Sizes. |
| `report` | dict | What produced the mesh (below). |

**Methods**

| method | inputs | returns | meaning |
|---|---|---|---|
| `areas()` | none | `(ntriangles,)` float64 | Triangle areas, positive. |
| `angles()` | none | `(ntriangles,)` float64 | The smallest angle of each triangle, **in degrees**. |
| `qualities()` | none | `(ntriangles,)` float64 | $4\sqrt3 A/(a^2+b^2+c^2)$, 1 for equilateral and 0 for degenerate. |
| `boundary_edge_counts()` | none | `(ntriangles,)` int32 | How many edges of each triangle lie on the boundary, 0 to 3. A 2 is a triangle filling a corner. |
| `has_interior_vertex()` | none | `(ntriangles,)` int32 | 1 where the triangle has a vertex off the boundary, 0 where every vertex is constrained. |
| `boundary_only_triangles()` | none | int array | Indices of the triangles with all three vertices on the boundary. |
| `locate(points, *, barycentric=False)` | `points` an `(x, y)` pair or an `(n, 2)` array_like. `barycentric` bool, also return barycentric coordinates. | int for one point, `(n,)` int32 array otherwise, $-1$ outside the mesh or in a hole. With `barycentric=True` also a `(3,)` or `(n, 3)` float64 array, relative to the vertices of `triangles[t]` in order, NaN outside. | The triangle containing each point. See the note below the table. |
| `boundary_vertices(marker=None)` | `marker` int or `None` | int array | Indices of boundary vertices, only those with `marker` when given. |
| `boundary_edges(marker=None)` | `marker` int or `None` | `(k, 2)` int array | Boundary edges, only those with `marker` when given. A copy. |
| `remark(rule)` | `rule(x, y, marker)` | `Mesh2D` | A copy with new side markers from a rule on the side midpoints ([L.9](elements-2d.md#l9-systems-on-2d-meshes)). |
| `validate()` | none | str | `""` when the mesh is sound, otherwise the first fault found. Checks orientation, neighbor symmetry, agreement of shared edges and that every input segment is an edge. |
| `triangulation()` | none | `matplotlib.tri.Triangulation` | For plotting and interpolation with matplotlib. |
| `plot(ax=None, **kw)` | `ax` a matplotlib axes, a new figure when `None`. `**kw` keywords passed to `triplot`. | the axes | Draws the mesh with equal aspect. |

**Point location.** `locate` walks through the mesh from its previous answer,
which is fast for points that follow one another in space. A walk that runs into
the boundary (the point is outside, in a hole, or across a hole or a reentrant
corner) builds a bucket grid over the mesh once, in $O(n_t)$, and the grid then
answers every query in $O(1)$. A batch of at least 1000 points and at least an
eighth of the triangle count builds the grid up front. The mesh keeps its
locator, so later calls reuse it. Containment uses the exact `orient2d`, so a
point on an edge or a vertex is reported in one of the triangles containing it,
and the rounded midpoint of a slanted boundary edge may rightly come out as
$-1$. On a 684 000-triangle mesh, 200 000 points take 0.04 to 0.07 s in any
order, including points in a hole or outside. With `barycentric=True`,
`np.einsum("ij,ij->i", lam, u[T[t]])` evaluates a $P_1$ nodal vector `u` at the
points.

**`report` keys**

| key | set by | meaning |
|---|---|---|
| `input_vertices` | `triangulate` | Vertices in the input rings. |
| `boundary_splits` | `triangulate` | Midpoints added to make the boundary conform. |
| `refine_inserts` | `triangulate` | Points inserted by quality refinement. |
| `smoothed` | `triangulate`, `smooth` | Smoothing moves accepted. |
| `min_angle`, `min_quality`, `max_area`, `total_area` | `triangulate` | Quality and size of the result. |
| `corner_flips`, `corner_splits`, `corner_centres`, `boundary_only`, `two_boundary_edges` | `triangulate` | Corner treatment counts, meaningful with `interior_vertex=True`. |
| `flips` | `restore_delaunay` | Edges flipped, 0 when the mesh already was Delaunay. |
| `structured` | `rectangle_mesh`, `mapped_mesh` | `(nx, ny, diagonal)`. |
| `flipped`, `boundary_edges`, `boundary_loops` | `mesh_from_arrays` | Triangles re-oriented, boundary edges found, and closed boundary loops (1 plus the number of holes). |
| `removed`, `vertex_map`, `kept`, `kept_reasons` | `remove_vertices` | See [J.12](#j12-remove_verticesmesh-vertices--min_angle00-max_edgenone-max_areanone-boundarytrue). |
| `local_refine` | the three refinement functions | `(scheme_name, vertices_added)`, with `scheme_name` one of `"barycentric"`, `"rivara"`, `"circumcentre"`. |

A function that takes a mesh copies the input's `report` and updates its own
keys.

### J.4 `smooth(mesh, passes=1, *, restore_delaunay=True)`

Moves every interior vertex to the centroid of its one-ring, accepting a move
only if every incident triangle keeps positive area and neither the worst
quality nor the worst angle of the fan gets worse. Boundary vertices never move.

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `mesh` | `Mesh2D` | required | The mesh to smooth. |
| `passes` | int | `1` | Sweeps. `passes <= 0` returns the mesh unchanged. Three to five is the useful range. |
| `restore_delaunay` | bool | `True` | Flip back to Delaunay after each pass. Set `False` after `bisect_triangles` or `split_triangles` to keep their structure. |

**Output.** A new `Mesh2D` with the same numbering of points and triangles.
`report["smoothed"]` counts accepted moves and does not fall to zero once the
mesh has settled, so compare `qualities()` for a convergence test.

### J.5 `restore_delaunay(mesh, max_sweeps=64)`

Flips edges until the mesh is constrained Delaunay again. Segments are never
flipped and no point is added or moved.

| argument | type | default | meaning |
|---|---|---|---|
| `mesh` | `Mesh2D` | required | The mesh. |
| `max_sweeps` | int | `64` | Limit on flipping sweeps over the mesh. |

**Output.** A new `Mesh2D` with the same points in the same order, and
`report["flips"]` set.

### J.6 Local refinement

`split_triangles(mesh, triangles)`, `bisect_triangles(mesh, triangles, levels=1)`,
`refine_triangles(mesh, triangles, *, min_angle=20.0, max_area=None, max_edge=None, max_points=500_000)`.

**Inputs**

| argument | used by | type | default | meaning |
|---|---|---|---|---|
| `mesh` | all | `Mesh2D` | required | The mesh to refine. |
| `triangles` | all | list of int, or bool mask of length `ntriangles` | required | The triangles to refine. An index out of range or a mask of the wrong length raises `ValueError`. An empty selection is a no-op. |
| `levels` | `bisect_triangles` | int | `1` | Passes of bisection over the same places. |
| `min_angle` | `refine_triangles` | float, degrees | `20.0` | Angle bound enforced over the whole mesh after insertion. |
| `max_area`, `max_edge` | `refine_triangles` | float or `None` | `None` | Size bounds enforced over the whole mesh. |
| `max_points` | `refine_triangles` | int | `500_000` | Vertex budget. |

**Output.** A new `Mesh2D`. Vertex numbering extends the input's, so a vector
indexed by vertex can be extended rather than re-projected.
`report["local_refine"]` is `(scheme_name, vertices_added)`.

| function | scheme | conforming | Delaunay | min angle | element diameter |
|---|---|---|---|---|---|
| `split_triangles` | barycentric $1\to3$ | on its own | no | falls about $3\times$ per level | **unchanged** |
| `bisect_triangles` | Rivara longest-edge | through propagation, which can refine more than named | no | at least half the starting minimum | shrinks, count grows about $2\times$ per level |
| `refine_triangles` | circumcenter (Ruppert) | yes | yes | meets `min_angle` | shrinks fastest, refines most |

`split_triangles` is for splitting one element, not for an adaptive loop (Part
I, Section 10.2).

### J.7 `Domain()`

The input region of the mesher. `triangulate` builds one from its arrays and
does not accept a `Domain`, so in Python it serves for geometric queries on a
region.

| member | inputs | returns | meaning |
|---|---|---|---|
| `Domain()` | none | `Domain` | An empty region. |
| `boundary(points, vertex_markers=[], marker=1)` | `points` a C-contiguous `(n, 2)` float64 array. `vertex_markers` list of int, one per point, default `marker` for all. `marker` int, the ring's marker. | the `Domain` | Sets the outer ring. Returns itself, for chaining. |
| `hole(points, vertex_markers=[], marker=2)` | as `boundary` | the `Domain` | Adds a hole. |
| `contains(x, y)` | `x`, `y` float | bool | Whether the point is inside the outer ring and outside every hole. |
| `bbox()` | none | `(xmin, ymin, xmax, ymax)` | The bounding box. |
| `nrings`, `nholes`, `nvertices` | attribute | int | Counts. |
| `smallest_input_angle` | attribute | float, degrees | The sharpest corner of the input, which caps `min_angle`. |

### J.8 `orient2d(ax, ay, bx, by, cx, cy)` and `incircle(ax, ay, bx, by, cx, cy, dx, dy)`

The exact predicates the mesher is built on. They are evaluated in double
precision with an error bound and fall back to double-double arithmetic when
the bound is not cleared, so the sign is exact.

| function | inputs | returns |
|---|---|---|
| `orient2d` | the coordinates of points $a$, $b$, $c$, floats | $+1$ when $abc$ is counter-clockwise, $-1$ clockwise, $0$ exactly collinear |
| `incircle` | the coordinates of $a$, $b$, $c$ (counter-clockwise) and $d$, floats | $+1$ when $d$ is strictly inside the circle through $a$, $b$, $c$, $-1$ outside, $0$ cocircular |

### J.9 `rectangle_mesh(x0, y0, x1, y1, nx, ny, *, diagonal="right", markers=(1, 2, 3, 4))`

Structured mesh of a rectangle with $n_x \times n_y$ equal cells ([manual,
Section 10.3](../manual.md#103-structured-meshes)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `x0`, `y0` | float | required | The lower-left corner. |
| `x1`, `y1` | float | required | The upper-right corner, `x1 > x0` and `y1 > y0`. |
| `nx`, `ny` | int | required | Cells along $x$ and along $y$, at least 1. |
| `diagonal` | str | `"right"` | How each cell is split. `"right"`, the `/` diagonal. `"left"`, the `\` diagonal. `"alternate"`, `/` where $i + j$ is even and `\` where it is odd. `"crossed"` (also `"crisscross"`), a vertex at the cell center and four triangles. Case-insensitive. |
| `markers` | int or 4 ints | `(1, 2, 3, 4)` | Markers of the bottom, right, top and left sides. One int gives all four the same marker. |

**Output.** A `Mesh2D` with $2 n_x n_y$ triangles ($4 n_x n_y$ for `"crossed"`)
and $(n_x+1)(n_y+1)$ grid vertices, plus $n_x n_y$ centers for `"crossed"`.

| numbering | rule |
|---|---|
| grid vertex $(i, j)$ | point `j*(nx+1) + i`, $0 \le i \le n_x$, $0 \le j \le n_y$, coordinates exactly $x_0 + (x_1-x_0)\,i/n_x$ and $y_0 + (y_1-y_0)\,j/n_y$ |
| center of cell $(i, j)$ | point `(nx+1)*(ny+1) + j*nx + i` (`"crossed"` only) |
| triangles | cell by cell in the same order, 2 or 4 per cell |
| segment markers | the marker of the side |
| point markers | 0 inside. A boundary vertex takes the marker of the side leaving it counter-clockwise, so the corners $(x_0, y_0)$, $(x_1, y_0)$, $(x_1, y_1)$, $(x_0, y_1)$ carry the bottom, right, top and left markers. |

`report["structured"]` is `(nx, ny, diagonal)`. **Raises** `ValueError` for
`x1 <= x0`, `y1 <= y0`, a count below 1, an unknown diagonal or a wrong number
of markers, and `TypeError` for a non-integer count.

### J.10 `mapped_mesh(bottom, right, top, left, nx=None, ny=None, *, diagonal="right", markers=(1, 2, 3, 4))`

Structured mesh of a four-sided region by transfinite (Coons) interpolation.
Grid vertex $(i, j)$ is placed at $X(i/n_x, j/n_y)$ with

$$X(\xi, \eta) = (1-\eta) B(\xi) + \eta\, T(\xi) + (1-\xi) L(\eta) + \xi R(\eta) - \left[(1-\xi)(1-\eta) P_{00} + \xi(1-\eta) P_{10} + (1-\xi)\eta P_{01} + \xi\eta P_{11}\right].$$

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `bottom` | callable, `(m, 2)` array_like, or two points | required | $B$, from $P_{00}$ to $P_{10}$. |
| `right` | as `bottom` | required | $R$, from $P_{10}$ to $P_{11}$. |
| `top` | as `bottom` | required | $T$, from $P_{01}$ to $P_{11}$. |
| `left` | as `bottom` | required | $L$, from $P_{00}$ to $P_{01}$. |
| `nx` | int or `None` | `None` | Cells along bottom and top. Read from a side given as an array of more than two points, required otherwise. |
| `ny` | int or `None` | `None` | Cells along left and right, likewise. |
| `diagonal`, `markers` | | | As in `rectangle_mesh`. |

A side may be given in three ways.

- A **callable** $s \mapsto (x, y)$ for $s \in [0, 1]$, sampled at $s = k/n$. It
  may be vectorized (an `(n+1, 2)` or `(2, n+1)` result for an array `s`) or
  take one float at a time.
- An **array** of $n + 1$ points, used as they are. Uneven spacing is kept.
- **Two points**, a straight side sampled evenly.

A side given the other way round, as a counter-clockwise walk gives `top` and
`left`, is detected at the corners and reversed. The corners must agree to
$10^{-10}$ of the size of the region and are then made identical.

**Output.** A `Mesh2D` numbered and marked as in `rectangle_mesh`. The
boundary vertices are the side samples exactly. Straight sides give the bilinear
map, so a rectangle is reproduced as the tensor grid of its samples. A mirrored
parameterization (every triangle clockwise) is re-oriented.
`report["structured"]` is `(nx, ny, diagonal)`.

**Raises** `ValueError` when the sides do not meet at four corners (the message
lists where each side starts and ends), when opposite sides have different
counts, when a count is missing, or when the map is not one-to-one. The last
names how many triangles are inverted or degenerate and the first bad cell. The
cause is crossing sides or cells too distorted for the diagonal.

### J.11 `mesh_from_arrays(points, triangles, segments=None, segment_markers=None, point_markers=None, *, marker=1, check=True)`

A `Mesh2D` from any triangulation given as arrays ([manual, Section 10.3](../manual.md#103-structured-meshes)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `points` | `(npoints, 2)` array_like of float | required | Vertex coordinates. Every point must belong to some triangle. |
| `triangles` | `(ntriangles, 3)` array_like of int | required | Vertex indices, 0-based, either orientation. Triangle $t$ of the result is triangle $t$ of the input, re-oriented counter-clockwise if needed. |
| `segments` | `(k, 2)` array_like of int | `None` | Edges to constrain with their own markers. Each must be an edge of the mesh, boundary or interior. |
| `segment_markers` | `(k,)` array_like of int | `None` | One marker per segment, default `marker`. |
| `point_markers` | `(npoints,)` array_like of int | `None` | One marker per point. `None` gives 0 inside and, to a boundary vertex, the marker of the boundary edge leaving it counter-clockwise. A vertex only on interior segments takes the marker of such a segment. |
| `marker` | int | `1` | Marker of the boundary edges not listed in `segments`. Every boundary edge becomes a segment. |
| `check` | bool | `True` | Run `validate()` on the result and raise if it reports a fault. |

**Output.** A `Mesh2D` with its neighbor links built. `report` has
`flipped`, `boundary_edges` and `boundary_loops`.

**Raises** `ValueError` for a wrong array shape, a vertex index out of range, a
triangle that repeats a vertex or has collinear vertices, a point used by no
triangle, an edge shared by more than two triangles, two triangles on the same
side of a shared edge (the input overlaps or folds there), a segment that is not
an edge, or a marker array of the wrong length.

### J.12 `remove_vertices(mesh, vertices, *, min_angle=0.0, max_edge=None, max_area=None, boundary=True)`

Removes vertices and refills each hole by Delaunay ear clipping on the
surrounding vertices ([manual, Section 10.4](../manual.md#104-removing-vertices)).

**Inputs**

| argument | type | default | meaning |
|---|---|---|---|
| `mesh` | `Mesh2D` | required | The mesh. Not modified. |
| `vertices` | list of int, or bool mask of length `npoints` | required | The vertices to remove. Duplicates are ignored, an empty selection is a no-op. |
| `min_angle` | float, degrees | `0.0` | A removal whose new triangles would have a smaller angle is refused. `0` refuses only inverted or degenerate triangles, which are never made. |
| `max_edge` | float or `None` | `None` | A removal that would create a longer edge is refused. |
| `max_area` | float or `None` | `None` | A removal that would create a larger triangle is refused. |
| `boundary` | bool | `True` | Allow removing boundary vertices. Only a vertex inside a straight side can go (see below). |

A vertex is kept, with the reason recorded, when:

- it is an interior vertex at the end of an interior segment,
- it is a boundary vertex at a corner, where more than two segments meet, where
  its two segments carry different markers, or whose own marker differs from
  its side's and from its neighbors' along the side (a straight side is
  straight to $10^{-10}$ of the neighbors' distance),
- the fill would need an edge the mesh already has outside the hole,
- a new triangle would break `min_angle`, `max_edge` or `max_area`.

The list is swept repeatedly until a sweep removes nothing, since removing one
vertex can make another removable.

**Output.** A new `Mesh2D`. The removed points are dropped and the others keep
their coordinates and their relative order. For a removed boundary vertex its
two segments become one, with their marker.

| `report` key | type | meaning |
|---|---|---|
| `removed` | int | Vertices removed. |
| `vertex_map` | int array, length of the old `npoints` | New index of each old vertex, $-1$ for a removed one. `u[vertex_map >= 0]` carries a nodal vector over. |
| `kept` | int array | Requested vertices that are still there, as old indices. |
| `kept_reasons` | list of str | One reason per entry of `kept`. |

**Raises** `ValueError` for an index out of range or a mask of the wrong length.
