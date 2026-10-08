"""
femd.mesh2d -- two-dimensional triangular meshes: Delaunay meshing, structured meshes,
meshes from arrays, local refinement, vertex removal, smoothing and point location.
"""
from __future__ import annotations

import numpy as np

from . import _femd as _C
from ._femd import Domain
from .spaces import _integer

# --------------------------------------------------------------------------- 2D meshing
class Mesh2D:
    """
    A 2D triangular mesh: arrays first, topology when you want it.

        m = fd.triangulate([(0,0), (1,0), (1,1), (0,1)], min_angle=30, max_area=1e-3)
        m.points        # (npoints, 2) float64
        m.triangles     # (ntriangles, 3) int32, counter-clockwise
        m.point_markers # (npoints,) int32; 0 interior, otherwise the ring marker

    The arrays are views onto the C++ mesh that owns them, so they cost nothing
    to take and must not be written to.  Call ``.copy()`` on one if you want to
    edit it.
    """

    __slots__ = ("_m", "report", "_loc")

    def __init__(self, core, report=None):
        self._m = core
        self.report = dict(report or {})
        self._loc = None                     # PointLocator, made on the first locate()

    # ---- arrays -------------------------------------------------------------
    @property
    def points(self):
        """(npoints, 2) float64 view of the vertex coordinates."""
        return self._m.points

    @property
    def triangles(self):
        """(ntriangles, 3) int32 view; every triangle is counter-clockwise."""
        return self._m.triangles

    @property
    def neighbours(self):
        """(ntriangles, 3) int32 view; [t, i] is the triangle across the edge
        opposite local vertex i of triangle t, or -1 on the boundary."""
        return self._m.neighbours

    @property
    def segments(self):
        """(nsegments, 2) int32 view of the constrained boundary edges."""
        return self._m.segments

    @property
    def point_markers(self):
        """(npoints,) int32 view; 0 for interior vertices, otherwise the marker
        of the boundary ring the vertex came from."""
        return self._m.point_markers

    @property
    def segment_markers(self):
        """(nsegments,) int32 view."""
        return self._m.segment_markers

    # ---- sizes --------------------------------------------------------------
    @property
    def npoints(self):
        return self._m.npoints

    @property
    def ntriangles(self):
        return self._m.ntriangles

    @property
    def nsegments(self):
        return self._m.nsegments

    # ---- per-triangle quantities --------------------------------------------
    def areas(self):
        """(ntriangles,) float64, positive."""
        return self._m.areas()

    def angles(self):
        """(ntriangles,) float64: the smallest interior angle of each triangle,
        in degrees."""
        return self._m.angles()

    def boundary_edge_counts(self):
        """(ntriangles,) int32: how many of each triangle's edges lie on the
        boundary. A value of 2 is a triangle filling a corner of the polygon."""
        return self._m.boundary_edge_counts()

    def has_interior_vertex(self):
        """(ntriangles,) int32, 1 where the triangle has at least one vertex off
        the boundary. A 0 means every vertex is constrained, so under Dirichlet
        data everywhere that element has no free degree of freedom. Pass
        ``interior_vertex=True`` to ``triangulate`` to rule them out."""
        return self._m.has_interior_vertex()

    def boundary_only_triangles(self):
        """Indices of the triangles whose three vertices are all on the boundary."""
        return np.flatnonzero(np.asarray(self.has_interior_vertex()) == 0)

    def qualities(self):
        """(ntriangles,) float64 in [0, 1]: 1 is equilateral, 0 degenerate.
        This is 4*sqrt(3)*A / (a^2+b^2+c^2), the same measure MATLAB's pdetriq
        uses."""
        return self._m.qualities()

    # ---- queries -------------------------------------------------------------
    def locate(self, points, *, barycentric=False):
        """
        The triangle containing each point, -1 for a point outside the mesh (or in a hole).

            t = m.locate((0.3, 0.2))                    # one point: an int
            t = m.locate(xy)                            # (n, 2) array: an int array of length n
            t, lam = m.locate(xy, barycentric=True)     # and the barycentric coordinates

        Each query walks through the mesh from the previous answer.  Where a walk runs into
        the boundary (the point is outside, in a hole, or across a hole or a reentrant
        corner), a bucket grid over the mesh is built once, in O(ntriangles), and from then
        on answers every query in O(1).  A batch of at least 1000 points and an eighth of
        the triangle count builds the grid up front.  The mesh keeps its locator, so later
        calls reuse the grid.  Containment is decided by the exact orient2d predicate, so a
        point on an edge or a vertex is reported in one of the triangles that contain it,
        and the answer is never a guess.

        Parameters
        ----------
        points : (x, y) pair, or (n, 2) array_like
        barycentric : bool
            Also return the barycentric coordinates of each point with respect to the
            vertices of its triangle, in the order of ``triangles[t]``.  NaN outside.

        Returns
        -------
        int or (n,) int32 array, and with barycentric=True a (3,) or (n, 3) float64 array.
        """
        p = np.asarray(points, dtype=np.float64)
        single = p.shape == (2,)
        p = np.ascontiguousarray(p.reshape(-1, 2) if single else p)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"points must be an (x, y) pair or an (n, 2) array, got shape {np.shape(points)}")
        if self._loc is None:
            self._loc = _C.PointLocator(self._m)
        t = np.asarray(self._loc.locate(p, self.ntriangles), dtype=np.int32)
        if not barycentric:
            return int(t[0]) if single else t
        lam = np.full((len(t), 3), np.nan)
        ok = t >= 0
        if ok.any():
            P, T = np.asarray(self.points), np.asarray(self.triangles)
            A, B, C = (P[T[t[ok], k]] for k in range(3))
            q = p[ok]
            det = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
            l1 = ((q[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (q[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])) / det
            l2 = ((B[:, 0] - A[:, 0]) * (q[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (q[:, 0] - A[:, 0])) / det
            lam[ok] = np.c_[1.0 - l1 - l2, l1, l2]
        return (int(t[0]), lam[0]) if single else (t, lam)

    def boundary_vertices(self, marker=None):
        """Indices of vertices on the boundary, optionally only those carrying
        the given marker."""
        mk = np.asarray(self.point_markers)
        return np.flatnonzero(mk != 0) if marker is None else np.flatnonzero(mk == marker)

    def boundary_edges(self, marker=None):
        """(k, 2) array of boundary edges, optionally filtered by marker."""
        seg = np.asarray(self.segments)
        if marker is None:
            return seg.copy()
        return seg[np.asarray(self.segment_markers) == marker]

    def remark(self, rule):
        """A copy of the mesh with new side markers, for boundaries that one ring marker does not split.

            m2 = m.remark(lambda x, y, old: np.where(x < 1e-9, 5, old))    # the left side gets marker 5

        rule(x, y, marker) receives the midpoints of the segments and their current markers (arrays)
        and returns the new markers.  Points and triangles keep their numbering; point markers are
        recomputed from the new side markers."""
        from .mesh2d import mesh_from_arrays as _mfa
        S = np.array(self.segments)
        P = np.array(self.points)
        mid = 0.5 * (P[S[:, 0]] + P[S[:, 1]])
        new = np.asarray(rule(mid[:, 0], mid[:, 1], np.array(self.segment_markers)))
        new = np.broadcast_to(new, (S.shape[0],)).astype(np.int64)
        if np.any(new <= 0):
            raise ValueError("remark(): markers must be positive (0 is reserved for interior vertices)")
        return _mfa(P, np.array(self.triangles), S, new)

    def validate(self):
        """Check the mesh against its invariants.  Returns an empty string when
        it is sound, otherwise a description of the first fault found."""
        return self._m.validate()

    def triangulation(self):
        """A ``matplotlib.tri.Triangulation`` over this mesh."""
        from matplotlib.tri import Triangulation
        p = np.asarray(self.points)
        return Triangulation(p[:, 0], p[:, 1], np.asarray(self.triangles))

    def plot(self, ax=None, **kw):
        """Draw the mesh with matplotlib.  Extra keywords go to ``triplot``."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        kw.setdefault("linewidth", 0.4)
        kw.setdefault("color", "0.3")
        ax.triplot(self.triangulation(), **kw)
        ax.set_aspect("equal")
        return ax

    def __repr__(self):
        extra = f", min angle {np.min(self.angles()):.2f} deg" if self.ntriangles else ""
        return f"Mesh2D({self.npoints} points, {self.ntriangles} triangles, {self.nsegments} segments{extra})"


def _as_ring(pts):
    a = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"a ring must be an (n, 2) array of points, got shape {a.shape}")
    if a.shape[0] < 3:
        raise ValueError("a ring needs at least 3 points")
    return a


def triangulate(boundary, holes=None, *, min_angle=20.0, max_area=None, max_edge=None,
                smooth=0, marker=1, hole_markers=None, vertex_markers=None, edge_markers=None,
                interior_vertex=False, max_points=500_000, check=False):
    """
    Mesh a polygonal region with a quality-guaranteed Delaunay triangulation.

        m = fd.triangulate([(0,0), (2,0), (2,1), (0,1)],
                           holes=[fd.circle(1, 0.5, 0.2, 24)],
                           min_angle=30, max_area=2e-3, smooth=2)

    Parameters
    ----------
    boundary : (n, 2) array_like
        The outer polygon, first vertex not repeated.  Either winding is
        accepted; it is re-oriented counter-clockwise internally.
    holes : sequence of (n, 2) array_like, optional
        Polygons to cut out.  Either winding is accepted.
    min_angle : float
        Smallest interior angle to aim for, IN DEGREES.  Ruppert's algorithm is
        proved to terminate up to 20.7 and in practice reaches about 33; asking
        for 34 or more raises ValueError rather than running forever.  Pass 0 to
        turn the shape criterion off and refine on size alone.
    max_area : float, optional
        Largest triangle area allowed.
    max_edge : float, optional
        Longest edge allowed.
    smooth : int
        Laplacian smoothing passes after refinement.  Each pass moves interior
        vertices to the centroid of their neighbours, rejecting any move that
        would invert a triangle, worsen the worst angle or quality of its fan,
        or break max_area or max_edge there, and restores the Delaunay property
        by flipping.  A last refinement pass repairs what the flips may enlarge,
        so min_angle, max_area and max_edge hold for the smoothed mesh too.
    marker : int
        Marker for the outer boundary's vertices and segments.  0 is reserved
        for interior vertices, so boundary markers start at 1.
    hole_markers : sequence of int, optional
        One marker per hole; defaults to marker+1, marker+2, ..., or with
        edge_markers to the numbers after the largest edge marker.
    vertex_markers : (n,) array_like of int, optional
        Per-vertex markers for the outer boundary, overriding ``marker``.
        They mark the vertices only; the sides keep ``marker``.
    edge_markers : (n,) array_like of int, optional
        One marker per edge of the outer polygon, overriding ``marker``.  Edge i
        joins boundary[i] and boundary[i+1] (the last joins the last vertex and
        the first), in the order given.  Every boundary segment of the mesh on
        that edge gets the marker, and the point markers follow the sides, a
        boundary vertex taking the marker of the side leaving it counter-clockwise.
        Not combined with ``vertex_markers``.
    interior_vertex : bool
        Require every triangle to have at least one vertex OFF the boundary.

        Without it, the triangle filling a corner of the polygon typically has
        two of its edges on the boundary and therefore all three of its vertices
        constrained, so under Dirichlet data everywhere that element carries no
        free degree of freedom. The option removes those, and the thin-neck case
        too, by flipping the third edge to reach an interior vertex where that
        keeps the angle bound, and by inserting a vertex where it does not.

        It lowers the reachable ``min_angle``. Forcing an interior vertex at a
        corner of angle alpha means at least two triangles share that corner, so
        one of them has an angle of at most alpha/2. Asking for more than half
        the sharpest input corner raises ValueError and says so. On ordinary
        geometry with 90 degree corners this never bites, and the cost is a few
        extra triangles near the corners.
    max_points : int
        Refinement stops with a RuntimeError past this many vertices rather than
        exhausting memory.  The message names the smallest input angle, which is
        what caps the reachable min_angle near a sharp corner.
    check : bool
        Run the mesh invariant check before returning.

    Returns
    -------
    Mesh2D
    """
    d = Domain()
    b = _as_ring(boundary)
    em = None
    if edge_markers is not None:
        if vertex_markers is not None:
            raise ValueError("give edge_markers or vertex_markers, not both: with edge_markers the "
                             "vertex markers follow the sides")
        em = np.asarray(edge_markers).ravel().astype(np.int64)
        if em.shape[0] != b.shape[0]:
            raise ValueError(f"edge_markers must have one entry per edge of the boundary ({b.shape[0]}; "
                             f"edge i joins boundary[i] and boundary[i+1]), got {em.shape[0]}")
        if np.any(em <= 0):
            raise ValueError("edge_markers must be positive (0 is reserved for interior vertices)")
    vm = [] if vertex_markers is None else [int(v) for v in np.asarray(vertex_markers).ravel()]
    if vm and len(vm) != b.shape[0]:
        raise ValueError("vertex_markers must have one entry per boundary point")
    d.boundary(b, vm, int(marker))

    holes = list(holes or [])
    if hole_markers is None:
        first = (int(em.max()) if em is not None else marker) + 1
        hole_markers = [first + i for i in range(len(holes))]
    if len(hole_markers) != len(holes):
        raise ValueError("hole_markers must have one entry per hole")
    for h, hm in zip(holes, hole_markers):
        d.hole(_as_ring(h), [], int(hm))

    core, report = _C.triangulate(d, float(min_angle), float(max_edge or 0.0), float(max_area or 0.0),
                                  int(smooth), int(max_points), bool(check), bool(interior_vertex))
    m = Mesh2D(core, report)
    if em is not None:
        m = _mark_outer_edges(m, b, em, int(marker))
    return m


def _mark_outer_edges(m, ring, em, marker):
    """The mesh m with each boundary segment of the outer ring marked by the input edge it lies on."""
    S = np.array(m.segments)
    P = np.array(m.points)
    old = np.array(m.segment_markers).astype(np.int64)
    A, B = ring, np.roll(ring, -1, axis=0)
    mid = 0.5 * (P[S[:, 0]] + P[S[:, 1]])
    edge, dist = _C.nearest_segments(np.ascontiguousarray(mid, dtype=np.float64),
                                     np.ascontiguousarray(A, dtype=np.float64))      # mesh/point_match.hpp
    edge, dist = np.asarray(edge, dtype=np.int64), np.asarray(dist)
    ext = A.max(axis=0) - A.min(axis=0)
    diag = float(np.hypot(ext[0], ext[1]))
    on_outer = (old == marker) & (dist <= 1e-9 * diag)
    new = np.where(on_outer, em[edge], old)
    out = mesh_from_arrays(P, np.array(m.triangles), S, new)
    out.report = dict(m.report)
    return out


def circle(cx, cy, r, n=32, clockwise=False):
    """n points evenly spaced on a circle, as an (n, 2) array.  Handy for holes;
    the winding does not matter, triangulate() fixes it."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    if clockwise:
        t = -t
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


# --------------------------------------------------------------------- local refinement
def _marked(mesh, triangles):
    a = np.atleast_1d(np.asarray(triangles))
    if a.dtype == bool:
        if a.shape != (mesh.ntriangles,):
            raise ValueError(f"a boolean mask must have one entry per triangle "
                             f"({mesh.ntriangles}), got {a.shape}")
        a = np.flatnonzero(a)
    a = a.astype(np.int64, copy=False)
    if a.size and (a.min() < 0 or a.max() >= mesh.ntriangles):
        raise ValueError(f"triangle index out of range for a mesh of {mesh.ntriangles} triangles")
    return [int(i) for i in a]


def split_triangles(mesh, triangles):
    """
    Barycentric split: replace each named triangle by three meeting at its centroid.

    `triangles` is a list of indices or a boolean mask of length `ntriangles`.
    Returns a new `Mesh2D`; the one passed in is untouched, so any arrays taken
    from it stay valid.

    Conforming on its own, since it touches nothing outside the triangle, and
    it leaves the mesh non-Delaunay there. **Do not use it to refine
    repeatedly.** Each child keeps a whole edge of its parent, so the element
    diameter never shrinks, and the interpolation error of a finite element is
    governed by the diameter, not the area. Four levels on a 30-degree mesh
    take the smallest angle to under a degree while the local diameter does not
    move at all. Use `bisect_triangles` or `refine_triangles` for that.
    """
    core, n = _C.split_triangles(mesh._m, _marked(mesh, triangles))
    out = Mesh2D(core, dict(mesh.report))
    out.report["local_refine"] = ("barycentric", n)
    return out


def bisect_triangles(mesh, triangles, levels=1):
    """
    Rivara longest-edge bisection of each named triangle.

    Walks from each triangle to its longest-edge neighbour and on, until a pair
    is reached whose shared edge is longest for both, then halves that edge --
    the longest-edge propagation path. Bisecting the terminal edge splits both
    its triangles at once, so no hanging node is ever created and the mesh
    stays conforming without any closure step. Because the path can reach past
    the triangles you named, more of them get refined than you asked for; that
    is the price of conformity.

    Non-degenerate: every angle it can ever produce is at least half the
    smallest angle of the mesh it started from, no matter how many times it is
    applied. The angle drops once on the first level and then plateaus. The
    element diameter does shrink, and triangle count grows about 2x per level,
    which keeps refinement tightly local.

    The mesh it returns is conforming but NOT Delaunay; bisection and Delaunay
    are different triangulations of the same points.
    """
    core, n = _C.bisect_triangles(mesh._m, _marked(mesh, triangles), int(levels))
    out = Mesh2D(core, dict(mesh.report))
    out.report["local_refine"] = ("rivara", n)
    return out


def refine_triangles(mesh, triangles, *, min_angle=20.0, max_area=None, max_edge=None,
                     max_points=500_000):
    """
    Delaunay refinement: insert the circumcentre of each named triangle.

    The only one of the three that keeps the mesh a constrained Delaunay
    triangulation and keeps the quality guarantee, because it is the operation
    Ruppert refinement already performs and it obeys the same rule: a
    circumcentre that would fall inside the diametral circle of a boundary
    segment is not inserted, the segment is split instead. After the pass the
    ordinary quality loop runs over the whole mesh, so the result meets
    `min_angle`, `max_area` and `max_edge` everywhere.

    One pass over the marked set; call it again for another level. It refines
    hardest of the three, since each insertion can cascade through the quality
    loop, so check the triangle count before asking for several levels.
    """
    core, n = _C.refine_triangles(mesh._m, _marked(mesh, triangles), float(min_angle),
                                  float(max_edge or 0.0), float(max_area or 0.0),
                                  int(max_points), 180.0)
    out = Mesh2D(core, dict(mesh.report))
    out.report["local_refine"] = ("circumcentre", n)
    return out


# --------------------------------------------------------------------------- coarsening
def remove_vertices(mesh, vertices, *, min_angle=0.0, max_edge=None, max_area=None, boundary=True):
    """
    Remove vertices from a mesh, filling each hole with new triangles on the vertices
    around it.  No other vertex moves and none is added.

        m2 = fd.remove_vertices(m, eta_v < tol)            # a boolean mask over the vertices
        u2 = u[m2.report["vertex_map"] >= 0]               # a nodal vector carries over by injection

    Each hole is refilled by Delaunay ear clipping, so on a Delaunay mesh the result is
    the Delaunay triangulation of the vertices that remain.

    Parameters
    ----------
    mesh : Mesh2D
    vertices : list of int, or bool mask of length npoints
        The vertices to remove.  A vertex that cannot go is kept, and the report says why.
    min_angle : float
        Degrees.  A removal whose new triangles would have a smaller angle is refused.
        Default 0: only inverted or degenerate triangles are refused.
    max_edge, max_area : float, optional
        A removal whose new triangles would break these is refused.
    boundary : bool
        Allow removing boundary vertices.  Default True.  Only a vertex on a straight side
        can go (never a corner), with the same marker on both its segments and a point marker
        equal to that marker or to both its neighbours' along the side.  Its two segments
        merge into one, so the boundary and the area are unchanged.

    Returns
    -------
    Mesh2D
        A new mesh; the one passed in is untouched.  The removed points are dropped and the
        others renumbered in their original order.  report keys:
        "removed" (count), "vertex_map" (int array over the old vertices, the new index or -1),
        "kept" (requested vertices still present, old indices), "kept_reasons" (one string each).

    An interior vertex is kept when it ends an interior segment, and any vertex when the fill
    would need an edge the mesh already has outside the hole or would break the bounds.
    Removing one vertex can make a neighbour removable, so the list is swept until a sweep
    removes nothing.
    """
    a = np.atleast_1d(np.asarray(vertices))
    if a.dtype == bool:
        if a.shape != (mesh.npoints,):
            raise ValueError(f"a boolean mask must have one entry per vertex ({mesh.npoints}), got {a.shape}")
        a = np.flatnonzero(a)
    a = a.astype(np.int64, copy=False)
    if a.size and (a.min() < 0 or a.max() >= mesh.npoints):
        raise ValueError(f"vertex index out of range for a mesh of {mesh.npoints} vertices")
    core, n, pmap, kept, reasons, passes = _C.remove_vertices(
        mesh._m, [int(i) for i in a], float(min_angle), float(max_edge or 0.0), float(max_area or 0.0), bool(boundary))
    out = Mesh2D(core, dict(mesh.report))
    out.report.update(removed=int(n), vertex_map=np.asarray(pmap, dtype=np.int64),
                      kept=np.asarray(kept, dtype=np.int64), kept_reasons=list(reasons))
    return out


# --------------------------------------------------------------------------- smoothing
def smooth(mesh, passes=1, *, restore_delaunay=True):
    """
    Laplacian smoothing of an existing mesh, without retriangulating.

    `triangulate(..., smooth=n)` runs the same thing as its last step; this is
    the standalone form, for when you want more passes, or want to smooth a mesh
    that has since been refined.

    Each pass moves every interior vertex to the centroid of its one-ring. A
    move is taken only if every incident triangle keeps positive area **and**
    neither the worst shape quality nor the worst angle in the fan gets worse,
    so smoothing can only improve the mesh and cannot tangle it. Vertices on the
    boundary never move, so the geometry and the area are exact.

    It converges, and fast. Measured against the fully converged positions on a
    1000-element mesh, the vertices are within 7e-3 after 3 passes, 3e-4 after 5,
    3e-7 after 10, and at machine precision by 25; the worst quality reaches its
    final value by the third pass. Three to five passes is the useful range and
    more than about ten buys nothing.

    `result.report["smoothed"]` counts the moves accepted in this call. It does
    NOT fall to zero once the mesh has settled, because a move is accepted when
    it leaves its fan no worse, which includes moves too small to measure. Use it
    to see that something happened, not as a convergence test; compare
    `qualities()` between two pass counts for that.

    Parameters
    ----------
    mesh : Mesh2D
    passes : int
        How many sweeps. Returns the mesh unchanged for `passes <= 0`.
    restore_delaunay : bool
        Flip back to Delaunay at the end of each pass, as `triangulate` does.
        Set it False for a mesh that carries deliberate non-Delaunay edges you
        want to keep, which is the case after `bisect_triangles` or
        `split_triangles`: restoring Delaunay would flip their structure away.

    Returns
    -------
    Mesh2D
        A new mesh; the one passed in is untouched, so any arrays taken from it
        stay valid. Vertex and triangle numbering are unchanged, since smoothing
        moves points rather than adding them.
    """
    core, n = _C.smooth(mesh._m, int(passes), bool(restore_delaunay))
    out = Mesh2D(core, dict(mesh.report))
    out.report["smoothed"] = n
    return out


def restore_delaunay(mesh, max_sweeps=64):
    """
    Flip edges until the mesh is a constrained Delaunay triangulation again.

    Segments are never flipped, so the boundary is untouched. Useful after
    `bisect_triangles` or `split_triangles`, which leave a valid conforming mesh
    that is not Delaunay. Returns a new mesh with the same points in the same
    order; only the triangles change. `report["flips"]` is how many were needed,
    so `0` means it was already Delaunay.
    """
    core, n = _C.restore_delaunay(mesh._m, int(max_sweeps))
    out = Mesh2D(core, dict(mesh.report))
    out.report["flips"] = n
    return out


# --------------------------------------------------------------------------- structured meshes
def _markers4(markers):
    if np.isscalar(markers):
        return [int(markers)] * 4
    mk = [int(m) for m in markers]
    if len(mk) != 4:
        raise ValueError(f"markers must be one int or four (bottom, right, top, left), got {len(mk)}")
    return mk


def mesh_from_arrays(points, triangles, segments=None, segment_markers=None, point_markers=None, *, marker=1,
                     check=True):
    """
    A Mesh2D from a point array and a triangle array, such as a mesh read from a file.

        m = fd.mesh_from_arrays(points, triangles)

    The neighbour links are built from the connectivity, every boundary edge becomes a
    constrained segment and the boundary vertices are marked, so the result is an ordinary
    Mesh2D: validate(), smooth() and the local refinement functions work on it unchanged.

    Parameters
    ----------
    points : (npoints, 2) array_like of float
    triangles : (ntriangles, 3) array_like of int
        Vertex indices, 0-based, either orientation.  Clockwise triangles are re-oriented,
        and triangle t of the result is triangle t of the input.
    segments : (k, 2) array_like of int, optional
        Edges to constrain with their own markers.  Each must be an edge of the mesh.
        Boundary edges not listed are constrained anyway, with `marker`.
    segment_markers : (k,) array_like of int, optional
        One marker per segment, default `marker`.
    point_markers : (npoints,) array_like of int, optional
        Default: 0 for interior vertices, and for a boundary vertex the marker of the
        boundary edge leaving it counter-clockwise.
    marker : int
        Marker of the boundary edges not listed in `segments`.  Default 1.
    check : bool
        Run Mesh2D.validate() on the result.  Default True.

    Returns
    -------
    Mesh2D
        report keys: "flipped" (triangles re-oriented), "boundary_edges", "boundary_loops".

    Raises
    ------
    ValueError
        For an index out of range, a triangle with a repeated or collinear vertices, a point
        used by no triangle, an edge shared by more than two triangles, two triangles on the
        same side of an edge (the mesh overlaps or folds), or a segment that is not an edge.
    """
    p = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"points must be an (n, 2) array, got shape {p.shape}")
    t = np.asarray(triangles)
    if t.ndim != 2 or t.shape[1] != 3 or not np.issubdtype(t.dtype, np.integer):
        raise ValueError(f"triangles must be an (m, 3) integer array, got shape {t.shape} and dtype {t.dtype}")
    t = np.ascontiguousarray(t, dtype=np.int32)
    s = np.zeros((0, 2), dtype=np.int32) if segments is None else np.asarray(segments)
    if s.size == 0:
        s = np.zeros((0, 2), dtype=np.int32)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"segments must be a (k, 2) integer array, got shape {s.shape}")
    s = np.ascontiguousarray(s, dtype=np.int32)
    sm = [] if segment_markers is None else [int(v) for v in np.asarray(segment_markers).ravel()]
    pm = [] if point_markers is None else [int(v) for v in np.asarray(point_markers).ravel()]
    core, report = _C.mesh_from_arrays(p, t, s, sm, pm, int(marker))
    m = Mesh2D(core, report)
    if check:
        msg = m.validate()
        if msg:
            raise ValueError(f"mesh_from_arrays: the result fails validation: {msg}")
    return m


def rectangle_mesh(x0, y0, x1, y1, nx, ny, *, diagonal="right", markers=(1, 2, 3, 4)):
    """
    Structured triangular mesh of the rectangle [x0, x1] x [y0, y1] with nx x ny equal cells.

        m = fd.rectangle_mesh(0, 0, 2, 1, 40, 20, diagonal="crossed")

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Lower-left and upper-right corners, x1 > x0 and y1 > y0.
    nx, ny : int
        Cells along x and along y, at least 1.
    diagonal : str
        How each cell is split.  "right": the "/" diagonal, 2 triangles.  "left": the "\\"
        diagonal.  "alternate": "/" where i+j is even and "\\" where it is odd, which removes
        the directional bias.  "crossed" (also "crisscross"): a vertex at the cell centre and
        4 triangles.
    markers : int or 4 ints
        Markers of the bottom, right, top and left sides.  Default (1, 2, 3, 4).  Corner
        (x0, y0) carries the bottom marker, (x1, y0) right, (x1, y1) top, (x0, y1) left.

    Returns
    -------
    Mesh2D
        Grid vertex (i, j) is point j*(nx+1) + i, the centres of "crossed" follow as point
        (nx+1)*(ny+1) + j*nx + i, and triangles run cell by cell in the same order.
        report["structured"] is (nx, ny, diagonal).
    """
    core, report = _C.rectangle_mesh(float(x0), float(y0), float(x1), float(y1), _integer(nx, "nx"),
                                     _integer(ny, "ny"), str(diagonal), _markers4(markers))
    return Mesh2D(core, report)


def _side(side, n, what):
    """A side of mapped_mesh as n+1 points: from a callable on [0, 1], an array of n+1 points,
    or two end points (a straight side)."""
    if callable(side):
        if n is None:
            raise ValueError(f"mapped_mesh: {what} is a callable, so its number of cells must be given")
        s = np.linspace(0.0, 1.0, n + 1)
        try:
            r = np.asarray(side(s), dtype=np.float64)
        except Exception:
            r = None
        if r is None or r.shape not in ((n + 1, 2), (2, n + 1)):
            r = np.array([np.asarray(side(float(v)), dtype=np.float64).ravel() for v in s])
        elif r.shape == (2, n + 1) and r.shape != (n + 1, 2):
            r = r.T
        if r.shape != (n + 1, 2):
            raise ValueError(f"mapped_mesh: {what}(s) must return a point (x, y), got shape {r.shape[1:]}")
        return np.ascontiguousarray(r)
    a = np.asarray(side, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2:
        raise ValueError(f"mapped_mesh: {what} must be a callable or an (m, 2) array of points, got shape {a.shape}")
    if a.shape[0] == 2 and n is not None and n != 1:              # two end points: a straight side
        s = np.linspace(0.0, 1.0, n + 1)[:, None]
        a = (1.0 - s) * a[0] + s * a[1]
    if n is not None and a.shape[0] != n + 1:
        raise ValueError(f"mapped_mesh: {what} has {a.shape[0]} points, {n} cells need {n + 1}")
    return np.ascontiguousarray(a)


def mapped_mesh(bottom, right, top, left, nx=None, ny=None, *, diagonal="right", markers=(1, 2, 3, 4)):
    """
    Structured triangular mesh of a four-sided region with curved sides, by transfinite
    (Coons) interpolation of the unit square.

        R, r = 2.0, 1.0                                    # a quarter annulus
        m = fd.mapped_mesh(bottom=[(r, 0), (R, 0)],
                           right=lambda s: np.c_[R*np.cos(s*np.pi/2), R*np.sin(s*np.pi/2)],
                           top=[(0, r), (0, R)],
                           left=lambda s: np.c_[r*np.cos(s*np.pi/2), r*np.sin(s*np.pi/2)],
                           nx=8, ny=24)

    The grid vertex (i, j) sits at X(i/nx, j/ny), where
    X(xi, eta) = (1-eta) B(xi) + eta T(xi) + (1-xi) L(eta) + xi R(eta), minus the bilinear
    interpolant of the four corners.  Boundary vertices are the side samples exactly.

    Parameters
    ----------
    bottom, right, top, left : callable, (m, 2) array_like, or two points
        The four sides, meeting at corners P00, P10, P11, P01.  Bottom runs P00 to P10, right
        P10 to P11, top P01 to P11, left P00 to P01, but a side given the other way round (as
        a counter-clockwise walk gives top and left) is detected and reversed.
        A callable maps s in [0, 1] to a point, vectorized (an (n, 2) or (2, n) result for an
        array s) or scalar.  It is sampled at s = i/n.  An array gives the points directly
        and fixes the count.  Two points give a straight side.
    nx, ny : int, optional
        Cells along bottom/top and along left/right.  Read from a side given as an array of
        more than two points, and required otherwise.
    diagonal, markers
        As in rectangle_mesh.

    Returns
    -------
    Mesh2D
        Numbered as in rectangle_mesh.  report["structured"] is (nx, ny, diagonal).

    Raises
    ------
    ValueError
        When the sides do not meet at four corners, have inconsistent counts, or when the map
        is not one-to-one (some triangles inverted: the sides cross, or cells are too
        distorted for this diagonal).
    """
    nx = None if nx is None else _integer(nx, "nx")
    ny = None if ny is None else _integer(ny, "ny")
    if nx is None:
        for sd in (bottom, top):
            if not callable(sd) and np.asarray(sd).shape[0] > 2:
                nx = np.asarray(sd).shape[0] - 1
                break
    if ny is None:
        for sd in (left, right):
            if not callable(sd) and np.asarray(sd).shape[0] > 2:
                ny = np.asarray(sd).shape[0] - 1
                break
    if nx is None:
        raise ValueError("mapped_mesh: pass nx, the number of cells along bottom and top; it can only be read "
                         "from a side given as an array of more than two points")
    if ny is None:
        raise ValueError("mapped_mesh: pass ny, the number of cells along left and right; it can only be read "
                         "from a side given as an array of more than two points")
    B, T = _side(bottom, nx, "bottom"), _side(top, nx, "top")
    L, R = _side(left, ny, "left"), _side(right, ny, "right")
    core, report = _C.mapped_mesh(B, R, T, L, str(diagonal), _markers4(markers))
    return Mesh2D(core, report)
