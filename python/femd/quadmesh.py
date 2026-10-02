"""
femd.quadmesh -- quadrilateral meshes.

    m = fd.rectangle_quad_mesh(0, 0, 2, 1, 32, 16)              # structured, markers 1..4 by side
    m = fd.mapped_quad_mesh(bottom, right, top, left, 24, 16)    # a four-sided region, curved sides
    m = fd.quadrangulate(outer, holes=[hole], max_area=1e-3)     # any domain: all-quad, from triangles

`quadrangulate` triangulates the domain with `triangulate`, splits every triangle into three
quads at its edge midpoints and centroid (every quad convex), and smooths.  A QuadMesh feeds
fd.LagrangeSpace(m, k) exactly as a triangle mesh does, which then gives Q_k elements.
"""
from __future__ import annotations

import numpy as np

from . import _femd as _C

__all__ = ["QuadMesh", "quad_mesh_from_arrays", "quad_mesh_from_triangles", "rectangle_quad_mesh", "mapped_quad_mesh", "quadrangulate"]


class QuadMesh:
    """
    A 2D quadrilateral mesh: arrays first, as Mesh2D.

        m.points        # (npoints, 2) float64
        m.quads         # (nquads, 4) int32, counter-clockwise, every quad convex
        m.neighbours    # (nquads, 4) int32, [q, i] is the quad across edge i (vertex i to i+1), -1 outside
        m.segments, m.segment_markers, m.point_markers

    The arrays are views onto the C++ mesh and must not be written to.
    """

    __slots__ = ("_m", "report")

    def __init__(self, core, report=None):
        self._m = core
        self.report = dict(report or {})

    # ---- arrays -----------------------------------------------------------------------
    points = property(lambda self: self._m.points, doc="(npoints, 2) float64 view of the vertex coordinates.")
    quads = property(lambda self: self._m.quads, doc="(nquads, 4) int32 view, counter-clockwise.")
    neighbours = property(lambda self: self._m.neighbours, doc="(nquads, 4) int32 view; -1 on the boundary.")
    segments = property(lambda self: self._m.segments, doc="(nsegments, 2) int32 view of the boundary edges.")
    segment_markers = property(lambda self: self._m.segment_markers, doc="(nsegments,) int32 view.")
    point_markers = property(lambda self: self._m.point_markers, doc="(npoints,) int32 view.")

    @property
    def npoints(self): return self._m.npoints
    @property
    def nquads(self): return self._m.nquads
    @property
    def ncells(self): return self._m.nquads
    @property
    def nsegments(self): return self._m.nsegments

    # ---- geometry and quality --------------------------------------------------------------
    def areas(self): return self._m.areas()
    def min_angles(self):
        """Smallest interior angle of each quad, degrees (90 for a rectangle)."""
        return self._m.min_angles()
    def max_angles(self):
        """Largest interior angle of each quad, degrees (below 180 for a convex quad)."""
        return self._m.max_angles()
    def aspects(self):
        """Longest over shortest edge of each quad."""
        return self._m.aspects()

    def locate(self, points):
        """The quad containing each point, -1 outside."""
        P = np.ascontiguousarray(np.atleast_2d(np.asarray(points, dtype=np.float64)))
        return self._m.locate(P)

    def boundary_edges(self, marker=None):
        S = np.array(self.segments)
        return S if marker is None else S[np.asarray(self.segment_markers) == marker]

    def validate(self):
        """"" when the mesh is sound, otherwise the first fault found."""
        return self._m.validate()

    def smooth(self, passes=1):
        """A smoothed copy: every interior vertex moves towards the mean of its neighbours, a move that
        would make a quad non-convex is halved and then dropped.  Boundary vertices stay."""
        core = self._m.copy()
        core.smooth(int(passes))
        return QuadMesh(core, self.report)

    def remark(self, rule):
        """A copy with new side markers: rule(x, y, marker) on the side midpoints returns them."""
        S = np.array(self.segments)
        P = np.array(self.points)
        mid = 0.5 * (P[S[:, 0]] + P[S[:, 1]])
        new = np.broadcast_to(np.asarray(rule(mid[:, 0], mid[:, 1], np.array(self.segment_markers))), (S.shape[0],)).astype(np.int32)
        if np.any(new <= 0):
            raise ValueError("remark(): markers must be positive (0 is reserved for interior vertices)")
        return quad_mesh_from_arrays(P, np.array(self.quads), S, new)

    # ---- plotting -----------------------------------------------------------------------------
    def triangulation(self):
        """A matplotlib Triangulation with each quad split along its 0-2 diagonal (for plotting)."""
        import matplotlib.tri as mtri
        Q = np.asarray(self.quads)
        T = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
        return mtri.Triangulation(self.points[:, 0], self.points[:, 1], T)

    def plot(self, ax=None, *, color="k", lw=0.5, markers=False, **kw):
        """Draw the quads; markers=True colours the boundary sides by marker."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        if ax is None:
            ax = plt.gca()
        P, Q = self.points, np.asarray(self.quads)
        segs = P[np.stack([Q, np.roll(Q, -1, axis=1)], axis=-1)].reshape(-1, 2, 2)
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, **kw))
        if markers:
            S, M = np.asarray(self.segments), np.asarray(self.segment_markers)
            for mk in np.unique(M):
                ss = P[S[M == mk]]
                ax.add_collection(LineCollection(ss, linewidths=2.0 * lw + 1, colors=f"C{int(mk) % 10}", label=f"marker {mk}"))
            ax.legend(loc="best", fontsize=8)
        ax.autoscale_view()
        ax.set_aspect("equal")
        return ax

    def __repr__(self):
        mk = sorted(set(np.asarray(self.segment_markers).tolist()))
        return f"QuadMesh({self.npoints} points, {self.nquads} quads, markers {mk})"


def quad_mesh_from_arrays(points, quads, segments=None, segment_markers=None, point_markers=None, *, marker=1):
    """A QuadMesh from arrays.  Clockwise quads are re-oriented; a degenerate or non-convex quad is
    refused.  Boundary edges not listed in `segments` become segments with `marker`.  Point markers
    default to 0 inside and, on the boundary, the marker of the side leaving the vertex
    counter-clockwise."""
    P = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 2))
    Q = np.ascontiguousarray(np.asarray(quads, dtype=np.int32).reshape(-1, 4))
    S = np.ascontiguousarray(np.zeros((0, 2), dtype=np.int32) if segments is None else np.asarray(segments, dtype=np.int32).reshape(-1, 2))
    if segment_markers is None:
        SM = np.full(S.shape[0], int(marker), dtype=np.int32)
    else:
        SM = np.ascontiguousarray(np.asarray(segment_markers, dtype=np.int32).reshape(-1))
    PM = np.ascontiguousarray(np.zeros(0, dtype=np.int32) if point_markers is None else np.asarray(point_markers, dtype=np.int32))
    return QuadMesh(_C.QuadMesh(P, Q, S, SM, PM, int(marker)))


def _grid_quads(nx, ny):
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    v = (j * (nx + 1) + i).ravel()
    return np.stack([v, v + 1, v + nx + 2, v + nx + 1], axis=1)


def _from_structured(tri, nx, ny, kind):
    """The quads of a structured triangle mesh's grid (numbering j*(nx+1)+i), same sides and markers."""
    P = np.array(tri.points)[: (nx + 1) * (ny + 1)]
    out = quad_mesh_from_arrays(P, _grid_quads(nx, ny), np.array(tri.segments), np.array(tri.segment_markers),
                                np.array(tri.point_markers)[: P.shape[0]])
    out.report["structured"] = (nx, ny, kind)
    return out


def rectangle_quad_mesh(x0, y0, x1, y1, nx, ny, *, markers=(1, 2, 3, 4)):
    """Structured mesh of [x0, x1] x [y0, y1] with nx x ny equal rectangles.

    Numbered as rectangle_mesh: grid vertex (i, j) is point j*(nx+1) + i, quad (i, j) is quad
    j*nx + i with vertices (i, j), (i+1, j), (i+1, j+1), (i, j+1).  The sides carry markers
    (bottom, right, top, left), one int for all four."""
    from .mesh2d import rectangle_mesh
    tri = rectangle_mesh(x0, y0, x1, y1, nx, ny, diagonal="right", markers=markers)
    return _from_structured(tri, int(nx), int(ny), "rectangle")


def mapped_quad_mesh(bottom, right, top, left, nx=None, ny=None, *, markers=(1, 2, 3, 4)):
    """Structured quad mesh of a four-sided region by transfinite (Coons) interpolation, with the
    sides given as in mapped_mesh (callables s -> (x, y), arrays of points, or two points).
    Numbered as rectangle_quad_mesh.  Refused when the map folds a cell (not convex)."""
    from .mesh2d import mapped_mesh
    tri = mapped_mesh(bottom, right, top, left, nx, ny, diagonal="right", markers=markers)
    nx_, ny_, _ = tri.report["structured"]
    return _from_structured(tri, int(nx_), int(ny_), "mapped")


def quad_mesh_from_triangles(mesh, smooth=3):
    """The all-quad mesh of a triangle mesh: every triangle split into three quads at its edge midpoints
    and centroid (each convex), side markers carried over, then `smooth` smoothing passes.  Use it
    after refining a triangulation locally (fd.refine_triangles), which quadrangulate does not do."""
    core = _C.QuadMesh.from_triangles(mesh._m)
    if smooth:
        core.smooth(int(smooth))
    m = QuadMesh(core, {"triangles": mesh.ntriangles})
    if m.nquads:
        m.report["min_angle"] = float(m.min_angles().min())
        m.report["max_angle"] = float(m.max_angles().max())
    return m


def quadrangulate(boundary, holes=None, *, max_area=None, max_edge=None, min_angle=28.0, smooth=3, **options):
    """An all-quadrilateral mesh of any polygonal domain with holes.

        m = fd.quadrangulate([(0, 0), (2, 0), (2, 1), (0, 1)], holes=[fd.circle(0.6, 0.5, 0.2, 48)],
                             max_area=2e-3)

    The domain is triangulated with fd.triangulate (holes, markers and every other option as there),
    each triangle is split into three quads at its edge midpoints and centroid, and the interior
    vertices are smoothed `smooth` times.  Every quad is convex.  max_area and max_edge refer to the
    QUADS: the triangles are made three times larger in area and twice as long in edge, so the quads
    come out at about the requested size.  min_angle applies to the triangles (default 28 degrees,
    which leaves the quads with angles of roughly 45 to 135 degrees after smoothing).

    report: the triangulate report under "triangulate", plus "triangles", and the smallest and largest
    quad angles."""
    from .mesh2d import triangulate
    tri = triangulate(boundary, holes, min_angle=min_angle,
                      max_area=None if max_area is None else 3.0 * float(max_area),
                      max_edge=None if max_edge is None else 2.0 * float(max_edge), **options)
    m = quad_mesh_from_triangles(tri, smooth)
    m.report["triangulate"] = dict(tri.report)
    return m
