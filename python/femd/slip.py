"""
femd.slip -- the strong slip condition u . n = 0 for a vector field on a 2D mesh.

    V = fd.VectorFunctionSpace(m, 2, dirichlet=[4], slip=[1, 3])

At every node on a slip side the two velocity coefficients (u_x, u_y) are replaced by one,
the tangential component: u_i = a_i t_i with t_i the unit tangent.  The normal component is
not a degree of freedom, so u . n = 0 holds exactly at the nodes.  In matrix terms the
coefficients of the system are x_B = Z x, with Z sparse (base dim x dim) and orthonormal
columns: a form is assembled on the space without the slip condition and reduced to
Z^T A Z, a residual to Z^T r.

The normal at a node:
  - a node inside a boundary edge: the normal of that edge;
  - a vertex between two slip edges: the length-weighted mean of their normals, which is
    the discrete "mass-conserving" normal, n_i proportional to  int phi_i n ds  (Engelman,
    Sani and Gresho 1982), so that  int u_h . n ds = 0  over the slip sides exactly;
  - a corner, where the two edge normals differ by more than corner_angle degrees: both
    components are fixed, u_i = 0 (the only value with u . n = 0 for both normals);
  - a vertex between a slip edge and a side without a Dirichlet condition: that edge's normal.
Nodes on a Dirichlet side keep their Dirichlet condition (Dirichlet wins).

The nodes, normals and Z are computed in C++ (fe/slip.hpp); Z is a _femd.CSRMatrix.
"""
from __future__ import annotations
import numpy as np

from . import _femd as _C


def _markers(v):
    if v is None:
        return []
    if isinstance(v, (int, np.integer)):
        return [int(v)]
    return sorted({int(m) for m in v})


def slip_nodes(f, markers, corner_angle=45.0, normal="conservative"):
    """The raw nodes of the scalar space f on the sides `markers`, with their unit normals, and
    the raw nodes that are corners.  Returns (nodes, normals (n, 2), corners).

    normal: "conservative" gives a node inside an edge the edge's normal, so that
    int u_h . n ds = 0 exactly; "smooth" blends the two vertex normals linearly along the edge,
    which follows a curved wall to second order (better for k >= 3 on curved walls)."""
    nodes, normals, corners = _C.slip_nodes(f, _markers(markers), float(corner_angle), str(normal))
    return np.asarray(nodes, dtype=np.int64), np.asarray(normals), np.asarray(corners, dtype=np.int64)


def slip_matrix(B, specs):
    """Z (a _femd.CSRMatrix, B.dim x dim) for the slip specs on the ProductSpace2D B (which has no
    slip condition), and per spec a dict with the nodes, their coordinates and normals, and the corners."""
    spec_t = [(int(s["fields"][0]), int(s["fields"][1]), _markers(s["markers"]), float(s.get("corner_angle", 45.0)),
               str(s.get("normal", "conservative"))) for s in specs]
    Z, raw = _C.slip_matrix(list(B.fields), [int(o) for o in B.offsets], spec_t)
    info = []
    for s, (nodes, normals, corners) in zip(specs, raw):
        i0, i1 = s["fields"]
        X = np.asarray(B.fields[i0].node_coordinates())
        nodes = np.asarray(nodes, dtype=np.int64)
        corners = np.asarray(corners, dtype=np.int64)
        info.append(dict(markers=list(s["markers"]), nodes=nodes, points=X[nodes], normals=np.asarray(normals),
                         corners=corners, corner_points=X[corners], fields=(i0, i1)))
    return Z, info
