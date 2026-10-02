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
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


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
    cell, _local, mark, ext, edge = (np.asarray(a) for a in f.facets())
    sel = ext.astype(bool) & np.isin(mark, markers)
    if not sel.any():
        raise ValueError(f"slip: no exterior side carries marker(s) {markers}")
    E = np.asarray(f.edges())[edge[sel]]
    P = np.asarray(f.mesh.points, dtype=np.float64)
    a, b = P[E[:, 0]], P[E[:, 1]]
    L = np.hypot(*(b - a).T)
    tau = (b - a) / L[:, None]
    nu = np.c_[tau[:, 1], -tau[:, 0]]
    CV = np.asarray(f.cell_vertices())[cell[sel]]
    centroid = P[CV].mean(axis=1)
    flip = np.einsum("ij,ij->i", 0.5 * (a + b) - centroid, nu) < 0
    nu[flip] *= -1.0                                                  # outward

    nv, k = f.mesh.npoints, f.degree
    # vertices: every slip edge at the vertex, with its normal and length
    ends = np.r_[E[:, 0], E[:, 1]]
    en = np.r_[nu, nu]
    el = np.r_[L, L]
    order = np.argsort(ends, kind="stable")
    ends, en, el = ends[order], en[order], el[order]
    verts, start, count = np.unique(ends, return_index=True, return_counts=True)
    acc = np.add.reduceat(en * el[:, None], start, axis=0)
    cos_min = np.cos(np.deg2rad(float(corner_angle)))
    corner = np.zeros(verts.size, dtype=bool)
    for j in np.flatnonzero(count > 1):
        ns = en[start[j]:start[j] + count[j]]
        c = ns @ ns.T
        corner[j] = c.min() < cos_min - 1e-12
    vnorm = acc / np.hypot(*acc.T)[:, None]
    nodes, normals = [verts[~corner]], [vnorm[~corner]]
    if k > 1:                                                        # nodes inside the slip edges
        inner = (nv + edge[sel][:, None] * (k - 1) + np.arange(k - 1)[None, :]).ravel()
        nodes.append(inner)
        nin = np.repeat(nu, k - 1, axis=0)
        if normal == "smooth":
            X = np.asarray(f.node_coordinates())[inner]
            ea, eb = np.repeat(E[:, 0], k - 1), np.repeat(E[:, 1], k - 1)
            t = np.einsum("ij,ij->i", X - P[ea], np.repeat(tau, k - 1, axis=0)) / np.repeat(L, k - 1)
            good = (count == 2) & ~corner                            # a vertex inside a smooth slip wall
            vn = np.zeros((nv, 2))
            vn[verts[good]] = vnorm[good]
            na, nb = vn[ea], vn[eb]
            ga, gb = np.any(na != 0, axis=1), np.any(nb != 0, axis=1)
            # an end of the wall (or a corner): mirror the other vertex's normal in the edge normal,
            # which is exact on a circle; with no good vertex at all, the edge normal
            ref = lambda m_: 2 * np.einsum("ij,ij->i", m_, nin)[:, None] * nin - m_      # noqa: E731
            na = np.where(ga[:, None], na, np.where(gb[:, None], ref(nb), nin))
            nb = np.where(gb[:, None], nb, np.where(ga[:, None], ref(na), nin))
            blend = (1 - t)[:, None] * na + t[:, None] * nb
            nin = blend / np.hypot(*blend.T)[:, None]
        elif normal != "conservative":
            raise ValueError(f"slip_normal: 'conservative' or 'smooth', got {normal!r}")
        normals.append(nin)
    return np.concatenate(nodes).astype(np.int64), np.concatenate(normals), verts[corner].astype(np.int64)


def slip_matrix(B, specs):
    """Z (B.dim x dim) for the slip specs on the ProductSpace2D B (which has no slip condition),
    and per spec a dict with the nodes, their coordinates and normals, and the corners."""
    nB = B.dim
    keep = np.ones(nB, dtype=bool)
    pair_r0, pair_r1, tx, ty, info = [], [], [], [], []
    for s in specs:
        i0, i1 = s["fields"]
        f0, f1 = B.fields[i0], B.fields[i1]
        r2a = np.asarray(f0.raw_to_adapted())
        if not np.array_equal(r2a, np.asarray(f1.raw_to_adapted())):
            raise ValueError("slip: the two components must have the same Dirichlet conditions")
        nodes, normals, corners = slip_nodes(f0, s["markers"], s.get("corner_angle", 45.0),
                                             s.get("normal", "conservative"))
        free = r2a[nodes] >= 0                                       # Dirichlet wins
        nodes, normals = nodes[free], normals[free]
        corners = corners[r2a[corners] >= 0]
        ad = r2a[nodes]
        if ad.size and np.unique(ad).size != ad.size:              # periodic: copies of one node, one normal
            u_, first, inv = np.unique(ad, return_index=True, return_inverse=True)
            acc = np.zeros((u_.size, 2))
            np.add.at(acc, inv, normals)
            nodes, normals, ad = nodes[first], acc / np.hypot(*acc.T)[:, None], u_
            corners = corners[np.unique(r2a[corners], return_index=True)[1]] if corners.size else corners
            keep_ = ~np.isin(ad, r2a[corners])                   # a corner in one copy wins: u = 0 there
            nodes, normals, ad = nodes[keep_], normals[keep_], ad[keep_]
        r0, r1 = B.offsets[i0] + ad, B.offsets[i1] + ad
        keep[r1] = False
        pair_r0.append(r0); pair_r1.append(r1)
        tx.append(-normals[:, 1]); ty.append(normals[:, 0])
        ac = r2a[corners]
        keep[B.offsets[i0] + ac] = False
        keep[B.offsets[i1] + ac] = False
        X = np.asarray(f0.node_coordinates())
        info.append(dict(markers=list(s["markers"]), nodes=nodes, points=X[nodes], normals=normals,
                         corners=corners, corner_points=X[corners], fields=(i0, i1)))
    col = np.cumsum(keep) - 1
    rows = np.flatnonzero(keep)
    vals = np.ones(rows.size)
    pos = np.full(nB, -1)
    pos[rows] = np.arange(rows.size)
    r0 = np.concatenate(pair_r0) if pair_r0 else np.zeros(0, int)
    r1 = np.concatenate(pair_r1) if pair_r1 else np.zeros(0, int)
    tx = np.concatenate(tx) if tx else np.zeros(0)
    ty = np.concatenate(ty) if ty else np.zeros(0)
    vals[pos[r0]] = tx
    R = np.r_[rows, r1]
    C = np.r_[col[rows], col[r0]]
    Vv = np.r_[vals, ty]
    Z = sp.csr_matrix((Vv, (R, C)), shape=(nB, int(keep.sum())))
    return Z, info
