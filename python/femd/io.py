"""
femd.io -- files other programs read and write: VTK output for ParaView, Gmsh import.

    fd.write_vtk("flow.vtu", uh, ph)                 # Functions on one 2D mesh, their names as labels
    fd.write_vtk("mesh.vtu", mesh=m)                 # the mesh alone, with its side markers
    series = fd.VTKSeries("run.pvd")                 # a time series: run_000000.vtu, ... and run.pvd
    series.write(t, w)                               # one file per call, the .pvd rewritten each time
    m = fd.read_gmsh("domain.msh")                   # a Mesh2D (triangles) or QuadMesh (quads)

VTK output uses the XML unstructured grid format (.vtu).  A field of degree k > 1 is written on
VTK's Lagrange cells of order k (triangle 69, quadrilateral 70), which ParaView renders as the
element polynomial, so nothing is lost to a linear sub-triangulation.  Gmsh import reads the ASCII
formats 2.2 and 4.1 (Gmsh's default).
"""
from __future__ import annotations

import base64
import os
import warnings

import numpy as np

__all__ = ["write_vtk", "VTKSeries", "read_gmsh"]


# ============================================================================ VTK output
_VTK_LINE, _VTK_TRIANGLE, _VTK_QUAD = 3, 5, 9
_VTK_LAGRANGE_CURVE, _VTK_LAGRANGE_TRIANGLE, _VTK_LAGRANGE_QUAD = 68, 69, 70


def _vtk_triangle_lattice(n):
    """(i, j) lattice points of VTK's Lagrange triangle of order n, in VTK's order: the vertices,
    the edge points of edges 0-1, 1-2, 2-0, then the interior as a triangle of order n - 3."""
    if n == 0:
        return [(0, 0)]
    pts = [(0, 0), (n, 0), (0, n)]
    pts += [(k, 0) for k in range(1, n)]
    pts += [(n - k, k) for k in range(1, n)]
    pts += [(0, n - k) for k in range(1, n)]
    if n >= 3:
        pts += [(i + 1, j + 1) for i, j in _vtk_triangle_lattice(n - 3)]
    return pts


def _vtk_quad_lattice(n):
    """(i, j) lattice points of VTK's Lagrange quadrilateral of order n, in VTK's order: the vertices,
    the edges 0-1, 1-2, 3-2, 0-3 (each in increasing parameter), then the interior, i fastest."""
    pts = [(0, 0), (n, 0), (n, n), (0, n)]
    pts += [(k, 0) for k in range(1, n)]
    pts += [(n, k) for k in range(1, n)]
    pts += [(k, n) for k in range(1, n)]
    pts += [(0, k) for k in range(1, n)]
    pts += [(i, j) for j in range(1, n) for i in range(1, n)]
    return pts


def _slot_entities(nv, n, lattice):
    """Per lattice slot: ("v", local vertex) | ("e", local vertex a, local vertex b, k from a) | ("c",)."""
    out = []
    for i, j in lattice:
        if nv == 3:
            corners = {(0, 0): 0, (n, 0): 1, (0, n): 2}
            if (i, j) in corners:
                out.append(("v", corners[(i, j)]))
            elif j == 0:
                out.append(("e", 0, 1, i))
            elif i + j == n:
                out.append(("e", 1, 2, j))
            elif i == 0:
                out.append(("e", 2, 0, n - j))
            else:
                out.append(("c",))
        else:
            corners = {(0, 0): 0, (n, 0): 1, (n, n): 2, (0, n): 3}
            if (i, j) in corners:
                out.append(("v", corners[(i, j)]))
            elif j == 0:
                out.append(("e", 0, 1, i))
            elif i == n:
                out.append(("e", 1, 2, j))
            elif j == n:
                out.append(("e", 3, 2, i))
            elif i == 0:
                out.append(("e", 0, 3, j))
            else:
                out.append(("c",))
    return out


class _Lattice:
    """The output points of a 2D mesh at order n: every cell's VTK lattice, shared points merged by
    topology (a vertex, a position along a global edge, or a point inside one cell).  broken=True
    keeps every cell's points apart, for discontinuous fields (DG, Raviart-Thomas, Nedelec)."""

    def __init__(self, mesh, n, broken=False):
        from .spaces2d import _space_for
        self.mesh, self.n = mesh, int(n)
        G = _space_for(mesh)(mesh, 1)                   # the geometry: cell maps and reference vertices
        self.G = G
        nv = G.nverts
        self.nv = nv
        cellv = np.asarray(G.cell_vertices(), dtype=np.int64)
        nc = cellv.shape[0]
        lattice = _vtk_triangle_lattice(n) if nv == 3 else _vtk_quad_lattice(n)
        ents = _slot_entities(nv, n, lattice)
        R = np.asarray(G.ref_nodes(), dtype=np.float64)[:nv]            # reference vertices
        st = np.array(lattice, dtype=np.float64) / n
        if nv == 3:
            ref = R[0] + st[:, :1] * (R[1] - R[0]) + st[:, 1:] * (R[2] - R[0])
        else:
            ref = R[0] + st[:, :1] * (R[1] - R[0]) + st[:, 1:] * (R[3] - R[0])
        self.ref = ref                                                  # (nslots, 2)
        keys = np.zeros((nc, len(lattice), 4), dtype=np.int64)
        for s, e in enumerate(ents):
            if broken:
                keys[:, s, 0], keys[:, s, 1], keys[:, s, 2] = 2, np.arange(nc), s
            elif e[0] == "v":
                keys[:, s, 0], keys[:, s, 1] = 0, cellv[:, e[1]]
            elif e[0] == "e":
                ga, gb = cellv[:, e[1]], cellv[:, e[2]]
                keys[:, s, 0] = 1
                keys[:, s, 1], keys[:, s, 2] = np.minimum(ga, gb), np.maximum(ga, gb)
                keys[:, s, 3] = np.where(ga < gb, e[3], n - e[3])
            else:
                keys[:, s, 0], keys[:, s, 1], keys[:, s, 2] = 2, np.arange(nc), s
        flat = keys.reshape(-1, 4)
        uniq, first, inv = np.unique(flat, axis=0, return_index=True, return_inverse=True)
        self.conn = inv.reshape(nc, len(lattice))                      # (ncells, nslots) point ids
        self.npoints = uniq.shape[0]
        cells = np.repeat(np.arange(nc, dtype=np.int32), len(lattice))
        XY, _ = G.map_points(np.ascontiguousarray(cells), np.ascontiguousarray(np.tile(ref[:, 0], nc)),
                             np.ascontiguousarray(np.tile(ref[:, 1], nc)))
        self.points = np.zeros((self.npoints, 3))
        self.points[inv.ravel(), :2] = XY
        self.ncells = nc
        self.broken = bool(broken)

    def values(self, space, coeffs):
        """The Function with these coefficients (of `space`, on this mesh) at the output points:
        (npoints,) for a scalar space, (npoints, 2) for a vector element."""
        if getattr(space, "value_shape", ()) == (2,):
            nc, ns = self.conn.shape
            cells = np.repeat(np.arange(nc, dtype=np.int32), ns)
            V = space.evaluate_cells(coeffs, cells, np.tile(self.ref[:, 0], nc), np.tile(self.ref[:, 1], nc), "value")
            out = np.zeros((self.npoints, 2))
            out[self.conn.ravel()] = V
            return out
        raw = space.prolongate(coeffs) if space.n_constraints else np.asarray(coeffs, dtype=np.float64)
        loc = np.asarray(raw)[np.asarray(space.cell_raw_dofs())]         # (ncells, nloc)
        T = np.array([space.ref_eval(a, b, 0)[0] for a, b in self.ref])  # (nslots, nloc)
        out = np.zeros(self.npoints)
        out[self.conn.ravel()] = (loc @ T.T).ravel()
        return out


def _fields(items):
    """[(name, kind, [(space, coeffs), ...]), ...] from Functions, ProductFunctions, (name, Function) pairs."""
    from .forms import Function, ProductFunction, VectorFunction, VectorElementFunction
    out = []
    for it in items:
        name = None
        if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
            name, it = it
        if isinstance(it, VectorFunction):
            parts = it.product.split(it.vector)
            out.append((name or it.name, "vector", list(zip(it.product.fields, parts))))
        elif isinstance(it, ProductFunction):
            P = it.product
            parts = P.split(it.vector)
            for b, (kind, idx) in enumerate(P.blocks):
                nm = f"{name or it.name}_{b}"
                out.append((nm, kind if kind in ("vector", "element") else "scalar", [(P.fields[i], parts[i]) for i in idx]))
        elif isinstance(it, VectorElementFunction):
            out.append((name or it.name, "element", [(it.space, it.vector)]))
        elif isinstance(it, Function):
            if getattr(it.space, "tdim", 1) != 2:
                raise TypeError(f"write_vtk: '{it.name}' is a Function on a 1D space; VTK output is for 2D meshes")
            out.append((name or it.name, "scalar", [(it.space, it.vector)]))
        else:
            raise TypeError(f"write_vtk: a field is a Function, a VectorFunction, a ProductFunction or a "
                            f"(name, Function) pair, got {type(it).__name__}")
    names = [f[0] for f in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"write_vtk: two fields are named {dup}; name them with (name, Function) pairs")
    return out


def _encode(a, binary):
    a = np.ascontiguousarray(a)
    if not binary:
        if a.dtype.kind == "f":
            return " ".join(repr(float(v)) for v in a.ravel())
        return " ".join(str(int(v)) for v in a.ravel())
    raw = a.tobytes()
    return base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode("ascii")


def _array(name, a, ncomp, vtype, binary):
    nm = f' Name="{name}"' if name else ""
    fmt = "binary" if binary else "ascii"
    return (f'<DataArray type="{vtype}"{nm} NumberOfComponents="{ncomp}" format="{fmt}">'
            f"{_encode(a, binary)}</DataArray>\n")


def _vtu(lat, fields, cell_data, lines, binary):
    """The text of a .vtu file."""
    n, nc = lat.n, lat.ncells
    if n == 1:
        ctype = _VTK_TRIANGLE if lat.nv == 3 else _VTK_QUAD
    else:
        ctype = _VTK_LAGRANGE_TRIANGLE if lat.nv == 3 else _VTK_LAGRANGE_QUAD
    conn = [lat.conn.ravel()]
    offsets = [np.arange(1, nc + 1) * lat.conn.shape[1]]
    types = [np.full(nc, ctype, dtype=np.uint8)]
    ncl = 0
    if lines is not None:
        segs, marks = lines
        ncl = segs.shape[0]
        conn.append(segs.ravel())
        offsets.append(offsets[0][-1] + np.arange(1, ncl + 1) * segs.shape[1])
        types.append(np.full(ncl, _VTK_LINE if segs.shape[1] == 2 else _VTK_LAGRANGE_CURVE, dtype=np.uint8))
    total = nc + ncl
    body = ['<?xml version="1.0"?>\n',
            '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n',
            "<UnstructuredGrid>\n", f'<Piece NumberOfPoints="{lat.npoints}" NumberOfCells="{total}">\n']
    sc = next((f[0] for f in fields if f[1] == "scalar"), None)
    ve = next((f[0] for f in fields if f[1] in ("vector", "element")), None)
    attrs = (f' Scalars="{sc}"' if sc else "") + (f' Vectors="{ve}"' if ve else "")
    body.append(f"<PointData{attrs}>\n")
    for name, kind, parts in fields:
        vals = [lat.values(S, c) for S, c in parts]
        if kind == "scalar":
            body.append(_array(name, vals[0], 1, "Float64", binary))
        elif kind == "element":
            V = np.zeros((lat.npoints, 3))
            V[:, :2] = vals[0]
            body.append(_array(name, V, 3, "Float64", binary))
        else:
            V = np.zeros((lat.npoints, 3))
            for k, v in enumerate(vals[:3]):
                V[:, k] = v
            body.append(_array(name, V, 3, "Float64", binary))
    body.append("</PointData>\n<CellData>\n")
    for name, a in cell_data.items():
        a = np.asarray(a)
        if a.shape[0] == nc and ncl:
            a = np.concatenate([a, np.zeros((ncl,) + a.shape[1:], dtype=a.dtype)])
        if a.shape[0] != total:
            raise ValueError(f"write_vtk: cell_data['{name}'] has {a.shape[0]} entries, the mesh has {nc} cells")
        ncomp = 1 if a.ndim == 1 else a.shape[1]
        if a.dtype.kind in "iub":
            body.append(_array(name, a.astype(np.int32), ncomp, "Int32", binary))
        else:
            body.append(_array(name, a.astype(np.float64), ncomp, "Float64", binary))
    body.append("</CellData>\n<Points>\n")
    body.append(_array("", lat.points, 3, "Float64", binary))
    body.append("</Points>\n<Cells>\n")
    body.append(_array("connectivity", np.concatenate(conn).astype(np.int64), 1, "Int64", binary))
    body.append(_array("offsets", np.concatenate(offsets).astype(np.int64), 1, "Int64", binary))
    body.append(_array("types", np.concatenate(types), 1, "UInt8", binary))
    body.append("</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    return "".join(body)


def _segment_lines(lat):
    """The mesh's segments as line cells through the output points (order n), with their markers."""
    m, n = lat.mesh, lat.n
    S = np.asarray(m.segments, dtype=np.int64)
    marks = np.asarray(m.segment_markers, dtype=np.int32)
    if S.shape[0] == 0:
        return None
    # the output point of a vertex, and of position k along a global edge, from the cells' keys
    cellv = np.asarray(lat.G.cell_vertices(), dtype=np.int64)
    nv = lat.nv
    lattice = _vtk_triangle_lattice(n) if nv == 3 else _vtk_quad_lattice(n)
    ents = _slot_entities(nv, n, lattice)
    table = {}
    for s, e in enumerate(ents):
        if e[0] == "v":
            for v, p in zip(cellv[:, e[1]], lat.conn[:, s]):
                table[("v", int(v))] = int(p)
        elif e[0] == "e":
            ga, gb = cellv[:, e[1]], cellv[:, e[2]]
            for a, b, p in zip(ga, gb, lat.conn[:, s]):
                k = e[3] if a < b else n - e[3]
                table[("e", int(min(a, b)), int(max(a, b)), int(k))] = int(p)
    rows = []
    for a, b in S:
        row = [table[("v", int(a))], table[("v", int(b))]]
        lo, hi = min(a, b), max(a, b)
        for k in range(1, n):                                   # VTK Lagrange curve: ends, then inside a -> b
            kk = k if a == lo else n - k
            row.append(table[("e", int(lo), int(hi), int(kk))])
        rows.append(row)
    return np.array(rows, dtype=np.int64), marks


def write_vtk(filename, *fields, mesh=None, degree=None, cell_data=None, segments=None, binary=True):
    """Write Functions on a 2D mesh (or the mesh alone) to a VTK XML unstructured grid, a .vtu file.

        fd.write_vtk("out.vtu", uh)                       # a scalar Function, named by its name
        fd.write_vtk("out.vtu", w)                        # a system: one array per block (w_0, w_1, ...)
        fd.write_vtk("out.vtu", ("speed", s), ("u", uv))  # (name, Function) pairs name them
        fd.write_vtk("mesh.vtu", mesh=m)                  # the mesh, with its segments and their markers

    fields: Functions of 2D spaces on one mesh, VectorFunctions (written as 3-component vectors,
      z = 0), ProductFunctions (one array per block, named name_0, name_1, ...), or (name, field) pairs.
    mesh: the mesh, when no field is given (or to check that the fields are on it).
    degree: the order of the output cells.  Default: the highest degree among the fields (1 for the
      mesh alone).  Order 1 writes linear triangles or quads; order k > 1 VTK's Lagrange cells, on
      which ParaView draws the element polynomial.  A field of lower degree is written exactly on
      them; one of higher degree is interpolated at their points.
    cell_data: {name: array of one value per cell}, such as an error indicator or
      mesh.report["cell_markers"] from read_gmsh (written automatically when present).
    segments: also write the mesh's segments as line cells, with their markers as cell data
      "marker" (zero on the 2D cells).  Default: True for the mesh alone, False with fields.
    binary: base64-encoded binary arrays (default), or ASCII.

    Returns the file name.  The points are the output lattice of every cell, shared between
    cells, so a continuous field is continuous in ParaView too.  With a discontinuous field among
    them (DGSpace2D, RTSpace, N1curlSpace: a VectorElementFunction is written as a 3-component
    vector) every cell keeps its own points, so the jumps are drawn as they are.  Periodic spaces are written on
    the mesh as it is, the identified sides as separate points."""
    flds = _fields(fields)
    meshes = {id(p[0].mesh): p[0].mesh for _, _, parts in flds for p in parts}
    if mesh is not None:
        if meshes and set(meshes) != {id(mesh)}:
            raise ValueError("write_vtk: the fields are not on the mesh given")
    else:
        if not meshes:
            raise TypeError("write_vtk: give at least one field, or the mesh with mesh=")
        if len(meshes) > 1:
            raise ValueError("write_vtk: the fields live on different meshes; write one file per mesh")
        mesh = next(iter(meshes.values()))
    n = int(degree) if degree is not None else max([p[0].degree for _, _, parts in flds for p in parts] or [1])
    if n < 1:
        raise ValueError(f"write_vtk: degree must be >= 1, got {n}")
    broken = any(getattr(p[0], "broken", False) or getattr(p[0], "value_shape", ()) == (2,)
                 for _, _, parts in flds for p in parts)
    lat = _Lattice(mesh, n, broken=broken)
    cd = {}
    rep = getattr(mesh, "report", {}) or {}
    if "cell_markers" in rep and len(rep["cell_markers"]) == lat.ncells:
        cd["cell_marker"] = np.asarray(rep["cell_markers"], dtype=np.int32)
    for k, v in (cell_data or {}).items():
        v = np.asarray(v)
        if v.shape[0] != lat.ncells:
            raise ValueError(f"write_vtk: cell_data['{k}'] has {v.shape[0]} entries, the mesh has {lat.ncells} cells")
        cd[k] = v
    want_lines = (not flds) if segments is None else bool(segments)
    lines = _segment_lines(lat) if want_lines else None
    if lines is not None:
        cd["marker"] = np.concatenate([np.zeros(lat.ncells, dtype=np.int32), lines[1]])
    filename = os.fspath(filename)
    if not filename.endswith(".vtu"):
        filename += ".vtu"
    text = _vtu(lat, flds, cd, lines, binary)
    with open(filename, "w") as f:
        f.write(text)
    return filename


class VTKSeries:
    """A time series for ParaView: one .vtu file per write() and a .pvd collection listing them.

        series = fd.VTKSeries("results/run.pvd")
        for n in range(N):
            irk.step(w)
            if n % 10 == 0:
                series.write(irk.t, w)                   # results/run_000000.vtu, ... ; run.pvd updated

    write(t, *fields, **options) takes the arguments of write_vtk after the file name.  The .pvd is
    rewritten after every file, so a run that stops early still leaves a readable series.  Open the
    .pvd in ParaView to get the time slider."""

    def __init__(self, filename, **options):
        filename = os.fspath(filename)
        if not filename.endswith(".pvd"):
            filename += ".pvd"
        self.filename, self.options, self.entries = filename, options, []
        d = os.path.dirname(filename)
        if d:
            os.makedirs(d, exist_ok=True)
        self._stem = os.path.splitext(os.path.basename(filename))[0]
        self._dir = d

    def write(self, t, *fields, **options):
        opts = dict(self.options, **options)
        name = f"{self._stem}_{len(self.entries):06d}.vtu"
        write_vtk(os.path.join(self._dir, name), *fields, **opts)
        self.entries.append((float(t), name))
        rows = "".join(f'<DataSet timestep="{tt!r}" group="" part="0" file="{f}"/>\n' for tt, f in self.entries)
        with open(self.filename, "w") as fh:
            fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                     f"<Collection>\n{rows}</Collection>\n</VTKFile>\n")
        return os.path.join(self._dir, name)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"VTKSeries({self.filename!r}, {len(self.entries)} files)"


# ============================================================================ Gmsh import
# element type -> (dimension, corner count, total nodes)
_GMSH = {1: (1, 2, 2), 8: (1, 2, 3), 26: (1, 2, 4), 27: (1, 2, 5), 28: (1, 2, 6),
         2: (2, 3, 3), 9: (2, 3, 6), 21: (2, 3, 10), 23: (2, 3, 15), 25: (2, 3, 21), 20: (2, 3, 9), 22: (2, 3, 12),
         3: (2, 4, 4), 10: (2, 4, 9), 16: (2, 4, 8), 36: (2, 4, 16), 37: (2, 4, 25),
         15: (0, 1, 1)}


def _sections(text):
    """{name: [lines]} of the $Name ... $EndName blocks."""
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("$End"):
            if cur is not None:
                out.setdefault(cur, buf)
            cur, buf = None, []
        elif s.startswith("$"):
            cur, buf = s[1:], []
        elif cur is not None:
            buf.append(s)
    return out


def _names(sec):
    names = {}
    if "PhysicalNames" in sec:
        lines = sec["PhysicalNames"]
        for line in lines[1:1 + int(lines[0])]:
            dim, tag, rest = line.split(maxsplit=2)
            names[(int(dim), int(tag))] = rest.strip().strip('"')
    return names


def _read_v2(sec):
    L = sec["Nodes"]
    nn = int(L[0])
    tags, xyz = np.empty(nn, dtype=np.int64), np.empty((nn, 3))
    for i, line in enumerate(L[1:1 + nn]):
        p = line.split()
        tags[i] = int(p[0])
        xyz[i] = [float(p[1]), float(p[2]), float(p[3])]
    elems = []                                           # (type, marker, node tags)
    E = sec["Elements"]
    for line in E[1:1 + int(E[0])]:
        p = [int(v) for v in line.split()]
        et, nt = p[1], p[2]
        t = p[3:3 + nt]
        marker = t[0] if nt >= 1 and t[0] != 0 else (t[1] if nt >= 2 else 0)
        elems.append((et, marker, p[3 + nt:]))
    return tags, xyz, elems


def _read_v4(sec):
    # physical tags of the entities
    phys = {}
    if "Entities" in sec:
        L = sec["Entities"]
        counts = [int(v) for v in L[0].split()]
        row = 1
        for dim, cnt in enumerate(counts):
            for _ in range(cnt):
                p = L[row].split()
                row += 1
                tag = int(p[0])
                k = 4 if dim == 0 else 7                 # points: tag x y z; others: tag + bounding box
                npt = int(p[k])
                phys[(dim, tag)] = [int(v) for v in p[k + 1:k + 1 + npt]]
    L = sec["Nodes"]
    nb, nn = (int(v) for v in L[0].split()[:2])
    tags, xyz = np.empty(nn, dtype=np.int64), np.empty((nn, 3))
    row, at = 1, 0
    for _ in range(nb):
        dim, etag, param, cnt = (int(v) for v in L[row].split())
        row += 1
        for i in range(cnt):
            tags[at + i] = int(L[row + i])
        row += cnt
        for i in range(cnt):
            v = L[row + i].split()
            xyz[at + i] = [float(v[0]), float(v[1]), float(v[2])]
        row += cnt
        at += cnt
    elems = []
    E = sec["Elements"]
    nb = int(E[0].split()[0])
    row = 1
    for _ in range(nb):
        dim, etag, et, cnt = (int(v) for v in E[row].split())
        row += 1
        pt = phys.get((dim, etag), [])
        marker = abs(pt[0]) if pt else etag
        for i in range(cnt):
            p = [int(v) for v in E[row + i].split()]
            elems.append((et, marker, p[1:]))
        row += cnt
    return tags, xyz, elems


def read_gmsh(filename, *, marker=0, tol=1e-9):
    """A 2D mesh from a Gmsh .msh file (ASCII format 2.2 or 4.1).

        m = fd.read_gmsh("channel.msh")                  # Mesh2D for triangles, QuadMesh for quads
        m.report["physical_names"]                       # {"inlet": 1, "wall": 3, ...}: side markers
        V = fd.LagrangeSpace(m, 2, dirichlet=[m.report["physical_names"]["wall"]])

    Triangles become a Mesh2D, quadrilaterals a QuadMesh (one kind per file; a mesh with both is
    refused, so recombine all or none).  Line elements become the segments, each with the tag of
    its physical group (or of its curve when it belongs to none) as its marker, which is what
    dirichlet= and ds(marker) name.  Boundary edges without a line element get `marker`.
    Higher-order elements are read by their corners: the sides become straight, and a warning
    says how far the dropped nodes were from them.  3D elements and a mesh that is not planar are
    refused.

    report keys: "physical_names" {name: tag} of the 1D groups (the side markers),
    "region_names" {name: tag} of the 2D groups, "cell_markers" (the physical, or else the
    surface, tag of every cell, in the cells' order), "gmsh_version", "order" (of the elements)."""
    from .mesh2d import mesh_from_arrays
    from .quadmesh import quad_mesh_from_arrays
    with open(os.fspath(filename), "r", errors="replace") as f:
        head = f.read(4096)
        if "\0" in head:
            raise ValueError(f"read_gmsh: {filename} is a binary .msh file; save it as ASCII (Mesh.Binary = 0, "
                             "or gmsh -format msh41 without -bin)")
        text = head + f.read()
    sec = _sections(text)
    if "MeshFormat" not in sec or "Nodes" not in sec or "Elements" not in sec:
        raise ValueError(f"read_gmsh: {filename} is not a Gmsh mesh file (no $MeshFormat, $Nodes and $Elements)")
    fmt = sec["MeshFormat"][0].split()
    version, ftype = fmt[0], int(fmt[1])
    if ftype != 0:
        raise ValueError(f"read_gmsh: {filename} is a binary .msh file; save it as ASCII (Mesh.Binary = 0)")
    major = int(float(version))
    if major == 2:
        tags, xyz, elems = _read_v2(sec)
    elif major == 4:
        tags, xyz, elems = _read_v4(sec)
    else:
        raise ValueError(f"read_gmsh: format {version} is not supported; save as 4.1 or 2.2")
    names = _names(sec)
    if np.ptp(xyz[:, 2]) > tol * max(float(np.ptp(xyz[:, :2], axis=0).max()), 1.0):
        raise ValueError("read_gmsh: the mesh is not planar (z varies); only 2D meshes are read")
    index = {int(t): i for i, t in enumerate(tags)}

    cells2, lines, kinds, orders, dropped = [], [], set(), set(), 0.0
    for et, mk, nodes in elems:
        info = _GMSH.get(et)
        if info is None:
            raise ValueError(f"read_gmsh: element type {et} is not a 2D element FEMd reads "
                             "(3D elements and incomplete high-order families are refused)")
        dim, nc, ntot = info
        if dim == 0:
            continue
        ids = [index[t] for t in nodes]
        if dim == 2:
            cells2.append((mk, ids[:nc], ids[nc:], et))
            kinds.add(nc)
        else:
            lines.append((mk, ids[:2], ids[2:], et))
        if ntot > nc:
            orders.add(et)
    if not cells2:
        raise ValueError("read_gmsh: the file has no triangles or quadrilaterals")
    if len(kinds) > 1:
        raise ValueError("read_gmsh: the mesh mixes triangles and quadrilaterals; FEMd reads one kind "
                         "(recombine all the surfaces, Mesh.RecombineAll = 1, or none)")
    nv = kinds.pop()
    # higher order: how far the edge nodes (Gmsh lists them after the corners, edge by edge) are from
    # the straight edges between the corners
    P = xyz[:, :2]

    def off(a, b, q):
        t = b - a
        return np.abs(t[0] * (q[:, 1] - a[1]) - t[1] * (q[:, 0] - a[0])) / max(float(np.hypot(*t)), 1e-300)
    for _, c, extra, et in (cells2 + lines) if orders else ():
        per = _order_of(et) - 1                          # nodes inside each edge
        ne = len(c) if len(c) > 2 else 1
        for e in range(ne):
            q = P[extra[e * per:(e + 1) * per]]
            if q.size:
                dropped = max(dropped, float(off(P[c[e]], P[c[(e + 1) % len(c)]], q).max()))
    # only the nodes the cells use, renumbered in their original order
    C = np.array([c for _, c, _, _ in cells2], dtype=np.int64)
    used = np.unique(C)
    new = np.full(len(tags), -1, dtype=np.int64)
    new[used] = np.arange(used.size)
    pts = P[used]
    cells = new[C]
    cm = np.array([mk for mk, _, _, _ in cells2], dtype=np.int64)
    segs, smk = [], []
    for mk, (a, b), _, _ in lines:
        if new[a] < 0 or new[b] < 0:
            continue
        segs.append((new[a], new[b]))
        smk.append(mk)
    segs = np.array(segs, dtype=np.int64).reshape(-1, 2)
    if segs.shape[0]:
        key = np.sort(segs, axis=1)
        _, keep = np.unique(key, axis=0, return_index=True)          # a side in two groups: the first wins
        keep = np.sort(keep)
        segs, smk = segs[keep], np.asarray(smk)[keep]
    if nv == 3:
        m = mesh_from_arrays(pts, cells, segs, np.asarray(smk, dtype=np.int64), marker=int(marker))
    else:
        m = quad_mesh_from_arrays(pts, cells, segs, np.asarray(smk, dtype=np.int64), marker=int(marker))
    size = float(np.ptp(pts, axis=0).max())
    if orders and dropped > 1e-10 * size:
        warnings.warn(f"read_gmsh: the mesh has higher-order elements; FEMd reads their corners, so curved sides "
                      f"become straight (the dropped nodes are up to {dropped:.3g} from them)", RuntimeWarning,
                      stacklevel=2)
    m.report["gmsh_version"] = version
    m.report["order"] = 1 if not orders else max(_order_of(et) for et in orders)
    m.report["physical_names"] = {nm: tag for (d, tag), nm in names.items() if d == 1}
    m.report["region_names"] = {nm: tag for (d, tag), nm in names.items() if d == 2}
    m.report["cell_markers"] = cm
    return m


def _order_of(et):
    return {8: 2, 9: 2, 10: 2, 16: 2, 20: 3, 21: 3, 26: 3, 36: 3, 22: 4, 23: 4, 27: 4, 37: 4, 25: 5, 28: 5}.get(et, 1)
