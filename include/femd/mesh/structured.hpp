//
//  structured.hpp  --  a Mesh2D from given arrays, and structured triangular
//  meshes of a rectangle or of a curved four-sided region.
//
//      Mesh2D M = mesh_from_arrays(points, triangles);          // any triangulation you already have
//      Mesh2D R = rectangle_mesh(0, 0, 2, 1, 40, 20);           // tensor grid, one diagonal per cell
//      Mesh2D Q = mapped_mesh(bottom, right, top, left);        // transfinite (Coons) map of the unit square
//
//  mesh_from_arrays() is the foundation.  It builds the neighbour links from the
//  connectivity, turns every boundary edge into a constrained segment, and marks
//  the boundary vertices, so the result is an ordinary Mesh2D: validate(),
//  smoothing and the three local refinement schemes work on it unchanged.
//
//  Structured numbering.  Grid vertex (i, j), 0 <= i <= nx, 0 <= j <= ny, is point
//  j*(nx+1) + i, so the grid is stored row by row from the bottom.  With the
//  crossed pattern the cell centres follow, centre (i, j) being point
//  (nx+1)*(ny+1) + j*nx + i.  Triangles are numbered cell by cell in the same
//  order, 2 per cell (4 for crossed).
//
//  Diagonals, for the cell with corners a = (i,j), b = (i+1,j), c = (i+1,j+1),
//  d = (i,j+1):
//      Right      a-c, the "/" diagonal         (a,b,c) (a,c,d)
//      Left       b-d, the "\" diagonal         (a,b,d) (b,c,d)
//      Alternate  Right where i+j is even, Left where it is odd
//      Crossed    centre m, four triangles      (a,b,m) (b,c,m) (c,d,m) (d,a,m)
//
//  Markers.  markers = {bottom, right, top, left}.  Every segment carries the
//  marker of its side.  A boundary vertex carries the marker of the boundary edge
//  leaving it counter-clockwise, so the corners (0,0), (nx,0), (nx,ny), (0,ny)
//  carry the bottom, right, top and left markers.  This is also the rule
//  mesh_from_arrays() applies to any mesh when no point markers are given.
//
#ifndef FEMD_MESH_STRUCTURED_HPP
#define FEMD_MESH_STRUCTURED_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/predicates.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace femd {

namespace detail {

inline std::int64_t edge_key(int a, int b)
{
    std::int64_t lo = std::min(a, b), hi = std::max(a, b);
    return (hi << 32) | (lo & 0xffffffffLL);
}

inline std::string pt_str(const Point &p)
{
    std::ostringstream s;
    s.precision(6);
    s << "(" << p.x << ", " << p.y << ")";
    return s.str();
}

} // namespace detail

/// @brief What mesh_from_arrays() did beyond copying the input.
struct FromArraysReport {
    int flipped = 0;          ///< clockwise triangles re-oriented counter-clockwise
    int boundary_edges = 0;   ///< edges with one triangle, all made segments
    int boundary_loops = 0;   ///< closed boundary loops (1 + number of holes for a connected mesh)
};

/**
 * @brief A Mesh2D from a point array and a triangle array.
 *
 * @param pts            vertex coordinates.
 * @param tris           vertex indices, three per triangle, either orientation.  A clockwise
 *                       triangle is re-oriented; triangle t of the result is triangle t of the input.
 * @param segs           optional constrained edges.  Each must be an edge of the mesh.  Boundary
 *                       edges not listed become segments with marker `marker`, so the whole
 *                       boundary is always constrained.  Interior edges may be listed too.
 * @param seg_markers    one marker per entry of segs, or empty for `marker`.
 * @param point_markers  one marker per point, or empty.  Empty gives 0 to interior vertices and
 *                       to each boundary vertex the marker of the boundary edge leaving it
 *                       counter-clockwise (then, for vertices only on interior segments, the
 *                       marker of such a segment).
 * @param marker         the marker of unlisted boundary edges.
 * @param report         optional: counts of what was done.
 *
 * @throws std::invalid_argument for an index out of range, a triangle with a repeated or
 *         collinear vertices, a point used by no triangle, an edge shared by more than two
 *         triangles, two triangles on the same side of an edge (the mesh overlaps or folds
 *         there), a listed segment that is not an edge, or mismatched marker counts.
 */
inline Mesh2D mesh_from_arrays(const std::vector<Point> &pts, std::vector<std::array<int, 3>> tris,
                               const std::vector<std::array<int, 2>> &segs = {},
                               const std::vector<int> &seg_markers = {},
                               const std::vector<int> &point_markers = {},
                               int marker = 1, FromArraysReport *report = nullptr)
{
    const int np = static_cast<int>(pts.size()), nt = static_cast<int>(tris.size());
    if (np < 3 || nt < 1) throw std::invalid_argument("mesh_from_arrays: need at least 3 points and 1 triangle");
    if (!seg_markers.empty() && seg_markers.size() != segs.size())
        throw std::invalid_argument("mesh_from_arrays: segment_markers must have one entry per segment");
    if (!point_markers.empty() && static_cast<int>(point_markers.size()) != np)
        throw std::invalid_argument("mesh_from_arrays: point_markers must have one entry per point");

    FromArraysReport rep;
    std::vector<char> used(np, 0);
    for (int t = 0; t < nt; ++t)
    {
        auto &T = tris[t];
        for (int k = 0; k < 3; ++k)
        {
            if (T[k] < 0 || T[k] >= np)
                throw std::invalid_argument("mesh_from_arrays: triangle " + std::to_string(t) + " has vertex index "
                                            + std::to_string(T[k]) + ", outside 0.." + std::to_string(np - 1));
            used[T[k]] = 1;
        }
        if (T[0] == T[1] || T[1] == T[2] || T[0] == T[2])
            throw std::invalid_argument("mesh_from_arrays: triangle " + std::to_string(t) + " repeats a vertex");
        const Point &a = pts[T[0]], &b = pts[T[1]], &c = pts[T[2]];
        int o = pred::orient2d(a.x, a.y, b.x, b.y, c.x, c.y);
        if (o == 0)
            throw std::invalid_argument("mesh_from_arrays: triangle " + std::to_string(t) + " is degenerate, its vertices "
                                        + detail::pt_str(a) + " " + detail::pt_str(b) + " " + detail::pt_str(c)
                                        + " are collinear");
        if (o < 0) { std::swap(T[1], T[2]); ++rep.flipped; }
    }
    for (int v = 0; v < np; ++v)
        if (!used[v])
            throw std::invalid_argument("mesh_from_arrays: point " + std::to_string(v) + " " + detail::pt_str(pts[v])
                                        + " is not a vertex of any triangle");

    Mesh2D M;
    for (const Point &p : pts) M.add_point(p, 0);
    for (const auto &T : tris) M.new_triangle(T[0], T[1], T[2]);

    // Pair up the edges.  After re-orientation a shared edge must appear once in each direction.
    struct Half { int t, i, count; };
    std::unordered_map<std::int64_t, Half> half;
    half.reserve(static_cast<std::size_t>(3 * nt));
    for (int t = 0; t < nt; ++t)
        for (int i = 0; i < 3; ++i)
        {
            int a = tris[t][(i + 1) % 3], b = tris[t][(i + 2) % 3];
            auto key = detail::edge_key(a, b);
            auto it = half.find(key);
            if (it == half.end()) { half.emplace(key, Half{t, i, 1}); continue; }
            Half &h = it->second;
            if (h.count >= 2)
                throw std::invalid_argument("mesh_from_arrays: edge (" + std::to_string(a) + ", " + std::to_string(b)
                                            + ") is shared by more than two triangles");
            int u = h.t, j = h.i;
            int c = tris[u][(j + 1) % 3], d = tris[u][(j + 2) % 3];
            if (!(c == b && d == a))
                throw std::invalid_argument("mesh_from_arrays: triangles " + std::to_string(u) + " and " + std::to_string(t)
                                            + " lie on the same side of their shared edge (" + std::to_string(a) + ", "
                                            + std::to_string(b) + "), so the mesh overlaps or folds there");
            M.link(t, i, u, j);
            h.count = 2;
        }

    // Boundary edges, directed counter-clockwise (as they appear in their triangle).
    std::vector<std::vector<int>> out(np);               // boundary edges leaving each vertex
    std::vector<std::array<int, 2>> bedge;
    for (int t = 0; t < nt; ++t)
        for (int i = 0; i < 3; ++i)
            if (M.neighbour(t, i) < 0)
            {
                int a = tris[t][(i + 1) % 3], b = tris[t][(i + 2) % 3];
                out[a].push_back(static_cast<int>(bedge.size()));
                bedge.push_back({a, b});
            }
    rep.boundary_edges = static_cast<int>(bedge.size());

    // User segments first, so their markers win.
    for (std::size_t s = 0; s < segs.size(); ++s)
    {
        int a = segs[s][0], b = segs[s][1];
        if (a < 0 || a >= np || b < 0 || b >= np || half.find(detail::edge_key(a, b)) == half.end())
            throw std::invalid_argument("mesh_from_arrays: segment " + std::to_string(s) + " (" + std::to_string(a) + ", "
                                        + std::to_string(b) + ") is not an edge of the mesh");
        M.add_segment(a, b, seg_markers.empty() ? marker : seg_markers[s]);
    }

    // Walk the boundary loops from the lowest vertex, adding unlisted edges in loop order.
    std::vector<char> done(bedge.size(), 0);
    std::vector<int> pmark(np, 0);
    std::vector<char> pset(np, 0);
    for (int start = 0; start < np; ++start)
        for (int e0 : out[start])
        {
            if (done[e0]) continue;
            ++rep.boundary_loops;
            int e = e0;
            while (e >= 0 && !done[e])
            {
                done[e] = 1;
                int a = bedge[e][0], b = bedge[e][1];
                if (!M.is_segment(a, b)) M.add_segment(a, b, marker);
                if (!pset[a]) { pmark[a] = M.segment_markers()[M.segment_index(a, b)]; pset[a] = 1; }
                int next = -1;
                for (int f : out[b]) if (!done[f]) { next = f; break; }
                e = next;
            }
        }

    // Vertices only on interior segments take the marker of such a segment.
    for (std::size_t s = 0; s < segs.size(); ++s)
        for (int v : {segs[s][0], segs[s][1]})
            if (!pset[v]) { pmark[v] = seg_markers.empty() ? marker : seg_markers[s]; pset[v] = 1; }

    for (int v = 0; v < np; ++v) M.set_point_marker(v, point_markers.empty() ? pmark[v] : point_markers[v]);
    if (report) *report = rep;
    return M;
}

// ---------------------------------------------------------------------------
enum class Diagonal { Right, Left, Alternate, Crossed };

/// @brief "right", "left", "alternate", "crossed" (also "crisscross", "union-jack"), any case.
inline Diagonal diagonal_from_name(std::string s)
{
    for (auto &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (s == "right" || s == "/") return Diagonal::Right;
    if (s == "left" || s == "\\") return Diagonal::Left;
    if (s == "alternate" || s == "alternating" || s == "right/left") return Diagonal::Alternate;
    if (s == "crossed" || s == "crisscross" || s == "criss-cross" || s == "union-jack" || s == "unionjack") return Diagonal::Crossed;
    throw std::invalid_argument("unknown diagonal '" + s + "': use 'right', 'left', 'alternate' or 'crossed'");
}

inline const char *diagonal_name(Diagonal d)
{
    switch (d) {
    case Diagonal::Right: return "right";
    case Diagonal::Left: return "left";
    case Diagonal::Alternate: return "alternate";
    default: return "crossed";
    }
}

/**
 * @brief Triangulate a structured grid of vertices.
 *
 * @param grid     (nx+1)*(ny+1) points, vertex (i, j) at index j*(nx+1) + i.  (0,0), (nx,0),
 *                 (nx,ny), (0,ny) are the corners, bottom, right, top and left the four sides.
 * @param nx, ny   cells in each direction, at least 1.
 * @param diag     the diagonal pattern.
 * @param markers  {bottom, right, top, left}.
 * @throws std::invalid_argument when the grid does not give a valid mesh: some triangles are
 *         inverted or degenerate, which means the sides cross or a cell is too distorted for
 *         this diagonal.  A grid whose cells are all clockwise (a mirrored parameterization) is
 *         accepted and re-oriented.
 */
inline Mesh2D structured_grid_mesh(std::vector<Point> grid, int nx, int ny, Diagonal diag = Diagonal::Right,
                                   std::array<int, 4> markers = {1, 2, 3, 4})
{
    if (nx < 1 || ny < 1) throw std::invalid_argument("structured mesh: need nx, ny >= 1");
    if (static_cast<long>(grid.size()) != static_cast<long>(nx + 1) * (ny + 1))
        throw std::invalid_argument("structured mesh: need (nx+1)*(ny+1) grid points");
    auto V = [nx](int i, int j) { return j * (nx + 1) + i; };
    const int nc0 = (nx + 1) * (ny + 1);
    if (diag == Diagonal::Crossed)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const Point &a = grid[V(i, j)], &b = grid[V(i + 1, j)], &c = grid[V(i + 1, j + 1)], &d = grid[V(i, j + 1)];
                grid.push_back(Point{0.25 * (a.x + b.x + c.x + d.x), 0.25 * (a.y + b.y + c.y + d.y)});
            }

    std::vector<std::array<int, 3>> tris;
    tris.reserve(static_cast<std::size_t>(nx) * ny * (diag == Diagonal::Crossed ? 4 : 2));
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            int a = V(i, j), b = V(i + 1, j), c = V(i + 1, j + 1), d = V(i, j + 1);
            bool right = diag == Diagonal::Right || (diag == Diagonal::Alternate && (i + j) % 2 == 0);
            if (diag == Diagonal::Crossed)
            {
                int m = nc0 + j * nx + i;
                tris.push_back({a, b, m}); tris.push_back({b, c, m});
                tris.push_back({c, d, m}); tris.push_back({d, a, m});
            }
            else if (right) { tris.push_back({a, b, c}); tris.push_back({a, c, d}); }
            else            { tris.push_back({a, b, d}); tris.push_back({b, c, d}); }
        }

    // Every triangle must have the same, nonzero orientation.
    int pos = 0, neg = 0, zero = 0, first_bad = -1;
    std::vector<int> sgn(tris.size());
    for (std::size_t t = 0; t < tris.size(); ++t)
    {
        const Point &p = grid[tris[t][0]], &q = grid[tris[t][1]], &r = grid[tris[t][2]];
        sgn[t] = pred::orient2d(p.x, p.y, q.x, q.y, r.x, r.y);
        (sgn[t] > 0 ? pos : sgn[t] < 0 ? neg : zero)++;
    }
    const bool mirrored = neg > pos;
    for (std::size_t t = 0; t < tris.size() && first_bad < 0; ++t)
        if (sgn[t] == 0 || (sgn[t] < 0) != mirrored) first_bad = static_cast<int>(t);
    if (first_bad >= 0)
    {
        int per = diag == Diagonal::Crossed ? 4 : 2;
        int cell = first_bad / per, ci = cell % nx, cj = cell / nx;
        int bad = zero + (mirrored ? pos : neg);
        throw std::invalid_argument("structured mesh: the map from the unit square is not one-to-one: "
                                    + std::to_string(bad) + " of " + std::to_string(tris.size())
                                    + " triangles are inverted or degenerate, the first in cell (" + std::to_string(ci)
                                    + ", " + std::to_string(cj) + ").  The sides may cross, or the cells are too "
                                    "distorted for this diagonal; more cells or another diagonal can help");
    }
    if (mirrored) for (auto &T : tris) std::swap(T[1], T[2]);

    // Segments by side, and the corner rule for the vertex markers.
    std::vector<std::array<int, 2>> segs;
    std::vector<int> smk;
    for (int i = 0; i < nx; ++i) { segs.push_back({V(i, 0), V(i + 1, 0)});   smk.push_back(markers[0]); }
    for (int j = 0; j < ny; ++j) { segs.push_back({V(nx, j), V(nx, j + 1)}); smk.push_back(markers[1]); }
    for (int i = nx; i > 0; --i) { segs.push_back({V(i, ny), V(i - 1, ny)}); smk.push_back(markers[2]); }
    for (int j = ny; j > 0; --j) { segs.push_back({V(0, j), V(0, j - 1)});   smk.push_back(markers[3]); }
    std::vector<int> pm(grid.size(), 0);
    for (int i = 0; i < nx; ++i) pm[V(i, 0)] = markers[0];
    for (int j = 0; j < ny; ++j) pm[V(nx, j)] = markers[1];
    for (int i = nx; i > 0; --i) pm[V(i, ny)] = markers[2];
    for (int j = ny; j > 0; --j) pm[V(0, j)] = markers[3];

    return mesh_from_arrays(grid, std::move(tris), segs, smk, pm, markers[0]);
}

/**
 * @brief Structured mesh of the rectangle [x0, x1] x [y0, y1] with nx x ny equal cells.
 * @throws std::invalid_argument for x1 <= x0, y1 <= y0 or a count below 1.
 */
inline Mesh2D rectangle_mesh(double x0, double y0, double x1, double y1, int nx, int ny,
                             Diagonal diag = Diagonal::Right, std::array<int, 4> markers = {1, 2, 3, 4})
{
    if (!(x1 > x0 && y1 > y0)) throw std::invalid_argument("rectangle_mesh: need x1 > x0 and y1 > y0");
    if (nx < 1 || ny < 1) throw std::invalid_argument("rectangle_mesh: need nx, ny >= 1");
    std::vector<Point> g(static_cast<std::size_t>(nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; ++j)
    {
        double y = j == ny ? y1 : y0 + (y1 - y0) * j / ny;
        for (int i = 0; i <= nx; ++i)
            g[j * (nx + 1) + i] = Point{i == nx ? x1 : x0 + (x1 - x0) * i / nx, y};
    }
    return structured_grid_mesh(std::move(g), nx, ny, diag, markers);
}

/**
 * @brief Structured mesh of a four-sided region by transfinite (Coons) interpolation.
 *
 * The unit square is mapped by
 *     X(xi, eta) = (1-eta) B(xi) + eta T(xi) + (1-xi) L(eta) + xi R(eta)
 *                  - [(1-xi)(1-eta) P00 + xi(1-eta) P10 + (1-xi) eta P01 + xi eta P11]
 * at xi = i/nx, eta = j/ny, with the four sides given as samples at those parameters.
 *
 * @param bottom  nx+1 points from P00 to P10.
 * @param right   ny+1 points from P10 to P11.
 * @param top     nx+1 points from P01 to P11.
 * @param left    ny+1 points from P00 to P01.
 *                A side given the other way round (top from P11 to P01, as a counter-clockwise
 *                walk gives it) is detected at the corners and reversed.  Corners must agree to
 *                1e-10 of the size of the region; each is then snapped to one value.
 * @param diag, markers  as in structured_grid_mesh().
 *
 * The boundary vertices are the given samples exactly.  For straight sides the map reproduces
 * the tensor grid of the samples, so it is exact for a rectangle with any spacing along its
 * sides, and bilinear for any convex quadrilateral.
 */
inline Mesh2D mapped_mesh(std::vector<Point> bottom, std::vector<Point> right, std::vector<Point> top,
                          std::vector<Point> left, Diagonal diag = Diagonal::Right,
                          std::array<int, 4> markers = {1, 2, 3, 4})
{
    const int nx = static_cast<int>(bottom.size()) - 1, ny = static_cast<int>(left.size()) - 1;
    if (nx < 1 || ny < 1) throw std::invalid_argument("mapped_mesh: each side needs at least 2 points");
    if (static_cast<int>(top.size()) != nx + 1)
        throw std::invalid_argument("mapped_mesh: bottom and top need the same number of points ("
                                    + std::to_string(nx + 1) + " and " + std::to_string(top.size()) + ")");
    if (static_cast<int>(right.size()) != ny + 1)
        throw std::invalid_argument("mapped_mesh: left and right need the same number of points ("
                                    + std::to_string(ny + 1) + " and " + std::to_string(right.size()) + ")");

    double lx = 1e300, ly = 1e300, hx = -1e300, hy = -1e300;
    for (const auto *s : {&bottom, &right, &top, &left})
        for (const Point &p : *s) { lx = std::min(lx, p.x); ly = std::min(ly, p.y); hx = std::max(hx, p.x); hy = std::max(hy, p.y); }
    const double tol = 1e-10 * std::max(std::hypot(hx - lx, hy - ly), 1e-300);
    auto near = [tol](const Point &p, const Point &q) { return std::hypot(p.x - q.x, p.y - q.y) <= tol; };

    // Try each side as given and reversed; take the first combination that closes.
    bool ok = false;
    for (int mask = 0; mask < 16 && !ok; ++mask)
    {
        auto B = bottom, R = right, T = top, L = left;
        if (mask & 1) std::reverse(B.begin(), B.end());
        if (mask & 2) std::reverse(R.begin(), R.end());
        if (mask & 4) std::reverse(T.begin(), T.end());
        if (mask & 8) std::reverse(L.begin(), L.end());
        if (near(B.front(), L.front()) && near(B.back(), R.front()) && near(T.front(), L.back()) && near(T.back(), R.back()))
        {
            bottom.swap(B); right.swap(R); top.swap(T); left.swap(L);
            ok = true;
        }
    }
    if (!ok)
        throw std::invalid_argument("mapped_mesh: the four sides do not meet at four corners.  bottom runs "
                                    + detail::pt_str(bottom.front()) + " to " + detail::pt_str(bottom.back()) + ", right "
                                    + detail::pt_str(right.front()) + " to " + detail::pt_str(right.back()) + ", top "
                                    + detail::pt_str(top.front()) + " to " + detail::pt_str(top.back()) + ", left "
                                    + detail::pt_str(left.front()) + " to " + detail::pt_str(left.back())
                                    + ".  Expected bottom P00->P10, right P10->P11, top P01->P11, left P00->P01");

    const Point P00 = bottom.front(), P10 = bottom.back(), P01 = left.back(), P11 = right.back();
    left.front() = P00; right.front() = P10; top.front() = P01; top.back() = P11;

    std::vector<Point> g(static_cast<std::size_t>(nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i)
        {
            Point p;
            if (j == 0) p = bottom[i];
            else if (j == ny) p = top[i];
            else if (i == 0) p = left[j];
            else if (i == nx) p = right[j];
            else
            {
                const double s = static_cast<double>(i) / nx, t = static_cast<double>(j) / ny;
                const Point &B = bottom[i], &T = top[i], &L = left[j], &R = right[j];
                p.x = (1 - t) * B.x + t * T.x + (1 - s) * L.x + s * R.x
                      - ((1 - s) * (1 - t) * P00.x + s * (1 - t) * P10.x + (1 - s) * t * P01.x + s * t * P11.x);
                p.y = (1 - t) * B.y + t * T.y + (1 - s) * L.y + s * R.y
                      - ((1 - s) * (1 - t) * P00.y + s * (1 - t) * P10.y + (1 - s) * t * P01.y + s * t * P11.y);
            }
            g[j * (nx + 1) + i] = p;
        }
    return structured_grid_mesh(std::move(g), nx, ny, diag, markers);
}

} // namespace femd

#endif // FEMD_MESH_STRUCTURED_HPP
