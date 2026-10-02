//
//  coarsen.hpp  --  remove vertices from a mesh you already have.
//
//      RemoveReport r;
//      remove_vertices(M, {12, 40, 41}, RemoveOptions{}, &r);   // in place
//      M.compact();                                              // squeeze out the removed points
//
//  Removing a vertex v deletes the triangles around it and fills the hole, its
//  link polygon, with new triangles on the polygon's vertices.  No other vertex
//  moves and no vertex is added, so a nodal vector carries over to the coarser
//  mesh by dropping the entries of the removed vertices.
//
//  The hole is filled by Delaunay ear clipping (Devillers): repeatedly cut off
//  an ear of the polygon whose circumcircle holds no other polygon vertex.  When
//  the mesh around v is Delaunay this gives exactly the Delaunay triangulation of
//  the remaining points, since deleting a vertex changes nothing outside its
//  star.  When it is not (after bisection, say) no such ear may exist, and the
//  valid ear with the largest smallest angle is cut instead.
//
//  Which vertices can go:
//    - an interior vertex that is not the end of an interior segment;
//    - a boundary vertex with exactly two segments, both with the same marker,
//      lying on the straight line between its two boundary neighbours (to 1e-10 of
//      their distance), and whose own marker is that of its side or of both its
//      boundary neighbours.  Its two segments merge into one, so the boundary and
//      the area are unchanged.  Corners, and vertices a marker singles out, stay.
//  A removal is also refused when the fill would need an edge the mesh already has
//  outside the hole, or when a new triangle breaks the optional quality and size
//  bounds (it never leaves a triangle inverted or degenerate).
//
#ifndef FEMD_TRIANGULATE_COARSEN_HPP
#define FEMD_TRIANGULATE_COARSEN_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace femd {

struct RemoveOptions {
    double min_angle = 0.0;   ///< degrees; every new triangle must reach it (0: validity only)
    double max_edge  = 0.0;   ///< every new edge at most this long (0: unbounded)
    double max_area  = 0.0;   ///< every new triangle at most this large (0: unbounded)
    bool   boundary  = true;  ///< allow removing boundary vertices on straight sides
};

struct RemoveReport {
    int removed = 0;
    int passes = 0;
    std::vector<int>         kept;     ///< requested vertices still in the mesh
    std::vector<std::string> reasons;  ///< why, one per entry of kept
};

namespace detail {

inline bool strictly_inside_or_on(const Point &a, const Point &b, const Point &c, const Point &p)
{
    return pred::orient2d(a.x, a.y, b.x, b.y, p.x, p.y) >= 0 && pred::orient2d(b.x, b.y, c.x, c.y, p.x, p.y) >= 0
           && pred::orient2d(c.x, c.y, a.x, a.y, p.x, p.y) >= 0;
}

} // namespace detail

/**
 * @brief Remove vertex v, filling its hole with new triangles.  In place.
 * @param why  receives the reason when the vertex is kept.
 * @return true when v was removed.  On false the mesh is unchanged.
 */
inline bool remove_vertex(Mesh2D &M, int v, const RemoveOptions &opt = {}, std::string *why = nullptr)
{
    auto refuse = [why](const char *r) { if (why) *why = r; return false; };
    if (v < 0 || v >= M.npoints()) return refuse("not a vertex of the mesh");
    if (M.vertex_triangle(v) < 0) return refuse("not a vertex of the mesh");

    std::vector<int> fan;
    const bool interior = M.one_ring(v, fan);

    // The hole's boundary: one directed edge p -> q per fan triangle (v, p, q), counter-clockwise,
    // with the triangle outside it.
    struct Edge { int p, q, out, outj; };
    std::vector<Edge> E;
    for (int t : fan)
    {
        const auto &T = M.triangle(t);
        int k = M.local_index(t, v);
        int p = T[(k + 1) % 3], q = T[(k + 2) % 3];
        if (M.is_segment(v, p) && interior) return refuse("it is the end of an interior segment");
        int o = M.neighbour(t, k);
        E.push_back({p, q, o, o >= 0 ? M.local_neighbour(o, t) : -1});
    }

    // Chain the edges into a polygon, starting where the chain is open (boundary) or anywhere.
    int start = E[0].p;
    if (!interior)
    {
        for (const Edge &e : E)
        {
            bool is_q = false;
            for (const Edge &f : E) if (f.q == e.p) { is_q = true; break; }
            if (!is_q) { start = e.p; break; }
        }
    }
    std::vector<int> P;
    std::vector<Edge> PE;
    int cur = start;
    for (std::size_t n = 0; n < E.size(); ++n)
    {
        auto it = std::find_if(E.begin(), E.end(), [cur](const Edge &e) { return e.p == cur; });
        if (it == E.end()) return refuse("its star is not a simple fan");
        P.push_back(cur);
        PE.push_back(*it);
        cur = it->q;
    }
    int marker = 0;
    if (interior)
    {
        if (cur != start) return refuse("its star is not a simple fan");
    }
    else
    {
        if (!opt.boundary) return refuse("it is on the boundary and boundary=false");
        P.push_back(cur);                                   // the chain's far end
        PE.push_back({cur, start, -1, -1});                 // the closing edge, the new boundary edge
        const int a = start, b = cur;
        int sa = M.segment_index(v, a), sb = M.segment_index(v, b);
        if (sa < 0 || sb < 0) return refuse("its boundary edges are not segments");
        int nseg = 0;
        for (int s = 0; s < M.nsegments(); ++s) nseg += (M.segments()[s][0] == v || M.segments()[s][1] == v);
        if (nseg != 2) return refuse("more than two segments meet at it");
        const Point &A = M.point(a), &B = M.point(b), &V = M.point(v);
        double abx = B.x - A.x, aby = B.y - A.y, L = std::hypot(abx, aby);
        double dist = std::fabs(abx * (V.y - A.y) - aby * (V.x - A.x)) / L;
        double s = (abx * (V.x - A.x) + aby * (V.y - A.y)) / (L * L);
        if (!(dist <= 1e-10 * L && s > 0.0 && s < 1.0)) return refuse("it is a corner of the boundary");
        if (M.segment_markers()[sa] != M.segment_markers()[sb]) return refuse("its two sides carry different markers");
        marker = M.segment_markers()[sa];
        const int pm = M.point_markers()[v];
        if (pm != marker && !(pm == M.point_markers()[a] && pm == M.point_markers()[b]))
            return refuse("its marker singles it out on its side");
    }
    const int m = static_cast<int>(P.size());
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j)
            if (P[i] == P[j]) return refuse("its star is pinched");
    if (m < 3) return refuse("its star is too small");

    // Ear clipping.  ids index P.
    std::vector<int> ids(m);
    for (int i = 0; i < m; ++i) ids[i] = i;
    std::vector<std::array<int, 3>> newt;                  // vertex indices of the mesh
    auto pt = [&](int i) -> const Point & { return M.point(P[i]); };
    while (ids.size() > 3)
    {
        const int r = static_cast<int>(ids.size());
        int best = -1, best_empty = -1;
        double best_angle = -1.0;
        for (int k = 0; k < r; ++k)
        {
            int ia = ids[(k + r - 1) % r], ib = ids[k], ic = ids[(k + 1) % r];
            const Point &a = pt(ia), &b = pt(ib), &c = pt(ic);
            if (pred::orient2d(a.x, a.y, b.x, b.y, c.x, c.y) <= 0) continue;
            bool blocked = false, empty = true;
            for (int l = 0; l < r && !blocked; ++l)
            {
                int il = ids[l];
                if (il == ia || il == ib || il == ic) continue;
                const Point &x = pt(il);
                if (detail::strictly_inside_or_on(a, b, c, x)) blocked = true;
                else if (pred::incircle(a.x, a.y, b.x, b.y, c.x, c.y, x.x, x.y) > 0) empty = false;
            }
            if (blocked) continue;
            int t1, i1, t2, i2;
            if (M.edge_triangles(P[ia], P[ic], t1, i1, t2, i2)) continue;   // that edge exists outside the hole
            if (empty && best_empty < 0) best_empty = k;
            double ang = min_angle_of(a, b, c);
            if (ang > best_angle) { best_angle = ang; best = k; }
        }
        int k = best_empty >= 0 ? best_empty : best;
        if (k < 0) return refuse("its hole cannot be filled without inverting a triangle or duplicating an edge");
        newt.push_back({P[ids[(k + r - 1) % r]], P[ids[k]], P[ids[(k + 1) % r]]});
        ids.erase(ids.begin() + k);
    }
    {
        const Point &a = pt(ids[0]), &b = pt(ids[1]), &c = pt(ids[2]);
        if (pred::orient2d(a.x, a.y, b.x, b.y, c.x, c.y) <= 0)
            return refuse("its hole cannot be filled without inverting a triangle or duplicating an edge");
        newt.push_back({P[ids[0]], P[ids[1]], P[ids[2]]});
    }

    // Quality and size of the new triangles.
    for (const auto &T : newt)
    {
        const Point &a = M.point(T[0]), &b = M.point(T[1]), &c = M.point(T[2]);
        if (opt.min_angle > 0.0 && min_angle_of(a, b, c) < opt.min_angle) return refuse("a new triangle would be below min_angle");
        if (opt.max_edge > 0.0)
        {
            double e2 = std::max(sqdist(a, b), std::max(sqdist(b, c), sqdist(c, a)));
            if (e2 > opt.max_edge * opt.max_edge) return refuse("a new edge would exceed max_edge");
        }
        if (opt.max_area > 0.0)
        {
            double ar = 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
            if (ar > opt.max_area) return refuse("a new triangle would exceed max_area");
        }
    }

    // Commit.
    for (int t : fan) M.kill_triangle(t);
    struct Slot { int t, i; };
    std::vector<std::pair<std::array<int, 2>, Slot>> inner;   // diagonals, to pair up
    for (const auto &T : newt)
    {
        int t = M.new_triangle(T[0], T[1], T[2]);
        for (int i = 0; i < 3; ++i)
        {
            int p = T[(i + 1) % 3], q = T[(i + 2) % 3];
            auto it = std::find_if(PE.begin(), PE.end(), [p, q](const Edge &e) { return e.p == p && e.q == q; });
            if (it != PE.end()) { if (it->out >= 0) M.link(t, i, it->out, it->outj); continue; }
            auto jt = std::find_if(inner.begin(), inner.end(),
                                   [p, q](const std::pair<std::array<int, 2>, Slot> &x) { return x.first[0] == q && x.first[1] == p; });
            if (jt != inner.end()) { M.link(t, i, jt->second.t, jt->second.i); inner.erase(jt); }
            else inner.push_back({{p, q}, {t, i}});
        }
    }
    if (!interior) M.merge_segments_at(v);
    if (why) why->clear();
    return true;
}

/**
 * @brief Remove every listed vertex that can go.  In place.
 *
 * Sweeps the list until a sweep removes nothing, since removing one vertex can make a
 * neighbour removable.  The removed points stay in the point array, unreferenced, until
 * M.compact() drops them and returns the old-to-new map.
 */
inline int remove_vertices(Mesh2D &M, const std::vector<int> &vertices, const RemoveOptions &opt = {},
                           RemoveReport *report = nullptr)
{
    std::vector<int> todo = vertices;
    std::sort(todo.begin(), todo.end());
    todo.erase(std::unique(todo.begin(), todo.end()), todo.end());
    std::vector<std::string> why(todo.size());
    std::vector<char> gone(todo.size(), 0);
    int removed = 0, passes = 0;
    for (bool progress = true; progress;)
    {
        progress = false;
        ++passes;
        for (std::size_t k = 0; k < todo.size(); ++k)
        {
            if (gone[k]) continue;
            if (remove_vertex(M, todo[k], opt, &why[k])) { gone[k] = 1; ++removed; progress = true; }
        }
    }
    if (report)
    {
        report->removed = removed;
        report->passes = passes;
        report->kept.clear(); report->reasons.clear();
        for (std::size_t k = 0; k < todo.size(); ++k)
            if (!gone[k]) { report->kept.push_back(todo[k]); report->reasons.push_back(why[k]); }
    }
    return removed;
}

} // namespace femd

#endif // FEMD_TRIANGULATE_COARSEN_HPP
