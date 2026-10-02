//
//  mesh2d.hpp  --  2D triangular mesh: points, triangles, neighbours, segments.
//
//  The topology is held in flat index arrays, not in a pointer linked half-edge
//  structure:
//
//      tri_[t] = {v0, v1, v2}    the three vertices, ALWAYS counter-clockwise
//      nbr_[t] = {n0, n1, n2}    nbr_[t][i] is the triangle across the edge
//                                OPPOSITE local vertex i, that is the edge
//                                (v[(i+1)%3], v[(i+2)%3]); -1 when there is none
//
//  A directed edge is therefore the pair (t, i) and its twin is found in O(1)
//  through nbr_.  Everything a half-edge structure gives is still here, but no
//  iterator can be invalidated by a neighbouring insertion, and the arrays hand
//  straight to NumPy with no conversion pass.  Removed triangles go on a free
//  list and are reused; compact() squeezes them out at the end.
//
//  Orientation convention: every triangle is counter-clockwise, so area() is
//  positive and orient2d() on its vertices is +1.  The original C meshgen
//  emitted clockwise triangles; triangulate() emits counter-clockwise, which is
//  what every FE assembly routine and every plotting library expects.
//
#ifndef FEMD_MESH_MESH2D_HPP
#define FEMD_MESH_MESH2D_HPP

#include "femd/mesh/predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace femd {

struct Point {
    double x = 0.0, y = 0.0;
};

inline bool operator==(const Point &a, const Point &b) { return a.x == b.x && a.y == b.y; }

/// @brief Squared distance, used everywhere in preference to a square root.
inline double sqdist(const Point &a, const Point &b)
{
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

/**
 * @brief Circumcentre of a triangle, and its circumradius squared.
 * @note Throws if the triangle is degenerate (zero area).
 */
inline Point circumcentre(const Point &a, const Point &b, const Point &c, double *r2 = nullptr)
{
    double bx = b.x - a.x, by = b.y - a.y;
    double cx = c.x - a.x, cy = c.y - a.y;
    double d = 2.0 * (bx * cy - by * cx);
    if (d == 0.0) throw std::runtime_error("circumcentre: degenerate triangle");
    double bl = bx * bx + by * by;
    double cl = cx * cx + cy * cy;
    Point p{a.x + (cy * bl - by * cl) / d, a.y + (bx * cl - cx * bl) / d};
    if (r2) *r2 = sqdist(p, a);
    return p;
}

/**
 * @brief Smallest interior angle of the triangle abc, in degrees.
 *
 * From the circumradius-to-shortest-edge ratio, sin(theta_min) = s / (2R),
 * which stays accurate for slivers where differencing the coordinates to get
 * the angle directly loses most of its digits.  Returns 0 for a degenerate
 * triangle.
 */
inline double min_angle_of(const Point &a, const Point &b, const Point &c)
{
    double r2 = 0.0;
    try { circumcentre(a, b, c, &r2); } catch (...) { return 0.0; }
    if (!(r2 > 0.0)) return 0.0;
    double s2 = std::min(sqdist(a, b), std::min(sqdist(b, c), sqdist(c, a)));
    double s = std::sqrt(s2 / (4.0 * r2));
    if (s > 1.0) s = 1.0;
    return std::asin(s) * 180.0 / 3.14159265358979323846;
}

/**
 * @brief A 2D triangular mesh with full neighbour topology and marked boundary
 *        segments.
 *
 * Construct one with triangulate() (see femd/triangulate/triangulate.hpp).
 * The low-level topology methods below are the mesher's working interface; they
 * are public so that an adaptive loop can keep refining an existing mesh.
 */
class Mesh2D {
public:
    Mesh2D() = default;

    // ---- sizes ------------------------------------------------------------
    int npoints()    const { return static_cast<int>(pt_.size()); }
    /// @brief Number of live triangles (excludes those on the free list).
    int ntriangles() const { return static_cast<int>(tri_.size()) - static_cast<int>(free_.size()); }
    /// @brief Size of the triangle array including free slots; indices run over this.
    int tri_capacity() const { return static_cast<int>(tri_.size()); }
    int nsegments()  const { return static_cast<int>(seg_.size()); }

    bool alive(int t) const { return tri_[t][0] >= 0; }

    // ---- data -------------------------------------------------------------
    const std::vector<Point>              &points()          const { return pt_; }
    const std::vector<int>                &point_markers()   const { return pmark_; }
    const std::vector<std::array<int, 3>> &triangles()       const { return tri_; }
    const std::vector<std::array<int, 3>> &neighbours()      const { return nbr_; }
    const std::vector<std::array<int, 2>> &segments()        const { return seg_; }
    const std::vector<int>                &segment_markers() const { return smark_; }

    const Point &point(int v) const { return pt_[v]; }
    const std::array<int, 3> &triangle(int t) const { return tri_[t]; }
    int neighbour(int t, int i) const { return nbr_[t][i]; }

    // ---- construction primitives -------------------------------------------
    int add_point(const Point &p, int marker = 0)
    {
        pt_.push_back(p);
        pmark_.push_back(marker);
        onseg_.push_back(0);
        return static_cast<int>(pt_.size()) - 1;
    }

    /**
     * @brief Is v an endpoint of some constrained segment, that is, on the boundary?
     *
     * This is the topological question, not a geometric one: it asks whether the
     * vertex belongs to the boundary of the region as the mesher recorded it.
     * Markers would answer it too, but only by convention; this cannot drift.
     */
    bool vertex_on_boundary(int v) const { return v >= 0 && v < static_cast<int>(onseg_.size()) && onseg_[v] != 0; }

    /// @brief How many of triangle t's three edges are constrained segments (0..3).
    int boundary_edge_count(int t) const
    {
        int n = 0;
        for (int i = 0; i < 3; ++i) if (edge_is_segment(t, i)) ++n;
        return n;
    }

    /**
     * @brief Does triangle t have at least one vertex off the boundary?
     *
     * A triangle with all three vertices on the boundary has no free degree of
     * freedom once Dirichlet data is imposed everywhere, so a solver gets an
     * element that contributes nothing.  It is a strictly stronger condition
     * than "no two edges on the boundary": a corner triangle always fails both,
     * but a triangle spanning a thin neck, with one edge on each side and no
     * edge shared with either, fails only this one.
     */
    bool has_interior_vertex(int t) const
    {
        const auto &T = tri_[t];
        return !(vertex_on_boundary(T[0]) && vertex_on_boundary(T[1]) && vertex_on_boundary(T[2]));
    }

    void set_point_marker(int v, int m) { pmark_[v] = m; }

    /// @brief Undo the most recent add_point. Only valid if nothing references it.
    void pop_last_point() { pt_.pop_back(); pmark_.pop_back(); onseg_.pop_back(); if (vtri_.size() > pt_.size()) vtri_.pop_back(); }

    /// @brief Move an existing vertex. The caller is responsible for validity.
    void move_point(int v, const Point &p) { pt_[v] = p; }

    /// @brief Record (a,b) as a constrained segment. Idempotent.
    int add_segment(int a, int b, int marker = 1)
    {
        auto it = segmap_.find(key(a, b));
        if (it != segmap_.end()) { smark_[it->second] = marker; return it->second; }
        seg_.push_back({a, b});
        smark_.push_back(marker);
        int s = static_cast<int>(seg_.size()) - 1;
        segmap_[key(a, b)] = s;
        mark_on_segment(a); mark_on_segment(b);
        return s;
    }

    /**
     * @brief Merge the two segments meeting at v, (a,v) and (v,b), into one segment (a,b).
     *
     * The merged segment takes the slot and the marker of the first of the two, and the
     * second is erased (later segment indices shift down by one).  Only the segment list
     * changes: the caller is responsible for the triangles, and for (a,b) becoming an edge.
     * @return false, changing nothing, unless exactly two segments meet at v.
     */
    bool merge_segments_at(int v)
    {
        int s1 = -1, s2 = -1;
        for (int s = 0; s < nsegments(); ++s)
            if (seg_[s][0] == v || seg_[s][1] == v)
            {
                if (s1 < 0) s1 = s;
                else if (s2 < 0) s2 = s;
                else return false;
            }
        if (s1 < 0 || s2 < 0) return false;
        int a = seg_[s1][0] == v ? seg_[s1][1] : seg_[s1][0];
        int b = seg_[s2][0] == v ? seg_[s2][1] : seg_[s2][0];
        if (a == b) return false;
        // keep the direction of the first segment: (a,v) -> (a,b), (v,a) -> (b,a)
        std::array<int, 2> merged = seg_[s1][1] == v ? std::array<int, 2>{a, b} : std::array<int, 2>{b, a};
        segmap_.erase(key(seg_[s1][0], seg_[s1][1]));
        segmap_.erase(key(seg_[s2][0], seg_[s2][1]));
        seg_[s1] = merged;
        seg_.erase(seg_.begin() + s2);
        smark_.erase(smark_.begin() + s2);
        segmap_.clear();
        for (int s = 0; s < nsegments(); ++s) segmap_[key(seg_[s][0], seg_[s][1])] = s;
        if (v < static_cast<int>(onseg_.size())) onseg_[v] = 0;
        return true;
    }

    /// @brief Index of the segment joining a and b, or -1 if the edge is free.
    int segment_index(int a, int b) const
    {
        auto it = segmap_.find(key(a, b));
        return it == segmap_.end() ? -1 : it->second;
    }

    bool is_segment(int a, int b) const { return segmap_.count(key(a, b)) != 0; }

    /// @brief True when the edge opposite local vertex i of triangle t is constrained.
    bool edge_is_segment(int t, int i) const { return is_segment(tri_[t][(i + 1) % 3], tri_[t][(i + 2) % 3]); }

    /**
     * @brief Replace segment s by the two halves (a,m) and (m,b), keeping its marker.
     * @return The index of the second half.
     */
    int split_segment(int s, int m)
    {
        int a = seg_[s][0], b = seg_[s][1];
        int mk = smark_[s];
        segmap_.erase(key(a, b));
        seg_[s] = {a, m};
        segmap_[key(a, m)] = s;
        seg_.push_back({m, b});
        smark_.push_back(mk);
        int s2 = static_cast<int>(seg_.size()) - 1;
        segmap_[key(m, b)] = s2;
        mark_on_segment(m);
        return s2;
    }

    /**
     * @brief Split segment s at the given point (which must lie on it), keeping
     *        the segment list, the topology and the Delaunay property in step.
     * @return the new vertex index, or -1 when the segment is not an edge of
     *         the mesh or the point is not strictly inside it
     */
    int split_segment_at(int s, const Point &m, bool legalize = true)
    {
        int a = seg_[s][0], b = seg_[s][1];
        if ((pt_[a].x - m.x) * (pt_[b].x - m.x) + (pt_[a].y - m.y) * (pt_[b].y - m.y) >= 0.0) return -1;
        int t1, i1, t2, i2;
        if (!edge_triangles(a, b, t1, i1, t2, i2)) return -1;
        int v = add_point(m, pmark_[a]);
        if (!split_edge_at(a, b, v)) { pt_.pop_back(); pmark_.pop_back(); onseg_.pop_back(); return -1; }
        split_segment(s, v);
        if (legalize) legalize_around(v);
        return v;
    }

    /// @brief Split segment s at its midpoint. Returns the new vertex, or -1.
    int split_segment_midpoint(int s, bool legalize = true)
    {
        const Point &a = pt_[seg_[s][0]], &b = pt_[seg_[s][1]];
        return split_segment_at(s, Point{0.5 * (a.x + b.x), 0.5 * (a.y + b.y)}, legalize);
    }

    /**
     * @brief Halve the interior edge (a,b), giving both adjacent triangles a
     *        new vertex at its midpoint.  Does NOT legalize.
     * @return the new vertex, or -1 if (a,b) is not an edge or is a segment
     *
     * The bisection schemes need the split WITHOUT the Delaunay flip that
     * usually follows it: legalizing would undo the very structure they build.
     */
    int split_edge_midpoint(int a, int b)
    {
        if (is_segment(a, b)) return -1;
        int t1, i1, t2, i2;
        if (!edge_triangles(a, b, t1, i1, t2, i2)) return -1;
        Point m{0.5 * (pt_[a].x + pt_[b].x), 0.5 * (pt_[a].y + pt_[b].y)};
        int v = add_point(m, 0);
        if (!split_edge_at(a, b, v)) { pop_last_point(); return -1; }
        return v;
    }

    /**
     * @brief Replace triangle t by three, meeting at a new vertex at its centroid.
     * @return the new vertex index, or -1 if t is not alive
     *
     * The barycentric split, `trichotomy` in the C meshgen.  It touches only t,
     * so the mesh stays conforming with no help from its neighbours, and it
     * needs no flip.  It also leaves the mesh NOT Delaunay at t, which is
     * intrinsic: the Delaunay triangulation containing the centroid is a
     * cavity retriangulation, not these three triangles.
     *
     * Two properties make it a poor engine for iterative refinement, and they
     * are worth knowing before reaching for it.  The three children each keep a
     * whole edge of the parent, so the element DIAMETER does not shrink at all,
     * and the interpolation error is governed by the diameter.  And the angles
     * fall by roughly a factor of three per level: an equilateral triangle goes
     * 60 -> 30 -> 10.9 -> 3.7 degrees.  For repeated refinement use
     * bisect_triangles() or refine_triangles() instead.
     */
    int split_triangle(int t)
    {
        if (t < 0 || t >= tri_capacity() || !alive(t)) return -1;
        const std::array<int, 3> T = tri_[t];
        int n0 = nbr_[t][0], n1 = nbr_[t][1], n2 = nbr_[t][2];
        int j0 = n0 < 0 ? -1 : local_neighbour(n0, t);
        int j1 = n1 < 0 ? -1 : local_neighbour(n1, t);
        int j2 = n2 < 0 ? -1 : local_neighbour(n2, t);

        int v = add_point(centroid(t), 0);
        kill_triangle(t);
        int A = new_triangle(T[0], T[1], v);   // edge2 = (T0,T1), the old edge opposite 2
        int B = new_triangle(T[1], T[2], v);   // edge2 = (T1,T2), opposite 0
        int C = new_triangle(T[2], T[0], v);   // edge2 = (T2,T0), opposite 1
        link(A, 2, n2, j2);
        link(B, 2, n0, j0);
        link(C, 2, n1, j1);
        link(A, 0, B, 1);                      // (T1,v)
        link(B, 0, C, 1);                      // (T2,v)
        link(C, 0, A, 1);                      // (T0,v)
        return v;
    }

    /**
     * @brief Index of the segment p lies strictly inside, or -1.
     * @param hint a triangle to start the search from
     */
    int segment_through(const Point &p, int hint = -1) const
    {
        int t = locate(p, hint);
        if (t < 0) return -1;
        for (int i = 0; i < 3; ++i)
        {
            int a = tri_[t][(i + 1) % 3], b = tri_[t][(i + 2) % 3];
            if (pred::orient2d(pt_[a].x, pt_[a].y, pt_[b].x, pt_[b].y, p.x, p.y) != 0) continue;
            if ((pt_[a].x - p.x) * (pt_[b].x - p.x) + (pt_[a].y - p.y) * (pt_[b].y - p.y) >= 0.0) continue;
            return segment_index(a, b);
        }
        return -1;
    }

    int new_triangle(int a, int b, int c)
    {
        int t;
        if (!free_.empty()) { t = free_.back(); free_.pop_back(); }
        else { tri_.push_back({-1, -1, -1}); nbr_.push_back({-1, -1, -1}); t = static_cast<int>(tri_.size()) - 1; }
        tri_[t] = {a, b, c};
        nbr_[t] = {-1, -1, -1};
        if (static_cast<int>(vtri_.size()) < npoints()) vtri_.resize(npoints(), -1);
        vtri_[a] = vtri_[b] = vtri_[c] = t;
        return t;
    }

    void kill_triangle(int t)
    {
        tri_[t] = {-1, -1, -1};
        nbr_[t] = {-1, -1, -1};
        free_.push_back(t);
    }

    /// @brief Glue triangle t across its edge i to triangle u across its edge j.
    void link(int t, int i, int u, int j)
    {
        if (t >= 0) nbr_[t][i] = u;
        if (u >= 0) nbr_[u][j] = t;
    }

    /// @brief Local index of vertex v in triangle t, or -1.
    int local_index(int t, int v) const
    {
        for (int i = 0; i < 3; ++i) if (tri_[t][i] == v) return i;
        return -1;
    }

    /// @brief Local index i such that nbr_[t][i] == u, or -1.
    int local_neighbour(int t, int u) const
    {
        for (int i = 0; i < 3; ++i) if (nbr_[t][i] == u) return i;
        return -1;
    }

    /**
     * @brief Some live triangle incident on vertex v, or -1 if v is isolated.
     *
     * A hint is cached per vertex when triangles are created; it is verified on
     * every call and repaired by a scan when it has gone stale, so the answer
     * is always correct and is O(1) in the common case.
     */
    int vertex_triangle(int v) const
    {
        if (v >= 0 && v < static_cast<int>(vtri_.size()))
        {
            int t = vtri_[v];
            if (t >= 0 && t < tri_capacity() && alive(t) && local_index(t, v) >= 0) return t;
        }
        for (int t = 0; t < tri_capacity(); ++t)
            if (alive(t) && local_index(t, v) >= 0)
            {
                if (static_cast<int>(vtri_.size()) < npoints()) vtri_.resize(npoints(), -1);
                vtri_[v] = t;
                return t;
            }
        return -1;
    }

    /**
     * @brief Every triangle incident on vertex v, in rotational order.
     * @param out receives the triangle indices (cleared first)
     * @return true when v is interior, that is the fan closes on itself
     */
    bool one_ring(int v, std::vector<int> &out) const
    {
        out.clear();
        int t0 = vertex_triangle(v);
        if (t0 < 0) return false;
        // rotate one way
        int t = t0;
        while (true)
        {
            out.push_back(t);
            int k = local_index(t, v);
            int nxt = nbr_[t][(k + 2) % 3];
            if (nxt < 0) break;
            if (nxt == t0) return true;               // closed fan: interior vertex
            t = nxt;
            if (static_cast<int>(out.size()) > tri_capacity()) return false;
        }
        // hit a boundary, so sweep back the other way from t0
        t = t0;
        while (true)
        {
            int k = local_index(t, v);
            int prv = nbr_[t][(k + 1) % 3];
            if (prv < 0) break;
            t = prv;
            out.push_back(t);
            if (static_cast<int>(out.size()) > tri_capacity()) return false;
        }
        return false;
    }

    /**
     * @brief The (at most two) triangles sharing the edge (a,b).
     * @param t1,i1 first triangle and the local index the edge is opposite to
     * @param t2,i2 the other side, or -1 when the edge is on the boundary
     * @return false when no triangle has (a,b) as an edge
     */
    bool edge_triangles(int a, int b, int &t1, int &i1, int &t2, int &i2) const
    {
        t1 = t2 = -1; i1 = i2 = -1;
        int t0 = vertex_triangle(a);
        if (t0 < 0) return false;
        std::vector<int> fan;
        one_ring(a, fan);
        for (int t : fan)
        {
            int j = local_index(t, b);
            if (j < 0) continue;
            int k = local_index(t, a);
            int opp = 3 - j - k;              // the third local index
            if (t1 < 0) { t1 = t; i1 = opp; }
            else if (t != t1) { t2 = t; i2 = opp; }
        }
        if (t1 >= 0 && t2 < 0) { t2 = nbr_[t1][i1]; if (t2 >= 0) i2 = local_neighbour(t2, t1); }
        return t1 >= 0;
    }

    // ---- geometry ----------------------------------------------------------
    double area(int t) const
    {
        const auto &T = tri_[t];
        const Point &a = pt_[T[0]], &b = pt_[T[1]], &c = pt_[T[2]];
        return 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    }

    Point centroid(int t) const
    {
        const auto &T = tri_[t];
        return Point{(pt_[T[0]].x + pt_[T[1]].x + pt_[T[2]].x) / 3.0,
                     (pt_[T[0]].y + pt_[T[1]].y + pt_[T[2]].y) / 3.0};
    }

    Point circumcentre_of(int t, double *r2 = nullptr) const
    {
        const auto &T = tri_[t];
        return circumcentre(pt_[T[0]], pt_[T[1]], pt_[T[2]], r2);
    }

    /// @brief Squared length of the longest edge; *which receives its local index.
    double longest_edge2(int t, int *which = nullptr) const
    {
        double best = -1.0; int bi = 0;
        for (int i = 0; i < 3; ++i)
        {
            double l = sqdist(pt_[tri_[t][(i + 1) % 3]], pt_[tri_[t][(i + 2) % 3]]);
            if (l > best) { best = l; bi = i; }
        }
        if (which) *which = bi;
        return best;
    }

    double shortest_edge2(int t) const
    {
        double best = std::numeric_limits<double>::max();
        for (int i = 0; i < 3; ++i) best = std::min(best, sqdist(pt_[tri_[t][(i + 1) % 3]], pt_[tri_[t][(i + 2) % 3]]));
        return best;
    }

    /**
     * @brief Smallest interior angle of triangle t, IN DEGREES.
     *
     * Computed from the circumradius-to-shortest-edge ratio, which is the
     * quantity Ruppert's algorithm actually bounds: sin(theta_min) = s / (2 R).
     * That form is stable for sliver triangles, where differencing the
     * coordinates to get the angle directly loses most of its digits.
     */
    double min_angle(int t) const
    {
        const auto &T = tri_[t];
        return min_angle_of(pt_[T[0]], pt_[T[1]], pt_[T[2]]);
    }

    /// @brief Radius-edge ratio R/s: what Ruppert bounds, small is good (0.577 is equilateral).
    double radius_edge_ratio(int t) const
    {
        double r2 = 0.0;
        try { circumcentre_of(t, &r2); } catch (...) { return std::numeric_limits<double>::infinity(); }
        return std::sqrt(r2 / shortest_edge2(t));
    }

    /**
     * @brief Shape quality in [0,1]: 1 for equilateral, 0 for degenerate.
     * @note This is 4*sqrt(3)*A / (a^2+b^2+c^2), the same normalization MATLAB's
     *       pdetriq and the original meshgen used.
     */
    double quality(int t) const
    {
        double s = 0.0;
        for (int i = 0; i < 3; ++i) s += sqdist(pt_[tri_[t][(i + 1) % 3]], pt_[tri_[t][(i + 2) % 3]]);
        if (s == 0.0) return 0.0;
        return 4.0 * std::sqrt(3.0) * area(t) / s;
    }

    // ---- point location ----------------------------------------------------
    /**
     * @brief Straight walk towards p from hint.
     * @return The triangle containing p (inside or on its boundary), or -1 when the walk
     *         ran into the boundary of the mesh or hit its step cap.  -1 is inconclusive:
     *         p may lie outside, or on the far side of a hole or a reentrant corner.
     *
     * Each step crosses the first edge p lies strictly on the wrong side of, judged by
     * the exact orient2d, so a point on an edge or a vertex stops the walk in one of
     * the triangles that contain it.
     */
    int walk(const Point &p, int hint = -1) const
    {
        int t = (hint >= 0 && hint < tri_capacity() && alive(hint)) ? hint : first_live();
        if (t < 0) return -1;
        int cap = 4 * tri_capacity() + 64;
        for (int step = 0; step < cap; ++step)
        {
            int next = -1;
            for (int i = 0; i < 3; ++i)
            {
                const Point &u = pt_[tri_[t][(i + 1) % 3]];
                const Point &v = pt_[tri_[t][(i + 2) % 3]];
                if (pred::orient2d(u.x, u.y, v.x, v.y, p.x, p.y) < 0) { next = i; break; }
            }
            if (next < 0) return t;              // p is inside or on the boundary of t
            if (nbr_[t][next] < 0) return -1;    // the walk left the mesh
            t = nbr_[t][next];
        }
        return -1;
    }

    /**
     * @brief Triangle containing p, found by walking from hint.
     * @return The triangle index, or -1 when p lies outside the triangulation.
     *
     * The straight walk is only reliable on a CONVEX region.  Once the exterior has
     * been trimmed away the mesh has holes and reentrant corners, and a walk aimed
     * across one of them runs into the boundary and stops, reporting "outside" for a
     * point that is plainly inside.  That silently skipped every refinement seed on
     * the far side of a hole before the fallback existed, so an inconclusive walk is
     * followed by a scan.  For many queries use PointLocator (point_locator.hpp),
     * whose fallback is a bucket grid instead of a scan.
     */
    int locate(const Point &p, int hint = -1) const
    {
        int t = walk(p, hint);
        return t >= 0 ? t : locate_by_scan(p);
    }

    /// @brief Containment by exhaustive search. O(ntriangles); the fallback for locate().
    int locate_by_scan(const Point &p) const
    {
        for (int t = 0; t < tri_capacity(); ++t)
        {
            if (!alive(t)) continue;
            bool in = true;
            for (int i = 0; i < 3 && in; ++i)
            {
                const Point &u = pt_[tri_[t][(i + 1) % 3]];
                const Point &v = pt_[tri_[t][(i + 2) % 3]];
                if (pred::orient2d(u.x, u.y, v.x, v.y, p.x, p.y) < 0) in = false;
            }
            if (in) return t;
        }
        return -1;
    }

    int first_live() const
    {
        for (int t = 0; t < tri_capacity(); ++t) if (alive(t)) return t;
        return -1;
    }

    // ---- Bowyer-Watson insertion -------------------------------------------
    /**
     * @brief Insert p by the Bowyer-Watson rule and retriangulate its cavity.
     * @param p       the point to insert
     * @param marker  marker for the new vertex
     * @param hint    a triangle to start the walk from, or -1
     * @param respect_segments  when true the cavity never crosses a constrained
     *        segment, so the result is a CONSTRAINED Delaunay triangulation
     * @return the new vertex index, or -1 when the insertion was refused
     *
     * The insertion is refused, leaving the mesh untouched, when p falls outside
     * the triangulation, lands on an existing vertex, or would produce a cavity
     * that is not star-shaped about p.  A caller that gets -1 should do
     * something else (Ruppert splits an edge instead); it must never simply
     * carry on, which is the bug that made the C meshgen crash.
     */
    int insert_point(const Point &p, int marker = 0, int hint = -1, bool respect_segments = true)
    {
        int v = add_point(p, marker);
        if (!insert_vertex(v, hint, respect_segments)) { pt_.pop_back(); pmark_.pop_back(); onseg_.pop_back(); return -1; }
        return v;
    }

    /**
     * @brief Insert a vertex that is already in the point array.
     * @return false, with the mesh untouched, when the insertion is refused.
     *
     * Same contract as insert_point; this is the form the mesher uses when the
     * input vertices are laid down before the super-triangle so that their
     * indices match the caller's numbering.
     */
    bool insert_vertex(int vnew, int hint = -1, bool respect_segments = true)
    {
        const Point p = pt_[vnew];
        int t0 = locate(p, hint);
        if (t0 < 0) return false;
        for (int i = 0; i < 3; ++i) if (tri_[t0][i] != vnew && pt_[tri_[t0][i]] == p) return false;

        // p exactly on an edge: the cavity would want a zero-area triangle, so
        // split the edge topologically instead and legalize around the new vertex.
        for (int i = 0; i < 3; ++i)
        {
            int a = tri_[t0][(i + 1) % 3], b = tri_[t0][(i + 2) % 3];
            if (pred::orient2d(pt_[a].x, pt_[a].y, pt_[b].x, pt_[b].y, p.x, p.y) != 0) continue;
            // and strictly between a and b
            if ((pt_[a].x - p.x) * (pt_[b].x - p.x) + (pt_[a].y - p.y) * (pt_[b].y - p.y) >= 0.0) continue;
            // A point on a SEGMENT is refused: the segment list has to be split
            // in the same breath as the topology, and only split_segment_at()
            // does both.  Silently splitting the edge here would leave the
            // segment record pointing at an edge that no longer exists.
            if (is_segment(a, b)) return false;
            if (!split_edge_at(a, b, vnew)) return false;
            legalize_around(vnew);
            return true;
        }

        cav_.clear(); stack_.clear(); incav_.assign(tri_capacity(), 0);
        cav_.push_back(t0); incav_[t0] = 1; stack_.push_back(t0);

        while (!stack_.empty())
        {
            int t = stack_.back(); stack_.pop_back();
            for (int i = 0; i < 3; ++i)
            {
                int u = nbr_[t][i];
                if (u < 0 || incav_[u]) continue;
                if (respect_segments && edge_is_segment(t, i)) continue;
                const auto &U = tri_[u];
                if (pred::incircle(pt_[U[0]].x, pt_[U[0]].y, pt_[U[1]].x, pt_[U[1]].y,
                                   pt_[U[2]].x, pt_[U[2]].y, p.x, p.y) > 0)
                {
                    incav_[u] = 1; cav_.push_back(u); stack_.push_back(u);
                }
            }
        }

        // Collect the cavity boundary: edges whose far triangle is not in the cavity.
        ring_.clear();
        for (int t : cav_)
            for (int i = 0; i < 3; ++i)
            {
                int u = nbr_[t][i];
                if (u >= 0 && incav_[u]) continue;
                int a = tri_[t][(i + 1) % 3], b = tri_[t][(i + 2) % 3];
                // star-shapedness: p must see (a,b) from the inside
                if (pred::orient2d(pt_[a].x, pt_[a].y, pt_[b].x, pt_[b].y, p.x, p.y) <= 0) return false;
                ring_.push_back({a, b, u, u < 0 ? -1 : local_neighbour(u, t)});
            }
        if (ring_.size() < 3) return false;

        for (int t : cav_) kill_triangle(t);

        // One new triangle per ring edge, then stitch them to each other.
        newt_.clear();
        for (const auto &r : ring_)
        {
            int t = new_triangle(r.a, r.b, vnew);
            link(t, 2, r.out, r.outi);   // edge (a,b) is opposite local vertex 2
            newt_.push_back(t);
        }
        // Edge opposite local vertex 0 is (b, vnew); opposite 1 is (vnew, a).
        std::unordered_map<std::int64_t, std::pair<int, int>> open;
        open.reserve(newt_.size() * 2);
        for (int t : newt_)
            for (int i = 0; i < 2; ++i)
            {
                int a = tri_[t][(i + 1) % 3], b = tri_[t][(i + 2) % 3];
                std::int64_t k = key(a, b);
                auto it = open.find(k);
                if (it == open.end()) open.emplace(k, std::make_pair(t, i));
                else { link(it->second.first, it->second.second, t, i); open.erase(it); }
            }
        return true;
    }

    /**
     * @brief Flip the edge shared by t and its neighbour across local index i.
     * @return false when the edge is on the boundary, is a segment, or the two
     *         triangles do not form a convex quadrilateral; the mesh is then
     *         untouched.
     */
    bool flip_edge(int t, int i)
    {
        int u = nbr_[t][i];
        if (u < 0) return false;
        if (edge_is_segment(t, i)) return false;
        int j = local_neighbour(u, t);
        if (j < 0) return false;

        int p = tri_[t][i];
        int a = tri_[t][(i + 1) % 3], b = tri_[t][(i + 2) % 3];
        int q = tri_[u][j];

        if (pred::orient2d(pt_[p].x, pt_[p].y, pt_[a].x, pt_[a].y, pt_[q].x, pt_[q].y) <= 0) return false;
        if (pred::orient2d(pt_[p].x, pt_[p].y, pt_[q].x, pt_[q].y, pt_[b].x, pt_[b].y) <= 0) return false;

        int n_aq = nbr_[u][(j + 1) % 3];   // across edge (a,q)
        int n_qb = nbr_[u][(j + 2) % 3];   // across edge (q,b)
        int n_bp = nbr_[t][(i + 1) % 3];   // across edge (b,p)
        int n_pa = nbr_[t][(i + 2) % 3];   // across edge (p,a)
        int i_aq = n_aq < 0 ? -1 : local_neighbour(n_aq, u);
        int i_qb = n_qb < 0 ? -1 : local_neighbour(n_qb, u);
        int i_bp = n_bp < 0 ? -1 : local_neighbour(n_bp, t);
        int i_pa = n_pa < 0 ? -1 : local_neighbour(n_pa, t);

        kill_triangle(t); kill_triangle(u);
        int T1 = new_triangle(p, a, q);
        int T2 = new_triangle(p, q, b);
        link(T1, 0, n_aq, i_aq);
        link(T1, 2, n_pa, i_pa);
        link(T2, 0, n_qb, i_qb);
        link(T2, 1, n_bp, i_bp);
        link(T1, 1, T2, 2);
        return true;
    }

    /**
     * @brief Split the edge (a,b) at the already-added vertex v lying on it.
     * @return false when (a,b) is not an edge of the mesh
     *
     * The one or two triangles along the edge are replaced by two each.  This
     * is the operation a point insertion needs when the point lands exactly on
     * an edge, where the Bowyer-Watson cavity would otherwise want to build a
     * zero-area triangle; splitting a boundary segment at its midpoint is
     * exactly that case, and it happens on every second step of Ruppert
     * refinement.
     */
    bool split_edge_at(int a, int b, int v)
    {
        int t1, i1, t2, i2;
        if (!edge_triangles(a, b, t1, i1, t2, i2)) return false;
        if (t1 < 0) return false;

        // orient t1 so that its edge reads (a,b)
        if (tri_[t1][(i1 + 1) % 3] != a)
        {
            std::swap(t1, t2); std::swap(i1, i2);
            if (t1 < 0 || tri_[t1][(i1 + 1) % 3] != a) return false;
        }
        int p = tri_[t1][i1];
        int n_pa = nbr_[t1][(i1 + 2) % 3], i_pa = n_pa < 0 ? -1 : local_neighbour(n_pa, t1);
        int n_bp = nbr_[t1][(i1 + 1) % 3], i_bp = n_bp < 0 ? -1 : local_neighbour(n_bp, t1);

        int q = -1, n_qb = -1, i_qb = -1, n_aq = -1, i_aq = -1;
        if (t2 >= 0)
        {
            q = tri_[t2][i2];
            n_qb = nbr_[t2][(i2 + 2) % 3]; i_qb = n_qb < 0 ? -1 : local_neighbour(n_qb, t2);
            n_aq = nbr_[t2][(i2 + 1) % 3]; i_aq = n_aq < 0 ? -1 : local_neighbour(n_aq, t2);
        }

        kill_triangle(t1);
        if (t2 >= 0) kill_triangle(t2);

        int A = new_triangle(p, a, v);     // edge0 = (a,v), edge1 = (v,p), edge2 = (p,a)
        int B = new_triangle(p, v, b);     // edge0 = (v,b), edge1 = (b,p), edge2 = (p,v)
        link(A, 2, n_pa, i_pa);
        link(B, 1, n_bp, i_bp);
        link(A, 1, B, 2);

        if (t2 >= 0)
        {
            int C = new_triangle(q, b, v); // edge0 = (b,v), edge1 = (v,q), edge2 = (q,b)
            int D = new_triangle(q, v, a); // edge0 = (v,a), edge1 = (a,q), edge2 = (q,v)
            link(C, 2, n_qb, i_qb);
            link(D, 1, n_aq, i_aq);
            link(C, 1, D, 2);
            link(A, 0, D, 0);
            link(B, 0, C, 0);
        }
        return true;
    }

    /**
     * @brief Flip around v until every edge opposite it is locally Delaunay.
     *
     * Segments are never flipped, so this restores the CONSTRAINED Delaunay
     * property, which is the invariant the mesher maintains.
     */
    void legalize_around(int v, int max_flips = 4096)
    {
        std::vector<int> fan;
        for (int n = 0; n < max_flips; ++n)
        {
            one_ring(v, fan);
            bool flipped = false;
            for (int t : fan)
            {
                int k = local_index(t, v);
                if (k < 0) continue;
                if (edge_is_segment(t, k) || locally_delaunay(t, k)) continue;
                if (flip_edge(t, k)) { flipped = true; break; }
            }
            if (!flipped) return;
        }
    }

    /// @brief Is the edge opposite local vertex i of t locally Delaunay?
    bool locally_delaunay(int t, int i) const
    {
        int u = nbr_[t][i];
        if (u < 0) return true;
        int j = local_neighbour(u, t);
        if (j < 0) return true;
        int q = tri_[u][j];
        const auto &T = tri_[t];
        return pred::incircle(pt_[T[0]].x, pt_[T[0]].y, pt_[T[1]].x, pt_[T[1]].y,
                              pt_[T[2]].x, pt_[T[2]].y, pt_[q].x, pt_[q].y) <= 0;
    }

    // ---- bulk editing -------------------------------------------------------
    /// @brief Kill every triangle for which keep(t) is false, and fix neighbours.
    template <class Keep>
    void filter_triangles(Keep keep)
    {
        std::vector<char> k(tri_capacity(), 0);
        for (int t = 0; t < tri_capacity(); ++t) if (alive(t) && keep(t)) k[t] = 1;
        for (int t = 0; t < tri_capacity(); ++t)
        {
            if (!alive(t) || !k[t]) continue;
            for (int i = 0; i < 3; ++i)
            {
                int u = nbr_[t][i];
                if (u >= 0 && !k[u]) nbr_[t][i] = -1;
            }
        }
        for (int t = 0; t < tri_capacity(); ++t) if (alive(t) && !k[t]) kill_triangle(t);
    }

    /**
     * @brief Squeeze out dead triangles, and optionally renumber the points.
     * @param points_too  when false the point array is left exactly as it is,
     *        so every existing vertex index stays valid and newly added ones
     *        are simply appended.  Local refinement wants this: a solution
     *        vector indexed by vertex can then be extended rather than
     *        re-projected.  Unused points are kept.
     * @return The old-to-new point map (the identity when points_too is false).
     */
    std::vector<int> compact(bool points_too = true)
    {
        std::vector<int> tmap(tri_capacity(), -1);
        std::vector<std::array<int, 3>> nt, nn;
        for (int t = 0; t < tri_capacity(); ++t)
            if (alive(t)) { tmap[t] = static_cast<int>(nt.size()); nt.push_back(tri_[t]); nn.push_back(nbr_[t]); }
        for (auto &n : nn) for (int i = 0; i < 3; ++i) if (n[i] >= 0) n[i] = tmap[n[i]];
        tri_.swap(nt); nbr_.swap(nn); free_.clear(); vtri_.clear();

        if (!points_too)
        {
            vtri_.assign(pt_.size(), -1);
            for (std::size_t t = 0; t < tri_.size(); ++t)
                for (int i = 0; i < 3; ++i) vtri_[tri_[t][i]] = static_cast<int>(t);
            std::vector<int> ident(pt_.size());
            for (std::size_t i = 0; i < pt_.size(); ++i) ident[i] = static_cast<int>(i);
            return ident;
        }

        std::vector<int> used(pt_.size(), 0);
        for (const auto &T : tri_) for (int i = 0; i < 3; ++i) used[T[i]] = 1;
        for (const auto &s : seg_) { used[s[0]] = 1; used[s[1]] = 1; }

        std::vector<int> pmap(pt_.size(), -1);
        std::vector<Point> np; std::vector<int> nm;
        for (std::size_t v = 0; v < pt_.size(); ++v)
            if (used[v]) { pmap[v] = static_cast<int>(np.size()); np.push_back(pt_[v]); nm.push_back(pmark_[v]); }
        pt_.swap(np); pmark_.swap(nm); vtri_.assign(pt_.size(), -1); onseg_.assign(pt_.size(), 0);

        for (std::size_t t = 0; t < tri_.size(); ++t)
            for (int i = 0; i < 3; ++i) { tri_[t][i] = pmap[tri_[t][i]]; vtri_[tri_[t][i]] = static_cast<int>(t); }
        segmap_.clear();
        for (std::size_t s = 0; s < seg_.size(); ++s)
        {
            seg_[s][0] = pmap[seg_[s][0]];
            seg_[s][1] = pmap[seg_[s][1]];
            segmap_[key(seg_[s][0], seg_[s][1])] = static_cast<int>(s);
            mark_on_segment(seg_[s][0]); mark_on_segment(seg_[s][1]);
        }
        return pmap;
    }

    // ---- validation ---------------------------------------------------------
    /**
     * @brief Check the mesh against its own invariants.
     * @return An empty string when the mesh is sound, otherwise the first fault.
     *
     * Checks: every live triangle is counter-clockwise with positive area,
     * neighbour links are symmetric, shared edges agree on their two vertices,
     * and every segment appears as an edge of some triangle.
     */
    std::string validate() const
    {
        for (int t = 0; t < tri_capacity(); ++t)
        {
            if (!alive(t)) continue;
            // Exact orientation, not the double area: a legitimately thin
            // triangle can have an area that rounds to 0 while its three
            // vertices are genuinely counter-clockwise.
            const auto &T = tri_[t];
            if (pred::orient2d(pt_[T[0]].x, pt_[T[0]].y, pt_[T[1]].x, pt_[T[1]].y, pt_[T[2]].x, pt_[T[2]].y) <= 0)
                return "triangle " + std::to_string(t) + " is degenerate or clockwise (area "
                       + std::to_string(area(t)) + ")";
            for (int i = 0; i < 3; ++i)
            {
                int u = nbr_[t][i];
                if (u < 0) continue;
                if (!alive(u)) return "triangle " + std::to_string(t) + " neighbours dead triangle " + std::to_string(u);
                int j = local_neighbour(u, t);
                if (j < 0) return "neighbour link " + std::to_string(t) + "->" + std::to_string(u) + " is not symmetric";
                int a = tri_[t][(i + 1) % 3], b = tri_[t][(i + 2) % 3];
                int c = tri_[u][(j + 1) % 3], d = tri_[u][(j + 2) % 3];
                if (!(a == d && b == c)) return "shared edge of " + std::to_string(t) + " and " + std::to_string(u) + " disagrees";
            }
        }
        for (std::size_t s = 0; s < seg_.size(); ++s)
        {
            int t1, i1, t2, i2;
            if (!edge_triangles(seg_[s][0], seg_[s][1], t1, i1, t2, i2))
                return "segment " + std::to_string(s) + " is missing from the triangulation";
        }
        return std::string();
    }

private:
    void mark_on_segment(int v) const
    {
        if (v < 0) return;
        if (static_cast<int>(onseg_.size()) < static_cast<int>(pt_.size())) onseg_.resize(pt_.size(), 0);
        if (v < static_cast<int>(onseg_.size())) onseg_[v] = 1;
    }

    static std::int64_t key(int a, int b)
    {
        std::int64_t lo = std::min(a, b), hi = std::max(a, b);
        return (hi << 32) | (lo & 0xffffffffLL);
    }

    struct RingEdge { int a, b, out, outi; };

    std::vector<Point>              pt_;
    std::vector<int>                pmark_;
    std::vector<std::array<int, 3>> tri_, nbr_;
    std::vector<int>                free_;
    std::vector<std::array<int, 2>> seg_;
    std::vector<int>                smark_;
    mutable std::vector<char>       onseg_;   ///< vertex is an endpoint of some segment
    std::unordered_map<std::int64_t, int> segmap_;

    // insertion scratch, kept as members so a refinement loop does not reallocate
    mutable std::vector<int>  vtri_;
    mutable std::vector<int>  cav_, stack_, newt_;
    mutable std::vector<char> incav_;
    mutable std::vector<RingEdge> ring_;
};

} // namespace femd

#endif // FEMD_MESH_MESH2D_HPP
