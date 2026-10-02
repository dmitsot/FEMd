//
//  local_refine.hpp  --  refine a mesh you already have, around triangles you name.
//
//  Three schemes, with different guarantees.  The differences are not academic:
//  measured on an equilateral triangle, refining every child each time,
//
//      level        barycentric 1->3           longest-edge bisection
//                 min angle   diameter        min angle   diameter
//        0           60.00     1.0000            60.00     1.0000
//        1           30.00     1.0000            30.00     1.0000
//        2           10.89     1.0000            30.00     0.8660
//        3            3.67     1.0000            30.00     0.5000
//        5            0.41     1.0000            30.00     0.2500
//
//  The barycentric split never reduces the element DIAMETER, because each child
//  keeps a whole edge of its parent, and the interpolation error of a finite
//  element is governed by the diameter.  It also loses roughly two thirds of the
//  minimum angle per level.  It is the right tool for "split this one element"
//  and the wrong one for an adaptive loop; the other two are for the loop.
//
//  All three work in place on an existing Mesh2D and only append vertices, so
//  vertex indices that were valid before are still valid afterwards.  Call
//  M.compact(false) when done to squeeze out the dead triangle slots while
//  keeping that property.
//
#ifndef FEMD_TRIANGULATE_LOCAL_REFINE_HPP
#define FEMD_TRIANGULATE_LOCAL_REFINE_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/triangulate/refine.hpp"

#include <utility>
#include <vector>

namespace femd {

/**
 * @brief Barycentric split: replace each named triangle by three.
 * @return the number of vertices added
 *
 * Conforming on its own, since it touches nothing outside the triangle, and it
 * leaves the mesh non-Delaunay there. See Mesh2D::split_triangle for why this
 * is a poor way to refine repeatedly.
 *
 * The triangle indices are read before anything changes, so a list taken from
 * the mesh as it stands is safe even though splitting recycles slots.
 */
inline int split_triangles(Mesh2D &M, const std::vector<int> &marked)
{
    std::vector<int> ok;
    for (int t : marked)
        if (t >= 0 && t < M.tri_capacity() && M.alive(t)) ok.push_back(t);
    int added = 0;
    for (int t : ok) if (M.split_triangle(t) >= 0) ++added;
    return added;
}

namespace detail {

/// @brief Is (a,b) the longest edge of triangle t as well?
inline bool is_longest_of(const Mesh2D &M, int t, int a, int b)
{
    int j = 0;
    M.longest_edge2(t, &j);
    int c = M.triangle(t)[(j + 1) % 3], d = M.triangle(t)[(j + 2) % 3];
    return (c == a && d == b) || (c == b && d == a);
}

/// @brief Halve the edge (a,b), segment or not, without legalizing.
inline int halve_edge(Mesh2D &M, int a, int b)
{
    int s = M.segment_index(a, b);
    if (s >= 0) return M.split_segment_midpoint(s, /*legalize=*/false);
    return M.split_edge_midpoint(a, b);
}

} // namespace detail

/**
 * @brief Rivara longest-edge bisection of the triangle containing p.
 * @return the number of vertices added
 *
 * Walk from that triangle to its longest-edge neighbour, and on, until a pair
 * is reached whose shared edge is the longest edge of both (or is on the
 * boundary): that is the longest-edge propagation path, LEPP. Halving the
 * terminal edge bisects both its triangles at once, so no hanging node is ever
 * created. Repeat until the triangle we started from has itself been bisected.
 *
 * The walk terminates because edge lengths strictly increase along it. The
 * refinement is non-degenerate: every angle it can ever produce is at least
 * half the smallest angle of the mesh it started from, however many times it
 * is applied, which is the property the barycentric split lacks.
 *
 * Identified by a POINT rather than a triangle index because bisection
 * recycles indices as it goes.
 */
inline int bisect_at(Mesh2D &M, const Point &p, int hint = -1, int max_steps = 1 << 16)
{
    int added = 0;
    int t0 = M.locate(p, hint);
    if (t0 < 0) return 0;
    std::array<int, 3> id0 = M.triangle(t0);

    for (int outer = 0; outer < max_steps; ++outer)
    {
        int t = M.locate(p, t0 >= 0 && M.alive(t0) ? t0 : hint);
        if (t < 0) return added;
        // once the triangle around p is no longer the one we started with, the
        // original has been bisected and we are done
        std::array<int, 3> id = M.triangle(t);
        bool same = (id[0] == id0[0] || id[0] == id0[1] || id[0] == id0[2])
                 && (id[1] == id0[0] || id[1] == id0[1] || id[1] == id0[2])
                 && (id[2] == id0[0] || id[2] == id0[1] || id[2] == id0[2]);
        if (!same) return added;

        // walk the LEPP to a terminal pair
        int cur = t;
        for (int step = 0; step < max_steps; ++step)
        {
            int i = 0;
            M.longest_edge2(cur, &i);
            int a = M.triangle(cur)[(i + 1) % 3], b = M.triangle(cur)[(i + 2) % 3];
            int u = M.neighbour(cur, i);
            if (u < 0 || detail::is_longest_of(M, u, a, b))
            {
                if (detail::halve_edge(M, a, b) < 0) return added;   // refuse rather than corrupt
                ++added;
                break;
            }
            cur = u;
        }
    }
    return added;
}

/**
 * @brief Rivara bisection of each named triangle, once per level.
 * @return the number of vertices added
 *
 * Each triangle is remembered by its centroid before anything moves, so the
 * list stays meaningful while earlier bisections renumber the mesh. Bisecting
 * one triangle may bisect others along its propagation path; that is how
 * conformity is kept, and it is why the count of new vertices can exceed the
 * number of triangles asked for.
 */
inline int bisect_triangles(Mesh2D &M, const std::vector<int> &marked, int levels = 1)
{
    // A seed is a point plus one of its triangle's VERTICES.  Vertex indices are
    // stable (the mesh only appends), so vertex_triangle() gives a live triangle
    // next to the seed at any later level -- a hint that keeps locate()'s walk
    // short and local instead of aiming it across the whole mesh.
    std::vector<std::pair<Point, int>> seeds;
    for (int t : marked)
        if (t >= 0 && t < M.tri_capacity() && M.alive(t))
            seeds.emplace_back(M.centroid(t), M.triangle(t)[0]);

    int added = 0;
    for (int lev = 0; lev < levels; ++lev)
        for (const auto &s : seeds) added += bisect_at(M, s.first, M.vertex_triangle(s.second));
    return added;
}

/**
 * @brief Delaunay refinement of the named triangles: insert their circumcentres.
 * @return the number of vertices added
 *
 * Unlike the other two this keeps the mesh a constrained Delaunay
 * triangulation and keeps the quality guarantee, because it is the same
 * operation Ruppert refinement performs and it obeys the same rule: a
 * circumcentre that would fall inside the diametral circle of a boundary
 * segment is not inserted, the segment is split instead.  After the pass the
 * ordinary quality loop runs over the whole mesh, so the result still meets
 * `opt`.
 *
 * One pass over the marked set. Call it again for another level.
 */
inline int refine_triangles(Mesh2D &M, const std::vector<int> &marked, const RefineOptions &opt)
{
    // point plus a vertex of its triangle, so locate() gets a local hint later
    std::vector<std::pair<Point, int>> cc;
    for (int t : marked)
    {
        if (t < 0 || t >= M.tri_capacity() || !M.alive(t)) continue;
        try { cc.emplace_back(M.circumcentre_of(t), M.triangle(t)[0]); }
        catch (const std::runtime_error &) {}
    }

    detail::SegmentGrid grid;
    grid.build(M);
    int added = 0;
    for (const auto &item : cc)
    {
        const Point &c = item.first;
        if (grid.stale(M)) grid.build(M);
        std::vector<int> cand = grid.candidates(M, c);   // copy: splitting clobbers the buffer
        bool split_any = false;
        for (int s : cand)
        {
            if (s >= M.nsegments()) continue;
            int a = M.segments()[s][0], b = M.segments()[s][1];
            if (detail::encroaches(M.point(a), M.point(b), c) && M.split_segment_midpoint(s) >= 0)
            { ++added; split_any = true; }
        }
        if (split_any) continue;                 // the neighbourhood changed; the quality loop finishes it
        if (M.insert_point(c, 0, M.vertex_triangle(item.second), /*respect_segments=*/true) >= 0) ++added;
    }
    added += refine(M, opt);                     // restore the angle and size targets everywhere
    return added;
}

} // namespace femd

#endif // FEMD_TRIANGULATE_LOCAL_REFINE_HPP
