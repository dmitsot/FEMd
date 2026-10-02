//
//  delaunay.hpp  --  initial triangulation, segment recovery, trimming.
//
//  Three steps, in the same order the C meshgen took them, with the same
//  algorithms and different bookkeeping:
//
//    1. super-triangle plus Bowyer-Watson insertion of every input vertex
//       (meshgen: WatsonAlg / add_node_Watsons)
//    2. recover the input segments by splitting at midpoints until each one
//       appears as an edge (meshgen: respect_bound)
//    3. throw away the triangles outside the region (meshgen: cleanMesh)
//
//  What changed, and why:
//
//    * meshgen rebuilt the twin pointers after every single point insertion
//      with a double loop over the whole edge list, so building the initial
//      triangulation of n points cost O(n * E^2).  Here the cavity ring is
//      stitched locally, which is O(cavity).
//    * meshgen restarted the ENTIRE triangulation from scratch (goto start)
//      each time it found a missing boundary edge.  Here the midpoint is
//      inserted into the existing triangulation, so recovery costs about one
//      point insertion per split.
//    * meshgen decided inside-versus-outside by a point-in-polygon test on
//      every triangle centroid, O(T * B).  Here a flood fill that cannot cross
//      a segment does it in O(T), with one point-in-polygon test per connected
//      component to say which side that component is on.
//
#ifndef FEMD_TRIANGULATE_DELAUNAY_HPP
#define FEMD_TRIANGULATE_DELAUNAY_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/domain.hpp"

#include <array>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace femd {
namespace detail {

/**
 * @brief Add three vertices enclosing the box and the triangle spanning them.
 * @return the three super-triangle vertex indices
 *
 * The triangle is made large enough (ten diameters) that no circumcircle of a
 * real triangle can reach its vertices, which is what keeps the convex hull of
 * the input correct after the super-triangle is removed.
 */
inline std::array<int, 3> add_super_triangle(Mesh2D &M, const Point &lo, const Point &hi)
{
    double cx = 0.5 * (lo.x + hi.x), cy = 0.5 * (lo.y + hi.y);
    double d = std::max(hi.x - lo.x, hi.y - lo.y);
    if (!(d > 0.0)) d = 1.0;
    d *= 10.0;
    int a = M.add_point({cx - 2.0 * d, cy - d}, 0);
    int b = M.add_point({cx + 2.0 * d, cy - d}, 0);
    int c = M.add_point({cx, cy + 2.0 * d}, 0);
    M.new_triangle(a, b, c);   // counter-clockwise by construction
    return {a, b, c};
}

/**
 * @brief Insert vertices [first, last) of M into the triangulation.
 * @throws std::runtime_error when a vertex cannot be inserted, which for input
 *         that is inside the super-triangle means two vertices coincide.
 */
inline void insert_range(Mesh2D &M, int first, int last)
{
    int hint = M.first_live();
    for (int v = first; v < last; ++v)
    {
        if (!M.insert_vertex(v, hint, /*respect_segments=*/false))
            throw std::runtime_error("triangulate: could not insert input vertex " + std::to_string(v)
                                     + " (duplicate point, or a point outside the super-triangle)");
        hint = M.vertex_triangle(v);
    }
}

/// @brief Every undirected edge currently present in the triangulation.
inline std::unordered_set<std::int64_t> edge_set(const Mesh2D &M)
{
    std::unordered_set<std::int64_t> e;
    e.reserve(static_cast<std::size_t>(M.ntriangles()) * 3);
    for (int t = 0; t < M.tri_capacity(); ++t)
    {
        if (!M.alive(t)) continue;
        for (int i = 0; i < 3; ++i)
        {
            int a = M.triangle(t)[(i + 1) % 3], b = M.triangle(t)[(i + 2) % 3];
            std::int64_t lo = std::min(a, b), hi = std::max(a, b);
            e.insert((hi << 32) | lo);
        }
    }
    return e;
}

/**
 * @brief Split segments until every one of them is an edge of the triangulation.
 * @return the number of midpoints inserted
 * @throws std::runtime_error if recovery has not converged after max_sweeps
 *
 * This is the conforming-Delaunay route: rather than flipping edges into place,
 * a missing segment is halved and both halves are retried.  Each split strictly
 * shortens the missing segments, so the process terminates for any input whose
 * segments do not cross each other.
 */
inline int recover_segments(Mesh2D &M, int max_sweeps = 64, int max_points = 1 << 22)
{
    int added = 0;
    for (int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        auto have = edge_set(M);
        std::vector<int> missing;
        for (int s = 0; s < M.nsegments(); ++s)
        {
            int a = M.segments()[s][0], b = M.segments()[s][1];
            std::int64_t lo = std::min(a, b), hi = std::max(a, b);
            if (!have.count((hi << 32) | lo)) missing.push_back(s);
        }
        if (missing.empty()) return added;

        for (int s : missing)
        {
            int a = M.segments()[s][0], b = M.segments()[s][1];
            Point m{0.5 * (M.point(a).x + M.point(b).x), 0.5 * (M.point(a).y + M.point(b).y)};
            int v = M.insert_point(m, M.point_markers()[a], M.vertex_triangle(a), /*respect_segments=*/false);
            if (v < 0)
            {
                // The midpoint may have landed exactly on ANOTHER segment, which
                // insert_point refuses because splitting it is a two-sided
                // bookkeeping job.  Split that one instead; s is retried on the
                // next sweep with a shorter neighbour in its way.
                int blocked = M.segment_through(m, M.vertex_triangle(a));
                if (blocked >= 0 && M.split_segment_midpoint(blocked) >= 0) ++added;
                continue;
            }
            M.split_segment(s, v);
            ++added;
            if (M.npoints() > max_points)
                throw std::runtime_error("triangulate: segment recovery exceeded the point budget; "
                                         "the input polygon is probably self-intersecting");
        }
    }
    throw std::runtime_error("triangulate: segment recovery did not converge; "
                             "check the input rings for crossing or duplicated edges");
}

/**
 * @brief Delete every triangle outside the region, and the super-triangle with it.
 *
 * Flood fill from the super-triangle, never crossing a segment, marks the
 * exterior.  Whatever the flood does not reach falls into connected components
 * separated from it by segments: the region itself, and one component per hole.
 * A single point-in-region test on one triangle of each component decides it.
 */
inline void trim_exterior(Mesh2D &M, const Domain &D, const std::array<int, 3> &super)
{
    int cap = M.tri_capacity();
    std::vector<int> comp(cap, -1);
    std::vector<int> stack;

    // component 0: everything reachable from the super-triangle without crossing a segment
    for (int t = 0; t < cap; ++t)
    {
        if (!M.alive(t) || comp[t] >= 0) continue;
        bool touches_super = false;
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                if (M.triangle(t)[i] == super[k]) touches_super = true;
        if (!touches_super) continue;
        comp[t] = 0; stack.push_back(t);
    }
    int ncomp = 1;
    auto flood = [&](int c) {
        while (!stack.empty())
        {
            int t = stack.back(); stack.pop_back();
            for (int i = 0; i < 3; ++i)
            {
                int u = M.neighbour(t, i);
                if (u < 0 || comp[u] >= 0) continue;
                if (M.edge_is_segment(t, i)) continue;
                comp[u] = c; stack.push_back(u);
            }
        }
    };
    flood(0);

    for (int t = 0; t < cap; ++t)
    {
        if (!M.alive(t) || comp[t] >= 0) continue;
        comp[t] = ncomp; stack.push_back(t);
        flood(ncomp);
        ++ncomp;
    }

    std::vector<char> inside(ncomp, 0);
    inside[0] = 0;                                   // the super-triangle side is always outside
    std::vector<int> rep(ncomp, -1);
    for (int t = 0; t < cap; ++t) if (M.alive(t) && rep[comp[t]] < 0) rep[comp[t]] = t;
    for (int c = 1; c < ncomp; ++c)
        if (rep[c] >= 0) inside[c] = D.contains(M.centroid(rep[c])) ? 1 : 0;

    M.filter_triangles([&](int t) { return inside[comp[t]] != 0; });
}

} // namespace detail
} // namespace femd

#endif // FEMD_TRIANGULATE_DELAUNAY_HPP
