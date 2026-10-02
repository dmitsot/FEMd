//
//  smooth.hpp  --  Laplacian smoothing, with the mesh kept Delaunay.
//
//  This is meshgen's jiggleMesh with two things added.  meshgen moved every
//  interior vertex to the average of its one-ring and accepted the move
//  unconditionally, which near a concave boundary or a badly shaped fan can
//  push a vertex out of its own polygon and invert triangles; and it never
//  restored the Delaunay property afterwards, so the smoothed mesh was no
//  longer a Delaunay triangulation of its own vertices.  Here a move is taken
//  only when every incident triangle keeps positive area and does not get
//  worse than the fan already was, and the mesh is flipped back to Delaunay
//  after each pass.
//
#ifndef FEMD_TRIANGULATE_SMOOTH_HPP
#define FEMD_TRIANGULATE_SMOOTH_HPP

#include "femd/mesh/mesh2d.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace femd {

/**
 * @brief Restore the Delaunay property by flipping, leaving segments alone.
 * @return the number of flips performed
 */
inline int restore_delaunay(Mesh2D &M, int max_sweeps = 64)
{
    int flips = 0;
    for (int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        int done = 0;
        for (int t = 0; t < M.tri_capacity(); ++t)
        {
            if (!M.alive(t)) continue;
            for (int i = 0; i < 3; ++i)
            {
                if (M.edge_is_segment(t, i)) continue;
                if (M.locally_delaunay(t, i)) continue;
                if (M.flip_edge(t, i)) { ++done; break; }
            }
        }
        flips += done;
        if (done == 0) break;
    }
    return flips;
}

/**
 * @brief Move interior vertices toward the centroid of their one-ring.
 * @param passes how many sweeps to run
 * @param restore flip back to Delaunay after each pass.  Pass false when the
 *        mesh carries deliberate non-Delaunay edges that must survive, as it
 *        does after the interior-vertex repair: restoring Delaunay would flip
 *        those corner edges straight back.
 * @return the number of vertices actually moved
 *
 * A vertex is interior when it belongs to no segment and its fan closes.  A
 * move is rejected if it would make any incident triangle non-positive in
 * area, or would lower EITHER the worst shape quality or the worst angle in
 * the fan.  Guarding on quality alone is not enough: quality and smallest
 * angle are not monotonically related, so a move can raise the quality of a
 * fan while shaving a fraction of a degree off its worst angle, and that is
 * enough to break the bound refinement just established.
 *
 * max_area and max_edge (0: none) guard the size bounds the same way: a move is
 * rejected when it makes a triangle of the fan larger than max_area, or an edge
 * at v longer than max_edge, beyond what the fan already had.  Without them a
 * vertex drifting to its centroid enlarges the triangles on the far side, and
 * smoothing a mesh refined to max_area left triangles up to a third larger.
 */
inline int laplacian_smooth(Mesh2D &M, int passes = 1, bool restore = true, double max_area = 0.0,
                            double max_edge = 0.0)
{
    if (passes <= 0) return 0;

    std::vector<char> on_segment(M.npoints(), 0);
    for (const auto &s : M.segments()) { on_segment[s[0]] = 1; on_segment[s[1]] = 1; }

    int moved = 0;
    std::vector<int> fan;
    for (int pass = 0; pass < passes; ++pass)
    {
        for (int v = 0; v < M.npoints(); ++v)
        {
            if (on_segment[v]) continue;
            if (!M.one_ring(v, fan)) continue;      // open fan: v is on the boundary
            if (fan.size() < 3) continue;

            double sx = 0.0, sy = 0.0; int n = 0;
            double q_before = 1.0, a_before = 180.0;
            for (int t : fan)
            {
                q_before = std::min(q_before, M.quality(t));
                a_before = std::min(a_before, M.min_angle(t));
                for (int i = 0; i < 3; ++i)
                    if (M.triangle(t)[i] != v) { sx += M.point(M.triangle(t)[i]).x; sy += M.point(M.triangle(t)[i]).y; ++n; }
            }
            if (n == 0) continue;
            Point target{sx / n, sy / n};           // each neighbour counted twice, which cancels

            // the size of the fan: its largest triangle, and its longest edge at v
            auto sizes = [&](double &amax, double &emax) {
                amax = 0.0; emax = 0.0;
                const Point &pv = M.point(v);
                for (int t : fan)
                {
                    amax = std::max(amax, M.area(t));
                    for (int i = 0; i < 3; ++i)
                    {
                        const int w = M.triangle(t)[i];
                        if (w == v) continue;
                        const double dx = M.point(w).x - pv.x, dy = M.point(w).y - pv.y;
                        emax = std::max(emax, dx * dx + dy * dy);
                    }
                }
                emax = std::sqrt(emax);
            };
            double amax_before = 0.0, emax_before = 0.0;
            if (max_area > 0.0 || max_edge > 0.0) sizes(amax_before, emax_before);

            Point old = M.point(v);
            M.move_point(v, target);
            bool ok = true;
            double q_after = 1.0, a_after = 180.0;
            for (int t : fan)
            {
                if (!(M.area(t) > 0.0)) { ok = false; break; }
                q_after = std::min(q_after, M.quality(t));
                a_after = std::min(a_after, M.min_angle(t));
            }
            if (ok && (max_area > 0.0 || max_edge > 0.0))
            {
                double amax_after, emax_after;
                sizes(amax_after, emax_after);
                if (max_area > 0.0 && amax_after > std::max(max_area, amax_before)) ok = false;
                if (max_edge > 0.0 && emax_after > std::max(max_edge, emax_before)) ok = false;
            }
            if (!ok || q_after < q_before || a_after < a_before - 1e-12) M.move_point(v, old);
            else ++moved;
        }
        if (restore) restore_delaunay(M);
    }
    return moved;
}

} // namespace femd

#endif // FEMD_TRIANGULATE_SMOOTH_HPP
