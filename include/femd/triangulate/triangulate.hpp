//
//  triangulate.hpp  --  mesh a Domain: one call, one Mesh2D.
//
//      Domain d = Domain::rectangle(0, 0, 1, 1, 8, 8);
//      d.hole(Domain::circle(0.5, 0.5, 0.15, 24, true), 2);
//      Mesh2D m = triangulate(d, {.min_angle = 30.0, .max_area = 2e-3});
//
//  The pipeline is meshgen's, in meshgen's order: Bowyer-Watson over the input
//  vertices, conform the boundary, throw the outside away, refine to quality,
//  smooth.  See delaunay.hpp, refine.hpp and smooth.hpp for what changed in
//  each step and why.
//
#ifndef FEMD_TRIANGULATE_TRIANGULATE_HPP
#define FEMD_TRIANGULATE_TRIANGULATE_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/domain.hpp"
#include "femd/triangulate/delaunay.hpp"
#include "femd/triangulate/refine.hpp"
#include "femd/triangulate/smooth.hpp"

#include <cmath>
#include <stdexcept>

namespace femd {

/**
 * @brief Everything triangulate() can be asked for.
 *
 * @warning New fields go at the END. Callers brace-initialize this positionally
 *          (MeshOptions{30.0, 0.0, 1e-3}), so inserting a field in the middle
 *          silently changes what every existing call means.
 */
struct MeshOptions {
    /// Smallest interior angle to aim for, in degrees.  0 turns the shape test off.
    double min_angle = 20.0;
    /// Longest edge allowed anywhere.  0 means unbounded.
    double max_edge = 0.0;
    /// Largest triangle area allowed.  0 means unbounded.
    double max_area = 0.0;
    /// Laplacian smoothing passes after refinement.
    int smooth_passes = 0;
    /// Hard cap on the vertex count; exceeding it throws with a diagnosis.
    int max_points = 500000;
    /// Run Mesh2D::validate() at the end and throw if it complains.
    bool check = false;
    /**
     * Require every triangle to have at least one vertex OFF the boundary.
     *
     * A triangle with all three vertices on the boundary has no free degree of
     * freedom once Dirichlet data is imposed everywhere, so it contributes
     * nothing to the solve.  The commonest case is the triangle filling a
     * corner of the polygon, which has two of its edges on the boundary; this
     * option also covers the triangle that spans a thin neck with one edge on
     * each side.
     *
     * Enforced by a flip where a flip reaches an interior vertex and keeps the
     * angle bound, and by inserting a vertex where it does not, then refining
     * again to repair the shape, iterating until neither step changes anything.
     * The cost is a handful of vertices near corners; the report says exactly
     * how many of each kind of fix were needed.
     */
    bool interior_vertex = false;
};

/// @brief What the mesher did, for logging and for tests.
struct MeshReport {
    int input_vertices = 0;     ///< vertices in the Domain
    int boundary_splits = 0;    ///< midpoints added to make the boundary conform
    int refine_inserts = 0;     ///< vertices added by Ruppert refinement
    int smoothed = 0;           ///< vertices moved by smoothing
    int corner_flips = 0;       ///< interior_vertex fixes done by flipping an edge
    int corner_splits = 0;      ///< interior_vertex fixes done by halving an interior edge
    int corner_centres = 0;     ///< interior_vertex fixes done by inserting a centroid
    int boundary_only = 0;      ///< triangles STILL having all three vertices on the boundary
    int two_boundary_edges = 0; ///< triangles with two or three edges on the boundary
    double min_angle = 0.0;     ///< smallest angle in the result, degrees
    double min_quality = 0.0;   ///< smallest shape quality in the result
    double max_area = 0.0;      ///< largest triangle area in the result
    double total_area = 0.0;    ///< area of the meshed region
};

/// @brief Mesh a region.  See MeshOptions.
inline Mesh2D triangulate(const Domain &D, const MeshOptions &opt = {}, MeshReport *report = nullptr)
{
    if (D.nrings() == 0) throw std::invalid_argument("triangulate: the domain has no outer boundary");
    if (D.rings()[0].hole) throw std::invalid_argument("triangulate: the domain has holes but no outer boundary");

    Mesh2D M;
    D.emit(M);
    int nin = M.npoints();

    Point lo, hi;
    D.bbox(lo, hi);
    auto super = detail::add_super_triangle(M, lo, hi);

    detail::insert_range(M, 0, nin);
    int splits = detail::recover_segments(M, 64, opt.max_points);
    detail::trim_exterior(M, D, super);

    // An input corner of angle alpha forces a triangle with an angle at most
    // alpha, so no amount of refinement can lift the minimum above it.  Say so
    // now rather than after half a million points.
    // With interior_vertex the ceiling halves, and the reason is worth stating
    // because it is exact rather than a rule of thumb: forcing every triangle to
    // reach an interior vertex means the corner of angle alpha is spanned by at
    // least two triangles, so one of them has an angle of at most alpha/2.
    double ia = D.smallest_input_angle();
    double ceiling = opt.interior_vertex ? 0.5 * ia : ia;
    if (opt.min_angle > ceiling + 1e-9)
        throw std::invalid_argument(
            "triangulate: min_angle = " + detail::num(opt.min_angle) + " degrees is unreachable: the input geometry "
            "has a corner of " + detail::num(ia) + " degrees"
            + (opt.interior_vertex
                 ? std::string(", and interior_vertex forces at least two triangles to share that corner, so one of "
                               "them has an angle of at most half of it")
                 : std::string(", and a triangle filling that corner cannot have a larger smallest angle"))
            + ". Ask for at most " + detail::num(ceiling)
            + (opt.interior_vertex ? ", turn interior_vertex off, or round off the corner."
                                   : ", or round off the corner."));

    double diag = std::sqrt((hi.x - lo.x) * (hi.x - lo.x) + (hi.y - lo.y) * (hi.y - lo.y));

    RefineOptions ro;
    ro.min_feature = 1e-9 * diag;
    ro.min_angle   = opt.min_angle;
    ro.max_edge    = opt.max_edge;
    ro.max_area    = opt.max_area;
    ro.max_points  = opt.max_points;
    ro.input_angle = D.smallest_input_angle();
    RefineCounts rc;
    int added = refine(M, ro, &rc);
    int moved = laplacian_smooth(M, opt.smooth_passes, true, opt.max_area, opt.max_edge);
    // The moves keep the size bounds, but the Delaunay flips that end each pass can still make a
    // triangle or an edge too large (the quadrilateral of a flip is fixed, its diagonal is not).
    // Refining once more repairs that, with a handful of points where it is needed, and restores
    // every guarantee of the first refinement; on a mesh that already meets them it adds nothing.
    if (opt.smooth_passes > 0 && !opt.interior_vertex)
    {
        RefineCounts rc1;
        added += refine(M, ro, &rc1);
    }

    // The interior-vertex repair runs AFTER smoothing, because smoothing ends
    // each pass by flipping back to Delaunay and would undo the corner flips.
    // Refinement inside this second call repairs any angle the flips cost.
    if (opt.interior_vertex)
    {
        ro.interior_vertex = true;
        RefineCounts rc2;
        added += refine(M, ro, &rc2);
        rc.flips += rc2.flips; rc.splits += rc2.splits;
        rc.centres += rc2.centres; rc.unfixed = rc2.unfixed;
        moved += laplacian_smooth(M, opt.smooth_passes, /*restore=*/false, opt.max_area, opt.max_edge);
    }
    M.compact();

    if (opt.check)
    {
        std::string bad = M.validate();
        if (!bad.empty()) throw std::runtime_error("triangulate: the result failed validation: " + bad);
    }

    if (report)
    {
        report->input_vertices  = nin;
        report->boundary_splits = splits;
        report->refine_inserts  = added;
        report->smoothed        = moved;
        report->corner_flips    = rc.flips;
        report->corner_splits   = rc.splits;
        report->corner_centres  = rc.centres;
        report->boundary_only   = 0;
        report->two_boundary_edges = 0;
        report->min_angle       = 180.0;
        report->min_quality     = 1.0;
        report->max_area        = 0.0;
        report->total_area      = 0.0;
        for (int t = 0; t < M.tri_capacity(); ++t)
        {
            if (!M.alive(t)) continue;
            report->min_angle   = std::min(report->min_angle, M.min_angle(t));
            if (!M.has_interior_vertex(t))    ++report->boundary_only;
            if (M.boundary_edge_count(t) > 1) ++report->two_boundary_edges;
            report->min_quality = std::min(report->min_quality, M.quality(t));
            report->max_area    = std::max(report->max_area, M.area(t));
            report->total_area += M.area(t);
        }
    }
    return M;
}

} // namespace femd

#endif // FEMD_TRIANGULATE_TRIANGULATE_HPP
