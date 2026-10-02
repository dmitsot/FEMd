//
//  refine.hpp  --  Ruppert refinement: split encroached segments, insert
//                  circumcentres of bad triangles.
//
//  The algorithm is the one the C meshgen implemented in refine.c, and the two
//  rules are unchanged:
//
//    * a segment is encroached when some vertex lies strictly inside its
//      diametral circle; an encroached segment is split at its midpoint
//    * a triangle is bad when its smallest angle is below the target, or it is
//      larger than the size limits; a bad triangle is fixed by inserting its
//      circumcentre, unless that circumcentre would encroach a segment, in
//      which case the segment is split instead and the triangle is revisited
//
//  Three things are different, and all three are fixes.
//
//  1. TERMINATION.  meshgen looped "while nothing changed" with no bound, and
//     the angle it compared against was wrong: its PI macro was written
//     `#define PI 4.0*atan(1.0)` without parentheses, so `angle*180.0/PI`
//     expanded to `angle*180.0/4.0*atan(1.0)` and reported every angle 1.621
//     times too small.  Asking meshgen for 20 degrees really asked for 32.4,
//     and asking for 25 asked for 40.5, which no Ruppert refinement can reach:
//     the loop never terminated and the process died.  min_angle here is in
//     true degrees, and values above the ceiling are rejected up front with an
//     explanation rather than run until the machine gives up.
//
//  2. NO ITERATOR TO INVALIDATE.  meshgen walked the face list with Ftraverse
//     while the insertion routine freed faces out of that same list, so the
//     traversal continued through freed memory.  Here the work list holds
//     integer indices and every pop is checked against alive().
//
//  3. A REFUSED INSERTION IS HANDLED.  When the cavity around a circumcentre
//     is not star-shaped, insert_vertex declines and the triangle is split on
//     its longest edge instead, rather than the mesh being corrupted.
//
#ifndef FEMD_TRIANGULATE_REFINE_HPP
#define FEMD_TRIANGULATE_REFINE_HPP

#include "femd/mesh/mesh2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace femd {

/// @brief Size and shape targets for Ruppert refinement.
struct RefineOptions {
    /// Smallest interior angle to aim for, in degrees.  0 disables the shape test.
    double min_angle = 20.0;
    /// Longest edge allowed.  0 means unbounded.
    double max_edge = 0.0;
    /// Largest triangle area allowed.  0 means unbounded.
    double max_area = 0.0;
    /// Hard cap on the vertex count; exceeding it throws rather than hanging.
    int max_points = 500000;
    /// Smallest angle present in the input, in degrees; used only for diagnostics.
    double input_angle = 180.0;
    /**
     * Shortest segment refinement may create.  A split below this throws
     * instead of proceeding.  0 disables the check, which is never a good idea:
     * without it, an unreachable target at a sharp corner splits the two
     * segments meeting there against each other until their midpoints collapse
     * onto the same double, leaving a fan of zero-area triangles that every
     * later stage silently accepts.  triangulate() sets it from the domain size.
     */
    double min_feature = 0.0;
    /**
     * Require every triangle to have at least one vertex OFF the boundary.
     *
     * A triangle whose three vertices are all on the boundary has no free
     * degree of freedom once Dirichlet data is imposed everywhere; the corner
     * triangle with two boundary edges is the common case, but a triangle
     * spanning a thin neck fails too.  See fix_boundary_only() for how it is
     * repaired and what it costs.
     */
    bool interior_vertex = false;
};

namespace detail {

/// @brief A number for an error message: short, and never rendered as "0.000000".
inline std::string num(double v)
{
    std::ostringstream o;
    double a = std::abs(v);
    if (a != 0.0 && (a < 1e-4 || a >= 1e6)) o << std::scientific << std::setprecision(3) << v;
    else                                    o << std::defaultfloat << std::setprecision(6) << v;
    return o.str();
}

/**
 * @brief Why refinement ran out of budget, said in terms of what is still bad.
 *
 * Two very different situations end up here and the caller needs to be able to
 * tell them apart: an honest request for a very fine mesh, where the answer is
 * simply a larger max_points, and an angle target that the geometry will not
 * allow, where a larger budget only buys a longer wait.  The message names the
 * counts so the reader does not have to guess which one they are in.
 */
inline std::string budget_message(const Mesh2D &M, const RefineOptions &opt)
{
    int bad_area = 0, bad_edge = 0, bad_angle = 0;
    double e2 = opt.max_edge > 0.0 ? opt.max_edge * opt.max_edge : 0.0;
    for (int t = 0; t < M.tri_capacity(); ++t)
    {
        if (!M.alive(t)) continue;
        if (opt.max_area > 0.0 && M.area(t) > opt.max_area) ++bad_area;
        if (e2 > 0.0 && M.longest_edge2(t) > e2) ++bad_edge;
        if (opt.min_angle > 0.0 && M.min_angle(t) < opt.min_angle) ++bad_angle;
    }
    std::string m = "refine: stopped at the budget of " + std::to_string(opt.max_points) + " points with "
                  + std::to_string(M.ntriangles()) + " triangles. Still failing: "
                  + std::to_string(bad_angle) + " on min_angle (" + detail::num(opt.min_angle) + " deg), "
                  + std::to_string(bad_area) + " on max_area, " + std::to_string(bad_edge) + " on max_edge. ";

    // Being cut off mid-refinement always leaves some bad angles behind, so the
    // angle count on its own says nothing.  What separates "this mesh is simply
    // large" from "this target is out of reach" is whether the size criteria
    // were still doing the work, and how close min_angle is to what the
    // geometry and the algorithm allow.
    bool size_driven  = (bad_area + bad_edge) > 0;
    bool angle_at_risk = opt.min_angle > 0.0
                      && (opt.min_angle > opt.input_angle - 3.0 || opt.min_angle > 30.0);

    if (size_driven && !angle_at_risk)
        m += "The size criteria were still refining when the budget ran out, and min_angle is well inside what "
             "this geometry allows, so this is a large mesh rather than a stall: raise max_points, or ask for "
             "larger triangles.";
    else if (size_driven)
        m += "Two things could be happening. The size criteria had not finished, which only needs a larger "
             "max_points; but min_angle = " + detail::num(opt.min_angle) + " degrees is also close to the "
             "limit here (the sharpest input corner is " + detail::num(opt.input_angle) + " degrees, and "
             "Ruppert refinement reaches about 33 at best), so it may be stalling. Try again with a lower "
             "min_angle to tell them apart.";
    else
        m += "The size criteria are satisfied, so the angle target is what refinement is chasing. The sharpest "
             "input corner is " + detail::num(opt.input_angle) + " degrees, and Ruppert refinement reaches "
             "about 33 degrees at best: lower min_angle, or round off that corner.";
    return m;
}

/**
 * @brief Uniform grid over the segments' diametral circles.
 *
 * Answers "which segments could this point encroach" in about constant time.
 * Rebuilt whenever the segment count has doubled, which costs O(S) and happens
 * O(log S) times over a whole refinement.
 */
class SegmentGrid {
public:
    void build(const Mesh2D &M)
    {
        n_ = M.nsegments();
        lo_ = {1e300, 1e300}; hi_ = {-1e300, -1e300};
        for (int s = 0; s < n_; ++s)
            for (int k = 0; k < 2; ++k)
            {
                const Point &p = M.point(M.segments()[s][k]);
                lo_.x = std::min(lo_.x, p.x); lo_.y = std::min(lo_.y, p.y);
                hi_.x = std::max(hi_.x, p.x); hi_.y = std::max(hi_.y, p.y);
            }
        if (n_ == 0) { nx_ = ny_ = 1; cell_.assign(1, {}); return; }
        double w = std::max(hi_.x - lo_.x, 1e-300), h = std::max(hi_.y - lo_.y, 1e-300);
        int k = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(n_) / 2.0)));
        nx_ = std::max(1, std::min(k, 4096));
        ny_ = std::max(1, std::min(k, 4096));
        dx_ = w / nx_; dy_ = h / ny_;
        cell_.assign(static_cast<std::size_t>(nx_) * ny_, {});
        for (int s = 0; s < n_; ++s)
        {
            const Point &a = M.point(M.segments()[s][0]), &b = M.point(M.segments()[s][1]);
            Point c{0.5 * (a.x + b.x), 0.5 * (a.y + b.y)};
            double r = 0.5 * std::sqrt(sqdist(a, b));
            int i0 = clampi(static_cast<int>((c.x - r - lo_.x) / dx_), nx_);
            int i1 = clampi(static_cast<int>((c.x + r - lo_.x) / dx_), nx_);
            int j0 = clampi(static_cast<int>((c.y - r - lo_.y) / dy_), ny_);
            int j1 = clampi(static_cast<int>((c.y + r - lo_.y) / dy_), ny_);
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) cell_[static_cast<std::size_t>(j) * nx_ + i].push_back(s);
        }
    }

    /// @brief True when enough segments have appeared since build() to be worth rebuilding.
    bool stale(const Mesh2D &M) const { return M.nsegments() - n_ > std::max(64, n_ / 4); }

    /**
     * @brief Segments whose diametral circle might contain p.
     *
     * Segments created since the last build() are not in the cells yet, so they
     * are appended here and scanned in full.  That overflow is what stale()
     * watches: the grid is rebuilt before the tail grows long enough to matter,
     * which keeps the query near constant time without ever missing a segment.
     * Missing one is not a performance bug but a correctness bug -- Ruppert
     * refinement diverges if a circumcentre is inserted that encroaches a
     * segment nobody checked.
     */
    const std::vector<int> &candidates(const Mesh2D &M, const Point &p) const
    {
        out_.clear();
        if (!cell_.empty())
        {
            int i = clampi(static_cast<int>((p.x - lo_.x) / dx_), nx_);
            int j = clampi(static_cast<int>((p.y - lo_.y) / dy_), ny_);
            out_ = cell_[static_cast<std::size_t>(j) * nx_ + i];
        }
        for (int s = n_; s < M.nsegments(); ++s) out_.push_back(s);
        return out_;
    }

private:
    static int clampi(int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); }
    Point lo_{}, hi_{};
    int n_ = 0, nx_ = 1, ny_ = 1;
    double dx_ = 1.0, dy_ = 1.0;
    std::vector<std::vector<int>> cell_;
    mutable std::vector<int> out_;
};

/**
 * @brief Give triangle t a vertex off the boundary.
 * @param newv    receives the new vertex index, or stays -1 if none was added
 * @param touched receives the vertices whose fans changed, for requeueing
 * @return 0 nothing needed, 1 fixed by a flip, 2 by splitting an interior edge,
 *         3 by inserting the centroid, -1 could not be fixed
 *
 * Three routes, in order of what they cost.
 *
 * **Flip.**  If an edge of t is interior and the triangle across it has its
 * apex off the boundary, flipping that edge replaces the pair by two triangles
 * that both reach the interior apex.  For the usual case -- a corner of the
 * polygon whose two segments bound a single triangle -- this is exactly the
 * right move and it adds no vertices.  It is only taken when both resulting
 * triangles still meet the angle target: a flip moves the pair AWAY from
 * Delaunay, so it can only lower the minimum angle, and a flip that breaks the
 * bound would just be undone by the next insertion nearby and re-made here,
 * which is how this turns into an oscillation.
 *
 * **Split an interior edge.**  Halving the longest non-segment edge puts a new
 * vertex strictly inside the region and gives it to both children.  Costs one
 * vertex, cannot be undone by a later Bowyer-Watson insertion, and leaves the
 * mesh Delaunay after legalizing.
 *
 * **Insert the centroid.**  Only when all three edges are segments, which
 * happens when a triangle spans the whole region or a one-element-wide neck.
 */
inline int give_interior_vertex(Mesh2D &M, int t, double min_angle, int &newv, std::vector<int> &touched)
{
    newv = -1; touched.clear();
    if (M.has_interior_vertex(t)) return 0;

    const std::array<int, 3> T = M.triangle(t);

    // 1. flip, if it reaches an interior apex and keeps the angle bound
    for (int i = 0; i < 3; ++i)
    {
        if (M.edge_is_segment(t, i)) continue;
        int u = M.neighbour(t, i);
        if (u < 0) continue;
        int j = M.local_neighbour(u, t);
        if (j < 0) continue;
        int q = M.triangle(u)[j];
        if (M.vertex_on_boundary(q)) continue;

        int p = T[i], a = T[(i + 1) % 3], b = T[(i + 2) % 3];
        if (min_angle > 0.0)
        {
            double A1 = min_angle_of(M.point(p), M.point(a), M.point(q));
            double A2 = min_angle_of(M.point(p), M.point(q), M.point(b));
            if (std::min(A1, A2) < min_angle) continue;      // would break the bound
        }
        if (M.flip_edge(t, i))
        {
            touched.assign({p, a, b, q});
            return 1;
        }
    }

    // 2. halve the longest interior edge
    int best = -1; double blen = -1.0;
    for (int i = 0; i < 3; ++i)
    {
        if (M.edge_is_segment(t, i)) continue;
        if (M.neighbour(t, i) < 0) continue;
        double l = sqdist(M.point(T[(i + 1) % 3]), M.point(T[(i + 2) % 3]));
        if (l > blen) { blen = l; best = i; }
    }
    if (best >= 0)
    {
        int a = T[(best + 1) % 3], b = T[(best + 2) % 3];
        Point mid{0.5 * (M.point(a).x + M.point(b).x), 0.5 * (M.point(a).y + M.point(b).y)};
        int v = M.add_point(mid, 0);
        if (M.split_edge_at(a, b, v))
        {
            M.legalize_around(v);        // every triangle it touches still contains v
            newv = v; touched.assign({v});
            return 2;
        }
        M.pop_last_point();
    }

    // 3. all three edges are segments: put a point in the middle
    int v = M.insert_point(M.centroid(t), 0, t, /*respect_segments=*/true);
    if (v >= 0) { M.legalize_around(v); newv = v; touched.assign({v}); return 3; }
    return -1;
}

/// @brief Is p strictly inside the diametral circle of the segment (a,b)?
inline bool encroaches(const Point &a, const Point &b, const Point &p)
{
    return (a.x - p.x) * (b.x - p.x) + (a.y - p.y) * (b.y - p.y) < 0.0;
}

} // namespace detail

/**
 * @brief Refine M in place until every triangle meets the options.
 * @return the number of vertices inserted
 * @throws std::runtime_error when the point budget is exhausted, with a message
 *         that says which target is out of reach and why.
 *
 * M must already be trimmed to the region, with its boundary recorded as
 * segments.  The mesh stays a constrained Delaunay triangulation throughout.
 */
/// @brief What refine() did, beyond the vertex count.
struct RefineCounts {
    int inserted = 0;    ///< vertices added
    int flips    = 0;    ///< interior-vertex fixes done by a flip (no new vertex)
    int splits   = 0;    ///< interior-vertex fixes done by halving an interior edge
    int centres  = 0;    ///< interior-vertex fixes done by inserting a centroid
    int unfixed  = 0;    ///< triangles left with every vertex on the boundary
};

inline int refine(Mesh2D &M, const RefineOptions &opt, RefineCounts *counts = nullptr)
{
    // Ruppert's guarantee holds up to about 20.7 degrees and in practice the
    // algorithm still terminates to roughly 33; beyond that it is known not to.
    if (opt.min_angle >= 34.0)
        throw std::invalid_argument("refine: min_angle = " + detail::num(opt.min_angle)
            + " degrees is above what Ruppert refinement can reach (the practical ceiling is about 33; "
              "the proof covers 20.7). Ask for less, or use smoothing to improve the shape further.");
    if (opt.min_angle < 0.0) throw std::invalid_argument("refine: min_angle must not be negative");

    const double max_edge2 = opt.max_edge > 0.0 ? opt.max_edge * opt.max_edge : 0.0;

    detail::SegmentGrid grid;
    grid.build(M);

    std::vector<int> segq, triq, fan, touched;
    RefineCounts cnt;
    int inserted = 0;

    auto push_fan = [&](int v) {
        M.one_ring(v, fan);
        for (int t : fan)
        {
            triq.push_back(t);
            for (int i = 0; i < 3; ++i)
            {
                int a = M.triangle(t)[(i + 1) % 3], b = M.triangle(t)[(i + 2) % 3];
                int s = M.segment_index(a, b);
                if (s >= 0) segq.push_back(s);
            }
        }
    };

    auto note_new_vertex = [&](int v) {
        ++inserted; cnt.inserted = inserted;
        if (grid.stale(M)) grid.build(M);
        for (int s : grid.candidates(M, M.point(v)))
        {
            int a = M.segments()[s][0], b = M.segments()[s][1];
            if (a != v && b != v && detail::encroaches(M.point(a), M.point(b), M.point(v))) segq.push_back(s);
        }
        push_fan(v);
        if (M.npoints() > opt.max_points) throw std::runtime_error(detail::budget_message(M, opt));
    };

    /// true when some vertex lies in the diametral circle of segment s
    auto segment_encroached = [&](int s) {
        int a = M.segments()[s][0], b = M.segments()[s][1];
        int t1, i1, t2, i2;
        if (!M.edge_triangles(a, b, t1, i1, t2, i2)) return false;
        if (t1 >= 0 && detail::encroaches(M.point(a), M.point(b), M.point(M.triangle(t1)[i1]))) return true;
        if (t2 >= 0 && detail::encroaches(M.point(a), M.point(b), M.point(M.triangle(t2)[i2]))) return true;
        return false;
    };

    auto split_segment_at_midpoint = [&](int s) {
        if (opt.min_feature > 0.0)
        {
            const Point &a = M.point(M.segments()[s][0]), &b = M.point(M.segments()[s][1]);
            if (sqdist(a, b) < opt.min_feature * opt.min_feature)
                throw std::runtime_error(
                    "refine: refinement wants to split a boundary segment that is already "
                    + detail::num(std::sqrt(sqdist(a, b))) + " long, below the feature floor of "
                    + detail::num(opt.min_feature) + ", near (" + detail::num(a.x) + ", " + detail::num(a.y)
                    + "). min_angle = " + detail::num(opt.min_angle) + " degrees cannot be reached there: the "
                    "smallest angle in the input geometry is " + detail::num(opt.input_angle) + " degrees, and two "
                    "segments meeting at a sharp corner split each other without limit when the target exceeds what "
                    "the corner allows. Lower min_angle, or round off that corner.");
        }
        int before = M.nsegments();
        int v = M.split_segment_midpoint(s);
        if (v < 0) return false;
        segq.push_back(s);
        for (int k = before; k < M.nsegments(); ++k) segq.push_back(k);
        note_new_vertex(v);
        return true;
    };

    auto drain_segments = [&]() {
        while (!segq.empty())
        {
            int s = segq.back(); segq.pop_back();
            if (s >= M.nsegments()) continue;
            if (segment_encroached(s)) split_segment_at_midpoint(s);
        }
    };

    // The interior-vertex criterion is kept separate from is_bad(): it is fixed
    // by a different operation, and mixing the two would send a corner triangle
    // down the circumcentre path, which cannot fix it -- the circumcentre of a
    // triangle at a sharp corner encroaches the segments there, splitting them
    // makes more boundary vertices, and the triangle stays all-boundary forever.
    auto needs_interior = [&](int t) { return opt.interior_vertex && !M.has_interior_vertex(t); };

    auto is_bad = [&](int t) {
        if (opt.max_area > 0.0 && M.area(t) > opt.max_area) return true;
        if (max_edge2 > 0.0 && M.longest_edge2(t) > max_edge2) return true;
        if (opt.min_angle > 0.0 && M.min_angle(t) < opt.min_angle) return true;
        return false;
    };

    // seed the work lists
    for (int s = 0; s < M.nsegments(); ++s) segq.push_back(s);
    for (int t = 0; t < M.tri_capacity(); ++t) if (M.alive(t)) triq.push_back(t);

    for (int outer = 0; outer < 64; ++outer)
    {
        while (true)
        {
            drain_segments();

            int t = -1;
            while (!triq.empty())
            {
                int u = triq.back(); triq.pop_back();
                if (!(u < M.tri_capacity() && M.alive(u))) continue;
                if (needs_interior(u) || is_bad(u)) { t = u; break; }
            }
            if (t < 0) break;

            if (needs_interior(t))
            {
                int nv = -1;
                int how = detail::give_interior_vertex(M, t, opt.min_angle, nv, touched);
                if (how > 0)
                {
                    if (how == 1) ++cnt.flips; else if (how == 2) ++cnt.splits; else ++cnt.centres;
                    if (nv >= 0) note_new_vertex(nv);
                    for (int v : touched) push_fan(v);
                    continue;
                }
                if (how < 0 && !is_bad(t)) { ++cnt.unfixed; continue; }
                // how < 0 but the triangle is bad on size or shape as well:
                // fall through and let ordinary refinement have a go at it.
            }

            Point c;
            try { c = M.circumcentre_of(t); }
            catch (const std::runtime_error &) { continue; }   // degenerate, nothing sensible to do

            // Would the circumcentre encroach a segment?  Then split those
            // instead and revisit the triangle.  The split has to happen here
            // and not through segq: segq re-tests encroachment against the
            // vertices that exist, and c is not one of them yet, so a segment
            // that only c encroaches would be dropped and the triangle would
            // come round again forever.  This is Ruppert's rule as stated.
            if (grid.stale(M)) grid.build(M);
            std::vector<int> hit;
            for (int s : grid.candidates(M, c))
            {
                if (s >= M.nsegments()) continue;
                int a = M.segments()[s][0], b = M.segments()[s][1];
                if (detail::encroaches(M.point(a), M.point(b), c)) hit.push_back(s);
            }
            if (!hit.empty())
            {
                bool any = false;
                for (int s : hit) if (split_segment_at_midpoint(s)) any = true;
                if (any) { triq.push_back(t); continue; }
                // none of them could be split: fall through to the edge split
            }

            int v = M.insert_point(c, 0, t, /*respect_segments=*/true);
            if (v >= 0) { note_new_vertex(v); continue; }

            // The circumcentre was refused: fall back to halving the longest edge.
            int which = 0;
            M.longest_edge2(t, &which);
            int a = M.triangle(t)[(which + 1) % 3], b = M.triangle(t)[(which + 2) % 3];
            int s = M.segment_index(a, b);
            if (s >= 0) { if (!split_segment_at_midpoint(s)) continue; }
            else
            {
                Point m{0.5 * (M.point(a).x + M.point(b).x), 0.5 * (M.point(a).y + M.point(b).y)};
                int w = M.insert_point(m, 0, t, /*respect_segments=*/true);
                if (w < 0) continue;            // give this triangle up rather than corrupt the mesh
                M.legalize_around(w);
                note_new_vertex(w);
            }
        }

        // A full sweep confirms the work lists did not miss anything.
        bool any = false;
        for (int s = 0; s < M.nsegments(); ++s) if (segment_encroached(s)) { segq.push_back(s); any = true; }
        for (int t = 0; t < M.tri_capacity(); ++t)
            if (M.alive(t) && (is_bad(t) || needs_interior(t))) { triq.push_back(t); any = true; }
        if (!any) break;
    }

    // Whatever is left after the sweeps is reported rather than hidden.
    cnt.unfixed = 0;
    if (opt.interior_vertex)
        for (int t = 0; t < M.tri_capacity(); ++t)
            if (M.alive(t) && !M.has_interior_vertex(t)) ++cnt.unfixed;
    if (counts) *counts = cnt;
    return inserted;
}

} // namespace femd

#endif // FEMD_TRIANGULATE_REFINE_HPP
