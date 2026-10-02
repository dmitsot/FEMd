//
//  domain.hpp  --  the region to be meshed: one outer polygon and its holes.
//
//  This is the input the C meshgen read from its text file, with the two rules
//  that file format enforced by hand relaxed into something the code checks and
//  fixes itself.  meshgen required the outer boundary counter-clockwise and
//  every hole clockwise, and called exit(0) on anything else; a Domain accepts
//  either winding and normalizes it, because a caller that has just built a
//  polygon from a parametrization should not have to know which way round it
//  came out.
//
//  Each vertex and each segment carries an integer marker, which is what the
//  vbc field was for: it survives meshing untouched and comes back out on the
//  generated boundary vertices, so a solver can look up its boundary condition
//  by marker.  Marker 0 means "interior", so boundary markers start at 1.
//
#ifndef FEMD_MESH_DOMAIN_HPP
#define FEMD_MESH_DOMAIN_HPP

#include "femd/mesh/predicates.hpp"
#include "femd/mesh/mesh2d.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

/**
 * @brief A planar region bounded by one closed outer polygon, with holes.
 *
 * Rings are closed implicitly: do not repeat the first vertex at the end.
 */
class Domain {
public:
    struct Ring {
        std::vector<Point> pts;
        std::vector<int>   vmark;   ///< per-vertex marker, same length as pts
        int                smark;   ///< marker given to every segment of this ring
        bool               hole;
    };

    /// @brief Signed area of a ring; positive means counter-clockwise.
    static double signed_area(const std::vector<Point> &p)
    {
        double s = 0.0;
        for (std::size_t i = 0, n = p.size(); i < n; ++i)
        {
            const Point &a = p[i], &b = p[(i + 1) % n];
            s += a.x * b.y - b.x * a.y;
        }
        return 0.5 * s;
    }

    /**
     * @brief Set the outer boundary.  Re-orients to counter-clockwise if needed.
     * @param pts    the ring vertices, first vertex not repeated (at least 3)
     * @param marker marker for every vertex and segment of this ring
     */
    Domain &boundary(std::vector<Point> pts, int marker = 1)
    {
        std::vector<int> vm(pts.size(), marker);
        return boundary(std::move(pts), std::move(vm), marker);
    }

    /// @brief Outer boundary with a marker per vertex (the meshgen vbc field).
    Domain &boundary(std::vector<Point> pts, std::vector<int> vmark, int smarker = 1)
    {
        check_ring(pts, vmark, "boundary");
        if (!rings_.empty() && !rings_[0].hole) throw std::invalid_argument("Domain: the outer boundary is already set");
        if (signed_area(pts) < 0.0) { reverse_ring(pts, vmark); }
        rings_.insert(rings_.begin(), Ring{std::move(pts), std::move(vmark), smarker, false});
        return *this;
    }

    /// @brief Add a hole.  Re-orients to clockwise if needed.
    Domain &hole(std::vector<Point> pts, int marker = 2)
    {
        std::vector<int> vm(pts.size(), marker);
        return hole(std::move(pts), std::move(vm), marker);
    }

    Domain &hole(std::vector<Point> pts, std::vector<int> vmark, int smarker = 2)
    {
        check_ring(pts, vmark, "hole");
        if (signed_area(pts) > 0.0) { reverse_ring(pts, vmark); }
        rings_.push_back(Ring{std::move(pts), std::move(vmark), smarker, true});
        return *this;
    }

    // ---- convenience factories ---------------------------------------------
    /// @brief Axis-aligned rectangle with n subdivisions along x and m along y.
    static Domain rectangle(double x0, double y0, double x1, double y1, int n = 1, int m = 1, int marker = 1)
    {
        if (!(x1 > x0 && y1 > y0)) throw std::invalid_argument("Domain::rectangle: need x1 > x0 and y1 > y0");
        if (n < 1 || m < 1) throw std::invalid_argument("Domain::rectangle: need n, m >= 1");
        std::vector<Point> p;
        for (int i = 0; i < n; ++i) p.push_back({x0 + (x1 - x0) * i / n, y0});
        for (int j = 0; j < m; ++j) p.push_back({x1, y0 + (y1 - y0) * j / m});
        for (int i = n; i > 0; --i) p.push_back({x0 + (x1 - x0) * i / n, y1});
        for (int j = m; j > 0; --j) p.push_back({x0, y0 + (y1 - y0) * j / m});
        Domain d; d.boundary(std::move(p), marker); return d;
    }

    /// @brief Regular n-gon approximating a circle, as an outer boundary.
    static std::vector<Point> circle(double cx, double cy, double r, int n, bool clockwise = false)
    {
        if (n < 3) throw std::invalid_argument("Domain::circle: need n >= 3");
        std::vector<Point> p;
        for (int i = 0; i < n; ++i)
        {
            double t = 2.0 * 3.14159265358979323846 * i / n;
            if (clockwise) t = -t;
            p.push_back({cx + r * std::cos(t), cy + r * std::sin(t)});
        }
        return p;
    }

    // ---- queries -------------------------------------------------------------
    const std::vector<Ring> &rings() const { return rings_; }
    int nrings() const { return static_cast<int>(rings_.size()); }
    int nholes() const { return static_cast<int>(rings_.size()) - (rings_.empty() ? 0 : 1); }

    int nvertices() const
    {
        int n = 0;
        for (const auto &r : rings_) n += static_cast<int>(r.pts.size());
        return n;
    }

    void bbox(Point &lo, Point &hi) const
    {
        lo = {1e300, 1e300}; hi = {-1e300, -1e300};
        for (const auto &r : rings_)
            for (const auto &p : r.pts)
            {
                lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y);
                hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y);
            }
    }

    /**
     * @brief Is p inside the region (inside the outer ring and outside every hole)?
     *
     * Crossing-number test built on the exact orient2d predicate, so a point
     * that is genuinely on a boundary edge is decided consistently rather than
     * by a tolerance.
     */
    bool contains(const Point &p) const
    {
        if (rings_.empty()) return false;
        if (!ring_contains(rings_[0].pts, p)) return false;
        for (std::size_t i = 1; i < rings_.size(); ++i)
            if (ring_contains(rings_[i].pts, p)) return false;
        return true;
    }

    /**
     * @brief Smallest angle between two consecutive segments, in degrees.
     *
     * Ruppert's algorithm cannot produce triangles whose minimum angle exceeds
     * half the smallest input angle, so this is the number that decides whether
     * a requested min_angle is reachable at all.  triangulate() reports it when
     * refinement fails to terminate.
     */
    double smallest_input_angle() const
    {
        double best = 180.0;
        for (const auto &r : rings_)
        {
            std::size_t n = r.pts.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const Point &a = r.pts[(i + n - 1) % n], &b = r.pts[i], &c = r.pts[(i + 1) % n];
                double ux = a.x - b.x, uy = a.y - b.y, vx = c.x - b.x, vy = c.y - b.y;
                double ang = std::atan2(std::abs(ux * vy - uy * vx), ux * vx + uy * vy) * 180.0 / 3.14159265358979323846;
                best = std::min(best, ang);
            }
        }
        return best;
    }

    /**
     * @brief Lay the rings down as points and segments of a fresh mesh.
     *
     * Vertex i of the mesh is vertex i of the concatenated rings, so the
     * caller's numbering survives.  No triangles are created.
     */
    void emit(Mesh2D &M) const
    {
        for (const auto &r : rings_)
        {
            int base = M.npoints();
            int n = static_cast<int>(r.pts.size());
            for (int i = 0; i < n; ++i) M.add_point(r.pts[i], r.vmark[i]);
            for (int i = 0; i < n; ++i) M.add_segment(base + i, base + (i + 1) % n, r.smark);
        }
    }

private:
    static void check_ring(const std::vector<Point> &p, const std::vector<int> &m, const char *what)
    {
        if (p.size() < 3) throw std::invalid_argument(std::string("Domain::") + what + ": need at least 3 vertices");
        if (m.size() != p.size()) throw std::invalid_argument(std::string("Domain::") + what + ": marker count must match vertex count");
        if (signed_area(p) == 0.0) throw std::invalid_argument(std::string("Domain::") + what + ": ring has zero area");
    }

    static void reverse_ring(std::vector<Point> &p, std::vector<int> &m)
    {
        std::reverse(p.begin(), p.end());
        std::reverse(m.begin(), m.end());
    }

    static bool ring_contains(const std::vector<Point> &r, const Point &p)
    {
        int wind = 0;
        std::size_t n = r.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const Point &a = r[i], &b = r[(i + 1) % n];
            if (a.y <= p.y)
            {
                if (b.y > p.y && pred::orient2d(a.x, a.y, b.x, b.y, p.x, p.y) > 0) ++wind;
            }
            else
            {
                if (b.y <= p.y && pred::orient2d(a.x, a.y, b.x, b.y, p.x, p.y) < 0) --wind;
            }
        }
        return wind != 0;
    }

    std::vector<Ring> rings_;
};

} // namespace femd

#endif // FEMD_MESH_DOMAIN_HPP
