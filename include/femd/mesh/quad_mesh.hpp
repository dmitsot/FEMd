//
//  quad_mesh.hpp  --  2D quadrilateral mesh: points, quads, neighbours, segments.
//
//      quad_[q]  = {v0, v1, v2, v3}   the four vertices, counter-clockwise, every quad convex
//      nbr_[q]   = {n0, n1, n2, n3}   nbr_[q][i] is the quad across edge i, the edge from
//                                     v_i to v_{(i+1)%4}; -1 on the boundary
//
//  (The triangle mesh numbers edge i opposite vertex i; a quad has no opposite vertex, so
//  here edge i starts at vertex i.)  Segments are the constrained edges with their markers,
//  and every boundary edge is one.
//
//  Generators: from arrays, and from a triangulation by splitting every triangle into three
//  quads at its edge midpoints and centroid (always convex, holes and markers carried over),
//  with a Laplacian smoothing that never makes a quad non-convex.  Structured and mapped quad
//  meshes are built in Python on the grids of structured.hpp.
//
#ifndef FEMD_MESH_QUAD_MESH_HPP
#define FEMD_MESH_QUAD_MESH_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/predicates.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace femd {

class QuadMesh {
public:
    QuadMesh() = default;

    /**
     * @brief A quad mesh from arrays.
     * @param quads    vertex indices; clockwise quads are re-oriented
     * @param segs     constrained edges with their markers (each must be an edge of the mesh);
     *                 boundary edges not listed become segments with `marker`
     * @param pmarks   point markers, empty: 0 inside, and on the boundary the marker of the
     *                 segment leaving the vertex counter-clockwise
     * @throws std::invalid_argument for an index out of range, a degenerate or non-convex quad,
     *         an edge shared by more than two quads, or a segment that is not an edge
     */
    QuadMesh(std::vector<Point> pts, std::vector<std::array<int, 4>> quads, std::vector<std::array<int, 2>> segs = {},
             std::vector<int> smarks = {}, std::vector<int> pmarks = {}, int marker = 1)
        : pt_(std::move(pts)), q_(std::move(quads))
    {
        const int np = npoints();
        int bad = 0, first_bad = -1;
        for (std::size_t t = 0; t < q_.size(); ++t)
        {
            for (int v : q_[t])
                if (v < 0 || v >= np) throw std::invalid_argument("QuadMesh: quad " + std::to_string(t) + " has a vertex out of range");
            if (signed_area(static_cast<int>(t)) < 0.0) std::swap(q_[t][1], q_[t][3]);
            if (!convex(static_cast<int>(t))) { if (first_bad < 0) first_bad = static_cast<int>(t); ++bad; }
        }
        if (bad)
            throw std::invalid_argument("QuadMesh: " + std::to_string(bad) + " quad(s) degenerate or not convex, the first is quad " +
                                        std::to_string(first_bad));
        build_neighbours();
        if (smarks.empty()) smarks.assign(segs.size(), marker);
        if (smarks.size() != segs.size()) throw std::invalid_argument("QuadMesh: one marker per segment");
        for (std::size_t s = 0; s < segs.size(); ++s)
        {
            if (edge_map_.find(key(segs[s][0], segs[s][1])) == edge_map_.end())
                throw std::invalid_argument("QuadMesh: segment " + std::to_string(s) + " is not an edge of the mesh");
            add_segment(segs[s][0], segs[s][1], smarks[s]);
        }
        for (int t = 0; t < nquads(); ++t)
            for (int i = 0; i < 4; ++i)
                if (nbr_[t][i] < 0)
                {
                    const int a = q_[t][i], b = q_[t][(i + 1) % 4];
                    if (segment_index(a, b) < 0) add_segment(a, b, marker);
                }
        if (!pmarks.empty())
        {
            if (static_cast<int>(pmarks.size()) != np) throw std::invalid_argument("QuadMesh: one point marker per point");
            pmark_ = std::move(pmarks);
        }
        else
        {
            pmark_.assign(static_cast<std::size_t>(np), 0);
            for (int t = 0; t < nquads(); ++t)
                for (int i = 0; i < 4; ++i)
                    if (nbr_[t][i] < 0)
                    {
                        const int a = q_[t][i], b = q_[t][(i + 1) % 4];
                        pmark_[a] = smark_[segment_index(a, b)];     // the side leaving a counter-clockwise
                    }
        }
    }

    // ---- sizes and data ------------------------------------------------------------
    int npoints()   const { return static_cast<int>(pt_.size()); }
    int nquads()    const { return static_cast<int>(q_.size()); }
    int nsegments() const { return static_cast<int>(seg_.size()); }
    const std::vector<Point> &points() const { return pt_; }
    const std::vector<std::array<int, 4>> &quads() const { return q_; }
    const std::vector<std::array<int, 4>> &neighbours() const { return nbr_; }
    const std::vector<std::array<int, 2>> &segments() const { return seg_; }
    const std::vector<int> &segment_markers() const { return smark_; }
    const std::vector<int> &point_markers() const { return pmark_; }
    const Point &point(int v) const { return pt_[v]; }
    const std::array<int, 4> &quad(int q) const { return q_[q]; }
    int neighbour(int q, int i) const { return nbr_[q][i]; }
    int segment_index(int a, int b) const
    {
        auto it = segmap_.find(key(a, b));
        return it == segmap_.end() ? -1 : it->second;
    }

    // ---- geometry --------------------------------------------------------------------
    double signed_area(int q) const
    {
        double s = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const Point &a = pt_[q_[q][i]], &b = pt_[q_[q][(i + 1) % 4]];
            s += a.x * b.y - b.x * a.y;
        }
        return 0.5 * s;
    }
    double area(int q) const { return std::abs(signed_area(q)); }
    /// @brief Strictly convex, counter-clockwise: every corner turns left (exact orient2d).
    bool convex(int q) const
    {
        for (int i = 0; i < 4; ++i)
        {
            const Point &a = pt_[q_[q][(i + 3) % 4]], &b = pt_[q_[q][i]], &c = pt_[q_[q][(i + 1) % 4]];
            if (pred::orient2d(a.x, a.y, b.x, b.y, c.x, c.y) <= 0) return false;
        }
        return true;
    }
    /// @brief Interior angle at corner i, degrees.
    double angle(int q, int i) const
    {
        const Point &a = pt_[q_[q][(i + 3) % 4]], &b = pt_[q_[q][i]], &c = pt_[q_[q][(i + 1) % 4]];
        const double ux = a.x - b.x, uy = a.y - b.y, vx = c.x - b.x, vy = c.y - b.y;
        return std::atan2(std::abs(ux * vy - uy * vx), ux * vx + uy * vy) * 180.0 / 3.14159265358979323846;
    }
    double min_angle(int q) const { double m = 180.0; for (int i = 0; i < 4; ++i) m = std::min(m, angle(q, i)); return m; }
    double max_angle(int q) const { double m = 0.0; for (int i = 0; i < 4; ++i) m = std::max(m, angle(q, i)); return m; }
    /// @brief Longest over shortest edge.
    double aspect(int q) const
    {
        double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const Point &a = pt_[q_[q][i]], &b = pt_[q_[q][(i + 1) % 4]];
            const double l = std::hypot(b.x - a.x, b.y - a.y);
            lo = std::min(lo, l); hi = std::max(hi, l);
        }
        return hi / lo;
    }

    /// @brief True when p is inside q or on its boundary (exact, convex quads).
    bool contains(int q, const Point &p) const
    {
        for (int i = 0; i < 4; ++i)
        {
            const Point &a = pt_[q_[q][i]], &b = pt_[q_[q][(i + 1) % 4]];
            if (pred::orient2d(a.x, a.y, b.x, b.y, p.x, p.y) < 0) return false;
        }
        return true;
    }

    /**
     * @brief The quad containing p, or -1.  Tries `hint`, then a walk, then a bucket grid over the
     *        bounding box (built on first use, O(1) per query on average afterwards).
     */
    int locate(const Point &p, int hint = -1) const
    {
        if (nquads() == 0) return -1;
        int t = (hint >= 0 && hint < nquads()) ? hint : last_;
        for (int step = 0; step < 4 * nquads() + 16; ++step)
        {
            if (contains(t, p)) { last_ = t; return t; }
            int next = -1;
            for (int i = 0; i < 4; ++i)
            {
                const Point &a = pt_[q_[t][i]], &b = pt_[q_[t][(i + 1) % 4]];
                if (pred::orient2d(a.x, a.y, b.x, b.y, p.x, p.y) < 0) { next = nbr_[t][i]; break; }
            }
            if (next < 0) break;
            t = next;
        }
        if (!grid_) build_grid();
        const int q = locate_prepared(p);
        if (q >= 0) last_ = q;
        return q;
    }

    /// @brief Build the bucket grid now (locate() builds it on its first miss).
    void prepare_locate() const { if (nquads() > 0 && !grid_) build_grid(); }

    /// @brief The quad containing p from the grid alone, -1 outside.  prepare_locate() first.
    ///        Const and stateless, so several threads may call it at once.
    int locate_prepared(const Point &p) const
    {
        if (nquads() == 0 || !grid_) return -1;
        const Grid &G = *grid_;
        if (p.x < G.x0 || p.y < G.y0 || p.x > G.x0 + G.w || p.y > G.y0 + G.h) return -1;
        const int i = std::min(G.nx - 1, static_cast<int>((p.x - G.x0) * G.nx / G.w));
        const int j = std::min(G.ny - 1, static_cast<int>((p.y - G.y0) * G.ny / G.h));
        const std::vector<int> &cell = G.cells[static_cast<std::size_t>(j) * G.nx + i];
        for (int q : cell) if (contains(q, p)) return q;
        return -1;
    }

    /// @brief "" when sound, otherwise the first fault.
    std::string validate() const
    {
        for (int t = 0; t < nquads(); ++t)
        {
            if (!convex(t)) return "quad " + std::to_string(t) + " is not convex and counter-clockwise";
            for (int i = 0; i < 4; ++i)
            {
                const int n = nbr_[t][i];
                if (n < 0) { if (segment_index(q_[t][i], q_[t][(i + 1) % 4]) < 0) return "boundary edge without a segment"; continue; }
                bool back = false;
                for (int j = 0; j < 4; ++j) back = back || nbr_[n][j] == t;
                if (!back) return "neighbour links of quads " + std::to_string(t) + " and " + std::to_string(n) + " disagree";
            }
        }
        return "";
    }

    /**
     * @brief Laplacian smoothing of the interior vertices, `passes` sweeps.  A move that would
     *        make an adjacent quad non-convex is halved, then dropped.
     */
    void smooth(int passes = 1)
    {
        std::vector<std::vector<int>> ring(pt_.size()), vquads(pt_.size());
        for (int t = 0; t < nquads(); ++t)
            for (int i = 0; i < 4; ++i)
            {
                const int a = q_[t][i], b = q_[t][(i + 1) % 4];
                ring[a].push_back(b); ring[b].push_back(a);
                vquads[a].push_back(t);
            }
        std::vector<char> fixed(pt_.size(), 0);
        for (const auto &s : seg_) fixed[s[0]] = fixed[s[1]] = 1;
        for (int pass = 0; pass < passes; ++pass)
            for (int v = 0; v < npoints(); ++v)
            {
                if (fixed[v] || ring[v].empty()) continue;
                std::vector<int> &r = ring[v];
                std::sort(r.begin(), r.end());
                r.erase(std::unique(r.begin(), r.end()), r.end());
                double sx = 0, sy = 0;
                for (int u : r) { sx += pt_[u].x; sy += pt_[u].y; }
                const Point old = pt_[v], target{sx / r.size(), sy / r.size()};
                for (double f : {1.0, 0.5, 0.25})
                {
                    pt_[v] = Point{old.x + f * (target.x - old.x), old.y + f * (target.y - old.y)};
                    bool ok = true;
                    for (int t : vquads[v]) ok = ok && convex(t);
                    if (ok) break;
                    pt_[v] = old;
                }
            }
        grid_.reset();
    }

    /**
     * @brief All-quad mesh of a triangulation: each triangle (a, b, c) becomes the quads
     *        (a, m_ab, g, m_ca), (b, m_bc, g, m_ab), (c, m_ca, g, m_bc) with the edge midpoints m and
     *        the centroid g.  Every quad is convex.  A segment is split in two with its marker.
     */
    static QuadMesh from_triangles(const Mesh2D &T)
    {
        std::vector<Point> pts(T.points().begin(), T.points().end());
        std::vector<int> pmark(T.point_markers().begin(), T.point_markers().end());
        std::unordered_map<std::int64_t, int> mid;
        auto midpoint = [&](int a, int b) {
            auto it = mid.find(key(a, b));
            if (it != mid.end()) return it->second;
            const Point &A = pts[a], &B = pts[b];
            const int m = static_cast<int>(pts.size());
            pts.push_back(Point{0.5 * (A.x + B.x), 0.5 * (A.y + B.y)});
            const int s = T.segment_index(a, b);
            pmark.push_back(s >= 0 ? T.segment_markers()[s] : 0);
            mid.emplace(key(a, b), m);
            return m;
        };
        std::vector<std::array<int, 4>> quads;
        for (int t = 0; t < T.tri_capacity(); ++t)
        {
            if (!T.alive(t)) continue;
            const auto &v = T.triangle(t);
            const int a = v[0], b = v[1], c = v[2];
            const int mab = midpoint(a, b), mbc = midpoint(b, c), mca = midpoint(c, a);
            const Point &A = pts[a], &B = pts[b], &C = pts[c];
            const int g = static_cast<int>(pts.size());
            pts.push_back(Point{(A.x + B.x + C.x) / 3.0, (A.y + B.y + C.y) / 3.0});
            pmark.push_back(0);
            quads.push_back({a, mab, g, mca});
            quads.push_back({b, mbc, g, mab});
            quads.push_back({c, mca, g, mbc});
        }
        std::vector<std::array<int, 2>> segs;
        std::vector<int> smarks;
        for (int s = 0; s < T.nsegments(); ++s)
        {
            const int a = T.segments()[s][0], b = T.segments()[s][1];
            auto it = mid.find(key(a, b));
            if (it == mid.end()) continue;                      // a segment that is no longer an edge
            segs.push_back({a, it->second}); smarks.push_back(T.segment_markers()[s]);
            segs.push_back({it->second, b}); smarks.push_back(T.segment_markers()[s]);
        }
        return QuadMesh(std::move(pts), std::move(quads), std::move(segs), std::move(smarks), std::move(pmark));
    }

private:
    static std::int64_t key(int a, int b)
    {
        if (a > b) std::swap(a, b);
        return (static_cast<std::int64_t>(a) << 32) | static_cast<std::uint32_t>(b);
    }

    void build_neighbours()
    {
        nbr_.assign(q_.size(), {-1, -1, -1, -1});
        edge_map_.clear();
        std::unordered_map<std::int64_t, std::array<int, 2>> first;     // edge -> (quad, local edge)
        for (int t = 0; t < nquads(); ++t)
            for (int i = 0; i < 4; ++i)
            {
                const std::int64_t k = key(q_[t][i], q_[t][(i + 1) % 4]);
                auto it = first.find(k);
                if (it == first.end()) { first.emplace(k, std::array<int, 2>{t, i}); edge_map_[k] = 1; continue; }
                if (edge_map_[k] >= 2) throw std::invalid_argument("QuadMesh: an edge is shared by more than two quads");
                edge_map_[k] = 2;
                nbr_[t][i] = it->second[0];
                nbr_[it->second[0]][it->second[1]] = t;
            }
    }

    void add_segment(int a, int b, int m)
    {
        if (segment_index(a, b) >= 0) return;
        segmap_.emplace(key(a, b), static_cast<int>(seg_.size()));
        seg_.push_back({a, b});
        smark_.push_back(m);
    }

    struct Grid {
        double x0 = 0, y0 = 0, w = 1, h = 1;
        int nx = 1, ny = 1;
        std::vector<std::vector<int>> cells;
    };
    void build_grid() const
    {
        auto G = std::make_unique<Grid>();
        double xmin = std::numeric_limits<double>::infinity(), ymin = xmin, xmax = -xmin, ymax = -xmin;
        for (const Point &p : pt_) { xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x); ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y); }
        G->x0 = xmin; G->y0 = ymin;
        G->w = std::max(xmax - xmin, 1e-300); G->h = std::max(ymax - ymin, 1e-300);
        const double n = std::max(1, nquads());
        G->nx = std::max(1, static_cast<int>(std::sqrt(n * G->w / G->h)));
        G->ny = std::max(1, static_cast<int>(n / G->nx));
        G->cells.assign(static_cast<std::size_t>(G->nx) * G->ny, {});
        auto ci = [&](double x) { return std::min(G->nx - 1, std::max(0, static_cast<int>((x - G->x0) * G->nx / G->w))); };
        auto cj = [&](double y) { return std::min(G->ny - 1, std::max(0, static_cast<int>((y - G->y0) * G->ny / G->h))); };
        for (int t = 0; t < nquads(); ++t)
        {
            double a = std::numeric_limits<double>::infinity(), b = a, c = -a, d = -a;
            for (int v : q_[t]) { a = std::min(a, pt_[v].x); c = std::max(c, pt_[v].x); b = std::min(b, pt_[v].y); d = std::max(d, pt_[v].y); }
            for (int j = cj(b); j <= cj(d); ++j)
                for (int i = ci(a); i <= ci(c); ++i) G->cells[static_cast<std::size_t>(j) * G->nx + i].push_back(t);
        }
        grid_ = std::move(G);
    }

    std::vector<Point> pt_;
    std::vector<std::array<int, 4>> q_, nbr_;
    std::vector<std::array<int, 2>> seg_;
    std::vector<int> smark_, pmark_;
    std::unordered_map<std::int64_t, int> segmap_, edge_map_;
    mutable int last_ = 0;
    mutable std::shared_ptr<Grid> grid_;
};

} // namespace femd

#endif // FEMD_MESH_QUAD_MESH_HPP
