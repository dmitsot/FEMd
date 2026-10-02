//
//  point_locator.hpp  --  which triangle contains a point, for many points.
//
//      PointLocator L(M);
//      std::vector<int> t = L.locate(points);     // -1 outside the mesh or in a hole
//
//  Each query first walks from the previous answer (Mesh2D::walk), which is O(1)
//  for queries that follow one another in space and O(sqrt(n)) for scattered ones.
//  A walk that runs into the boundary is inconclusive: the point may be outside,
//  in a hole, or across a hole or a reentrant corner.  Mesh2D::locate settles that
//  with an O(n) scan, which is fine for the refinement loop's occasional miss but
//  not for a batch where many points are outside.  Here it is settled by a bucket
//  grid over the bounding box, about one cell per triangle, each cell listing the
//  triangles whose bounding box meets it.  The grid is built on the first miss (or
//  up front with prepare(), for a batch of scattered points), in O(n), and from
//  then on answers each query in O(1) on average, rejecting a point outside the
//  bounding box of the mesh at once.  Keep one locator per mesh and reuse it.
//
//  Exactness.  Containment is decided by the exact orient2d, with a point on an
//  edge or a vertex counted inside.  A triangle is listed in every cell its
//  bounding box meets, and the cell of a coordinate is floor((x - x0) * nx / w),
//  which is monotone in x under rounding, so the cell of a point inside a triangle
//  is always among that triangle's cells.  The grid can therefore never miss.
//
//  The locator reads the mesh, which must not change while the locator is in use.
//
#ifndef FEMD_MESH_POINT_LOCATOR_HPP
#define FEMD_MESH_POINT_LOCATOR_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/predicates.hpp"
#include "femd/util/omp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace femd {

class PointLocator {
public:
    explicit PointLocator(const Mesh2D &M) : M_(M) {}

    /**
     * @brief Triangle containing p, or -1.
     *
     * Once the grid exists it answers directly, rejecting a point outside the mesh's
     * bounding box at once.  Before that the walk starts at hint (by default the previous
     * answer), and an inconclusive walk builds the grid.
     */
    int locate(const Point &p, int hint = -2)
    {
        int t;
        if (built_) t = grid_locate(p);
        else
        {
            t = M_.walk(p, hint == -2 ? last_ : hint);
            if (t < 0) t = grid_locate(p);
        }
        if (t >= 0) last_ = t;
        return t;
    }

    /**
     * @brief Build the grid now.  Worth it for a batch of scattered queries: each is then O(1)
     *        instead of an O(sqrt(n)) walk.  A batch that moves smoothly through the mesh is
     *        about as fast either way.
     */
    void prepare() { if (!built_) build(); }

    /// @brief One answer per point, each walk starting from the previous answer.  A batch above
    ///        FEMD_OMP_THRESHOLD builds the grid and is answered from it alone, in parallel.
    std::vector<int> locate(const std::vector<Point> &pts)
    {
        std::vector<int> out(pts.size());
        const long n = static_cast<long>(pts.size());
        if (n > FEMD_OMP_THRESHOLD)
        {
            prepare();
            FEMD_OMP_FOR_IF(true)
            for (long i = 0; i < n; ++i) out[i] = find(pts[i]);
            return out;
        }
        for (long i = 0; i < n; ++i) out[i] = locate(pts[i]);
        return out;
    }

    /// @brief Containment through the grid alone, which prepare() must have built.  Const and
    ///        stateless, so several threads may call it at once.
    int find(const Point &p) const
    {
        if (nx_ == 0) return -1;
        if (!(p.x >= x0_ && p.x <= x1_ && p.y >= y0_ && p.y <= y1_)) return -1;
        int c = cell(ix(p.x), iy(p.y));
        for (int k = start_[c]; k < start_[c + 1]; ++k)
            if (inside(tris_[k], p)) return tris_[k];
        return -1;
    }

    /// @brief Containment through the bucket grid alone, building it on first use.
    int grid_locate(const Point &p)
    {
        if (!built_) build();
        return find(p);
    }

    bool grid_built() const { return built_; }

private:
    int ix(double x) const { return std::min(nx_ - 1, std::max(0, static_cast<int>(std::floor((x - x0_) * idx_)))); }
    int iy(double y) const { return std::min(ny_ - 1, std::max(0, static_cast<int>(std::floor((y - y0_) * idy_)))); }
    int cell(int i, int j) const { return j * nx_ + i; }

    bool inside(int t, const Point &p) const
    {
        const auto &T = M_.triangle(t);
        for (int i = 0; i < 3; ++i)
        {
            const Point &u = M_.point(T[(i + 1) % 3]), &v = M_.point(T[(i + 2) % 3]);
            if (pred::orient2d(u.x, u.y, v.x, v.y, p.x, p.y) < 0) return false;
        }
        return true;
    }

    void bbox(int t, double &a, double &b, double &c, double &d) const
    {
        const auto &T = M_.triangle(t);
        a = c = 1e300; b = d = -1e300;
        for (int k = 0; k < 3; ++k)
        {
            const Point &q = M_.point(T[k]);
            a = std::min(a, q.x); b = std::max(b, q.x); c = std::min(c, q.y); d = std::max(d, q.y);
        }
    }

    void box()
    {
        have_box_ = true;
        x0_ = y0_ = 1e300; x1_ = y1_ = -1e300;
        for (int t = 0; t < M_.tri_capacity(); ++t)
        {
            if (!M_.alive(t)) continue;
            double a, b, c, d;
            bbox(t, a, b, c, d);
            x0_ = std::min(x0_, a); x1_ = std::max(x1_, b); y0_ = std::min(y0_, c); y1_ = std::max(y1_, d);
        }
    }

    void build()
    {
        built_ = true;
        const int n = M_.ntriangles();
        if (n == 0) return;
        if (!have_box_) box();
        const double w = std::max(x1_ - x0_, 1e-300), h = std::max(y1_ - y0_, 1e-300);
        nx_ = std::max(1, std::min(4096, static_cast<int>(std::round(std::sqrt(n * w / h)))));
        ny_ = std::max(1, std::min(4096, static_cast<int>(std::round(static_cast<double>(n) / nx_))));
        idx_ = nx_ / w; idy_ = ny_ / h;

        start_.assign(static_cast<std::size_t>(nx_) * ny_ + 1, 0);
        for (int pass = 0; pass < 2; ++pass)
        {
            std::vector<int> fill;
            if (pass == 1)
            {
                for (std::size_t c = 1; c < start_.size(); ++c) start_[c] += start_[c - 1];
                tris_.assign(static_cast<std::size_t>(start_.back()), -1);
                fill.assign(start_.begin(), start_.end() - 1);
            }
            for (int t = 0; t < M_.tri_capacity(); ++t)
            {
                if (!M_.alive(t)) continue;
                double a, b, c, d;
                bbox(t, a, b, c, d);
                const int i0 = ix(a), i1 = ix(b), j0 = iy(c), j1 = iy(d);
                for (int j = j0; j <= j1; ++j)
                    for (int i = i0; i <= i1; ++i)
                    {
                        if (pass == 0) ++start_[cell(i, j) + 1];
                        else tris_[fill[cell(i, j)]++] = t;
                    }
            }
        }
    }

    const Mesh2D &M_;
    int last_ = -1;
    bool built_ = false, have_box_ = false;
    int nx_ = 0, ny_ = 0;
    double x0_ = 0, y0_ = 0, x1_ = 0, y1_ = 0, idx_ = 0, idy_ = 0;
    std::vector<int> start_, tris_;
};

} // namespace femd

#endif // FEMD_MESH_POINT_LOCATOR_HPP
