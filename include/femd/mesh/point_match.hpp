//
//  point_match.hpp  --  nearest-neighbour searches for the setup code: the nearest of a set of
//  points to each query point (pairing the two sides of a periodic mesh, as scipy.spatial's
//  cKDTree did), and the nearest segment of a closed polygon to each point (which input edge a
//  boundary segment of a triangulation lies on).
//
#ifndef FEMD_MESH_POINT_MATCH_HPP
#define FEMD_MESH_POINT_MATCH_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace femd {

/**
 * @brief For each query q (m x 2, row-major) the index of the nearest point of P (n x 2) and the
 *        distance.  A uniform grid of about two points per cell, searched ring by ring until no
 *        closer point can remain; the first index among exact ties.
 */
inline void nearest_points(const std::vector<double> &P, const std::vector<double> &Q, std::vector<int> &idx,
                           std::vector<double> &dist)
{
    if (P.size() % 2 || Q.size() % 2) throw std::invalid_argument("nearest_points: (n, 2) arrays");
    const std::size_t n = P.size() / 2, m = Q.size() / 2;
    idx.assign(m, -1);
    dist.assign(m, std::numeric_limits<double>::infinity());
    if (n == 0) return;
    double x0 = P[0], x1 = P[0], y0 = P[1], y1 = P[1];
    for (std::size_t i = 0; i < n; ++i)
    {
        x0 = std::min(x0, P[2 * i]); x1 = std::max(x1, P[2 * i]);
        y0 = std::min(y0, P[2 * i + 1]); y1 = std::max(y1, P[2 * i + 1]);
    }
    const int g = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(n) / 2.0)));
    const double wx = std::max(x1 - x0, 1e-300) / g, wy = std::max(y1 - y0, 1e-300) / g;
    auto cell = [&](double v, double lo, double w) {
        int c = static_cast<int>(std::floor((v - lo) / w));
        return std::min(std::max(c, 0), g - 1);
    };
    std::vector<int> start(static_cast<std::size_t>(g) * g + 1, 0), items(n);
    for (std::size_t i = 0; i < n; ++i) ++start[static_cast<std::size_t>(cell(P[2 * i + 1], y0, wy)) * g + cell(P[2 * i], x0, wx) + 1];
    for (std::size_t c = 0; c + 1 < start.size(); ++c) start[c + 1] += start[c];
    {
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (std::size_t i = 0; i < n; ++i)
            items[static_cast<std::size_t>(fill[static_cast<std::size_t>(cell(P[2 * i + 1], y0, wy)) * g + cell(P[2 * i], x0, wx)]++)] = static_cast<int>(i);
    }
    const double w = std::min(wx, wy);
    for (std::size_t q = 0; q < m; ++q)
    {
        const double qx = Q[2 * q], qy = Q[2 * q + 1];
        const int cx = cell(qx, x0, wx), cy = cell(qy, y0, wy);
        double best = std::numeric_limits<double>::infinity();
        int bi = -1;
        for (int r = 0; r <= g; ++r)
        {
            for (int j = cy - r; j <= cy + r; ++j)
                for (int i = cx - r; i <= cx + r; ++i)
                {
                    if (i < 0 || j < 0 || i >= g || j >= g) continue;
                    if (std::max(std::abs(i - cx), std::abs(j - cy)) != r) continue;      // the ring only
                    const std::size_t c = static_cast<std::size_t>(j) * g + i;
                    for (int t = start[c]; t < start[c + 1]; ++t)
                    {
                        const int p = items[static_cast<std::size_t>(t)];
                        const double d = std::hypot(P[2 * static_cast<std::size_t>(p)] - qx, P[2 * static_cast<std::size_t>(p) + 1] - qy);
                        if (d < best || (d == best && p < bi)) { best = d; bi = p; }
                    }
                }
            if (bi >= 0 && best <= r * w) break;          // every cell beyond ring r is at least r w away
        }
        idx[q] = bi;
        dist[q] = best;
    }
}

/**
 * @brief For each point X[k] (k x 2) the segment A_e B_e of the closed polygon `ring` (n x 2,
 *        B_e = A_{e+1}) nearest to it, and the distance: the foot s = clip((X - A).d / |d|^2, 0, 1)
 *        on d = B - A, the first segment among ties.
 */
inline void nearest_segments(const std::vector<double> &X, const std::vector<double> &ring, std::vector<int> &edge,
                             std::vector<double> &dist)
{
    if (X.size() % 2 || ring.size() % 2) throw std::invalid_argument("nearest_segments: (n, 2) arrays");
    const std::size_t k = X.size() / 2, n = ring.size() / 2;
    edge.assign(k, -1);
    dist.assign(k, std::numeric_limits<double>::infinity());
    for (std::size_t p = 0; p < k; ++p)
        for (std::size_t e = 0; e < n; ++e)
        {
            const double ax = ring[2 * e], ay = ring[2 * e + 1];
            const std::size_t f = (e + 1) % n;
            const double dx = ring[2 * f] - ax, dy = ring[2 * f + 1] - ay;
            const double L2 = std::max(dx * dx + dy * dy, std::numeric_limits<double>::min());
            const double s = std::min(std::max(((X[2 * p] - ax) * dx + (X[2 * p + 1] - ay) * dy) / L2, 0.0), 1.0);
            const double d = std::hypot(X[2 * p] - (ax + s * dx), X[2 * p + 1] - (ay + s * dy));
            if (d < dist[p]) { dist[p] = d; edge[p] = static_cast<int>(e); }
        }
}

} // namespace femd

#endif // FEMD_MESH_POINT_MATCH_HPP
