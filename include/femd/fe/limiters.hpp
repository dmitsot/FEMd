//
//  limiters.hpp  --  slope limiters for DG coefficient vectors, the loops behind
//  fd.TVBLimiter, fd.SlopeLimiter (1D) and fd.VertexLimiter (2D).  Python builds the
//  reference tables (nodal <-> modal maps, cell means, the P1 projection) once; the limiting
//  of a coefficient vector, done after every stage of an SSP Runge-Kutta step, is here.
//
//  Limiter1D (Cockburn-Shu TVB minmod, and the finite volume slopes of Dutykh, Katsaounis and
//  Mitsotakis 2011): on each element the coefficients are taken to the Legendre scale, the end
//  values are compared with the neighbours' means through the TVB-modified minmod, and a
//  troubled element keeps its mean, gets a limited slope (the minmod of its own, or a finite
//  volume slope of the means with TVD2 or UNO2 reconstruction) and loses its higher modes.
//  VertexLimiter2D (Kuzmin 2010): the P1 part of each triangle is scaled about the cell mean
//  so that its vertex values stay between the smallest and the largest mean around the vertex.
//
//  The elements are independent once the means are known, so both run in parallel over the
//  elements with OpenMP, with the same result for any number of threads.
//
#ifndef FEMD_FE_LIMITERS_HPP
#define FEMD_FE_LIMITERS_HPP

#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

inline double minmod2(double a, double b)
{
    if (a > 0.0 && b > 0.0) return std::min(a, b);
    if (a < 0.0 && b < 0.0) return std::max(a, b);
    return 0.0;
}
inline double minmod3(double a, double b, double c) { return minmod2(a, minmod2(b, c)); }

/// phi(a/b) b for the slope limiters: "minmod", "mc", "vanleer", "vanalbada" (ids 0..3).
inline double limited_slope_of(int limiter, double a, double b)
{
    switch (limiter)
    {
    case 0: return minmod2(a, b);
    case 1: return minmod3(2.0 * a, 0.5 * (a + b), 2.0 * b);
    case 2: {                                        // van Leer: zero where the slopes differ in sign (stated, so that no fused multiply-add leaves a residue)
        if (!(a * b > 0.0)) return 0.0;
        const double den = std::abs(a) + std::abs(b);
        return (a * std::abs(b) + std::abs(a) * b) / den; }
    default: { const double den = a * a + b * b; return den > 0.0 ? a * b * (a + b) / den : 0.0; }
    }
}

class Limiter1D {
public:
    /**
     * @param p, ne        degree and number of elements
     * @param periodic     whether the space is periodic
     * @param h            element widths (ne)
     * @param to_modal     (p+1)^2 row-major matrix taking the element's coefficients to the Legendre scale (empty for a Legendre basis)
     * @param to_nodal     its inverse (empty for a Legendre basis)
     * @param M            the TVB constant
     * @param slope        -1 for the TVB limiter (the slope minmod(c1, dp, dm)), else the finite volume limiter 0..3
     * @param uno2         UNO2 reconstruction instead of TVD2 (slope limiters)
     * @param all          limit every element instead of the troubled ones (slope limiters)
     */
    Limiter1D(int p, int ne, bool periodic, const std::vector<double> &h, const std::vector<double> &to_modal,
              const std::vector<double> &to_nodal, double M, int slope, bool uno2, bool all)
        : p_(p), ne_(ne), periodic_(periodic), h_(h), tm_(to_modal), tn_(to_nodal), M_(M), slope_(slope), uno2_(uno2), all_(all)
    {
        if (static_cast<int>(h.size()) != ne) throw std::invalid_argument("Limiter1D: h must have one entry per element");
        const std::size_t q = static_cast<std::size_t>(p + 1) * static_cast<std::size_t>(p + 1);
        if (!tm_.empty() && (tm_.size() != q || tn_.size() != q)) throw std::invalid_argument("Limiter1D: the modal maps must be (p+1) x (p+1)");
        // centre distances to the neighbours (one-sided at the ends)
        dp_.resize(static_cast<std::size_t>(ne)); dm_.resize(static_cast<std::size_t>(ne));
        for (int j = 0; j < ne; ++j)
        {
            const int jp = j + 1 < ne ? j + 1 : 0, jm = j > 0 ? j - 1 : ne - 1;
            dp_[j] = 0.5 * (h[j] + h[jp]);
            dm_[j] = 0.5 * (h[j] + h[jm]);
        }
        if (!periodic && ne > 1) { dp_[ne - 1] = dm_[ne - 1]; dm_[0] = dp_[0]; }
        mean_.resize(static_cast<std::size_t>(ne));
    }

    int dim() const { return ne_ * (p_ + 1); }

    /// out = the limited copy of c; returns the number of troubled elements.
    int limit(const double *c, double *out) const
    {
        const int p = p_, ne = ne_, q = p + 1;
        if (p == 0 || ne < 2)
        {
            std::copy(c, c + static_cast<std::size_t>(ne) * q, out);
            return 0;
        }
        const bool par = ne * q > FEMD_OMP_THRESHOLD;
        (void)par;
        double *mean = mean_.data();
        // pass 1: every element to the Legendre scale (into out), its mean
        FEMD_OMP_FOR_IF(par)
        for (int j = 0; j < ne; ++j)
        {
            const double *cj = c + static_cast<std::size_t>(j) * q;
            double *oj = out + static_cast<std::size_t>(j) * q;
            to_modal(cj, oj);
            mean[j] = oj[0];
        }
        // pass 2: the test and the limiting, element by element
        int troubled = 0;
        FEMD_OMP(parallel for reduction(+ : troubled) if(par))
        for (int j = 0; j < ne; ++j)
        {
            double *oj = out + static_cast<std::size_t>(j) * q;
            const int jp = j + 1 < ne ? j + 1 : 0, jm = j > 0 ? j - 1 : ne - 1;
            double Dp = mean[jp] - mean[j], Dm = mean[j] - mean[jm];
            if (!periodic_) { if (j == ne - 1) Dp = Dm; if (j == 0) Dm = Dp; }
            bool bad = all_ && slope_ >= 0;
            if (!bad)
            {
                double right = 0.0, left = 0.0;
                for (int l = 0; l <= p; ++l) { right += oj[l]; left += (l % 2 == 0 ? oj[l] : -oj[l]); }
                right -= mean[j]; left = mean[j] - left;
                const double thr = M_ * h_[j] * h_[j];
                const double tr = std::abs(right) <= thr ? right : minmod3(right, Dp, Dm);
                const double tl = std::abs(left) <= thr ? left : minmod3(left, Dp, Dm);
                bad = tr != right || tl != left;
            }
            if (bad)
            {
                ++troubled;
                if (slope_ < 0) oj[1] = minmod3(oj[1], Dp, Dm);
                else oj[1] = 0.5 * h_[j] * fv_slope(j, Dp, Dm, mean);
                for (int l = 2; l <= p; ++l) oj[l] = 0.0;
            }
            to_nodal_inplace(oj);
        }
        return troubled;
    }

private:
    int p_, ne_;
    bool periodic_;
    std::vector<double> h_, tm_, tn_;
    double M_;
    int slope_;
    bool uno2_, all_;
    std::vector<double> dp_, dm_;
    mutable std::vector<double> mean_;

    void to_modal(const double *cj, double *oj) const
    {
        const int q = p_ + 1;
        if (tm_.empty()) { std::copy(cj, cj + q, oj); return; }
        for (int i = 0; i < q; ++i)
        {
            double s = 0.0;
            for (int l = 0; l < q; ++l) s += tm_[static_cast<std::size_t>(i) * q + l] * cj[l];
            oj[i] = s;
        }
    }
    void to_nodal_inplace(double *oj) const
    {
        const int q = p_ + 1;
        if (tn_.empty()) return;
        double tmp[64];
        std::vector<double> big;
        double *t = q <= 64 ? tmp : (big.resize(static_cast<std::size_t>(q)), big.data());
        for (int i = 0; i < q; ++i)
        {
            double s = 0.0;
            for (int l = 0; l < q; ++l) s += tn_[static_cast<std::size_t>(i) * q + l] * oj[l];
            t[i] = s;
        }
        std::copy(t, t + q, oj);
    }
    // the finite volume slope of element j from the neighbouring means
    double fv_slope(int j, double Dp, double Dm, const double *mean) const
    {
        const int ne = ne_;
        const double sp = Dp / dp_[j], sm = Dm / dm_[j];
        if (!uno2_) return limited_slope_of(slope_, sm, sp);
        auto second = [&](int i) {                      // 2 (sp_i - sm_i) / (dm_i + dp_i) of element i
            const int ip = i + 1 < ne ? i + 1 : 0, im = i > 0 ? i - 1 : ne - 1;
            double dpi = mean[ip] - mean[i], dmi = mean[i] - mean[im];
            if (!periodic_) { if (i == ne - 1) dpi = dmi; if (i == 0) dmi = dpi; }
            return 2.0 * (dpi / dp_[i] - dmi / dm_[i]) / (dm_[i] + dp_[i]);
        };
        bool interior = periodic_ || (ne >= 5 && j >= 2 && j <= ne - 3);
        if (!interior) return limited_slope_of(slope_, sm, sp);          // TVD2 in the two cells next to each end
        const int jp = j + 1 < ne ? j + 1 : 0, jm = j > 0 ? j - 1 : ne - 1;
        const double D0 = second(j), Dpp = second(jp), Dmm = second(jm);
        const double Splus = sp - 0.5 * dp_[j] * minmod2(D0, Dpp);
        const double Sminus = sm + 0.5 * dm_[j] * minmod2(Dmm, D0);
        return limited_slope_of(slope_, Sminus, Splus);
    }
};

class VertexLimiter2D {
public:
    /**
     * @param k, nc, n   degree, cells, local size
     * @param mean       (n) weights giving the cell mean from the local coefficients
     * @param Pi         (3 x n) row-major: vertex values of the L2 projection onto P1
     * @param E          (n x 3) row-major: local coefficients of the P1 function with given vertex values
     * @param rv         (nc x 3) vertex classes of each cell
     * @param nv         number of vertex classes
     */
    VertexLimiter2D(int k, int nc, int n, const std::vector<double> &mean, const std::vector<double> &Pi, const std::vector<double> &E,
                    const std::vector<int> &rv, int nv)
        : k_(k), nc_(nc), n_(n), nv_(nv), mean_(mean), Pi_(Pi), E_(E), rv_(rv)
    {
        if (static_cast<int>(mean.size()) != n || static_cast<int>(Pi.size()) != 3 * n || static_cast<int>(E.size()) != 3 * n || static_cast<int>(rv.size()) != 3 * nc)
            throw std::invalid_argument("VertexLimiter2D: table sizes do not match");
        ubar_.resize(static_cast<std::size_t>(nc)); Uv_.resize(static_cast<std::size_t>(3) * nc);
        umax_.resize(static_cast<std::size_t>(nv)); umin_.resize(static_cast<std::size_t>(nv));
        alpha_.resize(static_cast<std::size_t>(nc));
    }

    int dim() const { return nc_ * n_; }
    const std::vector<double> &alpha() const { return alpha_; }

    /// out = the limited copy of c; returns the number of troubled cells.
    int limit(const double *c, double *out) const
    {
        const int nc = nc_, n = n_;
        if (k_ == 0) { std::copy(c, c + static_cast<std::size_t>(nc) * n, out); std::fill(alpha_.begin(), alpha_.end(), 1.0); return 0; }
        const bool par = nc * n > FEMD_OMP_THRESHOLD;
        (void)par;
        double *ubar = ubar_.data(), *Uv = Uv_.data();
        double scale_u = 0.0, scale_d = 0.0;
        FEMD_OMP(parallel for reduction(max : scale_u, scale_d) if(par))
        for (int e = 0; e < nc; ++e)
        {
            const double *ce = c + static_cast<std::size_t>(e) * n;
            double m = 0.0;
            for (int i = 0; i < n; ++i) m += ce[i] * mean_[i];
            ubar[e] = m;
            for (int v = 0; v < 3; ++v)
            {
                double s = 0.0;
                for (int i = 0; i < n; ++i) s += Pi_[static_cast<std::size_t>(v) * n + i] * ce[i];
                Uv[3 * e + v] = s;
                scale_d = std::max(scale_d, std::abs(s - m));
            }
            scale_u = std::max(scale_u, std::abs(m));
        }
        std::fill(umax_.begin(), umax_.end(), -std::numeric_limits<double>::infinity());
        std::fill(umin_.begin(), umin_.end(), std::numeric_limits<double>::infinity());
        for (int e = 0; e < nc; ++e)
            for (int v = 0; v < 3; ++v)
            {
                const int r = rv_[3 * e + v];
                umax_[r] = std::max(umax_[r], ubar[e]);
                umin_[r] = std::min(umin_[r], ubar[e]);
            }
        const double tol = 1e-13 * std::max(std::max(scale_u, scale_d), 1e-300);
        int troubled = 0;
        FEMD_OMP(parallel for reduction(+ : troubled) if(par))
        for (int e = 0; e < nc; ++e)
        {
            double a = 1.0;
            for (int v = 0; v < 3; ++v)
            {
                const double d = Uv[3 * e + v] - ubar[e];
                const int r = rv_[3 * e + v];
                if (d > tol) a = std::min(a, (umax_[r] - ubar[e]) / d);
                else if (d < -tol) a = std::min(a, (umin_[r] - ubar[e]) / d);
            }
            a = std::min(1.0, std::max(0.0, a));
            alpha_[e] = a;
            const double *ce = c + static_cast<std::size_t>(e) * n;
            double *oe = out + static_cast<std::size_t>(e) * n;
            if (a < 1.0 - 1e-12)
            {
                ++troubled;
                for (int i = 0; i < n; ++i)
                {
                    double s = 0.0;
                    for (int v = 0; v < 3; ++v) s += (Uv[3 * e + v] - ubar[e]) * E_[static_cast<std::size_t>(i) * 3 + v];
                    oe[i] = ubar[e] + a * s;
                }
            }
            else std::copy(ce, ce + n, oe);
        }
        return troubled;
    }

private:
    int k_, nc_, n_, nv_;
    std::vector<double> mean_, Pi_, E_;
    std::vector<int> rv_;
    mutable std::vector<double> ubar_, Uv_, umax_, umin_, alpha_;
};

} // namespace femd

#endif // FEMD_FE_LIMITERS_HPP
