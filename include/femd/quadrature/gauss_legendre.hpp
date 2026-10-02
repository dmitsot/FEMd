//
//  gauss_legendre.hpp  --  Gauss-Legendre quadrature on the reference [-1,1].
//
//  Nodes/weights for 1..5 points are hand-tabulated (exact for polynomials up
//  to degree 2n-1), taken from old_poseidon.  A degree-p Lagrange mass matrix
//  needs n = p+1 points, so 1..5 already covers degree <= 4; for n >= 6 the
//  nodes/weights are computed on the fly by Newton iteration on the Legendre
//  polynomial (gauss_legendre_general), so any number of points is available.
//
//  (A GLL variant for lumped mass -- useful for explicit time stepping of wave
//   equations -- can slot in behind the same interface later.)
//
#ifndef FEMD_QUADRATURE_GAUSS_LEGENDRE_HPP
#define FEMD_QUADRATURE_GAUSS_LEGENDRE_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

namespace femd {

class GaussLegendre {
public:
    /**
     * @brief Build a Gauss-Legendre rule on the reference interval [-1,1].
     * @param npts  number of points, any n >= 1 (exact for polynomials up to
     *              degree 2*npts-1).  1..5 are tabulated; n >= 6 is computed.
     * @throws std::out_of_range if npts < 1.
     */
    explicit GaussLegendre(int npts) : n_(npts), x_(npts), w_(npts)
    {
        switch (npts)
        {
        case 1:
            x_ = {0.0};
            w_ = {2.0};
            break;
        case 2:
            x_ = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
            w_ = {1.0, 1.0};
            break;
        case 3:
            x_ = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
            w_ = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
            break;
        case 4:
        {
            double a = std::sqrt((3.0 + 2.0 * std::sqrt(6.0 / 5.0)) / 7.0);
            double b = std::sqrt((3.0 - 2.0 * std::sqrt(6.0 / 5.0)) / 7.0);
            x_ = {-a, -b, b, a};
            double wa = (18.0 - std::sqrt(30.0)) / 36.0;
            double wb = (18.0 + std::sqrt(30.0)) / 36.0;
            w_ = {wa, wb, wb, wa};
            break;
        }
        case 5:
        {
            double a = std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
            double b = std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
            x_ = {-a, -b, 0.0, b, a};
            double wa = (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
            double wb = (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
            w_ = {wa, wb, 128.0 / 225.0, wb, wa};
            break;
        }
        default:
            if (npts < 1) throw std::out_of_range("GaussLegendre: npts must be >= 1");
            gauss_legendre_general(npts, x_, w_);   // 6+ points: computed on the fly
            break;
        }
    }

    /// @brief Number of quadrature points. @return n.
    int size() const { return n_; }
    /// @brief Quadrature node q on [-1,1]. @param q index in [0,size()). @return the node coordinate.
    double node(int q)   const { return x_[q]; }
    /// @brief Quadrature weight q. @param q index in [0,size()). @return the weight.
    double weight(int q) const { return w_[q]; }

    /**
     * @brief A rule that integrates polynomials of a given degree exactly.
     * @param degree  polynomial degree to integrate exactly.
     * @return a GaussLegendre rule with ceil((degree+1)/2) points (>= 1).
     */
    static GaussLegendre forDegree(int degree)
    {
        int n = degree / 2 + 1;
        if (n < 1) n = 1;
        return GaussLegendre(n);
    }

private:
    /**
     * @brief Compute n-point Gauss-Legendre nodes/weights by Newton iteration.
     *        Used for n >= 6, where no hand-tabulated values are stored.  The
     *        nodes are the roots of the Legendre polynomial P_n (found by Newton
     *        from Chebyshev-spaced guesses) and the weights are
     *        w_i = 2 / ((1 - x_i^2) P_n'(x_i)^2).
     * @param n  number of points (>= 1).
     * @param x  [out] nodes on [-1,1], filled with n values.
     * @param w  [out] weights, filled with n values.
     */
    static void gauss_legendre_general(int n, std::vector<double> &x, std::vector<double> &w)
    {
        const double PI = 3.14159265358979323846;
        for (int i = 0; i < n; ++i)
        {
            double xi = std::cos(PI * (i + 0.75) / (n + 0.5));   // initial guess
            for (int it = 0; it < 100; ++it)                     // Newton on P_n
            {
                double p0 = 1.0, p1 = xi;                        // P_0, P_1
                for (int k = 2; k <= n; ++k)                     // 3-term recurrence
                {
                    double p2 = ((2.0 * k - 1.0) * xi * p1 - (k - 1.0) * p0) / k;
                    p0 = p1; p1 = p2;
                }
                double dp = n * (xi * p1 - p0) / (xi * xi - 1.0);  // P_n'(xi)
                double dx = -p1 / dp;
                xi += dx;
                if (std::abs(dx) <= 1e-15) break;
            }
            // re-evaluate P_n' at the converged node for the weight
            double p0 = 1.0, p1 = xi;
            for (int k = 2; k <= n; ++k)
            {
                double p2 = ((2.0 * k - 1.0) * xi * p1 - (k - 1.0) * p0) / k;
                p0 = p1; p1 = p2;
            }
            double dp = n * (xi * p1 - p0) / (xi * xi - 1.0);
            x[i] = xi;
            w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
        }
    }

    int n_;
    std::vector<double> x_, w_;
};

} // namespace femd

#endif // FEMD_QUADRATURE_GAUSS_LEGENDRE_HPP
