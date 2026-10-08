//
//  dense.hpp  --  small dense linear algebra for the setup code: LU with partial pivoting
//  (solve, several right-hand sides, inverse), the Legendre Vandermonde matrix, the
//  Gauss-Legendre rule as arrays, and a 2-norm without BLAS.
//
//  Matrices are row-major, n x n.  The LU is the textbook right-looking one with row
//  interchanges (the LAPACK getrf order of operations, unblocked), so it agrees with
//  numpy.linalg.solve / scipy.linalg.lu_factor to rounding.  A zero pivot throws: the
//  matrix is singular.
//
#ifndef FEMD_UTIL_DENSE_HPP
#define FEMD_UTIL_DENSE_HPP

#include "femd/quadrature/gauss_legendre.hpp"
#include "femd/util/omp.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace femd {

class DenseLU {
public:
    DenseLU() = default;
    /// @param A  n x n, row-major
    DenseLU(std::vector<double> A, int n) : n_(n), lu_(std::move(A)), piv_(static_cast<std::size_t>(n))
    {
        if (n < 0 || lu_.size() != static_cast<std::size_t>(n) * n) throw std::invalid_argument("DenseLU: A must be n x n");
        double *a = lu_.data();
        for (int k = 0; k < n; ++k)
        {
            int p = k;
            double best = std::abs(a[static_cast<std::size_t>(k) * n + k]);
            for (int i = k + 1; i < n; ++i)
            {
                const double v = std::abs(a[static_cast<std::size_t>(i) * n + k]);
                if (v > best) { best = v; p = i; }
            }
            piv_[k] = p;
            if (best == 0.0 || !std::isfinite(best)) throw std::runtime_error("DenseLU: the matrix is singular");
            if (p != k)
                for (int j = 0; j < n; ++j) std::swap(a[static_cast<std::size_t>(k) * n + j], a[static_cast<std::size_t>(p) * n + j]);
            const double d = a[static_cast<std::size_t>(k) * n + k];
            for (int i = k + 1; i < n; ++i)
            {
                double *ri = a + static_cast<std::size_t>(i) * n;
                const double l = ri[k] / d;
                ri[k] = l;
                if (l == 0.0) continue;
                const double *rk = a + static_cast<std::size_t>(k) * n;
                for (int j = k + 1; j < n; ++j) ri[j] -= l * rk[j];
            }
        }
    }

    int size() const { return n_; }

    /// @brief x = A^{-1} b in place (one right-hand side).
    void solve(double *b) const
    {
        const double *a = lu_.data();
        for (int k = 0; k < n_; ++k) if (piv_[k] != k) std::swap(b[k], b[piv_[k]]);
        for (int i = 1; i < n_; ++i)
        {
            double s = b[i];
            const double *ri = a + static_cast<std::size_t>(i) * n_;
            for (int j = 0; j < i; ++j) s -= ri[j] * b[j];
            b[i] = s;
        }
        for (int i = n_ - 1; i >= 0; --i)
        {
            double s = b[i];
            const double *ri = a + static_cast<std::size_t>(i) * n_;
            for (int j = i + 1; j < n_; ++j) s -= ri[j] * b[j];
            b[i] = s / ri[i];
        }
    }

    /// @brief X = A^{-1} B for B n x k (row-major), column by column.
    std::vector<double> solve_many(const std::vector<double> &B, int k) const
    {
        if (B.size() != static_cast<std::size_t>(n_) * k) throw std::invalid_argument("DenseLU::solve_many: B must be n x k");
        std::vector<double> X(B.size()), col(static_cast<std::size_t>(n_));
        for (int c = 0; c < k; ++c)
        {
            for (int i = 0; i < n_; ++i) col[i] = B[static_cast<std::size_t>(i) * k + c];
            solve(col.data());
            for (int i = 0; i < n_; ++i) X[static_cast<std::size_t>(i) * k + c] = col[i];
        }
        return X;
    }

    std::vector<double> inverse() const
    {
        std::vector<double> I(static_cast<std::size_t>(n_) * n_, 0.0);
        for (int i = 0; i < n_; ++i) I[static_cast<std::size_t>(i) * n_ + i] = 1.0;
        return solve_many(I, n_);
    }

private:
    int n_ = 0;
    std::vector<double> lu_;
    std::vector<int> piv_;
};

/// @brief V[i][k] = P_k(x_i), k = 0..p (row-major, len(x) x (p+1)), by the three-term recurrence
///        as numpy.polynomial.legendre.legvander computes it.
inline std::vector<double> legendre_vandermonde(const std::vector<double> &x, int p)
{
    if (p < 0) throw std::invalid_argument("legendre_vandermonde: degree must be >= 0");
    const std::size_t m = x.size(), w = static_cast<std::size_t>(p) + 1;
    std::vector<double> V(m * w);
    for (std::size_t i = 0; i < m; ++i)
    {
        double *v = &V[i * w];
        v[0] = 1.0;
        if (p >= 1) v[1] = x[i];
        for (int k = 2; k <= p; ++k) v[k] = (v[k - 1] * x[i] * (2 * k - 1) - v[k - 2] * (k - 1)) / k;
    }
    return V;
}

/// @brief sqrt(sum x_i^2 + y_i^2) (y may be null), the ordered dot product of omp.hpp: the same
///        bits for any thread count.
inline double norm2(const double *x, const double *y, std::size_t n)
{
    double s = ddot(x, x, n);
    if (y != nullptr) s += ddot(y, y, n);
    return std::sqrt(s);
}

} // namespace femd

#endif // FEMD_UTIL_DENSE_HPP
