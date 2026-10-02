//
//  smw.hpp  --  Sherman-Morrison-Woodbury splitting iteration.
//
//  For A = M - N (N = U V^T low rank / sparse) with rho(M^{-1} N) < 1, solve
//  A x = b by
//      SMW :  x_{k+1} = M^{-1} ( N x_k + b )
//      eSMW:  x_{k+1} = (1-w) x_k + w M^{-1} ( N x_k + b )         (relaxation w)
//  (Mitsotakis, "On iterative methods based on Sherman-Morrison-Woodbury
//   splitting", eqs. for the new method and its extrapolated variant.)
//
//  The iteration is generic: it needs only a callable that applies M^{-1}
//  (e.g. CirculantMatrix::solve via FFT, or any Poseidon direct solver) and a
//  callable that applies N (sparse matvec).  This decouples the iteration from
//  the FFT backend and lets it reuse the existing band/cyclic solvers.
//
#ifndef FEMD_ITERATIVE_SMW_HPP
#define FEMD_ITERATIVE_SMW_HPP

#include "femd/util/omp.hpp"
#include <vector>
#include <cmath>
#include <cstddef>

namespace femd {

using Vec = std::vector<double>;

/// @brief Euclidean norm of a vector. @param v the vector. @return ||v||_2.
inline double l2norm(const Vec &v)
{
    double s = 0.0;
    const int n = static_cast<int>(v.size());
    FEMD_OMP_FOR_REDUCE_IF(+, s, n > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

/// @brief Euclidean norm of the difference. @param a,b the vectors. @return ||a-b||_2.
inline double l2diff(const Vec &a, const Vec &b)
{
    double s = 0.0;
    const int n = static_cast<int>(a.size());
    FEMD_OMP_FOR_REDUCE_IF(+, s, n > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < n; ++i) { double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

/// @brief Minimal sparse operator in coordinate (triplet) form; apply() computes y = S x.
struct Sparse {
    int rows = 0, cols = 0;
    std::vector<int> ii, jj;
    std::vector<double> vv;

    Sparse() = default;
    Sparse(int r, int c) : rows(r), cols(c) {}

    /// @brief Append a triplet. @param i row (0-based). @param j col (0-based). @param v value.
    void add(int i, int j, double v) { ii.push_back(i); jj.push_back(j); vv.push_back(v); }

    /// @brief Sparse matrix-vector product. @param x input vector (length cols). @return y = S x (length rows).
    Vec apply(const Vec &x) const
    {
        Vec y(rows, 0.0);
        for (std::size_t k = 0; k < vv.size(); ++k) y[ii[k]] += vv[k] * x[jj[k]];
        return y;
    }
};

/// @brief Result of smw_solve: solution, iterations, final step residual, convergence flag.
struct SMWResult {
    Vec x;
    int iters = 0;
    double residual = 0.0;   // ||x_k - x_{k-1}||
    bool converged = false;
};

/**
 * @brief Generic (extrapolated) SMW iteration for A x = b with A = M - N.
 *
 * Iterates x_{k+1} = M^{-1}(N x_k + b); omega != 1 gives the extrapolated eSMW.
 * @param Minv   callable applying M^{-1} (e.g. an FFT circulant solve or any direct solver).
 * @param N      callable applying N (e.g. a Sparse matvec).
 * @param b      right-hand side.
 * @param omega  relaxation parameter (1 = plain SMW).
 * @param tol    stopping tolerance on the step norm ||x_{k+1}-x_k||.
 * @param maxit  maximum iterations.
 * @param x0     optional initial guess (default zero).
 * @return SMWResult holding the solution, iteration count, residual and convergence flag.
 */
template <class ApplyMinv, class ApplyN>
SMWResult smw_solve(ApplyMinv Minv, ApplyN N, const Vec &b,
                    double omega = 1.0, double tol = 1e-10, int maxit = 100000,
                    const Vec *x0 = nullptr)
{
    int n = static_cast<int>(b.size());
    SMWResult R;
    R.x.assign(n, 0.0);
    if (x0) R.x = *x0;

    for (int k = 1; k <= maxit; ++k)
    {
        Vec t = N(R.x);                          // t = N x
        FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n; ++i) t[i] += b[i]; // t = N x + b
        Vec xnew = Minv(t);                       // xnew = M^{-1}(N x + b)

        if (omega != 1.0)
        {
            FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
            for (int i = 0; i < n; ++i) xnew[i] = (1.0 - omega) * R.x[i] + omega * xnew[i];
        }

        double err = l2diff(xnew, R.x);
        R.x.swap(xnew);
        R.iters = k;
        R.residual = err;
        if (err < tol) { R.converged = true; break; }
    }
    return R;
}

} // namespace femd

#endif // FEMD_ITERATIVE_SMW_HPP
