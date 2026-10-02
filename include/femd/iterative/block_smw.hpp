//
//  block_smw.hpp  --  block extensions of the SMW iteration.
//
//  Implements the saddle-point and 2D block sweeps of
//    D. Mitsotakis, "On iterative methods based on Sherman-Morrison-Woodbury
//    splitting":
//
//    * block_smw_2x2  -- Algorithms 1 (block-Jacobi-SMW) and 2 (block-GS-SMW)
//        for  [[A1, B1],[B2, A2]] [x;y] = [b1;b2],  with Ai = Mi - Ni and each
//        Mi inverted by its own solver (e.g. an FFT circulant solve).
//
//    * poisson2d_block_gs_smw -- Algorithm 3, the block-Gauss-Seidel-SMW sweep
//        for the 5-point Laplacian on an m x m grid: block-tridiagonal with
//        identity coupling and diagonal blocks D = tridiag(1,-4,1) written as
//        D = M - N with M = circ(-4,1,...,1).
//
//  All blocks reuse CirculantMatrix::solve (M^{-1}) and Sparse matvecs.
//
#ifndef FEMD_ITERATIVE_BLOCK_SMW_HPP
#define FEMD_ITERATIVE_BLOCK_SMW_HPP

#include "femd/iterative/smw.hpp"
#include "femd/fft/circulant.hpp"
#include <functional>
#include <vector>
#include <cmath>

namespace femd {

/// @brief A linear operator as a callable Vec->Vec (the block solver's M^{-1}, N or B).
using Apply = std::function<Vec(const Vec &)>;

/// @brief Result of block_smw_2x2: solution blocks x,y, iterations, residual, convergence flag.
struct BlockResult2 {
    Vec x, y;
    int iters = 0;
    double residual = 0.0;
    bool converged = false;
};

/**
 * @brief Two-block SMW for [[A1,B1],[B2,A2]] [x;y] = [b1;b2], with Ai = Mi - Ni.
 * @param M1inv,M2inv  callables applying M1^{-1}, M2^{-1}.
 * @param N1,N2        callables applying N1, N2.
 * @param B1,B2        callables applying the coupling blocks.
 * @param b1,b2        right-hand side blocks.
 * @param gauss_seidel false = block-Jacobi (Alg. 1); true = block-GS (Alg. 2, uses fresh x).
 * @param tol          stopping tolerance on the combined step norm.
 * @param maxit        maximum iterations.
 * @return BlockResult2 with x, y, iterations, residual and convergence flag.
 */
inline BlockResult2 block_smw_2x2(Apply M1inv, Apply N1, Apply B1,
                                  Apply M2inv, Apply N2, Apply B2,
                                  const Vec &b1, const Vec &b2,
                                  bool gauss_seidel,
                                  double tol = 1e-10, int maxit = 100000)
{
    int n = (int) b1.size(), m = (int) b2.size();
    BlockResult2 R;
    R.x.assign(n, 0.0);
    R.y.assign(m, 0.0);

    for (int k = 1; k <= maxit; ++k)
    {
        // x^{k+1} = M1^{-1}( N1 x - B1 y + b1 )
        Vec r1 = N1(R.x);
        Vec by = B1(R.y);
        for (int i = 0; i < n; ++i) r1[i] += b1[i] - by[i];
        Vec xn = M1inv(r1);

        // y^{k+1} = M2^{-1}( N2 y - B2 x_used + b2 ),  x_used = xn (GS) or x (Jacobi)
        Vec r2 = N2(R.y);
        Vec bx = B2(gauss_seidel ? xn : R.x);
        for (int i = 0; i < m; ++i) r2[i] += b2[i] - bx[i];
        Vec yn = M2inv(r2);

        double err = std::sqrt(l2diff(xn, R.x) * l2diff(xn, R.x) +
                               l2diff(yn, R.y) * l2diff(yn, R.y));
        R.x.swap(xn);
        R.y.swap(yn);
        R.iters = k;
        R.residual = err;
        if (err < tol) { R.converged = true; break; }
    }
    return R;
}

/// @brief Result of poisson2d_block_gs_smw: solution blocks, sweeps, residual, convergence flag.
struct Poisson2DResult {
    std::vector<Vec> x;   // m blocks of length m
    int iters = 0;
    double residual = 0.0;
    bool converged = false;
};

/**
 * @brief Algorithm 3: block-Gauss-Seidel-SMW for the 5-point 2D Poisson Laplacian.
 *
 * On an m x m grid each block row is solved by an FFT circulant solve of
 * M = circ(-4,1,...,1) (with N the corner correction).
 * @param m      grid size (the operator is m^2 x m^2).
 * @param b      right-hand side as m blocks of length m.
 * @param tol    stopping tolerance on the step norm.
 * @param maxit  maximum sweeps.
 * @return Poisson2DResult with the m solution blocks, sweep count, residual and convergence flag.
 */
inline Poisson2DResult poisson2d_block_gs_smw(int m, const std::vector<Vec> &b,
                                              double tol = 1e-7, int maxit = 100000)
{
    std::vector<double> c(m, 0.0);
    c[0] = -4.0; c[1] = 1.0; c[m - 1] = 1.0;
    CirculantMatrix M(c);
    Sparse N(m, m);
    N.add(0, m - 1, 1.0);
    N.add(m - 1, 0, 1.0);

    Poisson2DResult R;
    R.x.assign(m, Vec(m, 0.0));

    for (int k = 1; k <= maxit; ++k)
    {
        double err2 = 0.0;
        for (int i = 0; i < m; ++i)              // sweep block rows (GS, in place)
        {
            Vec rhs = N.apply(R.x[i]);
            for (int j = 0; j < m; ++j) rhs[j] += b[i][j];
            if (i > 0)     for (int j = 0; j < m; ++j) rhs[j] -= R.x[i - 1][j]; // fresh
            if (i < m - 1) for (int j = 0; j < m; ++j) rhs[j] -= R.x[i + 1][j]; // old
            Vec xi = M.solve(rhs);
            for (int j = 0; j < m; ++j) { double d = xi[j] - R.x[i][j]; err2 += d * d; }
            R.x[i].swap(xi);
        }
        R.iters = k;
        R.residual = std::sqrt(err2);
        if (R.residual < tol) { R.converged = true; break; }
    }
    return R;
}

} // namespace femd

#endif // FEMD_ITERATIVE_BLOCK_SMW_HPP
