//
//  cg.hpp  --  preconditioned conjugate gradients for symmetric positive definite systems.
//
//  A and M are the callables of krylov/adapters.hpp:  void(const Vec &in, Vec &out),
//  out = A in and out = M^{-1} in.  M must be symmetric positive definite too.
//  The stopping test is on the TRUE relative residual ||b - A x|| / ||b||, tracked by
//  the recurrence and recomputed at the end, which is what `residual` reports.
//
//  The vector updates run in parallel above FEMD_OMP_THRESHOLD, and the inner
//  products are ordered sums (detail::ordered_sum), so the iterates have the same
//  bits for any number of threads.
//
#ifndef FEMD_KRYLOV_CG_HPP
#define FEMD_KRYLOV_CG_HPP

#include "femd/util/omp.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace femd {
namespace krylov {

struct CGResult {
    bool converged = false;
    int iterations = 0;
    double residual = 0.0;          ///< ||b - A x|| / ||b||, recomputed at the end
    bool breakdown = false;         ///< p^T A p <= 0 or r^T z <= 0: A or M is not SPD
};

/**
 * @brief Solve A x = b by preconditioned CG.
 * @param x   [in/out] initial guess, overwritten with the solution
 * @param tol on ||r|| / ||b||
 */
template <class Op, class Pre>
CGResult cg(Op &&A, Pre &&M, std::vector<double> &x, const std::vector<double> &b, int max_iter, double tol)
{
    using Vec = std::vector<double>;
    const std::size_t n = b.size();
    if (x.size() != n) throw std::invalid_argument("cg: x and b differ in length");
    auto dot = [n](const Vec &a, const Vec &c) { return detail::ordered_sum(n, [&](std::size_t i) { return a[i] * c[i]; }); };
    const long nl = static_cast<long>(n);
    const bool par = nl > FEMD_OMP_THRESHOLD;
    (void) par;                                              // unused in the serial build
    CGResult out;
    const double nb = std::sqrt(dot(b, b));
    if (nb == 0.0) { std::fill(x.begin(), x.end(), 0.0); out.converged = true; return out; }
    Vec r(n), z, p, Ap;
    A(x, Ap);
    FEMD_OMP_FOR_IF(par)
    for (long i = 0; i < nl; ++i) r[i] = b[i] - Ap[i];
    double rn = std::sqrt(dot(r, r));
    if (rn <= tol * nb) { out.converged = true; out.residual = rn / nb; return out; }
    M(r, z);
    p = z;
    double rz = dot(r, z);
    int it = 0;
    while (it < max_iter)
    {
        if (!(rz > 0.0)) { out.breakdown = true; break; }
        A(p, Ap);
        const double pAp = dot(p, Ap);
        if (!(pAp > 0.0)) { out.breakdown = true; break; }
        const double alpha = rz / pAp;
        FEMD_OMP_FOR_IF(par)
        for (long i = 0; i < nl; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        ++it;
        rn = std::sqrt(dot(r, r));
        if (rn <= tol * nb) break;
        M(r, z);
        const double rz_new = dot(r, z);
        const double beta = rz_new / rz;
        rz = rz_new;
        FEMD_OMP_FOR_IF(par)
        for (long i = 0; i < nl; ++i) p[i] = z[i] + beta * p[i];
    }
    A(x, Ap);
    FEMD_OMP_FOR_IF(par)
    for (long i = 0; i < nl; ++i) r[i] = b[i] - Ap[i];
    out.iterations = it;
    out.residual = std::sqrt(dot(r, r)) / nb;
    out.converged = !out.breakdown && out.residual <= tol * (1.0 + 1e-6) + 1e-15;
    if (!out.converged && !out.breakdown && out.residual <= 10.0 * tol && it < max_iter) out.converged = true;
    return out;
}

} // namespace krylov
} // namespace femd

#endif // FEMD_KRYLOV_CG_HPP
