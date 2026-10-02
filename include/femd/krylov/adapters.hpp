//
//  adapters.hpp  --  FEMd operators and solvers as the matvec / preconditioner
//  callables of the vendored csnewton Krylov solvers (gmres, lgmres).
//
//  csnewton takes an operator and a LEFT preconditioner, each a callable
//      void(const std::vector<double> &in, std::vector<double> &out).
//  These builders turn an AssembledMatrix into the first and a LinearSolver
//  (banded, cyclic, FFT circulant, dense) into the second, and `solve` wraps a
//  whole preconditioned solve with two corrections to the vendored defaults,
//  made here so the vendored file stays verbatim:
//
//    * csnewton stops on ||M^{-1} r|| / ||b||, which mixes a preconditioned
//      residual with an unpreconditioned norm, so the meaning of `tol` changes
//      with the scaling of M.  `solve` passes tol * ||M^{-1} b|| / ||b||, which
//      turns the criterion into the standard ||M^{-1} r|| / ||M^{-1} b|| < tol.
//    * the returned residual is the TRUE relative residual ||b - A x|| / ||b||,
//      recomputed at the end, alongside the preconditioned one.
//
//  Replaces Poseidon's krylov/adapters.hpp, which spoke the 1-based matrix<double>.
//
#ifndef FEMD_KRYLOV_ADAPTERS_HPP
#define FEMD_KRYLOV_ADAPTERS_HPP

#include "femd/forms/assembly.hpp"
#include "femd/solve/linear_solver.hpp"
// The vendored file carries `#pragma omp` lines that a build without OpenMP flags as
// unknown.  Silence that one warning around it, since the file is never edited here.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif
#include "femd/vendor/csnewton/csnewton.hpp"
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
#include <cmath>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace femd {
namespace krylov {

/// The vendored csnewton parallelises every vector operation with an unguarded
/// `#pragma omp parallel for`.  Below FEMD_OMP_THRESHOLD the fork-join costs more
/// than the loop, so for small systems this runs the solver on one thread, and
/// restores the previous thread count on the way out.
struct SerialBelowThreshold {
    int saved = 0;
    explicit SerialBelowThreshold(std::size_t n)
    {
#ifdef _OPENMP
        if (n < static_cast<std::size_t>(FEMD_OMP_THRESHOLD)) { saved = omp_get_max_threads(); omp_set_num_threads(1); }
#else
        (void) n;
#endif
    }
    ~SerialBelowThreshold()
    {
#ifdef _OPENMP
        if (saved > 0) omp_set_num_threads(saved);
#endif
    }
    SerialBelowThreshold(const SerialBelowThreshold &) = delete;
    SerialBelowThreshold &operator=(const SerialBelowThreshold &) = delete;
};

using Vec = std::vector<double>;

/// @brief out = K in, through the banded kernel.  @param K the operator (kept by reference).
inline auto matvec(const AssembledMatrix &K)
{
    return [&K](const Vec &x, Vec &y) {
        y.resize(static_cast<std::size_t>(K.n));
        if (K.n > 0) K.apply(x.data(), y.data());
    };
}

/// @brief out = M^{-1} in, through a factored solver.  @param S the solver (kept by reference).
inline auto precon(const LinearSolver &S)
{
    return [&S](const Vec &r, Vec &z) { z = S.solve(r); };
}

/// @brief out = in.
inline auto identity()
{
    return [](const Vec &r, Vec &z) { z = r; };
}

enum class Method { GMRES, LGMRES };

struct Result {
    bool converged = false;
    int iterations = 0;
    double residual = 0.0;          ///< true relative residual ||b - A x|| / ||b||
    double precond_residual = 0.0;  ///< ||M^{-1}(b - A x)|| / ||M^{-1} b||, what the stopping test measures
};

inline double norm2(const Vec &v)
{
    double s = 0.0;
    for (double a : v) s += a * a;
    return std::sqrt(s);
}

/**
 * @brief Solve A x = b by preconditioned GMRES(restart) or LGMRES(restart, k_aug).
 * @param A      matvec callable, out = A in
 * @param M      left preconditioner callable, out = M^{-1} in
 * @param x      [in/out] initial guess, overwritten with the solution
 * @param tol    on ||M^{-1} r|| / ||M^{-1} b||
 */
template <class Op, class Pre>
Result solve(Op &&A, Pre &&M, Vec &x, const Vec &b, Method method, int restart, int max_iter, double tol, int k_aug = 2)
{
    if (x.size() != b.size()) throw std::invalid_argument("krylov::solve: x and b differ in length");
    SerialBelowThreshold serial(b.size());
    const double nb = norm2(b);
    Result out;
    if (nb == 0.0)
    {
        std::fill(x.begin(), x.end(), 0.0);
        out.converged = true;
        return out;
    }
    Vec Mb;
    M(b, Mb);
    const double nMb = norm2(Mb);
    const double scaled = nMb > 0.0 ? tol * nMb / nb : tol;
    csnewton::GMRESResult<double> g;
    if (method == Method::LGMRES)
        g = csnewton::lgmres<double>(A, M, x, b, restart, k_aug, max_iter, scaled);
    else
        g = csnewton::gmres<double>(A, M, x, b, restart, max_iter, scaled);
    Vec Ax, r(b.size()), Mr;
    A(x, Ax);
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = b[i] - Ax[i];
    M(r, Mr);
    out.iterations = g.iterations;
    out.residual = norm2(r) / nb;
    out.precond_residual = nMb > 0.0 ? norm2(Mr) / nMb : 0.0;
    out.converged = out.precond_residual <= tol * (1.0 + 1e-8) || (g.converged && out.precond_residual <= 10.0 * tol);
    return out;
}

} // namespace krylov
} // namespace femd

#endif // FEMD_KRYLOV_ADAPTERS_HPP
