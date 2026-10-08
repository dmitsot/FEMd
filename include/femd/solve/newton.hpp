//
//  newton.hpp  --  Newton's method for an algebraic system  F(x) = 0, the iteration behind
//  fd.newton, fd.newton_system and the stage equations of fd.IRK.
//
//  The caller supplies two callables:
//
//      void residual(const std::vector<double> &x, std::vector<double> &r)            r = F(x)
//      bool direction(const std::vector<double> &x, const std::vector<double> &r,
//                     std::vector<double> &d, int &linear_iterations, std::string &message)
//
//  direction solves J(x) d = -r however it likes (a factorization made now, one kept from
//  before for a simplified Newton, a Krylov method), reports the linear iterations it took,
//  and returns false with a message when it cannot (a singular Jacobian).  Each iteration
//  then takes the step x + s d, with s halved from 1 until ||F|| decreases by the Armijo
//  factor (1 - 1e-4 s) when line_search is on, and stops when
//
//      ||F|| <= max(tol, rtol ||F(x_0)||),
//
//  or, with xtol > 0, when the error left in x is estimated below xtol ||x|| from the
//  contraction rate theta = ||d_k|| / ||d_{k-1}||, theta / (1 - theta) ||d_k|| <= xtol ||x||
//  (as in Hairer and Wanner's implicit Runge-Kutta codes), on the first iteration or once
//  the steps no longer contract when ||d_k|| <= xtol ||x|| itself, and when a full step
//  leaves ||F|| where it was while ||d_k|| <= sqrt(xtol) ||x||: ||F|| is then at its
//  round-off floor and the steps are noise.  The floor grows with the size of the problem,
//  so this lets large problems keep an absolute tol.
//
#ifndef FEMD_SOLVE_NEWTON_HPP
#define FEMD_SOLVE_NEWTON_HPP

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace femd {

struct NewtonOptions {
    double tol = 1e-10;              ///< stop when ||F|| <= max(tol, rtol ||F_0||)
    double rtol = 0.0;
    double xtol = 0.0;               ///< and, when > 0, when the error left in x is estimated below xtol ||x||
    int maxiter = 50;
    bool line_search = true;         ///< halve the step until ||F|| decreases (Armijo)
};

/// @brief What Newton did.
struct NewtonReport {
    bool converged = false;
    int iterations = 0;
    std::vector<double> residuals;   ///< ||F|| at every iterate, the first is the initial one
    std::vector<double> steps;       ///< the step length taken at each iteration
    std::vector<int> linear_iterations;   ///< per iteration, what direction() reported (0 for a direct solve)
    std::string message;
    double residual() const { return residuals.empty() ? 0.0 : residuals.back(); }
};

namespace detail {
inline double two_norm(const std::vector<double> &v)
{
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}
inline bool all_finite(const std::vector<double> &v)
{
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}
} // namespace detail

/**
 * @brief Newton's method on F(x) = 0, x updated in place to the last accepted iterate.
 * @param progress  optional, called after each iteration with (iteration, ||F||, step, linear iterations)
 */
template <class Residual, class Direction>
NewtonReport newton_solve(std::vector<double> &x, Residual &&residual, Direction &&direction, const NewtonOptions &opt,
                          const std::function<void(int, double, double, int)> &progress = nullptr)
{
    using detail::two_norm;
    NewtonReport rep;
    std::vector<double> r, d, xnew, rnew;
    residual(x, r);
    double nr = two_norm(r);
    rep.residuals.push_back(nr);
    const double target = std::max(opt.tol, opt.rtol * nr);
    double nd_prev = 0.0;
    for (int it = 1; it <= opt.maxiter; ++it)
    {
        if (nr <= target) break;
        int lin = 0;
        std::string msg;
        if (!direction(x, r, d, lin, msg))
        {
            rep.message = msg.empty() ? "the Jacobian could not be factored" : msg;
            break;
        }
        if (d.size() != x.size() || !detail::all_finite(d))
        {
            rep.message = "the Newton direction is not finite; the Jacobian is singular or nearly so";
            break;
        }
        const double nd = two_norm(d);
        const bool contracting = nd_prev > 0.0 && nd < nd_prev;
        const double theta = nd_prev > 0.0 ? nd / nd_prev : 0.0;
        nd_prev = nd;
        if (opt.xtol > 0.0)
        {
            const double bound = opt.xtol * two_norm(x);
            const bool done = contracting ? theta / (1.0 - theta) * nd <= bound : nd <= bound;
            if (done)
            {
                for (std::size_t q = 0; q < x.size(); ++q) x[q] += d[q];
                residual(x, r);
                nr = two_norm(r);
                rep.iterations = it;
                rep.steps.push_back(1.0);
                rep.linear_iterations.push_back(lin);
                rep.residuals.push_back(nr);
                rep.converged = true;
                rep.message = "converged on the step: the error left is below xtol, ||F|| is at its round-off floor";
                if (progress) progress(it, nr, 1.0, lin);
                return rep;
            }
        }
        double step = 1.0, n_new = 0.0;
        bool ok = false;
        xnew.resize(x.size());
        while (true)
        {
            for (std::size_t q = 0; q < x.size(); ++q) xnew[q] = x[q] + step * d[q];
            residual(xnew, rnew);
            n_new = two_norm(rnew);
            ok = std::isfinite(n_new) && n_new <= (1.0 - 1e-4 * step) * nr;
            if (ok || (!opt.line_search && std::isfinite(n_new)) || step < std::ldexp(1.0, -30)) break;
            step *= 0.5;
        }
        if (!std::isfinite(n_new) || (opt.line_search && !ok))
        {
            rep.message = !std::isfinite(n_new) ? "the residual is not finite along the Newton direction"
                                                : "the line search could not reduce ||F||; the Jacobian may be singular or the guess too far";
            break;
        }
        const bool stalled = opt.xtol > 0.0 && n_new >= 0.9 * nr && nd <= std::sqrt(opt.xtol) * two_norm(x);
        if (!(stalled && n_new > nr))                       // when stalled keep the better of the two
        {
            x.swap(xnew);
            r.swap(rnew);
            nr = n_new;
        }
        rep.iterations = it;
        rep.steps.push_back(step);
        rep.linear_iterations.push_back(lin);
        rep.residuals.push_back(nr);
        if (progress) progress(it, nr, step, lin);
        if (stalled)
        {
            rep.converged = true;
            rep.message = "converged: ||F|| stalls at its round-off floor while the step is tiny";
            return rep;
        }
    }
    rep.converged = nr <= target;
    return rep;
}

} // namespace femd

#endif // FEMD_SOLVE_NEWTON_HPP
