//
//  implicit_rk.hpp  --  implicit Runge-Kutta methods for stiff and differential-algebraic
//  systems  M u' = f(t, u), the stage iteration in C++.
//
//  One step from u^n with the stages K_i,
//
//      M K_i = f(t_n + c_i dt, U_i),   U_i = u^n + dt sum_j a_ij K_j,   u^{n+1} = u^n + dt sum_i b_i K_i,
//
//  is the nonlinear system  R(K) = 0,  R_i = M K_i - f(t_i, U_i),  for K = [K_1; ...; K_s]
//  (the stages stacked), solved by Newton's method with the tests of a simplified Newton
//  iteration (Hairer and Wanner): stop on ||R|| <= max(tol, rtol ||R_0||), or when the
//  error left in K is estimated below xtol ||K|| from the contraction rate of the steps,
//  or when ||R|| stalls at its round-off floor while the step is tiny.
//
//  The stepper sees the problem through three callables, so it serves 1D banded and 2D
//  sparse problems, boundary data lifted by the caller, and Python forms alike:
//
//      residual(double t, const std::vector<double> &U, std::vector<double> &f)   f = f(t, U)
//      mass(const std::vector<double> &K, std::vector<double> &MK)                MK = M K
//      solve(const std::vector<double> &K, const std::vector<double> &r, std::vector<double> &d)
//
//  solve applies the (approximate) inverse of the stage Jacobian  I (x) M - dt A (x) f'  to r,
//  given the current stages K: a simplified Newton passes a factorization made once per step
//  (or kept over several steps), the classical one refactors at each call.  Which, and how
//  the s n system is factored (decoupled through the eigenvalues of A, or as one block
//  matrix), is the caller's decision.
//
//  Named tableaux: "gauss" (Gauss-Legendre, s = 1, 2, 3, order 2s, A-stable, symplectic) and
//  "radau" (Radau IIA, s = 1, 2, 3, order 2s-1, L-stable, stiffly accurate, the choice for
//  differential-algebraic systems).  Any other tableau goes in as (A, b).
//
#ifndef FEMD_TIMESTEP_IMPLICIT_RK_HPP
#define FEMD_TIMESTEP_IMPLICIT_RK_HPP

#include "femd/solve/newton.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

struct ImplicitTableau {
    int s = 0;                       ///< stages
    int order = 0;                   ///< classical order (0 when not known)
    std::string name;
    std::vector<double> A;           ///< s x s row-major
    std::vector<double> b, c;
    double a(int i, int j) const { return A[static_cast<std::size_t>(i) * s + j]; }

    void validate()
    {
        if (s < 1) throw std::invalid_argument("implicit RK: at least one stage");
        if (static_cast<int>(A.size()) != s * s || static_cast<int>(b.size()) != s)
            throw std::invalid_argument("implicit RK: A must be s x s and b of length s");
        if (c.empty())
        {
            c.assign(static_cast<std::size_t>(s), 0.0);
            for (int i = 0; i < s; ++i) for (int j = 0; j < s; ++j) c[i] += a(i, j);
        }
        if (static_cast<int>(c.size()) != s) throw std::invalid_argument("implicit RK: c must have length s");
    }

    /// @brief "gauss" or "radau" with 1, 2 or 3 stages.
    static ImplicitTableau named(std::string m, int stages)
    {
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const double r3 = std::sqrt(3.0), r6 = std::sqrt(6.0), r15 = std::sqrt(15.0);
        ImplicitTableau T;
        T.s = stages;
        if (m == "gauss" || m == "gauss-legendre" || m == "gl")
        {
            T.name = "gauss"; T.order = 2 * stages;
            if (stages == 1) { T.A = {0.5}; T.b = {1.0}; }
            else if (stages == 2) { T.A = {0.25, 0.25 - r3 / 6, 0.25 + r3 / 6, 0.25}; T.b = {0.5, 0.5}; }
            else if (stages == 3)
            {
                T.A = {5.0 / 36, 2.0 / 9 - r15 / 15, 5.0 / 36 - r15 / 30,
                       5.0 / 36 + r15 / 24, 2.0 / 9, 5.0 / 36 - r15 / 24,
                       5.0 / 36 + r15 / 30, 2.0 / 9 + r15 / 15, 5.0 / 36};
                T.b = {5.0 / 18, 4.0 / 9, 5.0 / 18};
            }
            else throw std::invalid_argument("implicit RK: Gauss-Legendre with 1, 2 or 3 stages");
        }
        else if (m == "radau" || m == "radau iia" || m == "radauiia" || m == "radau-iia")
        {
            T.name = "radau"; T.order = 2 * stages - 1;
            if (stages == 1) { T.A = {1.0}; T.b = {1.0}; }
            else if (stages == 2) { T.A = {5.0 / 12, -1.0 / 12, 0.75, 0.25}; T.b = {0.75, 0.25}; }
            else if (stages == 3)
            {
                T.A = {(88 - 7 * r6) / 360, (296 - 169 * r6) / 1800, (-2 + 3 * r6) / 225,
                       (296 + 169 * r6) / 1800, (88 + 7 * r6) / 360, (-2 - 3 * r6) / 225,
                       (16 - r6) / 36, (16 + r6) / 36, 1.0 / 9};
                T.b = {(16 - r6) / 36, (16 + r6) / 36, 1.0 / 9};
            }
            else throw std::invalid_argument("implicit RK: Radau IIA with 1, 2 or 3 stages");
        }
        else
            throw std::invalid_argument("implicit RK: unknown method '" + m + "'; use gauss or radau, or give a tableau (A, b)");
        T.validate();
        return T;
    }
};

struct ImplicitRKOptions {
    double tol = 1e-12;              ///< stop when ||R|| <= max(tol, rtol ||R_0||)
    double rtol = 0.0;
    double xtol = 1e-12;             ///< and when the error left in K is estimated below xtol ||K||
    int maxiter = 20;
    bool line_search = false;        ///< halve the step until ||R|| decreases (Armijo)
};

class ImplicitRK {
public:
    explicit ImplicitRK(ImplicitTableau T, ImplicitRKOptions opt = ImplicitRKOptions()) : T_(std::move(T)), opt_(opt) { T_.validate(); }
    ImplicitRK(const std::string &method, int stages, ImplicitRKOptions opt = ImplicitRKOptions())
        : T_(ImplicitTableau::named(method, stages)), opt_(opt) {}

    const ImplicitTableau &tableau() const { return T_; }
    const ImplicitRKOptions &options() const { return opt_; }
    ImplicitRKOptions &options() { return opt_; }

    /// @brief The stages of the last step, stacked [K_1; ...; K_s] (empty before the first step).
    const std::vector<double> &stages() const { return K_; }
    /// @brief Start the next step's Newton iteration from these stages (s n values), or from nothing.
    void set_stages(std::vector<double> K) { K_ = std::move(K); }
    void forget_stages() { K_.clear(); }

    /// @brief The stage values U_i = u + dt sum_j a_ij K_j, stacked.
    void stage_values(const std::vector<double> &u, double dt, const std::vector<double> &K, std::vector<double> &U) const
    {
        const std::size_t n = u.size();
        const long nl = static_cast<long>(n);
        const bool par = n > detail::vector_par_threshold;
        (void)par;
        const int s = T_.s;
        U.resize(n * static_cast<std::size_t>(s));
        for (int i = 0; i < s; ++i)
        {
            double *Ui = U.data() + static_cast<std::size_t>(i) * n;
            std::copy(u.begin(), u.end(), Ui);
            for (int j = 0; j < s; ++j)
            {
                const double w = dt * T_.a(i, j);
                if (w == 0.0) continue;
                const double *Kj = K.data() + static_cast<std::size_t>(j) * n;
                FEMD_OMP_FOR_IF(par)
                for (long q = 0; q < nl; ++q) Ui[q] += w * Kj[q];
            }
        }
    }

    /// @brief R_i = M K_i - f(t + c_i dt, U_i), stacked.
    template <class Residual, class Mass>
    void stage_residual(const std::vector<double> &u, double t, double dt, const std::vector<double> &K,
                        Residual &&residual, Mass &&mass, std::vector<double> &R) const
    {
        const std::size_t n = u.size();
        const int s = T_.s;
        stage_values(u, dt, K, U_);
        R.resize(n * static_cast<std::size_t>(s));
        for (int i = 0; i < s; ++i)
        {
            const std::size_t off = static_cast<std::size_t>(i) * n;
            Ki_.assign(K.begin() + static_cast<std::ptrdiff_t>(off), K.begin() + static_cast<std::ptrdiff_t>(off + n));
            mass(Ki_, MK_);
            if (MK_.size() != n) throw std::invalid_argument("implicit RK: the mass operator returned the wrong length");
            Ui_.assign(U_.begin() + static_cast<std::ptrdiff_t>(off), U_.begin() + static_cast<std::ptrdiff_t>(off + n));
            residual(t + T_.c[i] * dt, Ui_, f_);
            if (f_.size() != n) throw std::invalid_argument("implicit RK: the residual returned the wrong length");
            double *Ri = R.data() + off;
            for (std::size_t q = 0; q < n; ++q) Ri[q] = MK_[q] - f_[q];
        }
    }

    /**
     * @brief One step, u overwritten with the value at t + dt when Newton converges (and left
     *        alone otherwise, so the caller can refactor and call again).
     * @param k0  the first Newton iterate for the stages (s n values), or empty: the stages of the
     *            last step, or zero before the first one
     */
    template <class Residual, class Mass, class Solve>
    NewtonReport step(std::vector<double> &u, double t, double dt, Residual &&residual, Mass &&mass, Solve &&solve,
                      const std::vector<double> &k0 = std::vector<double>())
    {
        const std::size_t n = u.size(), N = n * static_cast<std::size_t>(T_.s);
        std::vector<double> K = !k0.empty() ? k0 : (K_.size() == N ? K_ : std::vector<double>(N, 0.0));
        if (K.size() != N) throw std::invalid_argument("implicit RK: k0 must have s n values");
        NewtonReport rep = newton(u, t, dt, K, residual, mass, solve);
        if (!rep.converged) return rep;
        K_ = K;
        const long nl = static_cast<long>(n);
        const bool par = n > detail::vector_par_threshold;
        (void)par;
        double *uu = u.data();
        for (int i = 0; i < T_.s; ++i)
        {
            const double w = dt * T_.b[i];
            if (w == 0.0) continue;
            const double *Ki = K.data() + static_cast<std::size_t>(i) * n;
            FEMD_OMP_FOR_IF(par)
            for (long q = 0; q < nl; ++q) uu[q] += w * Ki[q];
        }
        return rep;
    }

private:
    ImplicitTableau T_;
    ImplicitRKOptions opt_;
    std::vector<double> K_;
    mutable std::vector<double> U_, Ui_, Ki_, MK_, f_;

    // Newton on R(K) = 0 (solve/newton.hpp), K updated in place; the stage solver gives the direction
    template <class Residual, class Mass, class Solve>
    NewtonReport newton(const std::vector<double> &u, double t, double dt, std::vector<double> &K,
                        Residual &residual, Mass &mass, Solve &solve)
    {
        NewtonOptions o;
        o.tol = opt_.tol; o.rtol = opt_.rtol; o.xtol = opt_.xtol; o.maxiter = opt_.maxiter; o.line_search = opt_.line_search;
        std::vector<double> negR;
        auto res = [&](const std::vector<double> &Kc, std::vector<double> &R) { stage_residual(u, t, dt, Kc, residual, mass, R); };
        auto dir = [&](const std::vector<double> &Kc, const std::vector<double> &R, std::vector<double> &d, int &lin, std::string &) {
            negR.resize(R.size());
            for (std::size_t q = 0; q < R.size(); ++q) negR[q] = -R[q];
            solve(Kc, negR, d);
            if (d.size() != Kc.size()) throw std::invalid_argument("implicit RK: the stage solver returned the wrong length");
            lin = 0;
            return true;
        };
        return newton_solve(K, res, dir, o);
    }
};

} // namespace femd

#endif // FEMD_TIMESTEP_IMPLICIT_RK_HPP
