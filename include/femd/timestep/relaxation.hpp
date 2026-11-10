//
//  relaxation.hpp  --  relaxation Runge-Kutta (RRK) for any Runge-Kutta step, explicit or implicit
//  (Ketcheson 2019; Ranocha, Sayyari, Dalcin, Parsani and Ketcheson 2020).
//
//  A step from u_n with stage values U_i and stage derivatives k_i becomes
//
//      u_{n+1} = u_n + gamma dt d,      d = sum_i b_i k_i,
//      r(gamma) = eta(u_n + gamma dt d) - eta(u_n) - gamma dt e = 0,
//      e = sum_i b_i <eta'(U_i), k_i>   (0 with RelaxOptions::conserved),
//
//  and is taken to t_n + gamma dt (the incremental direction technique, which keeps the order of
//  the method).  gamma = 1 + O(dt^{p-1}).  The entropy is
//    * quadratic, eta(v) = 1/2 <v, E v> with E symmetric: gamma in closed form,
//          gamma = 2 (e - <u, E d>) / (dt <d, E d>);
//    * or general, eta(v) and, when known, the directional derivative deta(v, w) = <eta'(v), w>.
//      Without deta the entropy must be conserved by f (e = 0), and the root is found
//      without derivatives.
//  The root of r is found by Newton (r'(gamma) = dt deta(u + gamma dt d, d) - dt e), the
//  secant method or Brent's method on a bracket that excludes the trivial root gamma = 0.
//  RelaxSolver::Auto takes the closed form for a quadratic entropy, Newton when deta is
//  known and the secant method otherwise, and falls back to Brent when that fails.
//
//  RelaxOptions::conserved takes e = 0 even when the derivative is known: the entropy then
//  returns to eta(u_n) to rounding, whatever error the stages carry (a Newton tolerance in an
//  implicit method, a Krylov tolerance in the mass solves).
//
//  Used by ExplicitRK::step_relaxed, SSPRK::step_relaxed and ImplicitRK::step_relaxed.
//
#ifndef FEMD_TIMESTEP_RELAXATION_HPP
#define FEMD_TIMESTEP_RELAXATION_HPP

#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

enum class RelaxSolver { Auto, Closed, Newton, Secant, Brent };

inline RelaxSolver relax_solver(std::string m)
{
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (m == "auto") return RelaxSolver::Auto;
    if (m == "newton") return RelaxSolver::Newton;
    if (m == "secant") return RelaxSolver::Secant;
    if (m == "brent") return RelaxSolver::Brent;
    throw std::invalid_argument("relaxation: unknown root solver '" + m + "'; use auto, newton, secant or brent");
}

inline const char *relax_solver_name(RelaxSolver s)
{
    switch (s)
    {
    case RelaxSolver::Auto: return "auto";
    case RelaxSolver::Closed: return "closed form";
    case RelaxSolver::Newton: return "newton";
    case RelaxSolver::Secant: return "secant";
    case RelaxSolver::Brent: return "brent";
    }
    return "?";
}

/// The entropy of a relaxation step: quadratic (E set) or general (eta, and deta when known).
struct Entropy {
    using Vec = std::vector<double>;
    std::function<void(const Vec &, Vec &)> E;              ///< y = E x, E symmetric: eta = 1/2 <v, E v>
    std::function<double(const Vec &)> eta;                 ///< eta(v)
    std::function<double(const Vec &, const Vec &)> deta;   ///< <eta'(v), w>, may be empty

    bool quadratic() const { return static_cast<bool>(E); }
    bool has_derivative() const { return quadratic() || static_cast<bool>(deta); }

    static Entropy quadratic_form(std::function<void(const Vec &, Vec &)> op) { Entropy H; H.E = std::move(op); return H; }
    static Entropy general(std::function<double(const Vec &)> f, std::function<double(const Vec &, const Vec &)> df = {})
    {
        Entropy H; H.eta = std::move(f); H.deta = std::move(df); return H;
    }
};

struct RelaxOptions {
    RelaxSolver solver = RelaxSolver::Auto;
    bool conserved = false;          ///< take e = 0: the entropy is conserved by f, whatever the stages say
    double tol = 1e-14;              ///< stop when |gamma_{k+1} - gamma_k| <= tol max(1, |gamma|)
    int maxiter = 20;                ///< iterations of Newton or the secant method (Brent: 5 maxiter)
};

struct RelaxInfo {
    double gamma = 1.0;              ///< the relaxation parameter, 1 when the step was not relaxed
    int iterations = 0;              ///< root-solver iterations (entropy evaluations for Brent)
    bool converged = false;
    RelaxSolver solver = RelaxSolver::Auto;   ///< the method that produced gamma
    double residual = 0.0;           ///< |r(gamma)|, 0 for the closed form
    double production = 0.0;         ///< dt e = dt sum b_i <eta'(U_i), k_i>, the entropy change of the step
    std::string message;             ///< why it failed, empty on success
};

namespace detail {

inline std::string relax_num(double x)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return buf;
}

/// The root gamma > 0 of r, the trivial root 0 excluded.  r and dr (may be empty) are given.
struct RelaxRoot {
    std::function<double(double)> r, dr;
    double floor;                    ///< the rounding level of r (and of rho near gamma = 1)
    double tol;
    int maxiter;

    static bool admissible(double g) { return std::isfinite(g) && g >= 1e-3; }

    // Newton and the secant method work on rho(gamma) = r(gamma) / gamma, which has the same positive
    // root and not the trivial one at 0, and is nearly linear near gamma = 1.  gamma is determined
    // only to about eps |eta| / |r'(gamma)|, and r' = O(dt^2), so for small dt the iterates soon
    // wander in rounding noise.  The iteration stops when the step in gamma falls below tol, when
    // rho is exactly zero, or once |rho| has reached the rounding level `floor` and either stops
    // falling or stays there one more iteration.  The iterate with the smallest |rho| is returned.
    struct Best {
        double g = 1.0, R = std::numeric_limits<double>::infinity();
        void see(double gg, double RR) { if (std::abs(RR) < R) { g = gg; R = std::abs(RR); } }
    };

    bool accept(RelaxInfo &I, const Best &B, const char *who) const
    {
        I.gamma = B.g; I.residual = B.R * std::abs(B.g);          // |r| = gamma |rho|
        if (!admissible(B.g))
        {
            I.message = std::string(who) + ": converged to gamma = " + relax_num(B.g) + ", not a positive root";
            return false;
        }
        return true;
    }

    double rho(double g) const { return r(g) / g; }
    bool noise_done(double R, double prev) const { return std::abs(R) <= floor && (prev <= floor || std::abs(R) >= prev); }

    bool newton(RelaxInfo &I) const
    {
        Best B;
        double g = 1.0, prev = std::numeric_limits<double>::infinity();
        for (int it = 0;; ++it)
        {
            const double R = rho(g);
            if (!std::isfinite(R)) { I.message = "newton: r is not finite"; return false; }
            B.see(g, R);
            if (R == 0.0 || noise_done(R, prev)) return accept(I, B, "newton");
            if (it == maxiter) break;
            const double Rp = (dr(g) - R) / g;                      // rho' = (r' - rho) / gamma
            if (!(std::isfinite(Rp) && Rp != 0.0))
            {
                if (B.R <= floor) return accept(I, B, "newton");
                I.message = "newton: r'(gamma) is zero or not finite";
                return false;
            }
            const double dg = R / Rp;
            g -= dg;
            I.iterations = it + 1;
            if (!admissible(g)) { I.message = "newton: gamma = " + relax_num(g) + " left the positive axis"; return false; }
            if (std::abs(dg) <= tol * std::max(1.0, std::abs(g)))
            {
                const double Rn = rho(g);
                if (!std::isfinite(Rn)) { I.message = "newton: r is not finite"; return false; }
                B.see(g, Rn);
                return accept(I, B, "newton");
            }
            prev = std::abs(R);
        }
        if (B.R <= floor) return accept(I, B, "newton");
        I.residual = B.R;
        I.message = "newton: no convergence in " + std::to_string(maxiter) + " iterations";
        return false;
    }

    bool secant(RelaxInfo &I) const
    {
        Best B;
        double g0 = 1.0, g1 = 0.99, r0 = rho(g0);
        if (!std::isfinite(r0)) { I.message = "secant: r is not finite"; return false; }
        B.see(g0, r0);
        if (r0 == 0.0) return accept(I, B, "secant");
        double prev = std::numeric_limits<double>::infinity();
        for (int it = 0;; ++it)
        {
            const double r1 = rho(g1);
            if (!std::isfinite(r1)) { I.message = "secant: r is not finite"; return false; }
            B.see(g1, r1);
            if (r1 == 0.0 || (it > 0 && noise_done(r1, prev))) return accept(I, B, "secant");
            if (it == maxiter) break;
            if (r1 == r0)
            {
                if (B.R <= floor) return accept(I, B, "secant");
                I.message = "secant: r(gamma_k) = r(gamma_{k-1})";
                return false;
            }
            const double g2 = g1 - r1 * (g1 - g0) / (r1 - r0);
            I.iterations = it + 1;
            if (!admissible(g2)) { I.message = "secant: gamma = " + relax_num(g2) + " left the positive axis"; return false; }
            g0 = g1; r0 = r1; g1 = g2;
            prev = std::abs(r1);
            if (std::abs(g1 - g0) <= tol * std::max(1.0, std::abs(g1)))
            {
                const double Rn = rho(g1);
                if (!std::isfinite(Rn)) { I.message = "secant: r is not finite"; return false; }
                B.see(g1, Rn);
                return accept(I, B, "secant");
            }
        }
        if (B.R <= floor) return accept(I, B, "secant");
        I.residual = B.R;
        I.message = "secant: no convergence in " + std::to_string(maxiter) + " iterations";
        return false;
    }

    /// Brent's method (zeroin) on a bracket [a, b] around 1 inside [1e-3, 16].
    bool brent(RelaxInfo &I) const
    {
        double a = 0.5, b = 1.5, fa = r(a), fb = r(b);
        int evals = 2;
        while (std::isfinite(fa) && std::isfinite(fb) && (fa > 0) == (fb > 0) && fa != 0.0 && fb != 0.0)
        {
            if (a <= 1e-3 && b >= 16.0)
            {
                I.message = "brent: no sign change of r on [1e-3, 16], no relaxation parameter (dt too large?)";
                I.iterations = evals;
                return false;
            }
            if (a > 1e-3) { a = std::max(1e-3, 0.5 * a); fa = r(a); ++evals; }
            if (b < 16.0) { b = std::min(16.0, 2.0 * b); fb = r(b); ++evals; }
        }
        if (!(std::isfinite(fa) && std::isfinite(fb))) { I.message = "brent: r is not finite on the bracket"; return false; }
        if (fa == 0.0) { I.gamma = a; I.residual = 0.0; I.iterations = evals; return true; }
        if (fb == 0.0) { I.gamma = b; I.residual = 0.0; I.iterations = evals; return true; }
        double c = a, fc = fa, d = b - a, e = d, prev = std::numeric_limits<double>::infinity();
        const double eps = std::numeric_limits<double>::epsilon();
        for (int it = 0; it < 5 * maxiter; ++it)
        {
            if ((fb > 0) == (fc > 0)) { c = a; fc = fa; d = b - a; e = d; }
            if (std::abs(fc) < std::abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
            const double tol1 = 2.0 * eps * std::abs(b) + 0.5 * tol * std::max(1.0, std::abs(b));
            const double xm = 0.5 * (c - b);
            // b holds the smallest |r| so far; in rounding noise the bracket shrinks slowly, so stop as
            // Newton does once |r| is at the rounding level twice or stops falling there
            if (std::abs(xm) <= tol1 || (std::abs(fb) <= floor && (prev <= floor || std::abs(fb) >= prev)))
            {
                I.gamma = b; I.residual = std::abs(fb); I.iterations = evals;
                return admissible(b);
            }
            if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb))
            {
                double p, q, s = fb / fa;
                if (a == c) { p = 2.0 * xm * s; q = 1.0 - s; }             // secant
                else
                {                                                          // inverse quadratic interpolation
                    const double qq = fa / fc, rr = fb / fc;
                    p = s * (2.0 * xm * qq * (qq - rr) - (b - a) * (rr - 1.0));
                    q = (qq - 1.0) * (rr - 1.0) * (s - 1.0);
                }
                if (p > 0) q = -q; else p = -p;
                if (2.0 * p < std::min(3.0 * xm * q - std::abs(tol1 * q), std::abs(e * q))) { e = d; d = p / q; }
                else { d = xm; e = d; }                                    // bisection
            }
            else { d = xm; e = d; }
            prev = std::abs(fb);
            a = b; fa = fb;
            b += std::abs(d) > tol1 ? d : (xm > 0 ? tol1 : -tol1);
            fb = r(b); ++evals;
            if (!std::isfinite(fb)) { I.message = "brent: r is not finite"; return false; }
        }
        I.message = "brent: no convergence";
        I.iterations = evals;
        return false;
    }
};

} // namespace detail

/**
 * The relaxation of one Runge-Kutta step, for any tableau, explicit or implicit: the stepper calls
 * begin(), then stage(U_i, k_i, b_i) for every stage (U_i the stage value, k_i = M^{-1} f(t_i, U_i)
 * or the stage derivative K_i), then finish(u, dt), which finds gamma and sets
 * u <- u + gamma dt sum_i b_i k_i.  A Relaxation keeps its work vectors between steps.
 */
class Relaxation {
public:
    using Vec = std::vector<double>;

    void begin(std::size_t n, const Entropy &H, const RelaxOptions &opt)
    {
        if (!H.quadratic() && !H.eta) throw std::invalid_argument("relaxation: the entropy has neither E nor eta");
        if (opt.solver == RelaxSolver::Newton && !H.has_derivative())
            throw std::invalid_argument("relaxation: Newton needs the derivative of the entropy (deta, or a quadratic entropy)");
        if (opt.solver == RelaxSolver::Closed && !H.quadratic())
            throw std::invalid_argument("relaxation: the closed form needs a quadratic entropy");
        if (!(opt.tol > 0.0) || opt.maxiter < 1) throw std::invalid_argument("relaxation: tol must be > 0 and maxiter >= 1");
        H_ = &H; opt_ = opt; n_ = n; e_ = 0.0;
        d_.assign(n, 0.0);
    }

    /// d += b k and, unless the entropy is declared conserved, e += b <eta'(U), k>.
    void stage(const Vec &U, const Vec &k, double b)
    {
        if (b == 0.0) return;
        if (U.size() != n_ || k.size() != n_) throw std::invalid_argument("relaxation: a stage has the wrong length");
        const long nl = static_cast<long>(n_);
        const bool par = n_ > detail::vector_par_threshold;
        const double *kk = k.data();
        double *dd = d_.data();
        FEMD_OMP_FOR_IF(par)
        for (long q = 0; q < nl; ++q) dd[q] += b * kk[q];
        if (!opt_.conserved && H_->has_derivative()) e_ += b * deta(U, k);
    }

    /// Find gamma and, when one is found, set u <- u + gamma dt d.  u is left alone otherwise.
    RelaxInfo finish(Vec &u, double dt)
    {
        if (u.size() != n_) throw std::invalid_argument("relaxation: u has the wrong length");
        const Entropy &H = *H_;
        const std::size_t n = n_;
        const long nl = static_cast<long>(n);
        const bool par = n > detail::vector_par_threshold;
        const double e = e_;
        RelaxInfo I;
        I.production = dt * e;
        const double eps = std::numeric_limits<double>::epsilon();
        bool ok = false;
        if (H.quadratic() && (opt_.solver == RelaxSolver::Auto || opt_.solver == RelaxSolver::Closed))
        {
            apply_E(d_, Ev_);
            const double dEd = dot(d_, Ev_), uEd = dot(u, Ev_);
            I.solver = RelaxSolver::Closed;
            if (dEd == 0.0) { I.gamma = 1.0; ok = true; }        // nothing moves the entropy
            else
            {
                I.gamma = 2.0 * (e - uEd) / (dt * dEd);
                ok = detail::RelaxRoot::admissible(I.gamma);
                if (!ok) I.message = "closed form: gamma = " + detail::relax_num(I.gamma) + " is not a positive root (dt too large?)";
            }
        }
        else
        {
            const bool deriv = H.has_derivative();
            const double eta0 = eta(u);
            W_.resize(n);
            auto trial = [&](double g) {                         // W = u + g dt d
                const double w = g * dt;
                const double *uu = u.data(), *dd = d_.data();
                double *W = W_.data();
                FEMD_OMP_FOR_IF(par)
                for (long q = 0; q < nl; ++q) W[q] = uu[q] + w * dd[q];
            };
            detail::RelaxRoot R;
            R.r = [&](double g) { trial(g); return eta(W_) - eta0 - g * dt * e; };
            if (deriv) R.dr = [&](double g) { trial(g); return dt * deta(W_, d_) - dt * e; };
            R.floor = 64.0 * eps * (std::abs(eta0) + std::abs(dt * e));
            R.tol = opt_.tol;
            R.maxiter = opt_.maxiter;
            RelaxSolver m = opt_.solver;
            if (m == RelaxSolver::Auto) m = deriv ? RelaxSolver::Newton : RelaxSolver::Secant;
            I.solver = m;
            ok = m == RelaxSolver::Newton ? R.newton(I) : m == RelaxSolver::Secant ? R.secant(I) : R.brent(I);
            if (!ok && opt_.solver == RelaxSolver::Auto)
            {
                const std::string first = I.message;
                RelaxInfo J;
                J.production = I.production;
                J.solver = RelaxSolver::Brent;
                ok = R.brent(J);
                J.iterations += I.iterations;
                if (!ok) J.message = first + "; then " + J.message;
                I = J;
            }
        }
        I.converged = ok;
        if (!ok) return I;
        const double w = I.gamma * dt;
        const double *dd = d_.data();
        double *uu = u.data();
        FEMD_OMP_FOR_IF(par)
        for (long q = 0; q < nl; ++q) uu[q] += w * dd[q];
        return I;
    }

    const Vec &direction() const { return d_; }      ///< d = sum_i b_i k_i of the current step
    double production_rate() const { return e_; }   ///< e = sum_i b_i <eta'(U_i), k_i>

private:
    const Entropy *H_ = nullptr;
    RelaxOptions opt_;
    std::size_t n_ = 0;
    double e_ = 0.0;
    Vec d_, W_, Ev_;

    double dot(const Vec &x, const Vec &y) const { return ddot(x.data(), y.data(), n_); }
    void apply_E(const Vec &x, Vec &y) const
    {
        H_->E(x, y);
        if (y.size() != n_) throw std::invalid_argument("relaxation: E x has the wrong length");
    }
    double eta(const Vec &v)
    {
        if (H_->quadratic()) { apply_E(v, Ev_); return 0.5 * dot(v, Ev_); }
        return H_->eta(v);
    }
    double deta(const Vec &v, const Vec &w)                  // E symmetric: <E v, w>
    {
        if (H_->quadratic()) { apply_E(v, Ev_); return dot(Ev_, w); }
        return H_->deta(v, w);
    }
};

} // namespace femd

#endif // FEMD_TIMESTEP_RELAXATION_HPP
