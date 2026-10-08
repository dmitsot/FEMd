//
//  ssp_rk.hpp  --  explicit strong-stability-preserving Runge-Kutta methods for  M u' = f(t, u),
//  written as convex combinations of forward Euler steps (Shu and Osher), with an optional
//  limiter applied after every stage.  The stepper sees the rate  k = M^{-1} f(t, u)  and the
//  limiter as callables, as explicit_rk.hpp does:
//
//      void rate(double t, const std::vector<double> &u, std::vector<double> &k)
//      void limit(std::vector<double> &u)                      // in place; a no-op when there is none
//
//  Methods (stages, order): (1,1) forward Euler, (2,2) Heun, (3,3) Shu-Osher, (4,3) with the
//  SSP coefficient 2, and (10,4), Ketcheson's low-storage method with SSP coefficient 6.
//
#ifndef FEMD_TIMESTEP_SSP_RK_HPP
#define FEMD_TIMESTEP_SSP_RK_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class SSPRK {
public:
    SSPRK(int stages, int order) : s_(stages), p_(order)
    {
        if (!(key() == 11 || key() == 22 || key() == 33 || key() == 43 || key() == 104))
            throw std::invalid_argument("SSPRK: (stages, order) must be (1,1), (2,2), (3,3), (4,3) or (10,4), got (" +
                                        std::to_string(stages) + "," + std::to_string(order) + ")");
    }
    int stages() const { return s_; }
    int order() const { return p_; }
    /// @brief The SSP coefficient: the step may be this many times the forward Euler step.
    double cfl() const
    {
        switch (key()) { case 43: return 2.0; case 104: return 6.0; default: return 1.0; }
    }
    std::string name() const
    {
        switch (key())
        {
        case 11: return "forward Euler";
        case 22: return "Heun (SSPRK(2,2))";
        case 33: return "Shu-Osher (SSPRK(3,3))";
        case 43: return "SSPRK(4,3)";
        default: return "Ketcheson SSPRK(10,4)";
        }
    }

    /// @brief One step, u overwritten with the value at t + dt.
    template <class Rate, class Limit>
    void step(std::vector<double> &u, double t, double dt, Rate &&rate, Limit &&limit)
    {
        const std::size_t n = u.size();
        std::vector<double> &k = k_, &u1 = u1_, &u2 = u2_;
        // out = from + h L(tt, from), without the limiter: it is applied to the stage values below,
        // exactly where the Shu-Osher form puts it (after each convex combination)
        auto euler = [&](std::vector<double> &out, const std::vector<double> &from, double tt, double h) {
            rate(tt, from, k);
            if (k.size() != n) throw std::invalid_argument("SSPRK: the rate returned the wrong length");
            out.resize(n);
            for (std::size_t q = 0; q < n; ++q) out[q] = from[q] + h * k[q];
        };
        switch (key())
        {
        case 11:
            euler(u1, u, t, dt);
            limit(u1);
            u.swap(u1);
            return;
        case 22:
            euler(u1, u, t, dt); limit(u1);                       // u1 = lim(u0 + dt L(u0))
            euler(u2, u1, t + dt, dt);                            // u1 + dt L(u1)
            for (std::size_t q = 0; q < n; ++q) u[q] = 0.5 * u[q] + 0.5 * u2[q];
            limit(u);
            return;
        case 33:
            euler(u1, u, t, dt); limit(u1);
            euler(u2, u1, t + dt, dt);
            for (std::size_t q = 0; q < n; ++q) u2[q] = 0.75 * u[q] + 0.25 * u2[q];
            limit(u2);
            euler(u1, u2, t + 0.5 * dt, dt);
            for (std::size_t q = 0; q < n; ++q) u[q] = u[q] / 3.0 + 2.0 / 3.0 * u1[q];
            limit(u);
            return;
        case 43:
        {
            const double h = 0.5 * dt;
            euler(u1, u, t, h); limit(u1);
            euler(u2, u1, t + h, h); limit(u2);
            euler(u1, u2, t + dt, h);
            for (std::size_t q = 0; q < n; ++q) u1[q] = 2.0 / 3.0 * u[q] + u1[q] / 3.0;
            limit(u1);
            euler(u, u1, t + h, h); limit(u);
            return;
        }
        default:                                                   // Ketcheson (2008), low storage
        {
            const double h = dt / 6.0;
            std::vector<double> &q1 = u1, &q2 = u2;
            q1 = u; q2 = u;
            double tq = t;
            for (int i = 0; i < 5; ++i) { euler(t_, q1, tq, h); limit(t_); q1.swap(t_); tq += h; }
            for (std::size_t q = 0; q < n; ++q) q2[q] = q2[q] / 25.0 + 9.0 / 25.0 * q1[q];
            for (std::size_t q = 0; q < n; ++q) q1[q] = 15.0 * q2[q] - 5.0 * q1[q];
            limit(q1);
            tq = t + dt / 3.0;
            for (int i = 0; i < 4; ++i) { euler(t_, q1, tq, h); limit(t_); q1.swap(t_); tq += h; }
            rate(t + dt, q1, k_);
            for (std::size_t q = 0; q < n; ++q) u[q] = q2[q] + 0.6 * q1[q] + 0.1 * dt * k_[q];
            limit(u);
            return;
        }
        }
    }

private:
    int s_, p_;
    std::vector<double> k_, u1_, u2_, t_;
    int key() const { return 10 * s_ + p_; }
};

} // namespace femd

#endif // FEMD_TIMESTEP_SSP_RK_HPP
