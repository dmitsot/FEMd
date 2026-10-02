//
//  explicit_rk.hpp  --  explicit Runge-Kutta methods for non-stiff systems  M u' = f(t, u).
//
//  The stepper sees only the RATE  k = M^{-1} f(t, U), a callable
//      void rate(double t, const std::vector<double> &U, std::vector<double> &k)
//  so it serves a mass matrix (factored once by the caller), a diagonal DG mass, or
//  a plain ODE (M = I) alike.  One step of an s-stage explicit tableau (A strictly
//  lower triangular):
//
//      U_i = u + dt sum_{j<i} a_ij k_j,     k_i = rate(t + c_i dt, U_i),
//      u  <- u + dt sum_i b_i k_i.
//
//  Named methods: "euler" (order 1), "midpoint" = "rk2" (order 2), "heun" (order 2,
//  the explicit trapezoidal rule), "ralston" (order 2, smallest error constant),
//  "rk3" (Kutta's third order), "rk4" (the classical fourth order method) and "3/8"
//  (Kutta's 3/8 rule, order 4).  Any other explicit tableau goes in as (A, b, c).
//
#ifndef FEMD_TIMESTEP_EXPLICIT_RK_HPP
#define FEMD_TIMESTEP_EXPLICIT_RK_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

struct ExplicitTableau {
    int s = 0;                       ///< stages
    int order = 0;                   ///< classical order (0 when not known)
    std::string name;
    std::vector<double> A;           ///< s x s row-major, strictly lower triangular
    std::vector<double> b, c;

    double a(int i, int j) const { return A[static_cast<std::size_t>(i) * s + j]; }

    /// @brief Check shape and explicitness; fill c = row sums of A when it is empty.
    void validate()
    {
        if (s < 1) throw std::invalid_argument("explicit RK: at least one stage");
        if (static_cast<int>(A.size()) != s * s || static_cast<int>(b.size()) != s)
            throw std::invalid_argument("explicit RK: A must be s x s and b of length s");
        for (int i = 0; i < s; ++i)
            for (int j = i; j < s; ++j)
                if (a(i, j) != 0.0)
                    throw std::invalid_argument("explicit RK: A must be strictly lower triangular (a_" + std::to_string(i) + std::to_string(j) +
                                                " != 0); an implicit tableau belongs to fd.IRK");
        if (c.empty())
        {
            c.assign(static_cast<std::size_t>(s), 0.0);
            for (int i = 0; i < s; ++i) for (int j = 0; j < i; ++j) c[i] += a(i, j);
        }
        if (static_cast<int>(c.size()) != s) throw std::invalid_argument("explicit RK: c must have length s");
        double sb = 0.0;
        for (double v : b) sb += v;
        if (std::abs(sb - 1.0) > 1e-12) throw std::invalid_argument("explicit RK: the weights b must sum to 1 (consistency)");
    }

    static ExplicitTableau named(std::string m)
    {
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        ExplicitTableau T;
        auto set = [&T](int s, int order, const char *name, std::vector<double> A, std::vector<double> b) {
            T.s = s; T.order = order; T.name = name; T.A = std::move(A); T.b = std::move(b); T.c.clear();
        };
        if (m == "euler" || m == "forward euler" || m == "rk1")
            set(1, 1, "euler", {0.0}, {1.0});
        else if (m == "midpoint" || m == "rk2" || m == "explicit midpoint")
            set(2, 2, "midpoint", {0.0, 0.0, 0.5, 0.0}, {0.0, 1.0});
        else if (m == "heun" || m == "trapezoid" || m == "explicit trapezoid")
            set(2, 2, "heun", {0.0, 0.0, 1.0, 0.0}, {0.5, 0.5});
        else if (m == "ralston")
            set(2, 2, "ralston", {0.0, 0.0, 2.0 / 3.0, 0.0}, {0.25, 0.75});
        else if (m == "rk3" || m == "kutta3")
            set(3, 3, "rk3", {0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -1.0, 2.0, 0.0}, {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0});
        else if (m == "rk4" || m == "classical" || m == "classical rk4")
            set(4, 4, "rk4", {0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1.0, 0}, {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0});
        else if (m == "3/8" || m == "rk38" || m == "3/8 rule")
            set(4, 4, "3/8", {0, 0, 0, 0, 1.0 / 3.0, 0, 0, 0, -1.0 / 3.0, 1.0, 0, 0, 1.0, -1.0, 1.0, 0}, {0.125, 0.375, 0.375, 0.125});
        else
            throw std::invalid_argument("explicit RK: unknown method '" + m + "'; use euler, midpoint (rk2), heun, ralston, rk3, rk4 "
                                        "or 3/8, or give a tableau (A, b)");
        T.validate();
        return T;
    }
};

class ExplicitRK {
public:
    explicit ExplicitRK(ExplicitTableau T) : T_(std::move(T)) { T_.validate(); }
    explicit ExplicitRK(const std::string &name) : T_(ExplicitTableau::named(name)) {}

    const ExplicitTableau &tableau() const { return T_; }

    /**
     * @brief One step, u overwritten with the value at t + dt.
     * @param rate  void(double t, const std::vector<double> &U, std::vector<double> &k), k = M^{-1} f(t, U)
     */
    template <class Rate>
    void step(std::vector<double> &u, double t, double dt, Rate &&rate)
    {
        const std::size_t n = u.size();
        const int s = T_.s;
        if (k_.size() != static_cast<std::size_t>(s)) k_.assign(static_cast<std::size_t>(s), std::vector<double>());
        U_.resize(n);
        for (int i = 0; i < s; ++i)
        {
            // U_i = u + dt sum_{j<i} a_ij k_j  (U_0 = u, no copy of the sum when the row is zero)
            std::copy(u.begin(), u.end(), U_.begin());
            for (int j = 0; j < i; ++j)
            {
                const double w = dt * T_.a(i, j);
                if (w == 0.0) continue;
                const std::vector<double> &kj = k_[j];
                for (std::size_t q = 0; q < n; ++q) U_[q] += w * kj[q];
            }
            rate(t + T_.c[i] * dt, U_, k_[i]);
            if (k_[i].size() != n) throw std::invalid_argument("explicit RK: the rate returned the wrong length");
        }
        for (int i = 0; i < s; ++i)
        {
            const double w = dt * T_.b[i];
            if (w == 0.0) continue;
            const std::vector<double> &ki = k_[i];
            for (std::size_t q = 0; q < n; ++q) u[q] += w * ki[q];
        }
    }

private:
    ExplicitTableau T_;
    std::vector<std::vector<double>> k_;
    std::vector<double> U_;
};

} // namespace femd

#endif // FEMD_TIMESTEP_EXPLICIT_RK_HPP
