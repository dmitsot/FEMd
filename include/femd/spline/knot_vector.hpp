//
//  knot_vector.hpp  --  knot vector built FROM A GRID, never user input.
//
//  Grid a = x_0 < ... < x_ne = b, degree p, interior multiplicities r_j (from
//  the requested continuity k_j through r_j = p - k_j).  Two flavors:
//
//    Open:     [a]*(p+1)  +  interior knots with multiplicity  +  [b]*(p+1)
//    Periodic: one period  P = [x_0] + interior knots  (M = 1 + sum r_j knots),
//              extended by p knots below (shifted by -L) and p+1 above (+L), so
//              that xi_{i+M} = xi_i + L and B_{j+M}(x) = B_j(x - L).
//
//  In both cases the knot vector has M + 2p + 1 entries, the raw basis has
//  n_raw = M + p functions, and element e occupies the span starting at knot
//  index  span(e) = p + sum_{j<=e} r_j.
//
#ifndef FEMD_SPLINE_KNOT_VECTOR_HPP
#define FEMD_SPLINE_KNOT_VECTOR_HPP

#include "femd/mesh/mesh1d.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class KnotVector {
public:
    enum class Flavor { Open, Periodic };

    /**
     * @param mesh        the grid
     * @param degree      p >= 1
     * @param flavor      Open or Periodic
     * @param continuity  default continuity k at interior nodes, -1 => p-1 (maximal)
     * @param at          per-node overrides {node index in 1..ne-1 : continuity}
     */
    KnotVector(const Mesh1D &mesh, int degree, Flavor flavor, int continuity = -1,
               const std::map<int, int> &at = {})
        : p_(degree), flavor_(flavor)
    {
        if (degree < 1) throw std::invalid_argument("KnotVector: degree must be >= 1");
        // -1 is the one sentinel for "maximal"; any other negative value is an error, not a default
        if (continuity < -1 || continuity > degree - 1)
            throw std::invalid_argument("KnotVector: continuity " + std::to_string(continuity) +
                                        " must lie in [0, p-1] = [0, " + std::to_string(degree - 1) +
                                        "], or be -1 for the default p-1");
        int ne = mesh.nelem();
        int kdef = continuity < 0 ? degree - 1 : continuity;
        mult_.assign(ne + 1, 0);                        // mult_[j] for interior j = 1..ne-1
        for (int j = 1; j < ne; ++j)
        {
            int k = kdef;
            auto it = at.find(j);
            if (it != at.end()) k = it->second;
            if (k < 0 || k > degree - 1)
                throw std::invalid_argument("KnotVector: continuity at node " + std::to_string(j) +
                                            " must lie in [0, p-1] = [0, " + std::to_string(degree - 1) + "]");
            mult_[j] = degree - k;
        }
        for (const auto &kv : at)
            if (kv.first < 1 || kv.first > ne - 1)
                throw std::invalid_argument("KnotVector: continuity override at node " + std::to_string(kv.first) + " is not an interior node");

        // one period / the interior core
        std::vector<double> core;                       // interior knots with multiplicity
        for (int j = 1; j < ne; ++j) for (int r = 0; r < mult_[j]; ++r) core.push_back(mesh.vertices()[j]);
        M_ = 1 + static_cast<int>(core.size());
        double a = mesh.a(), b = mesh.b(), L = b - a;

        if (flavor == Flavor::Open)
        {
            knots_.assign(degree + 1, a);
            knots_.insert(knots_.end(), core.begin(), core.end());
            knots_.insert(knots_.end(), degree + 1, b);
        }
        else
        {
            std::vector<double> P;                      // one period, M knots
            P.push_back(a);
            P.insert(P.end(), core.begin(), core.end());
            // indices -p .. M+p
            for (int i = -degree; i <= M_ + degree; ++i)
            {
                int q = i, shift = 0;
                while (q < 0)   { q += M_; --shift; }
                while (q >= M_) { q -= M_; ++shift; }
                knots_.push_back(P[q] + shift * L);
            }
        }
        // span index of each element
        span_.assign(ne, 0);
        int acc = degree;
        for (int e = 0; e < ne; ++e)
        {
            if (e > 0) acc += mult_[e];
            span_[e] = acc;
        }
    }

    int degree() const { return p_; }
    Flavor flavor() const { return flavor_; }
    bool periodic() const { return flavor_ == Flavor::Periodic; }
    const std::vector<double> &knots() const { return knots_; }
    int size() const { return static_cast<int>(knots_.size()); }
    /// @brief Number of raw basis functions, M + p.
    int n_raw() const { return M_ + p_; }
    /// @brief Knots per period (periodic) or interior core size plus one (open). The periodic dimension.
    int M() const { return M_; }
    /// @brief Knot index where element e's span starts:  knots[span(e)] = x_e, knots[span(e)+1] = x_{e+1}.
    int span(int e) const { return span_[e]; }
    /// @brief Multiplicity of interior node j (1..ne-1).
    int multiplicity(int j) const { return mult_[j]; }
    /// @brief Greville abscissa of raw function j.
    double greville(int j) const
    {
        double s = 0.0;
        for (int i = 1; i <= p_; ++i) s += knots_[j + i];
        return s / p_;
    }

private:
    int p_;
    Flavor flavor_;
    int M_ = 0;
    std::vector<double> knots_;
    std::vector<int> mult_;
    std::vector<int> span_;
};

} // namespace femd

#endif // FEMD_SPLINE_KNOT_VECTOR_HPP
