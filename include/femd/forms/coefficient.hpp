//
//  coefficient.hpp  --  the scalar field c(x) in  integral c * (D^a N_i)(D^b N_j).
//
//  Three ways to give it, one way to consume it: at_quad(Q) samples it at the
//  cache nodes into a flat array [e*nq + q].  Kernels only ever see that array,
//  which is what makes a nonlinearity (values of u_h at the nodes) and a
//  constant identical to the assembler (design doc, Section 15.1).
//
#ifndef FEMD_FORMS_COEFFICIENT_HPP
#define FEMD_FORMS_COEFFICIENT_HPP

#include "femd/fe/quadrature_cache.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

namespace femd {

class Coefficient {
public:
    Coefficient() : kind_(Const), c_(1.0) {}
    Coefficient(double c) : kind_(Const), c_(c) {}                          // NOLINT implicit on purpose
    Coefficient(std::function<double(double)> f) : kind_(Func), f_(std::move(f)) {}
    /// @brief Values already at the quadrature nodes, laid out [e*nq + q].
    explicit Coefficient(std::vector<double> at_nodes) : kind_(Nodes), v_(std::move(at_nodes)) {}

    bool is_constant() const { return kind_ == Const; }
    double constant() const { return c_; }

    std::vector<double> at_quad(const QuadratureCache &Q) const
    {
        std::size_t n = static_cast<std::size_t>(Q.nelem()) * Q.nq();
        if (kind_ == Nodes)
        {
            if (v_.size() != n) throw std::invalid_argument("Coefficient: node array length does not match the cache");
            return v_;
        }
        std::vector<double> out(n);
        if (kind_ == Const) { for (auto &x : out) x = c_; return out; }
        for (int e = 0; e < Q.nelem(); ++e)
            for (int q = 0; q < Q.nq(); ++q) out[e * Q.nq() + q] = f_(Q.x(e, q));
        return out;
    }

private:
    enum Kind { Const, Func, Nodes } kind_;
    double c_ = 0.0;
    std::function<double(double)> f_;
    std::vector<double> v_;
};

} // namespace femd

#endif // FEMD_FORMS_COEFFICIENT_HPP
