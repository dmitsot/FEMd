//
//  gauss_lobatto.hpp  --  Gauss-Lobatto quadrature on the reference [-1,1].
//
//  n >= 2 points: the ends and the n-2 roots of P_{n-1}', with weights
//  w_i = 2 / (n (n-1) P_{n-1}(x_i)^2).  Exact for polynomials up to degree 2n-3.
//  The nodes are those of LagrangeElement(n-1, lobatto), bit for bit, so a nodal
//  basis on them is exactly the identity at the quadrature points and the mass
//  matrix of the rule is exactly diagonal (the lumped mass).
//
#ifndef FEMD_QUADRATURE_GAUSS_LOBATTO_HPP
#define FEMD_QUADRATURE_GAUSS_LOBATTO_HPP

#include "femd/fe/lagrange_element.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class GaussLobatto {
public:
    /// @brief The n-point Gauss-Lobatto rule on [-1,1]. @throws std::out_of_range if npts < 2.
    explicit GaussLobatto(int npts) : n_(npts)
    {
        if (npts < 2) throw std::out_of_range("GaussLobatto: need at least 2 points, got " + std::to_string(npts));
        const int p = npts - 1;
        x_ = LagrangeElement(p, false).refnodes();
        w_.resize(npts);
        for (int i = 0; i < npts; ++i)
        {
            const double z = x_[i];
            double P0 = 1.0, P1 = z;                              // P_p(z) by the three-term recurrence
            for (int k = 2; k <= p; ++k) { double P2 = ((2.0 * k - 1) * z * P1 - (k - 1.0) * P0) / k; P0 = P1; P1 = P2; }
            const double Pp = (p == 0) ? 1.0 : P1;
            w_[i] = 2.0 / (p * (p + 1.0) * Pp * Pp);
        }
    }

    int size() const { return n_; }
    double node(int q)   const { return x_[q]; }
    double weight(int q) const { return w_[q]; }

private:
    int n_;
    std::vector<double> x_, w_;
};

} // namespace femd

#endif // FEMD_QUADRATURE_GAUSS_LOBATTO_HPP
