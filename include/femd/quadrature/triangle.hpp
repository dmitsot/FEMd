//
//  triangle.hpp  --  quadrature on the reference triangle T = {xi, eta >= 0, xi + eta <= 1}.
//
//  TriangleQuadrature(d) is exact for polynomials of total degree <= d.  Weights sum
//  to |T| = 1/2.  Low degrees use symmetric rules (1, 3 and Radon's 7 points, exact to
//  degrees 1, 2 and 5).  Above that, the collapsed (Duffy / Stroud conical product)
//  rule: Gauss-Legendre in u and v on the unit square mapped by xi = u (1 - v), eta = v,
//  whose Jacobian 1 - v is integrated exactly by one extra point in v.  It is exact at
//  any degree, with ceil((d+1)/2) * ceil((d+2)/2) points.
//
#ifndef FEMD_QUADRATURE_TRIANGLE_HPP
#define FEMD_QUADRATURE_TRIANGLE_HPP

#include "femd/quadrature/gauss_legendre.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class TriangleQuadrature {
public:
    explicit TriangleQuadrature(int degree) : degree_(degree < 0 ? 0 : degree)
    {
        const int d = degree_;
        if (d <= 1)
            add(1.0 / 3.0, 1.0 / 3.0, 0.5);
        else if (d == 2)
        {
            add(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0);
            add(2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0);
            add(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0);
        }
        else if (d <= 5)
        {
            const double s15 = std::sqrt(15.0);
            add(1.0 / 3.0, 1.0 / 3.0, 9.0 / 80.0);
            const double a1 = (6.0 - s15) / 21.0, w1 = (155.0 - s15) / 2400.0;
            const double a2 = (6.0 + s15) / 21.0, w2 = (155.0 + s15) / 2400.0;
            add(a1, a1, w1); add(1.0 - 2.0 * a1, a1, w1); add(a1, 1.0 - 2.0 * a1, w1);
            add(a2, a2, w2); add(1.0 - 2.0 * a2, a2, w2); add(a2, 1.0 - 2.0 * a2, w2);
        }
        else
        {
            const int nu = (d + 2) / 2, nv = (d + 3) / 2;          // ceil((d+1)/2), ceil((d+2)/2)
            GaussLegendre gu(nu), gv(nv);
            for (int j = 0; j < nv; ++j)
            {
                const double v = 0.5 * (gv.node(j) + 1.0), wv = 0.5 * gv.weight(j);
                for (int i = 0; i < nu; ++i)
                {
                    const double u = 0.5 * (gu.node(i) + 1.0), wu = 0.5 * gu.weight(i);
                    add(u * (1.0 - v), v, wu * wv * (1.0 - v));
                }
            }
        }
    }

    int degree() const { return degree_; }
    int size() const { return static_cast<int>(w_.size()); }
    double xi(int q)  const { return xi_[q]; }
    double eta(int q) const { return eta_[q]; }
    double weight(int q) const { return w_[q]; }

private:
    void add(double x, double y, double w) { xi_.push_back(x); eta_.push_back(y); w_.push_back(w); }
    int degree_;
    std::vector<double> xi_, eta_, w_;
};

} // namespace femd

#endif // FEMD_QUADRATURE_TRIANGLE_HPP
