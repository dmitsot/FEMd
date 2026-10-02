//
//  lagrange_element.hpp  --  Lagrange reference element of arbitrary degree on [-1,1].
//
//  Gauss-Lobatto nodes (the default) or equally spaced ones, endpoints included.  Provides:
//    * direct evaluation phi_at / dphi_at at any reference point xi
//    * precomputed tables phi(q,a), dphi_ref(q,a) at quadrature points, so the
//      assembly hot loop just reads from arrays (the efficiency goal).
//
//  Derivatives here are d/dxi on the reference element; the assembler converts
//  to physical via the element Jacobian.
//
#ifndef FEMD_FE_LAGRANGE_ELEMENT_HPP
#define FEMD_FE_LAGRANGE_ELEMENT_HPP

#include "femd/quadrature/gauss_legendre.hpp"
#include <cmath>
#include <vector>
#include <cassert>

namespace femd {

class LagrangeElement {
public:
    /**
     * @brief Build a Lagrange reference element of a given degree on [-1,1].
     * @param degree      polynomial degree p >= 1
     * @param equispaced  true (the default): p+1 equally spaced nodes. false: the p+1 Gauss-Lobatto points,
     *                    symmetrized so that node p-a is exactly -(node a).  Both include the endpoints.
     */
    explicit LagrangeElement(int degree, bool equispaced = true) : degree_(degree), nodes_(degree + 1), equi_(equispaced)
    {
        assert(degree >= 1);
        if (degree == 1) { nodes_ = {-1.0, 1.0}; return; }
        if (equispaced)
            for (int a = 0; a <= degree; ++a) nodes_[a] = -1.0 + 2.0 * a / degree_;
        else
        {
            nodes_ = gll(degree);
            for (int a = 0; a <= degree; ++a) { double s = 0.5 * (nodes_[a] - nodes_[degree - a]); nodes_[a] = s; }
            nodes_[0] = -1.0; nodes_[degree] = 1.0;
        }
    }

    bool equispaced() const { return equi_; }

    /// @brief Polynomial degree. @return p.
    int degree() const { return degree_; }
    /// @brief Number of local basis functions. @return p+1.
    int nloc() const { return degree_ + 1; }
    /// @brief The reference nodes on [-1,1]. @return const ref to the p+1 node coordinates.
    const std::vector<double> &refnodes() const { return nodes_; }

    // ---- direct evaluation (used for point evaluation of FE functions) -----
    /// @brief Value of local basis a at a reference point. @param xi point in [-1,1]. @param a local index. @return phi_a(xi).
    double phi_at(double xi, int a) const { return lag(xi, nodes_, a); }
    /// @brief Reference derivative of local basis a. @param xi point in [-1,1]. @param a local index. @return phi_a'(xi).
    double dphi_at(double xi, int a) const { return dlag(xi, nodes_, a); }

    // ---- precomputed tables against a quadrature rule ----------------------
    /**
     * @brief Precompute basis values and reference derivatives at quadrature nodes.
     * @param q  the quadrature rule to tabulate against.
     * @post phi(iq,a) and dphi_ref(iq,a) become valid for all quad points and local nodes.
     */
    void tabulate(const GaussLegendre &q)
    {
        nq_ = q.size();
        int nl = nloc();
        phi_.assign(nq_ * nl, 0.0);
        dphi_.assign(nq_ * nl, 0.0);
        for (int iq = 0; iq < nq_; ++iq)
        {
            double xi = q.node(iq);
            for (int a = 0; a < nl; ++a)
            {
                phi_[iq * nl + a] = lag(xi, nodes_, a);
                dphi_[iq * nl + a] = dlag(xi, nodes_, a);
            }
        }
    }

    /// @brief Tabulated basis value at quadrature node iq. @param iq quad index. @param a local index. @return phi_a at node iq.
    double phi(int iq, int a)      const { return phi_[iq * nloc() + a]; }
    /// @brief Tabulated reference derivative at quad node iq. @param iq quad index. @param a local index. @return phi_a' at node iq.
    double dphi_ref(int iq, int a) const { return dphi_[iq * nloc() + a]; }

private:
    int degree_;
    std::vector<double> nodes_;     // p+1 nodes on [-1,1]
    bool equi_ = false;

    /// The p+1 Gauss-Lobatto points on [-1,1] (the ends and the roots of P_p'), by Newton on
    /// (1 - x^2) P_p'(x) from Chebyshev-Gauss-Lobatto starting points.
    static std::vector<double> gll(int p)
    {
        std::vector<double> x(static_cast<std::size_t>(p) + 1);
        for (int i = 0; i <= p; ++i)
        {
            double z = -std::cos(3.14159265358979323846 * i / p);
            if (i > 0 && i < p)
                for (int it = 0; it < 100; ++it)
                {
                    // P_p, P_{p-1} by the three-term recurrence; q = (1 - z^2) P_p' = p (P_{p-1} - z P_p)
                    double P0 = 1.0, P1 = z;
                    for (int n = 2; n <= p; ++n) { double P2 = ((2.0 * n - 1) * z * P1 - (n - 1.0) * P0) / n; P0 = P1; P1 = P2; }
                    const double q = p * (P0 - z * P1);
                    // q' = -p (p + 1) P_p (Legendre equation)
                    const double dq = -p * (p + 1.0) * P1;
                    const double dz = q / dq;
                    z -= dz;
                    if (std::abs(dz) < 1e-16) break;
                }
            x[i] = z;
        }
        x[0] = -1.0; x[p] = 1.0;
        return x;
    }
    int nq_ = 0;
    std::vector<double> phi_, dphi_; // nq x nloc, row-major

    /// @brief Lagrange basis i at xi over node set nd. @param xi point. @param nd nodes. @param i basis index. @return L_i(xi).
    static double lag(double xi, const std::vector<double> &nd, int i)
    {
        double p = 1.0;
        for (int j = 0; j < (int) nd.size(); ++j)
            if (j != i) p *= (xi - nd[j]) / (nd[i] - nd[j]);
        return p;
    }

    /// @brief Derivative of Lagrange basis i at xi. @param xi point. @param nd nodes. @param i basis index. @return L_i'(xi).
    static double dlag(double xi, const std::vector<double> &nd, int i)
    {
        double sum = 0.0;
        int n = (int) nd.size();
        for (int k = 0; k < n; ++k)
        {
            if (k == i) continue;
            double term = 1.0 / (nd[i] - nd[k]);
            for (int j = 0; j < n; ++j)
                if (j != i && j != k) term *= (xi - nd[j]) / (nd[i] - nd[j]);
            sum += term;
        }
        return sum;
    }
};

} // namespace femd

#endif // FEMD_FE_LAGRANGE_ELEMENT_HPP
