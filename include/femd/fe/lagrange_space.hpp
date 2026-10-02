//
//  lagrange_space.hpp  --  classical C^0 nodal Lagrange family.
//
//  Raw dofs are the global nodes e*p + l.  With "free" ends (the default) the
//  coefficients ARE nodal values.  Endpoint conditions come from the
//  family-agnostic builder in boundary_condition.hpp; periodicity identifies
//  the last raw node with the first.
//
//  Nodes: the p+1 Gauss-Lobatto points of each element (the default) or equally spaced
//  ones.  The space is the same either way; the basis, its conditioning and the
//  interpolant differ at high degree (manual, Section 3.2).
//
//  raw_eval: the values L_j(xi) from the product formula with barycentric weights, and
//  derivatives of ANY order m through the differentiation matrix at the nodes,
//  L_l^{(m)}(xi) = sum_j L_j(xi) (D^m)_{jl} with D_{jl} = L_l'(x_j), exact for polynomials
//  of degree p and free of the monomial expansion's ill conditioning at high degree.
//  Scaled by (2/h_e)^m.  LagrangeElement (inherited from Poseidon) supplies the node set
//  and serves as an independent cross-check in the tests.
//
#ifndef FEMD_FE_LAGRANGE_SPACE_HPP
#define FEMD_FE_LAGRANGE_SPACE_HPP

#include "femd/fe/function_space.hpp"
#include "femd/fe/lagrange_element.hpp"
#include <vector>

namespace femd {

class LagrangeSpace : public FunctionSpace {
public:
    LagrangeSpace(const Mesh1D &mesh, int degree, const BCSpec &bc = BCSpec::free(), bool equispaced = true)
        : FunctionSpace(mesh, degree, mesh.nelem() * degree + 1), elem_(degree, equispaced)
    {
        build_tables();
        apply_boundary_conditions(bc);
    }

    bool equispaced() const { return elem_.equispaced(); }

    /// @brief Periodic: raw node n-1 is identified with raw node 0.
    ConstraintOperator make_periodic() const override
    {
        int n = raw_dim();
        std::vector<ConstraintOperator::Row> rows(n - 1);
        rows[0].push_back(std::make_pair(0, 1.0));
        rows[0].push_back(std::make_pair(n - 1, 1.0));
        for (int i = 1; i < n - 1; ++i) rows[i].push_back(std::make_pair(i, 1.0));
        return ConstraintOperator(n, std::move(rows));
    }

    const LagrangeElement &element() const { return elem_; }

    int raw_dof(int e, int l) const override { return e * degree() + l; }

    double raw_coordinate(int i) const override
    {
        int p = degree(), ne = nelem();
        if (i == raw_dim() - 1) return mesh().b();
        int e = i / p, l = i - e * p;
        if (e > ne - 1) { e = ne - 1; l = p; }
        return mesh().from_reference(e, elem_.refnodes()[l]);
    }

    void raw_eval(int e, double x, int nder, double *out) const override
    {
        raw_eval_ref(e, mesh().to_reference(e, x), nder, out);
    }

    void raw_eval_ref(int e, double xi, int nder, double *out) const override
    {
        const int p = degree(), nl = p + 1;
        const double s = 2.0 / mesh().h(e);                 // d xi / dx
        const std::vector<double> &nd = elem_.refnodes();
        double Lv[64];
        std::vector<double> Lbuf;
        double *L = Lv;
        if (nl > 64) { Lbuf.resize(static_cast<std::size_t>(nl)); L = Lbuf.data(); }
        for (int j = 0; j < nl; ++j)
        {
            double v = w_[j];
            for (int m = 0; m < nl; ++m) if (m != j) v *= (xi - nd[m]);
            L[j] = v;
        }
        double scale = 1.0;
        for (int m = 0; m <= nder; ++m)
        {
            double *o = out + static_cast<std::size_t>(m) * nl;
            if (m == 0) for (int l = 0; l < nl; ++l) o[l] = L[l];
            else if (m > p) for (int l = 0; l < nl; ++l) o[l] = 0.0;
            else
            {
                const double *Dm = &Dpow_[static_cast<std::size_t>(m) * nl * nl];
                for (int l = 0; l < nl; ++l)
                {
                    double v = 0.0;
                    for (int j = 0; j < nl; ++j) v += L[j] * Dm[static_cast<std::size_t>(j) * nl + l];
                    o[l] = v * scale;
                }
            }
            scale *= s;
        }
    }

    /// @brief Coordinates of the raw global nodes (== adapted dofs for the free space).
    std::vector<double> nodes() const
    {
        int p = degree();
        std::vector<double> x(raw_dim());
        for (int e = 0; e < nelem(); ++e)
            for (int l = 0; l <= p; ++l)
                x[raw_dof(e, l)] = mesh().from_reference(e, elem_.refnodes()[l]);
        x.back() = mesh().b();
        return x;
    }

private:
    /// Barycentric weights w_j = 1 / prod_{m != j} (x_j - x_m), and the powers D^m (m = 0..p) of the
    /// differentiation matrix D_{jl} = L_l'(x_j) (row-major, nl x nl each).
    void build_tables()
    {
        const int p = degree(), nl = p + 1;
        const std::vector<double> &nd = elem_.refnodes();
        w_.assign(static_cast<std::size_t>(nl), 1.0);
        for (int j = 0; j < nl; ++j)
            for (int m = 0; m < nl; ++m) if (m != j) w_[j] /= (nd[j] - nd[m]);
        std::vector<double> D(static_cast<std::size_t>(nl) * nl, 0.0);
        for (int j = 0; j < nl; ++j)
        {
            double diag = 0.0;
            for (int l = 0; l < nl; ++l)
            {
                if (l == j) continue;
                const double v = (w_[l] / w_[j]) / (nd[j] - nd[l]);
                D[static_cast<std::size_t>(j) * nl + l] = v;
                diag -= v;                                  // rows of D sum to 0 (derivative of 1)
            }
            D[static_cast<std::size_t>(j) * nl + j] = diag;
        }
        Dpow_.assign(static_cast<std::size_t>(nl) * nl * nl, 0.0);
        for (int i = 0; i < nl; ++i) Dpow_[static_cast<std::size_t>(i) * nl + i] = 1.0;     // D^0 = I
        for (int m = 1; m <= p; ++m)
        {
            const double *A = &Dpow_[static_cast<std::size_t>(m - 1) * nl * nl];
            double *C = &Dpow_[static_cast<std::size_t>(m) * nl * nl];
            for (int i = 0; i < nl; ++i)                    // D^m = D * D^{m-1}: L^{(m)} at the nodes from L^{(m-1)}
                for (int l = 0; l < nl; ++l)
                {
                    double v = 0.0;
                    for (int k = 0; k < nl; ++k) v += D[static_cast<std::size_t>(i) * nl + k] * A[static_cast<std::size_t>(k) * nl + l];
                    C[static_cast<std::size_t>(i) * nl + l] = v;
                }
        }
    }

    LagrangeElement elem_;
    std::vector<double> w_, Dpow_;   // barycentric weights, D^0..D^p
};

} // namespace femd

#endif // FEMD_FE_LAGRANGE_SPACE_HPP
