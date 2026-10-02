//
//  dg_space.hpp  --  discontinuous piecewise polynomials of degree p >= 0.
//
//  Each element carries its own p+1 functions and nothing is shared across a
//  vertex, so the raw basis IS the adapted basis (C is the identity) and the
//  dimension is nelem*(p+1).  Two local bases:
//
//      Legendre   P_0..P_p mapped to the element, P_l(1) = 1 and P_l(-1) = (-1)^l.
//                 Orthogonal: the mass matrix is diagonal, (h/2) * 2/(2l+1).
//      Lobatto    the nodal (Lagrange) basis on the p+1 Gauss-Lobatto points, so a
//                 coefficient is the value at a node, and the two end nodes of an
//                 element are its traces.  Needs p >= 1.
//
//  Periodicity does not change the basis at all: it only adds the seam between the
//  last and the first element as an interior facet (facet_cache.hpp).  The space is
//  then flagged periodic, so its matrices keep the seam's coupling as cyclic corners
//  and the cyclic solvers apply.  There are no built-in endpoint conditions: data
//  enter weakly, through ds.
//
//  Facet terms couple the dofs of neighbouring elements, so the half-bandwidth is
//  widened from p (one element) to 2p+1 (two neighbours).
//
#ifndef FEMD_FE_DG_SPACE_HPP
#define FEMD_FE_DG_SPACE_HPP

#include "femd/fe/function_space.hpp"
#include "femd/quadrature/gauss_legendre.hpp"
#include "femd/util/omp.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

namespace detail {

/// @brief P_n^{(m)}(x) for n = 0..p, m = 0..nder, row-major out[m*(p+1) + n].
inline void legendre_table(int p, int nder, double x, double *out)
{
    const int nl = p + 1;
    for (int m = 0; m <= nder; ++m)
        for (int n = 0; n <= p; ++n)
        {
            double v;
            if (n == 0) v = (m == 0) ? 1.0 : 0.0;
            else
            {
                // n P_n^(m) = (2n-1) (x P_{n-1}^(m) + m P_{n-1}^(m-1)) - (n-1) P_{n-2}^(m)
                double a = out[m * nl + n - 1];
                double b = m > 0 ? out[(m - 1) * nl + n - 1] : 0.0;
                double c = n >= 2 ? out[m * nl + n - 2] : 0.0;
                v = ((2 * n - 1) * (x * a + m * b) - (n - 1) * c) / n;
            }
            out[m * nl + n] = v;
        }
}

/// @brief The p+1 Gauss-Lobatto points on [-1,1]: the ends and the roots of P_p'.
inline std::vector<double> gauss_lobatto_nodes(int p)
{
    std::vector<double> x(p + 1);
    if (p < 1) throw std::invalid_argument("Gauss-Lobatto nodes need p >= 1");
    x[0] = -1.0; x[p] = 1.0;
    std::vector<double> t(3 * (p + 1));
    for (int i = 1; i < p; ++i)
    {
        double z = -std::cos(3.14159265358979323846 * i / p);     // Chebyshev-Gauss-Lobatto start
        for (int it = 0; it < 100; ++it)
        {
            legendre_table(p, 2, z, t.data());
            double d = t[1 * (p + 1) + p] / t[2 * (p + 1) + p];    // P_p' / P_p''
            z -= d;
            if (std::fabs(d) < 1e-16) break;
        }
        x[i] = z;
    }
    return x;
}

/// @brief Inverse of a small dense n x n matrix (row-major), Gauss-Jordan with partial pivoting.
inline std::vector<double> small_inverse(std::vector<double> A, int n)
{
    std::vector<double> I(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;
    for (int c = 0; c < n; ++c)
    {
        int piv = c;
        for (int r = c + 1; r < n; ++r) if (std::fabs(A[r * n + c]) > std::fabs(A[piv * n + c])) piv = r;
        if (A[piv * n + c] == 0.0) throw std::runtime_error("small_inverse: singular matrix");
        if (piv != c)
            for (int k = 0; k < n; ++k) { std::swap(A[c * n + k], A[piv * n + k]); std::swap(I[c * n + k], I[piv * n + k]); }
        double d = 1.0 / A[c * n + c];
        for (int k = 0; k < n; ++k) { A[c * n + k] *= d; I[c * n + k] *= d; }
        for (int r = 0; r < n; ++r)
        {
            if (r == c) continue;
            double f = A[r * n + c];
            if (f == 0.0) continue;
            for (int k = 0; k < n; ++k) { A[r * n + k] -= f * A[c * n + k]; I[r * n + k] -= f * I[c * n + k]; }
        }
    }
    return I;
}

} // namespace detail

class DGSpace : public FunctionSpace {
public:
    enum class Basis { Legendre, Lobatto };

    DGSpace(const Mesh1D &mesh, int degree, bool periodic = false, Basis basis = Basis::Legendre)
        : FunctionSpace(mesh, degree, mesh.nelem() * (degree + 1), /*min_degree=*/0), basis_(basis)
    {
        const int p = degree, nl = p + 1;
        if (basis_ == Basis::Lobatto)
        {
            if (p < 1) throw std::invalid_argument("DGSpace: the Lobatto basis needs degree >= 1");
            ref_ = detail::gauss_lobatto_nodes(p);
            std::vector<double> V(static_cast<std::size_t>(nl) * nl), t(nl);
            for (int i = 0; i < nl; ++i)
            {
                detail::legendre_table(p, 0, ref_[i], t.data());
                for (int j = 0; j < nl; ++j) V[i * nl + j] = t[j];
            }
            vinv_ = detail::small_inverse(V, nl);             // phi_l = sum_j vinv[j][l] P_j
        }
        else
        {
            GaussLegendre rule(nl);
            ref_.resize(nl);
            for (int i = 0; i < nl; ++i) ref_[i] = rule.node(i);
            std::vector<double> V(static_cast<std::size_t>(nl) * nl), t(nl);
            for (int i = 0; i < nl; ++i)
            {
                detail::legendre_table(p, 0, ref_[i], t.data());
                for (int j = 0; j < nl; ++j) V[i * nl + j] = t[j];
            }
            vinv_ = detail::small_inverse(V, nl);             // coefficients from values at the Gauss points
        }
        apply_boundary_conditions(periodic ? BCSpec::make_periodic() : BCSpec::free());

        // facet coupling: the dofs of neighbouring elements, and across the seam when periodic
        const int ne = nelem(), n = raw_dim();
        int u = 0;
        auto pair = [&](int e1, int e2) {
            for (int a = 0; a < nl; ++a)
                for (int b = 0; b < nl; ++b)
                {
                    int d = std::abs(raw_dof(e1, a) - raw_dof(e2, b));
                    if (periodic) d = std::min(d, n - d);
                    u = std::max(u, d);
                }
        };
        for (int e = 0; e + 1 < ne; ++e) pair(e, e + 1);
        if (periodic && ne > 1) pair(ne - 1, 0);
        widen_uband(u);
    }

    Basis basis() const { return basis_; }
    /// @brief The local reference points: Gauss points (Legendre) or Gauss-Lobatto nodes.
    const std::vector<double> &reference_points() const { return ref_; }

    int raw_dof(int e, int l) const override { return e * (degree() + 1) + l; }

    double raw_coordinate(int i) const override
    {
        int nl = degree() + 1, e = i / nl, l = i - e * nl;
        return mesh().from_reference(e, ref_[l]);
    }

    bool broken() const override { return true; }

    void raw_eval(int e, double x, int nder, double *out) const override
    {
        raw_eval_ref(e, mesh().to_reference(e, x), nder, out);
    }

    void raw_eval_ref(int e, double xi, int nder, double *out) const override
    {
        const int p = degree(), nl = p + 1;
        const double s = 2.0 / mesh().h(e);                  // d xi / dx
        thread_local std::vector<double> tab_;            // raw_eval runs in parallel (cache build)
        tab_.assign(static_cast<std::size_t>(nder + 1) * nl, 0.0);
        detail::legendre_table(p, nder, xi, tab_.data());
        double scale = 1.0;
        for (int m = 0; m <= nder; ++m)
        {
            for (int l = 0; l < nl; ++l)
            {
                double v;
                if (basis_ == Basis::Legendre) v = tab_[m * nl + l];
                else
                {
                    v = 0.0;
                    for (int j = 0; j < nl; ++j) v += vinv_[j * nl + l] * tab_[m * nl + j];
                }
                out[m * nl + l] = v * scale;
            }
            scale *= s;
        }
    }

    /**
     * @brief Coefficients of the element-wise interpolant from values at raw_coordinate(i),
     *        element by element: the identity for Lobatto (nodal), the inverse Vandermonde at
     *        the Gauss points for Legendre.
     */
    std::vector<double> interpolate_values(const std::vector<double> &vals) const
    {
        const int nl = degree() + 1, ne = nelem();
        if (static_cast<int>(vals.size()) != ne * nl) throw std::invalid_argument("interpolate_values: need nelem*(p+1) values");
        if (basis_ == Basis::Lobatto) return vals;
        std::vector<double> c(vals.size(), 0.0);
        FEMD_OMP_FOR_IF(detail::parallel_elements(ne, nl))
        for (int e = 0; e < ne; ++e)
            for (int j = 0; j < nl; ++j)
            {
                double v = 0.0;
                for (int i = 0; i < nl; ++i) v += vinv_[j * nl + i] * vals[e * nl + i];
                c[e * nl + j] = v;
            }
        return c;
    }

    /// @brief Periodicity leaves the basis alone: C is the identity.
    ConstraintOperator make_periodic() const override
    {
        int n = raw_dim();
        std::vector<ConstraintOperator::Row> rows(n);
        for (int i = 0; i < n; ++i) rows[i].push_back(std::make_pair(i, 1.0));
        return ConstraintOperator(n, std::move(rows));
    }

private:
    Basis basis_;
    std::vector<double> ref_, vinv_;
};

} // namespace femd

#endif // FEMD_FE_DG_SPACE_HPP
