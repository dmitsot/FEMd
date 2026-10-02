//
//  field.hpp  --  a member of a FunctionSpace: an adapted coefficient vector
//  plus point evaluation and sampling at the quadrature nodes.
//
//  Templated on Scalar (double, std::complex<double>, later a dual type) so
//  that the same evaluation code serves real fields, complex-step
//  differentiation and forward-mode AD.  Geometry stays real.
//
//  Coefficients are the ADAPTED ones (length V.dim()).  For LagrangeSpace they
//  are nodal values; for SplineSpace they are not.  The design doc (Section
//  10.2) keeps that difference visible in the API rather than hiding it.
//
#ifndef FEMD_FE_FIELD_HPP
#define FEMD_FE_FIELD_HPP

#include "femd/fe/function_space.hpp"
#include "femd/fe/quadrature_cache.hpp"
#include <vector>

namespace femd {

template <class Scalar = double>
class Field {
public:
    explicit Field(const FunctionSpace &V) : V_(&V), c_(V.dim(), Scalar(0)) {}
    Field(const FunctionSpace &V, std::vector<Scalar> coeffs) : V_(&V), c_(std::move(coeffs))
    {
        if (static_cast<int>(c_.size()) != V.dim()) throw std::invalid_argument("Field: coefficient length != V.dim()");
    }

    const FunctionSpace &space() const { return *V_; }
    int dim() const { return static_cast<int>(c_.size()); }
    std::vector<Scalar>       &coeffs()       { return c_; }
    const std::vector<Scalar> &coeffs() const { return c_; }
    Scalar  operator[](int j) const { return c_[j]; }
    Scalar &operator[](int j)       { return c_[j]; }

    /// @brief u(x)
    Scalar operator()(double x) const { return derivative(x, 0); }

    /// @brief d^k u / dx^k (x)
    Scalar derivative(double x, int k) const
    {
        int e = V_->eval(x, k, buf_);
        const std::vector<int> &d = V_->element_dofs(e);
        int na = static_cast<int>(d.size());
        Scalar s(0);
        for (int kk = 0; kk < na; ++kk) s += c_[d[kk]] * buf_[k * na + kk];
        return s;
    }

    /**
     * @brief d^k u / dx^k at every quadrature node of the cache, laid out as
     *        out[e*nq + q].  These are the c[e][q] arrays of design doc 15.1.
     */
    std::vector<Scalar> at_quad(const QuadratureCache &Q, int k = 0) const
    {
        if (k > Q.nder()) throw std::invalid_argument("Field::at_quad: cache holds fewer derivatives");
        int ne = Q.nelem(), nq = Q.nq();
        std::vector<Scalar> out(static_cast<std::size_t>(ne) * nq, Scalar(0));
        FEMD_OMP_FOR_IF(detail::parallel_elements(ne, nq))
        for (int e = 0; e < ne; ++e)
        {
            int nl = Q.nloc(e);
            for (int q = 0; q < nq; ++q)
            {
                const double *row = Q.basis_row(e, q, k);
                Scalar s(0);
                for (int kk = 0; kk < nl; ++kk) s += c_[Q.dof(e, kk)] * row[kk];
                out[e * nq + q] = s;
            }
        }
        return out;
    }

private:
    const FunctionSpace *V_;
    std::vector<Scalar> c_;
    mutable std::vector<double> buf_;
};

} // namespace femd

#endif // FEMD_FE_FIELD_HPP
