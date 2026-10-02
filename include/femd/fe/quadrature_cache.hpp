//
//  quadrature_cache.hpp  --  precomputed ADAPTED basis values and derivatives
//  at the quadrature nodes, in PHYSICAL coordinates (design doc, Section 14).
//
//  Assembly never calls a basis evaluator; it reads these tables.  Layout is
//  flat with the local index innermost:
//      tab[((e*nq + q)*(nder+1) + m)*nloc_max + k]  =  d^m N_{dof(e,k)} / dx^m  (x_{e,q})
//  Weights already carry the Jacobian h_e/2.  nloc(e) may be smaller than
//  nloc_max near a boundary (never larger); dof(e,k) for k >= nloc(e) is -1.
//
#ifndef FEMD_FE_QUADRATURE_CACHE_HPP
#define FEMD_FE_QUADRATURE_CACHE_HPP

#include "femd/fe/function_space.hpp"
#include "femd/quadrature/gauss_legendre.hpp"
#include "femd/quadrature/gauss_lobatto.hpp"
#include "femd/util/omp.hpp"
#include <vector>

namespace femd {

class QuadratureCache {
public:
    QuadratureCache() = default;

    /// @brief Tabulate space V against an npts Gauss-Legendre rule, derivatives 0..nder, or with
    ///        lobatto = true the npts Gauss-Lobatto rule (npts >= 2): with a Lobatto nodal basis of
    ///        degree npts-1 its mass matrix is diagonal, the lumped mass.
    QuadratureCache(const FunctionSpace &V, int npts, int nder, bool lobatto = false)
        : nelem_(V.nelem()), nq_(npts), nder_(nder), nloc_max_(V.nloc_max()), lobatto_(lobatto)
    {
        if (lobatto) build(V, GaussLobatto(npts));
        else build(V, GaussLegendre(npts));
    }

    bool lobatto() const { return lobatto_; }

private:
    template <class Rule>
    void build(const FunctionSpace &V, const Rule &rule)
    {
        const Mesh1D &m = V.mesh();
        xq_.assign(static_cast<std::size_t>(nelem_) * nq_, 0.0);
        w_.assign(xq_.size(), 0.0);
        tab_.assign(static_cast<std::size_t>(nelem_) * nq_ * (nder_ + 1) * nloc_max_, 0.0);
        nloc_.assign(nelem_, 0);
        dofs_.assign(static_cast<std::size_t>(nelem_) * nloc_max_, -1);

        // Elements are independent and each writes its own slices of the tables, so
        // they run in parallel, each thread with its own evaluation buffers.
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(nelem_, nq_))
        {
        std::vector<double> vals, raw;
        FEMD_OMP_FOR
        for (int e = 0; e < nelem_; ++e)
        {
            const std::vector<int> &d = V.element_dofs(e);
            nloc_[e] = static_cast<int>(d.size());
            for (int k = 0; k < nloc_[e]; ++k) dofs_[e * nloc_max_ + k] = d[k];
            double jac = 0.5 * m.h(e);
            for (int q = 0; q < nq_; ++q)
            {
                double x = m.from_reference(e, rule.node(q));
                xq_[e * nq_ + q] = x;
                w_[e * nq_ + q]  = jac * rule.weight(q);
                V.eval_on_element_ref(e, rule.node(q), nder_, vals, raw);     // (nder+1) x nloc(e), at the exact xi
                for (int mm = 0; mm <= nder_; ++mm)
                    for (int k = 0; k < nloc_[e]; ++k)
                        tab_[idx(e, q, mm, k)] = vals[mm * nloc_[e] + k];
            }
        }
        }
    }

public:
    int nelem()    const { return nelem_; }
    int nq()       const { return nq_; }
    int nder()     const { return nder_; }
    int nloc_max() const { return nloc_max_; }
    int nloc(int e) const { return nloc_[e]; }
    int dof(int e, int k) const { return dofs_[e * nloc_max_ + k]; }
    double x(int e, int q)      const { return xq_[e * nq_ + q]; }
    double weight(int e, int q) const { return w_[e * nq_ + q]; }
    /// @brief d^m N_{dof(e,k)} / dx^m at node (e,q).
    double basis(int e, int q, int m, int k) const { return tab_[idx(e, q, m, k)]; }
    /// @brief Pointer to the nloc_max contiguous values for (e,q,m): the inner assembly loop.
    const double *basis_row(int e, int q, int m) const { return &tab_[idx(e, q, m, 0)]; }

    const std::vector<double> &nodes()   const { return xq_; }
    const std::vector<double> &weights() const { return w_; }

private:
    std::size_t idx(int e, int q, int m, int k) const
    {
        return ((static_cast<std::size_t>(e) * nq_ + q) * (nder_ + 1) + m) * nloc_max_ + k;
    }
    int nelem_ = 0, nq_ = 0, nder_ = 0, nloc_max_ = 0;
    bool lobatto_ = false;
    std::vector<double> xq_, w_, tab_;
    std::vector<int> nloc_, dofs_;
};

} // namespace femd

#endif // FEMD_FE_QUADRATURE_CACHE_HPP
