//
//  spline_space.hpp  --  the B-spline family.
//
//  Built from a grid and a degree.  The knot vector is an implementation detail
//  (KnotVector), the boundary conditions come from the family-agnostic builder,
//  periodicity identifies B_{j+M} with B_j for j < p.  Default continuity is
//  C^{p-1}; `continuity` lowers it everywhere, `continuity_at` at chosen nodes.
//  continuity = 0 spans the same space as LagrangeSpace(p) on a Bernstein
//  basis (design doc, Section 6.1): coefficients are NOT nodal values here.
//
#ifndef FEMD_SPLINE_SPLINE_SPACE_HPP
#define FEMD_SPLINE_SPLINE_SPACE_HPP

#include "femd/fe/function_space.hpp"
#include "femd/spline/knot_vector.hpp"
#include "femd/spline/bspline_basis.hpp"
#include <map>
#include <vector>

namespace femd {

class SplineSpace : public FunctionSpace {
public:
    SplineSpace(const Mesh1D &mesh, int degree, const BCSpec &bc = BCSpec::free(),
                int continuity = -1, const std::map<int, int> &continuity_at = {})
        : FunctionSpace(mesh, degree, KnotVector(mesh, degree, bc.periodic ? KnotVector::Flavor::Periodic : KnotVector::Flavor::Open,
                                                 continuity, continuity_at).n_raw()),
          kv_(mesh, degree, bc.periodic ? KnotVector::Flavor::Periodic : KnotVector::Flavor::Open, continuity, continuity_at)
    {
        apply_boundary_conditions(bc);
    }

    const KnotVector &knots() const { return kv_; }

    int raw_dof(int e, int l) const override { return kv_.span(e) - degree() + l; }

    void raw_eval(int e, double x, int nder, double *out) const override
    {
        ders_basis_funs(kv_.knots(), kv_.span(e), x, degree(), nder, out);
    }

    /// @brief Periodic: B_{j+M} is B_j shifted by one period, identify them for j < p.
    ConstraintOperator make_periodic() const override
    {
        int M = kv_.M(), p = degree();
        std::vector<ConstraintOperator::Row> rows(M);
        for (int j = 0; j < M; ++j)
        {
            rows[j].push_back(std::make_pair(j, 1.0));
            if (j < p) rows[j].push_back(std::make_pair(j + M, 1.0));
        }
        return ConstraintOperator(raw_dim(), std::move(rows));
    }

    /// @brief Greville abscissae of the RAW functions (the natural coefficient locations).
    std::vector<double> greville_raw() const
    {
        std::vector<double> g(raw_dim());
        for (int j = 0; j < raw_dim(); ++j) g[j] = kv_.greville(j);
        return g;
    }

    double raw_coordinate(int i) const override { return kv_.greville(i); }

    /// @brief Greville abscissa of each ADAPTED function (== dof_coordinates()).
    std::vector<double> greville() const { return dof_coordinates(); }

private:
    KnotVector kv_;
};

} // namespace femd

#endif // FEMD_SPLINE_SPLINE_SPACE_HPP
