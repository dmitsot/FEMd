//
//  function_space.hpp  --  the interface every element family presents.
//
//  A family provides two things:
//    * a RAW local basis per element:  raw_dof(e,l) and raw_eval(e,x,nder,out)
//    * a ConstraintOperator C that turns raw functions into the adapted basis
//  Everything else (adapted element dofs, adapted point evaluation, the
//  half-bandwidth) is derived here, once, from those two.  Downstream code
//  (cache, kernels, solvers, Field) sees only this class and never asks which
//  family it was given.
//
//  Indices are 0-based.  1-based equation indices exist only at the solver
//  boundary (design doc, Section 11.3).
//
#ifndef FEMD_FE_FUNCTION_SPACE_HPP
#define FEMD_FE_FUNCTION_SPACE_HPP

#include "femd/mesh/mesh1d.hpp"
#include "femd/fe/constraint_operator.hpp"
#include "femd/fe/boundary_condition.hpp"
#include <map>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace femd {

class FunctionSpace {
public:
    virtual ~FunctionSpace() = default;

    // ---- geometry and degree ---------------------------------------------
    const Mesh1D &mesh() const { return mesh_; }
    int nelem()  const { return mesh_.nelem(); }
    int degree() const { return degree_; }

    // ---- raw layer (family-specific) ---------------------------------------
    int raw_dim()  const { return raw_dim_; }
    /// @brief Raw local functions per element.  Both families have p+1.
    int raw_nloc() const { return degree_ + 1; }
    /// @brief Raw global index of local function l on element e.
    virtual int raw_dof(int e, int l) const = 0;
    /**
     * @brief Values and derivatives of the raw local basis on element e at
     *        physical x, PHYSICAL derivatives (chain rule applied inside).
     * @param out  row-major (nder+1) x raw_nloc():  out[m*raw_nloc() + l] = d^m B_l / dx^m (x)
     */
    virtual void raw_eval(int e, double x, int nder, double *out) const = 0;
    /// @brief raw_eval at the reference coordinate xi in [-1, 1] of element e.  Families with an
    ///        element-local basis override it so tables built at reference points (quadrature,
    ///        facets) never round xi through x: the error of xi = (2x - a - b)/h is eps |x| / h,
    ///        which on fine meshes leaves 1e-12 noise where an orthogonal basis has exact zeros.
    virtual void raw_eval_ref(int e, double xi, int nder, double *out) const
    {
        raw_eval(e, mesh().from_reference(e, xi), nder, out);
    }
    /// @brief Representative coordinate of raw function i: the node (Lagrange) or Greville abscissa (splines).
    virtual double raw_coordinate(int i) const = 0;

    // ---- boundary conditions ------------------------------------------------
    const BCSpec &bc() const { return bc_; }
    /// @brief Number of built-in constraint rows (left rows first, then right).
    int n_constraints() const { return build_.k; }
    /**
     * @brief Raw coefficient vector u_g with lambda_r(u_g) = g[r] for every
     *        built-in functional, in row order (left then right), supported on
     *        the eliminated raw functions only.  u = u_g + sum_j c_j N_j then
     *        carries the inhomogeneous data (design doc, Section 5.6).
     */
    std::vector<double> lift(const std::vector<double> &g) const { return build_.lift(g, raw_dim_); }

    // ---- adapted layer (derived here) --------------------------------------
    const ConstraintOperator &constraints() const { return C_; }
    int dim()   const { return C_.n_adapted(); }
    /// @brief Half-bandwidth of any matrix assembled on this space (cyclic distance if periodic).
    int uband() const { return uband_; }
    /// @brief Adapted functions with support on element e (sorted global indices).
    const std::vector<int> &element_dofs(int e) const { return edofs_[e]; }
    int nloc(int e) const { return static_cast<int>(edofs_[e].size()); }
    /// @brief Largest nloc(e) over the mesh.  Always <= raw_nloc().
    int nloc_max() const { return nloc_max_; }

    /**
     * @brief Adapted local functions on element e at physical x.
     * @param vals  row-major (nder+1) x nloc(e):  vals[m*nloc(e)+k] = d^m N_{dofs[k]} / dx^m
     */
    void eval_on_element(int e, double x, int nder, std::vector<double> &vals) const
    {
        eval_on_element(e, x, nder, vals, raw_buf_);
    }

    /// @brief Same, with the raw scratch buffer supplied by the caller: safe to call from
    ///        several threads at once, each with its own `vals` and `raw`.
    void eval_on_element(int e, double x, int nder, std::vector<double> &vals, std::vector<double> &raw) const
    {
        raw.assign(static_cast<std::size_t>(nder + 1) * raw_nloc(), 0.0);
        raw_eval(e, x, nder, raw.data());
        adapt_raw(e, nder, vals, raw);
    }

    /// @brief eval_on_element at the reference coordinate xi of element e (see raw_eval_ref).
    void eval_on_element_ref(int e, double xi, int nder, std::vector<double> &vals, std::vector<double> &raw) const
    {
        raw.assign(static_cast<std::size_t>(nder + 1) * raw_nloc(), 0.0);
        raw_eval_ref(e, xi, nder, raw.data());
        adapt_raw(e, nder, vals, raw);
    }

    /// @brief Adapted values on element e from raw values already in `raw`.
    void adapt_raw(int e, int nder, std::vector<double> &vals, const std::vector<double> &raw) const
    {
        int nl = raw_nloc(), na = nloc(e);
        vals.assign(static_cast<std::size_t>(nder + 1) * na, 0.0);
        for (int l = 0; l < nl; ++l)
            for (const auto &kc : adapt_[e][l])            // (local adapted k, coefficient)
                for (int m = 0; m <= nder; ++m)
                    vals[m * na + kc.first] += kc.second * raw[m * nl + l];
    }

    /// @brief Same, locating the element first.  Returns the element index.
    int eval(double x, int nder, std::vector<double> &vals) const
    {
        int e = mesh_.element_of(x);
        eval_on_element(e, x, nder, vals);
        return e;
    }

    /**
     * @brief Representative coordinate of each ADAPTED function: that of its
     *        unit-weight (free) raw function, wrapped into [a,b) for periodic
     *        spaces.  Position interleaving in ProductSpace sorts on these.
     */
    std::vector<double> dof_coordinates() const
    {
        std::vector<double> g(dim());
        double a = mesh_.a(), L = mesh_.b() - a;
        for (int j = 0; j < dim(); ++j)
        {
            const ConstraintOperator::Row &r = C_.row(j);
            int rep = r.front().first;
            for (const auto &en : r) if (en.second == 1.0) { rep = en.first; break; }
            double x = raw_coordinate(rep);
            if (bc_.periodic) { while (x < a) x += L; while (x >= a + L) x -= L; }
            g[j] = x;
        }
        return g;
    }

    /// @brief Local adapted position of a raw local function: list of (k, C_{dofs[k], raw_dof(e,l)}).
    const std::vector<std::pair<int, double>> &adapt_map(int e, int l) const { return adapt_[e][l]; }

protected:
    /// @param min_degree  the lowest degree the family allows: 1 for the continuous families,
    ///                    0 for a discontinuous one (piecewise constants).
    FunctionSpace(Mesh1D mesh, int degree, int raw_dim, int min_degree = 1)
        : mesh_(std::move(mesh)), degree_(degree), raw_dim_(raw_dim)
    {
        if (degree < min_degree)
            throw std::invalid_argument("FunctionSpace: degree must be >= " + std::to_string(min_degree));
    }

    /**
     * @brief Widen the half-bandwidth to cover couplings the elements alone do not show:
     *        the facet terms of a discontinuous family couple the dofs of neighbouring
     *        elements.  Call after apply_boundary_conditions().
     */
    void widen_uband(int u) { uband_ = std::max(uband_, u); }
public:
    /// @brief Whether the space is discontinuous across element boundaries, so interior-facet
    ///        (dS) terms couple the dofs of neighboring elements.
    virtual bool broken() const { return false; }
protected:

    /**
     * @brief Build the constraint operator for a BC specification.  Derived
     *        constructors call this LAST, once raw_dof() and raw_eval() work.
     *        Endpoint conditions are family-agnostic (boundary_condition.hpp);
     *        periodicity is delegated to the family via make_periodic().
     */
    void apply_boundary_conditions(const BCSpec &spec)
    {
        bc_ = spec;
        if (spec.periodic) { build_ = ConstraintBuild(); build_.C = make_periodic(); set_constraints(build_.C); return; }

        int p = degree_, nl = raw_nloc(), ne = nelem();
        std::vector<std::map<int, double>> rows;
        std::vector<double> vals;
        auto add_end = [&](const BoundaryCondition &bcnd, int e, double x)
        {
            int M = bcnd.max_order();
            if (M < 0) return;
            if (M > p) throw std::invalid_argument("boundary condition of derivative order " + std::to_string(M) +
                                                   " on a degree-" + std::to_string(p) + " space");
            vals.assign(static_cast<std::size_t>(M + 1) * nl, 0.0);
            raw_eval(e, x, M, vals.data());
            for (const auto &f : bcnd.functionals)
            {
                std::map<int, double> row;
                for (int l = 0; l < nl; ++l)
                {
                    double v = 0.0;
                    for (const auto &t : f) v += t.second * vals[t.first * nl + l];
                    if (v != 0.0) row[raw_dof(e, l)] += v;
                }
                rows.push_back(std::move(row));
            }
        };
        add_end(spec.left,  0,      mesh_.a());
        add_end(spec.right, ne - 1, mesh_.b());

        // preference: distance (in raw index) from the nearest end, boundary-most first
        std::vector<double> pref(raw_dim_);
        for (int i = 0; i < raw_dim_; ++i) pref[i] = std::min(i, raw_dim_ - 1 - i);

        build_ = detail::build_constraints(raw_dim_, rows, pref);
        if (build_.C.n_adapted() <= 0)
            throw std::invalid_argument("boundary conditions leave an empty space: use more elements or fewer constraints");
        set_constraints(build_.C);
    }

    /// @brief Periodic identification, family-specific.  Default: unsupported.
    virtual ConstraintOperator make_periodic() const
    {
        throw std::logic_error("periodic boundary conditions are not implemented for this family");
    }

    /// @brief Install C and derive element dofs, local maps, uband.
    void set_constraints(ConstraintOperator C)
    {
        if (C.n_raw() != raw_dim_) throw std::invalid_argument("FunctionSpace: constraint operator has wrong raw size");
        C_ = std::move(C);
        int ne = nelem(), nl = raw_nloc();
        edofs_.assign(ne, {});
        adapt_.assign(ne, std::vector<std::vector<std::pair<int, double>>>(nl));
        uband_ = 0; nloc_max_ = 0;
        for (int e = 0; e < ne; ++e)
        {
            std::vector<int> &d = edofs_[e];
            for (int l = 0; l < nl; ++l)
                for (const auto &en : C_.col(raw_dof(e, l))) d.push_back(en.first);
            std::sort(d.begin(), d.end());
            d.erase(std::unique(d.begin(), d.end()), d.end());
            for (int l = 0; l < nl; ++l)
                for (const auto &en : C_.col(raw_dof(e, l)))
                {
                    int k = static_cast<int>(std::lower_bound(d.begin(), d.end(), en.first) - d.begin());
                    adapt_[e][l].push_back(std::make_pair(k, en.second));
                }
            // half-bandwidth: index spread, measured cyclically for periodic spaces
            int na = C_.n_adapted();
            for (std::size_t x = 0; x < d.size(); ++x)
                for (std::size_t y = x + 1; y < d.size(); ++y)
                {
                    int dist = d[y] - d[x];
                    if (bc_.periodic) dist = std::min(dist, na - dist);
                    uband_ = std::max(uband_, dist);
                }
            nloc_max_ = std::max(nloc_max_, static_cast<int>(d.size()));
        }
    }

private:
    Mesh1D mesh_;
    int degree_;
    int raw_dim_;
    ConstraintOperator C_;
    BCSpec bc_;
    ConstraintBuild build_;
    std::vector<std::vector<int>> edofs_;                                  // e -> adapted dofs
    std::vector<std::vector<std::vector<std::pair<int, double>>>> adapt_;  // e -> l -> (k, c)
    int uband_ = 0, nloc_max_ = 0;
    mutable std::vector<double> raw_buf_;
};

} // namespace femd

#endif // FEMD_FE_FUNCTION_SPACE_HPP
