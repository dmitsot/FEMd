//
//  boundary_condition.hpp  --  boundary conditions as linear functionals, and
//  the family-agnostic construction of the ConstraintOperator from them
//  (design doc, Sections 4 and 5).
//
//  A BoundaryCondition at one endpoint is a list of functionals, each a sum
//      lambda(u) = sum_m alpha_m u^{(m)}(x_end) = 0
//  stored as (m, alpha_m) pairs.  A `bc` entry names what is BUILT IN:
//  "free" builds in nothing (the default), "neumann" builds in u' = 0.
//
//  The construction needs only the raw basis derivatives at the endpoint, so
//  it serves Lagrange and splines with one implementation:
//    1. every functional becomes a sparse row over the raw dofs of the end element,
//    2. rows are normalised, the involved columns collected,
//    3. Gauss-Jordan with a PREFERENCE order (boundary-most raw function first)
//       picks the eliminated (pivot) columns; a tiny pivot falls through to the
//       next column, which is the rank-revealing fallback of Section 5.2,
//    4. each free involved column c gives one adapted function
//       N = B_c - sum_r A[r][c] B_{P_r}   (reduced form, free coefficient 1),
//    5. uninvolved raw functions pass through unchanged.
//  Adapted functions keep raw order, so bandedness is preserved.
//
#ifndef FEMD_FE_BOUNDARY_CONDITION_HPP
#define FEMD_FE_BOUNDARY_CONDITION_HPP

#include "femd/fe/constraint_operator.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace femd {

// ---------------------------------------------------------------------------
struct BoundaryCondition {
    using Functional = std::vector<std::pair<int, double>>;   // (derivative order m, alpha_m)
    std::vector<Functional> functionals;                       // k of them
    std::string name = "free";

    static BoundaryCondition free()      { return BoundaryCondition(); }
    static BoundaryCondition dirichlet() { BoundaryCondition b; b.name = "dirichlet"; b.functionals = { {{0, 1.0}} }; return b; }
    static BoundaryCondition neumann()   { BoundaryCondition b; b.name = "neumann";   b.functionals = { {{1, 1.0}} }; return b; }
    static BoundaryCondition clamped()   { BoundaryCondition b; b.name = "clamped";   b.functionals = { {{0, 1.0}}, {{1, 1.0}} }; return b; }
    /// @brief alpha u + beta u' = 0
    static BoundaryCondition robin(double alpha, double beta)
    { BoundaryCondition b; b.name = "robin"; b.functionals = { {{0, alpha}, {1, beta}} }; return b; }
    /// @brief u^{(m)} = 0
    static BoundaryCondition derivative(int m)
    { BoundaryCondition b; b.name = "derivative(" + std::to_string(m) + ")"; b.functionals = { {{m, 1.0}} }; return b; }
    static BoundaryCondition custom(std::vector<Functional> f)
    { BoundaryCondition b; b.name = "custom"; b.functionals = std::move(f); return b; }

    /// @brief "free" | "dirichlet" | "neumann" | "clamped"
    static BoundaryCondition from_name(const std::string &s)
    {
        if (s == "free")      return free();
        if (s == "dirichlet") return dirichlet();
        if (s == "neumann")   return neumann();
        if (s == "clamped")   return clamped();
        throw std::invalid_argument("BoundaryCondition::from_name: unknown condition '" + s + "' (robin needs parameters, periodic is a BCSpec)");
    }

    int count() const { return static_cast<int>(functionals.size()); }
    int max_order() const
    {
        int M = -1;
        for (const Functional &f : functionals) for (const auto &t : f) M = std::max(M, t.first);
        return M;
    }
};

// ---------------------------------------------------------------------------
struct BCSpec {
    bool periodic = false;
    BoundaryCondition left, right;

    BCSpec() = default;
    BCSpec(BoundaryCondition l, BoundaryCondition r) : left(std::move(l)), right(std::move(r)) {}

    static BCSpec free()                 { return BCSpec(); }
    static BCSpec make_periodic()        { BCSpec s; s.periodic = true; return s; }
    static BCSpec both(const BoundaryCondition &b) { return BCSpec(b, b); }
    /// @brief ("dirichlet","neumann") style, or a single name for both ends, or "periodic".
    static BCSpec from_names(const std::string &l, const std::string &r)
    { return BCSpec(BoundaryCondition::from_name(l), BoundaryCondition::from_name(r)); }
    static BCSpec from_name(const std::string &s)
    { if (s == "periodic") return make_periodic(); return both(BoundaryCondition::from_name(s)); }

    int count() const { return periodic ? 0 : left.count() + right.count(); }
    std::string describe() const
    { return periodic ? std::string("periodic") : "(" + left.name + ", " + right.name + ")"; }
};

// ---------------------------------------------------------------------------
/// Result of building endpoint constraints: the operator plus what the lift needs.
struct ConstraintBuild {
    ConstraintOperator C;
    int k = 0;                         // number of constraint rows (left rows first)
    std::vector<int>    pivot_raw;     // raw index eliminated by row r
    std::vector<double> A_pivot;       // k x k, row-major: original (normalised) rows on the pivot columns
    std::vector<double> row_scale;     // 1/norm applied to each original row

    /**
     * @brief Raw coefficient vector u_g with  lambda_r(u_g) = g_r  for every
     *        constraint row, supported on the pivot raw functions only.
     */
    std::vector<double> lift(const std::vector<double> &g, int n_raw) const
    {
        if (static_cast<int>(g.size()) != k) throw std::invalid_argument("lift: need one value per constraint row");
        std::vector<double> u(n_raw, 0.0);
        if (k == 0) return u;
        // solve A_pivot x = D g   (D = row scaling), tiny dense LU with partial pivoting
        std::vector<double> A(A_pivot), b(k);
        for (int r = 0; r < k; ++r) b[r] = g[r] * row_scale[r];
        for (int c = 0; c < k; ++c)
        {
            int piv = c;
            for (int r = c + 1; r < k; ++r) if (std::abs(A[r * k + c]) > std::abs(A[piv * k + c])) piv = r;
            if (piv != c) { for (int j = 0; j < k; ++j) std::swap(A[c * k + j], A[piv * k + j]); std::swap(b[c], b[piv]); }
            for (int r = c + 1; r < k; ++r)
            {
                double f = A[r * k + c] / A[c * k + c];
                for (int j = c; j < k; ++j) A[r * k + j] -= f * A[c * k + j];
                b[r] -= f * b[c];
            }
        }
        for (int r = k - 1; r >= 0; --r)
        {
            double s = b[r];
            for (int j = r + 1; j < k; ++j) s -= A[r * k + j] * b[j];
            b[r] = s / A[r * k + r];
        }
        for (int r = 0; r < k; ++r) u[pivot_raw[r]] = b[r];
        return u;
    }
};

namespace detail {

/**
 * @brief Build the constraint operator from sparse constraint rows.
 * @param n_raw       raw dimension
 * @param rows        each row: raw index -> coefficient  (the functional applied to each raw function)
 * @param preference  value per raw index; SMALLER is eliminated FIRST (distance from the nearest constrained end)
 */
inline ConstraintBuild build_constraints(int n_raw,
                                         const std::vector<std::map<int, double>> &rows,
                                         const std::vector<double> &preference)
{
    ConstraintBuild out;
    int k = static_cast<int>(rows.size());
    out.k = k;
    if (k == 0) { out.C = ConstraintOperator::identity(n_raw); return out; }

    // involved columns
    std::vector<int> S;
    for (const auto &r : rows) for (const auto &t : r) S.push_back(t.first);
    std::sort(S.begin(), S.end()); S.erase(std::unique(S.begin(), S.end()), S.end());
    int ns = static_cast<int>(S.size());
    std::map<int, int> pos; for (int c = 0; c < ns; ++c) pos[S[c]] = c;

    // dense normalised A (k x ns), keep a copy for the lift
    std::vector<double> A(static_cast<std::size_t>(k) * ns, 0.0);
    out.row_scale.assign(k, 1.0);
    for (int r = 0; r < k; ++r)
    {
        double nrm = 0.0;
        for (const auto &t : rows[r]) nrm += t.second * t.second;
        nrm = std::sqrt(nrm);
        if (nrm == 0.0) throw std::invalid_argument("build_constraints: a boundary functional vanishes on every basis function (row " + std::to_string(r) + ")");
        out.row_scale[r] = 1.0 / nrm;
        for (const auto &t : rows[r]) A[r * ns + pos[t.first]] = t.second / nrm;
    }
    std::vector<double> A0(A);

    // column order: preference ascending, ties by index
    std::vector<int> order(ns);
    for (int c = 0; c < ns; ++c) order[c] = c;
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return preference[S[x]] < preference[S[y]]; });

    // Gauss-Jordan with preference; rank-revealing by falling through tiny pivots
    const double tol = 1e-10;
    std::vector<int> pivot_col(k, -1);          // r -> column position in S
    std::vector<bool> row_used(k, false), col_used(ns, false);
    int found = 0;
    for (int oc : order)
    {
        if (found == k) break;
        int best = -1; double bv = tol;
        for (int r = 0; r < k; ++r) if (!row_used[r] && std::abs(A[r * ns + oc]) > bv) { bv = std::abs(A[r * ns + oc]); best = r; }
        if (best < 0) continue;
        double d = A[best * ns + oc];
        for (int c = 0; c < ns; ++c) A[best * ns + c] /= d;
        for (int r = 0; r < k; ++r)
        {
            if (r == best) continue;
            double f = A[r * ns + oc];
            if (f == 0.0) continue;
            for (int c = 0; c < ns; ++c) A[r * ns + c] -= f * A[best * ns + c];
        }
        row_used[best] = true; col_used[oc] = true; pivot_col[best] = oc; ++found;
    }
    if (found < k)
        throw std::invalid_argument("build_constraints: boundary functionals are linearly dependent or inconsistent (rank " +
                                    std::to_string(found) + " < " + std::to_string(k) + ")");

    out.pivot_raw.assign(k, -1);
    out.A_pivot.assign(static_cast<std::size_t>(k) * k, 0.0);
    for (int r = 0; r < k; ++r)
    {
        out.pivot_raw[r] = S[pivot_col[r]];
        for (int rr = 0; rr < k; ++rr) out.A_pivot[rr * k + r] = A0[rr * ns + pivot_col[r]];
    }

    // assemble C rows in raw order
    std::vector<bool> is_pivot(n_raw, false);
    for (int r = 0; r < k; ++r) is_pivot[out.pivot_raw[r]] = true;
    std::vector<ConstraintOperator::Row> crow;
    crow.reserve(n_raw - k);
    for (int i = 0; i < n_raw; ++i)
    {
        if (is_pivot[i]) continue;
        ConstraintOperator::Row row;
        row.push_back(std::make_pair(i, 1.0));
        auto it = pos.find(i);
        if (it != pos.end())
        {
            int c = it->second;
            for (int r = 0; r < k; ++r)
            {
                double gamma = -A[r * ns + c];
                if (std::abs(gamma) > 1e-14) row.push_back(std::make_pair(out.pivot_raw[r], gamma));
            }
            std::sort(row.begin(), row.end());
        }
        crow.push_back(std::move(row));
    }
    out.C = ConstraintOperator(n_raw, std::move(crow));
    return out;
}

} // namespace detail
} // namespace femd

#endif // FEMD_FE_BOUNDARY_CONDITION_HPP
