//
//  ldlt.hpp  --  sparse symmetric indefinite LDL^T with pivoting, FEMd's direct solver for
//  symmetric matrices that are not positive definite: saddle points (Stokes, mixed methods),
//  Helmholtz-type operators, and whatever the Cholesky of cholesky.hpp refuses.
//
//      SparseLDLT F(A);                // order (AMD), analyse, factor: P A P^T = L D L^T
//      F.apply(b, x);                  // x = A^{-1} b
//      F.refactor(A2);                 // same pattern, new values: the numeric phase only
//
//  The method is multifrontal (Duff and Reid).  The symbolic phase orders the matrix (AMD),
//  post-orders the elimination tree and groups its chains into supernodes, each with the row
//  structure of its columns of L.  The numeric phase walks the supernodes children first.  A
//  supernode's front is a dense symmetric matrix on its columns, the rows below them and the
//  variables its children could not eliminate; it is assembled from the entries of A and the
//  children's contribution blocks, and its fully summed variables are eliminated with 1x1 and
//  2x2 pivots chosen by the threshold test of Duff and Reid (as in MA27/MA57): a 1x1 pivot
//  a_kk is taken when |a_kk| >= u max_i |a_ik|, and otherwise a 2x2 pivot P on (k, r), r the
//  largest fully summed entry of column k, when |P^{-1}| [g_k; g_r] <= [1/u; 1/u] with g the
//  largest entries outside P.  A variable that passes neither test is delayed to the parent
//  front, where more of its column is summed.  This bounds the growth of the factors by 1/u
//  per step (u = 0.01 by default), so the factorization is stable for any symmetric matrix
//  that is not singular, at the price of some extra fill where pivots are delayed.
//
//  The matrix must hold both triangles with a symmetric pattern (FEMd assembles full
//  symmetric patterns).  D is block diagonal with 1x1 and 2x2 blocks, which also gives the
//  inertia (the numbers of positive, negative and zero eigenvalues, by Sylvester's law).
//
#ifndef FEMD_SPARSE_LDLT_HPP
#define FEMD_SPARSE_LDLT_HPP

#include "femd/sparse/amd.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/sparse/precond.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class SparseLDLT : public SparsePreconditioner {
public:
    enum class Ordering { AMD, Natural };

    /**
     * @brief Order, analyse and factor A.
     * @param A         a symmetric matrix, both triangles stored, symmetric pattern
     * @param ordering  Ordering::AMD (default) or Ordering::Natural
     * @param u         pivot threshold in (0, 0.5]: larger is more stable, smaller delays less
     */
    explicit SparseLDLT(const CSRMatrix &A, Ordering ordering = Ordering::AMD, double u = 0.01)
        : ordering_(ordering), u_(u)
    {
        check_square(A, "LDL^T");
        if (!(u > 0.0 && u <= 0.5)) throw std::invalid_argument("LDL^T: the pivot threshold u must lie in (0, 0.5]");
        n_ = A.nrows();
        pattern_ = A.pattern;
        symbolic(A);
        numeric(A);
    }

    /// @brief New values on the same pattern: the numeric phase only (pivots are chosen afresh).
    void refactor(const CSRMatrix &A)
    {
        if (A.nrows() != n_ || A.ncols() != n_ || A.nnz() != pattern_->nnz() ||
            (A.pattern != pattern_ && (A.pattern->indptr != pattern_->indptr || A.pattern->indices != pattern_->indices)))
            throw std::invalid_argument("LDL^T::refactor: the matrix has a different pattern");
        numeric(A);
    }

    /// @brief z = A^{-1} r.
    void apply(const double *r, double *z) const override
    {
        std::vector<double> &y = work_;
        y.resize(static_cast<std::size_t>(n_));
        for (int k = 0; k < n_; ++k) y[k] = r[perm_[k]];
        for (const Front &f : fronts_)                                  // L y = b
        {
            const int m = f.m;
            const int *ix = f.idx.data();
            const double *L = f.L.data();
            for (int c = 0, b = 0; c < f.ne; c += f.bs[b], ++b)
            {
                if (f.bs[b] == 1)
                {
                    const double yv = y[ix[c]];
                    if (yv != 0.0)
                        for (int i = c + 1; i < m; ++i) y[ix[i]] -= L[static_cast<std::size_t>(c) * m + i] * yv;
                }
                else
                {
                    const double y1 = y[ix[c]], y2 = y[ix[c + 1]];
                    const double *L1 = L + static_cast<std::size_t>(c) * m, *L2 = L1 + m;
                    for (int i = c + 2; i < m; ++i) y[ix[i]] -= L1[i] * y1 + L2[i] * y2;
                }
            }
        }
        for (const Front &f : fronts_)                                  // D
            for (int c = 0, b = 0, d = 0; c < f.ne; c += f.bs[b], d += (f.bs[b] == 1 ? 1 : 3), ++b)
            {
                if (f.bs[b] == 1) y[f.idx[c]] /= f.D[d];
                else
                {
                    const double a = f.D[d], bb = f.D[d + 1], cc = f.D[d + 2], det = a * cc - bb * bb;
                    const double y1 = y[f.idx[c]], y2 = y[f.idx[c + 1]];
                    y[f.idx[c]] = (cc * y1 - bb * y2) / det;
                    y[f.idx[c + 1]] = (a * y2 - bb * y1) / det;
                }
            }
        for (auto it = fronts_.rbegin(); it != fronts_.rend(); ++it)   // L^T x = y
        {
            const Front &f = *it;
            const int m = f.m;
            const int *ix = f.idx.data();
            const double *L = f.L.data();
            std::vector<int> start;
            for (int c = 0, b = 0; c < f.ne; c += f.bs[b], ++b) start.push_back(c);
            for (int b = static_cast<int>(start.size()) - 1; b >= 0; --b)
            {
                const int c = start[b];
                if (f.bs[b] == 1)
                {
                    const double *Lc = L + static_cast<std::size_t>(c) * m;
                    double s = y[ix[c]];
                    for (int i = c + 1; i < m; ++i) s -= Lc[i] * y[ix[i]];
                    y[ix[c]] = s;
                }
                else
                {
                    const double *L1 = L + static_cast<std::size_t>(c) * m, *L2 = L1 + m;
                    double s1 = y[ix[c]], s2 = y[ix[c + 1]];
                    for (int i = c + 2; i < m; ++i) { s1 -= L1[i] * y[ix[i]]; s2 -= L2[i] * y[ix[i]]; }
                    y[ix[c]] = s1; y[ix[c + 1]] = s2;
                }
            }
        }
        for (int k = 0; k < n_; ++k) z[perm_[k]] = y[k];
    }
    std::string name() const override { return "ldlt"; }

    std::vector<double> solve(const std::vector<double> &b) const
    {
        if (static_cast<int>(b.size()) != n_) throw std::invalid_argument("LDL^T::solve: length mismatch");
        std::vector<double> x(static_cast<std::size_t>(n_));
        if (n_ > 0) apply(b.data(), x.data());
        return x;
    }

    // ---- statistics ----------------------------------------------------------------
    /// @brief Entries of L below the diagonal (the dense panels of the fronts).
    long long nnz_L() const
    {
        long long s = 0;
        for (const Front &f : fronts_) s += static_cast<long long>(f.ne) * f.m - static_cast<long long>(f.ne) * (f.ne + 1) / 2;
        return s;
    }
    /// @brief (positive, negative, zero) eigenvalue counts of A, from D.
    std::array<int, 3> inertia() const { return inertia_; }
    int delayed() const { return delayed_; }            ///< pivots delayed to a parent front, all fronts together
    int two_by_two() const { return n2x2_; }            ///< 2x2 pivots
    int supernodes() const { return static_cast<int>(sn_first_.size()) - 1; }
    int max_front() const { return max_front_; }
    double threshold() const { return u_; }
    std::string ordering_name() const { return ordering_ == Ordering::AMD ? "amd" : "natural"; }
    const std::vector<int> &permutation() const { return perm_; }
    /// @brief Smallest |eigenvalue| of a pivot block, over all blocks.
    double min_pivot() const { return min_piv_; }

private:
    struct Front {
        int m = 0, ne = 0;
        std::vector<int> idx;          // m variables (permuted indices), the first ne eliminated in this order
        std::vector<double> L;         // m x ne, column-major; column c holds L(i, c) for i > c
        std::vector<int> bs;           // pivot block sizes, 1 or 2
        std::vector<double> D;         // per block: d, or (a, b, c) of [[a, b], [b, c]]
    };

    Ordering ordering_;
    double u_;
    std::shared_ptr<const SparsityPattern> pattern_;
    std::vector<int> perm_, pinv_, parent_;
    std::vector<int> sn_first_, sn_parent_;            // supernode s has columns sn_first_[s] .. sn_first_[s+1]-1
    std::vector<std::vector<int>> sn_rows_, sn_children_;
    std::vector<Front> fronts_;
    std::array<int, 3> inertia_{{0, 0, 0}};
    int delayed_ = 0, n2x2_ = 0, max_front_ = 0;
    double min_piv_ = 0.0;
    mutable std::vector<double> work_;

    // ---- symbolic -----------------------------------------------------------------------
    static std::vector<int> etree_of(int n, const int *Ap, const int *Ai, const std::vector<int> &p,
                                     const std::vector<int> &pinv)
    {
        std::vector<int> parent(static_cast<std::size_t>(n), -1), anc(static_cast<std::size_t>(n), -1);
        for (int k = 0; k < n; ++k)
            for (int q = Ap[p[k]]; q < Ap[p[k] + 1]; ++q)
            {
                int i = pinv[Ai[q]];
                while (i != -1 && i < k)
                {
                    const int next = anc[i];
                    anc[i] = k;
                    if (next == -1) { parent[i] = k; break; }
                    i = next;
                }
            }
        return parent;
    }

    void symbolic(const CSRMatrix &A)
    {
        const int n = n_;
        const int *Ap = A.indptr(), *Ai = A.indices();
        std::vector<int> p0 = (ordering_ == Ordering::AMD) ? ordering::amd_order(n, Ap, Ai) : ordering::natural_order(n);
        std::vector<int> pinv0(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) pinv0[p0[k]] = k;
        // post-order the elimination tree, so every subtree is a contiguous range of columns
        std::vector<int> par0 = etree_of(n, Ap, Ai, p0, pinv0);
        std::vector<int> head(static_cast<std::size_t>(n), -1), next(static_cast<std::size_t>(n), -1), stack, post;
        post.reserve(static_cast<std::size_t>(n));
        for (int j = n - 1; j >= 0; --j)
            if (par0[j] >= 0) { next[j] = head[par0[j]]; head[par0[j]] = j; }
        for (int r = 0; r < n; ++r)
        {
            if (par0[r] != -1) continue;
            stack.push_back(r);
            while (!stack.empty())
            {
                const int v = stack.back();
                if (head[v] != -1) { const int c = head[v]; head[v] = next[c]; stack.push_back(c); }
                else { stack.pop_back(); post.push_back(v); }
            }
        }
        perm_.resize(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) perm_[k] = p0[post[k]];
        pinv_.assign(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k) pinv_[perm_[k]] = k;
        parent_ = etree_of(n, Ap, Ai, perm_, pinv_);
        // column counts of L (below the diagonal), from the row subtrees
        std::vector<int> cc(static_cast<std::size_t>(n), 0), flag(static_cast<std::size_t>(n), -1), nchild(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k)
        {
            flag[k] = k;
            for (int q = Ap[perm_[k]]; q < Ap[perm_[k] + 1]; ++q)
                for (int i = pinv_[Ai[q]]; i < k && flag[i] != k; i = parent_[i]) { ++cc[i]; flag[i] = k; }
        }
        for (int j = 0; j < n; ++j) if (parent_[j] >= 0) ++nchild[parent_[j]];
        // fundamental supernodes: chains j -> j+1 with j the only child and one row less in L
        sn_first_.clear();
        std::vector<int> sn_of(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j)
        {
            const bool merge = j > 0 && parent_[j - 1] == j && nchild[j] == 1 && cc[j - 1] == cc[j] + 1;
            if (!merge) sn_first_.push_back(j);
            sn_of[j] = static_cast<int>(sn_first_.size()) - 1;
        }
        const int ns = static_cast<int>(sn_first_.size());
        sn_first_.push_back(n);
        sn_parent_.assign(static_cast<std::size_t>(ns), -1);
        sn_children_.assign(static_cast<std::size_t>(ns), {});
        for (int s = 0; s < ns; ++s)
        {
            const int last = sn_first_[s + 1] - 1;
            if (parent_[last] >= 0) { sn_parent_[s] = sn_of[parent_[last]]; sn_children_[sn_parent_[s]].push_back(s); }
        }
        // row structure of each supernode: the rows below its last column
        sn_rows_.assign(static_cast<std::size_t>(ns), {});
        std::vector<int> mark(static_cast<std::size_t>(n), -1);
        for (int s = 0; s < ns; ++s)
        {
            const int f = sn_first_[s], l = sn_first_[s + 1] - 1;
            std::vector<int> &rows = sn_rows_[s];
            auto take = [&](int i) { if (i > l && mark[i] != s) { mark[i] = s; rows.push_back(i); } };
            for (int j = f; j <= l; ++j)
                for (int q = Ap[perm_[j]]; q < Ap[perm_[j] + 1]; ++q) take(pinv_[Ai[q]]);
            for (int c : sn_children_[s]) for (int i : sn_rows_[c]) take(i);
            std::sort(rows.begin(), rows.end());
        }
    }

    // ---- numeric ------------------------------------------------------------------------
    static double &at(std::vector<double> &F, int m, int i, int j) { return i >= j ? F[static_cast<std::size_t>(j) * m + i] : F[static_cast<std::size_t>(i) * m + j]; }

    // symmetric swap of positions a < b in a lower-stored m x m front
    static void sym_swap(std::vector<double> &F, int m, int a, int b)
    {
        if (a == b) return;
        if (a > b) std::swap(a, b);
        auto L = [&](int i, int j) -> double & { return F[static_cast<std::size_t>(j) * m + i]; };   // i >= j
        std::swap(L(a, a), L(b, b));
        for (int j = 0; j < a; ++j) std::swap(L(a, j), L(b, j));
        for (int j = a + 1; j < b; ++j) std::swap(L(j, a), L(b, j));
        for (int i = b + 1; i < m; ++i) std::swap(L(i, a), L(i, b));
    }

    void numeric(const CSRMatrix &A)
    {
        const int n = n_;
        const int ns = static_cast<int>(sn_first_.size()) - 1;
        const int *Ap = A.indptr(), *Ai = A.indices();
        const double *Ax = A.data.data();
        fronts_.assign(static_cast<std::size_t>(ns), Front());
        inertia_ = {{0, 0, 0}};
        delayed_ = 0; n2x2_ = 0; max_front_ = 0;
        min_piv_ = std::numeric_limits<double>::infinity();
        std::vector<std::vector<int>> cb_idx(static_cast<std::size_t>(ns)), delay(static_cast<std::size_t>(ns));
        std::vector<std::vector<double>> cb(static_cast<std::size_t>(ns));
        std::vector<int> loc(static_cast<std::size_t>(n), -1);
        std::vector<double> F, w1, w2;
        const double u = u_;
        for (int s = 0; s < ns; ++s)
        {
            const int f = sn_first_[s], l = sn_first_[s + 1] - 1;
            Front &fr = fronts_[s];
            std::vector<int> &idx = fr.idx;
            idx.clear();
            for (int c : sn_children_[s]) idx.insert(idx.end(), delay[c].begin(), delay[c].end());
            for (int j = f; j <= l; ++j) idx.push_back(j);
            const int p = static_cast<int>(idx.size());
            idx.insert(idx.end(), sn_rows_[s].begin(), sn_rows_[s].end());
            const int m = static_cast<int>(idx.size());
            fr.m = m;
            max_front_ = std::max(max_front_, m);
            for (int t = 0; t < m; ++t) loc[idx[t]] = t;
            F.assign(static_cast<std::size_t>(m) * m, 0.0);
            // entries of A in the supernode's columns, lower triangle in the new order
            for (int j = f; j <= l; ++j)
                for (int q = Ap[perm_[j]]; q < Ap[perm_[j] + 1]; ++q)
                {
                    const int i = pinv_[Ai[q]];
                    if (i < j) continue;
                    at(F, m, loc[i], loc[j]) += Ax[q];
                }
            // the children's contribution blocks
            for (int c : sn_children_[s])
            {
                const std::vector<int> &ci = cb_idx[c];
                const std::vector<double> &C = cb[c];
                const int mc = static_cast<int>(ci.size());
                for (int b = 0; b < mc; ++b)
                {
                    const int lb = loc[ci[b]];
                    for (int a = b; a < mc; ++a) at(F, m, loc[ci[a]], lb) += C[static_cast<std::size_t>(b) * mc + a];
                }
                std::vector<double>().swap(cb[c]);
                std::vector<int>().swap(cb_idx[c]);
            }
            // eliminate the fully summed variables 0 .. p-1 with 1x1 and 2x2 pivots
            const bool root = sn_parent_[s] < 0;
            auto get = [&](int i, int j) { return i >= j ? F[static_cast<std::size_t>(j) * m + i] : F[static_cast<std::size_t>(i) * m + j]; };
            auto colmax = [&](int j, int k, int skip1, int skip2) {
                double g = 0.0;
                for (int i = k; i < m; ++i) if (i != skip1 && i != skip2) g = std::max(g, std::abs(get(i, j)));
                return g;
            };
            int k = 0;
            fr.bs.clear(); fr.D.clear();
            w1.resize(static_cast<std::size_t>(m)); w2.resize(static_cast<std::size_t>(m));
            while (k < p)
            {
                int piv1 = -1, pa = -1, pb = -1;
                for (int j = k; j < p && piv1 < 0 && pa < 0; ++j)
                {
                    const double ajj = std::abs(get(j, j));
                    const double g = colmax(j, k, j, -1);
                    if (ajj > 0.0 && ajj >= u * g) { piv1 = j; break; }
                    int r = -1; double big = 0.0;
                    for (int i = k; i < p; ++i) if (i != j && std::abs(get(i, j)) > big) { big = std::abs(get(i, j)); r = i; }
                    if (r < 0) continue;
                    const double a = get(j, j), b = get(r, j), c = get(r, r), det = a * c - b * b;
                    if (!(std::abs(det) > 0.0) || !std::isfinite(det)) continue;
                    const double gj = colmax(j, k, j, r), gr = colmax(r, k, j, r);
                    const double ad = std::abs(det);
                    if ((std::abs(c) * gj + std::abs(b) * gr) <= ad / u && (std::abs(b) * gj + std::abs(a) * gr) <= ad / u)
                    { pa = j; pb = r; }
                }
                if (piv1 < 0 && pa < 0)
                {
                    if (!root) break;                                  // delay the rest to the parent
                    // the root has no parent to delay to: take the largest pivot there is
                    double best = 0.0;
                    for (int j = k; j < p; ++j) if (std::abs(get(j, j)) > best) { best = std::abs(get(j, j)); piv1 = j; }
                    double bestdet = 0.0;
                    for (int j = k; j < p; ++j)
                        for (int r = j + 1; r < p; ++r)
                        {
                            const double det = std::abs(get(j, j) * get(r, r) - get(r, j) * get(r, j));
                            if (det > bestdet && std::abs(get(r, j)) > best) { bestdet = det; pa = j; pb = r; }
                        }
                    if (pa >= 0) piv1 = -1;
                    if (piv1 < 0 && pa < 0)
                        throw std::runtime_error("LDL^T: the matrix is singular (no nonzero pivot left, " +
                                                 std::to_string(p - k) + " variables)");
                }
                if (piv1 >= 0)
                {
                    sym_swap(F, m, k, piv1); std::swap(idx[k], idx[piv1]);
                    double *Fk = &F[static_cast<std::size_t>(k) * m];
                    const double d = Fk[k];
                    for (int i = k + 1; i < m; ++i) { w1[i] = Fk[i]; Fk[i] /= d; }
                    for (int j = k + 1; j < m; ++j)
                    {
                        const double wj = w1[j];
                        if (wj == 0.0) continue;
                        double *Fj = &F[static_cast<std::size_t>(j) * m];
                        for (int i = j; i < m; ++i) Fj[i] -= Fk[i] * wj;
                    }
                    fr.bs.push_back(1); fr.D.push_back(d);
                    ++inertia_[d > 0 ? 0 : 1];
                    min_piv_ = std::min(min_piv_, std::abs(d));
                    k += 1;
                }
                else
                {
                    sym_swap(F, m, k, pa); std::swap(idx[k], idx[pa]);
                    const int r = (pb == k) ? pa : pb;
                    sym_swap(F, m, k + 1, r); std::swap(idx[k + 1], idx[r]);
                    double *F1 = &F[static_cast<std::size_t>(k) * m], *F2 = &F[static_cast<std::size_t>(k + 1) * m];
                    const double a = F1[k], b = F1[k + 1], c = F2[k + 1], det = a * c - b * b;
                    for (int i = k + 2; i < m; ++i)
                    {
                        w1[i] = F1[i]; w2[i] = F2[i];
                        F1[i] = (c * w1[i] - b * w2[i]) / det;
                        F2[i] = (a * w2[i] - b * w1[i]) / det;
                    }
                    F1[k + 1] = 0.0;
                    for (int j = k + 2; j < m; ++j)
                    {
                        const double v1 = w1[j], v2 = w2[j];
                        if (v1 == 0.0 && v2 == 0.0) continue;
                        double *Fj = &F[static_cast<std::size_t>(j) * m];
                        for (int i = j; i < m; ++i) Fj[i] -= F1[i] * v1 + F2[i] * v2;
                    }
                    fr.bs.push_back(2); fr.D.push_back(a); fr.D.push_back(b); fr.D.push_back(c);
                    ++n2x2_;
                    // eigenvalues of [[a, b], [b, c]]: signs from the determinant and the trace
                    if (det < 0) { ++inertia_[0]; ++inertia_[1]; }
                    else if (a + c > 0) inertia_[0] += 2;
                    else inertia_[1] += 2;
                    const double tr = 0.5 * (a + c), disc = std::sqrt(std::max(0.0, 0.25 * (a - c) * (a - c) + b * b));
                    min_piv_ = std::min(min_piv_, std::min(std::abs(tr - disc), std::abs(tr + disc)));
                    k += 2;
                }
            }
            fr.ne = k;
            delay[s].assign(idx.begin() + k, idx.begin() + p);
            delayed_ += p - k;
            // store the panel of L, and hand the rest of the front to the parent
            fr.L.assign(F.begin(), F.begin() + static_cast<std::size_t>(k) * m);
            const int mc = m - k;
            if (mc > 0 && !root)
            {
                cb_idx[s].assign(idx.begin() + k, idx.end());
                cb[s].resize(static_cast<std::size_t>(mc) * mc);
                for (int b = 0; b < mc; ++b)
                    for (int a = b; a < mc; ++a) cb[s][static_cast<std::size_t>(b) * mc + a] = F[static_cast<std::size_t>(k + b) * m + (k + a)];
            }
            for (int t = 0; t < m; ++t) loc[idx[t]] = -1;
        }
        if (!(min_piv_ < std::numeric_limits<double>::infinity())) min_piv_ = 0.0;
    }
};

} // namespace femd

#endif // FEMD_SPARSE_LDLT_HPP
