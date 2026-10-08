//
//  lu_frontal.hpp  --  the multifrontal numeric phase of FEMd's sparse LU (lu.hpp):
//  Gaussian elimination with threshold partial pivoting inside dense frontal matrices,
//  after Duff and Reid and MA41, for matrices whose pattern is symmetric or nearly so
//  (finite element matrices: the Navier-Stokes Jacobian, saddle points, convection-diffusion).
//
//  The symbolic phase works on the pattern of A + A^T: it post-orders the elimination tree
//  of the ordered pattern and groups its chains into supernodes, each with the row structure
//  of its columns.  The numeric phase walks the supernodes children first.  A supernode's
//  front is a dense square matrix on its variables, the rows and columns below them, and the
//  variables its children could not eliminate; it is assembled from the entries of A and the
//  children's contribution blocks.  Its fully summed variables are eliminated in panels: a
//  pivot for column j is the entry of a fully summed row that is at least u times the
//  largest entry of the whole column (the diagonal when it passes, else the largest), the
//  pivot row and column are swapped into place, the column of L and the row of U are
//  computed with the panel's earlier pivots applied, and when the panel is full the trailing
//  matrix receives the panel's update as one dense product, in parallel over its columns.  A
//  column with no acceptable pivot is delayed, with its row, to the parent front, where more
//  of it is summed.  Pivoting inside the fronts keeps the fill of the symmetric pattern (plus
//  what the delays add) and bounds the growth by 1/u per step.
//
//  The factors come out in the form lu.hpp uses, L and U by columns in the pivoted numbering,
//  so the parallel solves of lu.hpp apply unchanged.
//
#ifndef FEMD_SPARSE_LU_FRONTAL_HPP
#define FEMD_SPARSE_LU_FRONTAL_HPP

#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FEMD_LU_PANEL
#define FEMD_LU_PANEL 32
#endif

namespace femd {
namespace frontal {

template <class T>
struct Factor {
    struct Front {
        int m = 0, ne = 0, p = 0;
        std::vector<int> ridx, cidx;   // the m row and column variables; the first ne are eliminated in this order
        std::vector<T> L;              // m x ne, column-major: L(i, c) for i > c, the pivot at (c, c)
        std::vector<T> Ut;             // ne x m, row-major: U(c, j) for j > c
    };
    int n = 0;
    double u = 0.1;
    std::vector<int> perm, pinv, parent;             // the symmetric permutation (post-ordered) and the elimination tree
    std::vector<int> sn_first, sn_parent;
    std::vector<std::vector<int>> sn_rows, sn_children;
    std::vector<Front> fronts;
    int delayed = 0, max_front = 0, offdiag = 0;
    double minpiv = 0.0, maxpiv = 0.0;

    // ---- symbolic: the pattern of A + A^T, a given ordering p0 -----------------------------
    static std::vector<int> etree_of(int n, const std::vector<int> &Sp, const std::vector<int> &Si, const std::vector<int> &p, const std::vector<int> &pinv)
    {
        std::vector<int> parent(static_cast<std::size_t>(n), -1), anc(static_cast<std::size_t>(n), -1);
        for (int k = 0; k < n; ++k)
            for (int q = Sp[p[k]]; q < Sp[p[k] + 1]; ++q)
            {
                int i = pinv[Si[q]];
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

    /// Sp/Si: the symmetric pattern (A + A^T, no diagonal needed); p0: the ordering to post-order.
    void symbolic(int n_, const std::vector<int> &Sp, const std::vector<int> &Si, const std::vector<int> &p0)
    {
        n = n_;
        std::vector<int> pinv0(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) pinv0[p0[k]] = k;
        std::vector<int> par0 = etree_of(n, Sp, Si, p0, pinv0);
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
        perm.resize(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) perm[k] = p0[post[k]];
        pinv.assign(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k) pinv[perm[k]] = k;
        parent = etree_of(n, Sp, Si, perm, pinv);
        // column counts of L (below the diagonal), from the row subtrees
        std::vector<int> cc(static_cast<std::size_t>(n), 0), flag(static_cast<std::size_t>(n), -1), nchild(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k)
        {
            flag[k] = k;
            for (int q = Sp[perm[k]]; q < Sp[perm[k] + 1]; ++q)
                for (int i = pinv[Si[q]]; i < k && flag[i] != k; i = parent[i]) { ++cc[i]; flag[i] = k; }
        }
        for (int j = 0; j < n; ++j) if (parent[j] >= 0) ++nchild[parent[j]];
        // fundamental supernodes: chains j -> j+1 with j the only child and one row less in L
        sn_first.clear();
        std::vector<int> sn_of(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j)
        {
            const bool merge = j > 0 && parent[j - 1] == j && nchild[j] == 1 && cc[j - 1] == cc[j] + 1;
            if (!merge) sn_first.push_back(j);
            sn_of[j] = static_cast<int>(sn_first.size()) - 1;
        }
        const int ns = static_cast<int>(sn_first.size());
        sn_first.push_back(n);
        sn_parent.assign(static_cast<std::size_t>(ns), -1);
        sn_children.assign(static_cast<std::size_t>(ns), {});
        for (int s = 0; s < ns; ++s)
        {
            const int last = sn_first[s + 1] - 1;
            if (parent[last] >= 0) { sn_parent[s] = sn_of[parent[last]]; sn_children[sn_parent[s]].push_back(s); }
        }
        // row structure of each supernode: the rows below its last column
        sn_rows.assign(static_cast<std::size_t>(ns), {});
        std::vector<int> mark(static_cast<std::size_t>(n), -1);
        for (int s = 0; s < ns; ++s)
        {
            const int f = sn_first[s], l = sn_first[s + 1] - 1;
            std::vector<int> &rows = sn_rows[s];
            auto take = [&](int i) { if (i > l && mark[i] != s) { mark[i] = s; rows.push_back(i); } };
            for (int j = f; j <= l; ++j)
                for (int q = Sp[perm[j]]; q < Sp[perm[j] + 1]; ++q) take(pinv[Si[q]]);
            for (int c : sn_children[s]) for (int i : sn_rows[c]) take(i);
            std::sort(rows.begin(), rows.end());
        }
    }

    // ---- numeric -------------------------------------------------------------------------
    struct Scratch {
        std::vector<int> locr, locc;
        std::vector<T> F, col;
    };
    struct Stats {
        int delayed = 0, max_front = 0, offdiag = 0;
        double minpiv = std::numeric_limits<double>::infinity(), maxpiv = 0.0;
        void merge(const Stats &o) { delayed += o.delayed; max_front = std::max(max_front, o.max_front); offdiag += o.offdiag; minpiv = std::min(minpiv, o.minpiv); maxpiv = std::max(maxpiv, o.maxpiv); }
    };

    /// A by rows (Ap, Ai, Ax) and by columns (Cp, Ci, Cx), original numbering.  The supernodes
    /// are processed children first: independent subtrees of the supernodal tree concurrently
    /// (each thread with its own scratch), then the top in order with the trailing updates of
    /// its fronts shared between the threads.  The arithmetic of a front does not depend on the
    /// thread that does it, so the factors are the same for any number of threads.
    void numeric(const int *Ap, const int *Ai, const T *Ax, const int *Cp, const int *Ci, const T *Cx)
    {
        const int ns = static_cast<int>(sn_first.size()) - 1;
        fronts.assign(static_cast<std::size_t>(ns), Front());
        cb_r_.assign(static_cast<std::size_t>(ns), {}); cb_c_.assign(static_cast<std::size_t>(ns), {}); cb_.assign(static_cast<std::size_t>(ns), {});
        delay_r_.assign(static_cast<std::size_t>(ns), {}); delay_c_.assign(static_cast<std::size_t>(ns), {});
        // the cut of the supernodal tree: subtrees of at most 1/32 of the estimated work (m^2 p per front)
        std::vector<double> W(static_cast<std::size_t>(ns));
        for (int s = 0; s < ns; ++s)
        {
            const double p = static_cast<double>(sn_first[s + 1] - sn_first[s]), m = p + static_cast<double>(sn_rows[s].size());
            W[s] = m * m * p + m * m;
        }
        for (int s = 0; s < ns; ++s) if (sn_parent[s] >= 0) W[sn_parent[s]] += W[s];
        double total = 0.0;
        for (int s = 0; s < ns; ++s) if (sn_parent[s] < 0) total += W[s];
        const double limit = n > 2 * FEMD_OMP_THRESHOLD ? total / 32.0 : -1.0;
        std::vector<int> part(static_cast<std::size_t>(ns), -1), roots;
        for (int s = ns - 1; s >= 0; --s)
        {
            const int pa = sn_parent[s];
            if (pa >= 0 && part[pa] >= 0) { part[s] = part[pa]; continue; }
            if (W[s] <= limit) { part[s] = static_cast<int>(roots.size()); roots.push_back(s); }
        }
        const int nparts = static_cast<int>(roots.size());
        std::vector<int> order(static_cast<std::size_t>(nparts));
        for (int k = 0; k < nparts; ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return W[roots[a]] > W[roots[b]]; });
        std::vector<int> partptr(static_cast<std::size_t>(nparts) + 1, 0), newid(static_cast<std::size_t>(nparts));
        for (int k = 0; k < nparts; ++k) newid[order[k]] = k;
        for (int s = 0; s < ns; ++s) if (part[s] >= 0) ++partptr[newid[part[s]] + 1];
        for (int k = 0; k < nparts; ++k) partptr[k + 1] += partptr[k];
        std::vector<int> nodes(static_cast<std::size_t>(partptr[nparts])), top;
        {
            std::vector<int> fill(partptr.begin(), partptr.end() - 1);
            for (int s = 0; s < ns; ++s) { if (part[s] >= 0) nodes[fill[newid[part[s]]]++] = s; else top.push_back(s); }
        }
        Stats total_stats;
        const bool par = nparts > 1;
        (void)par;
        FEMD_OMP_PARALLEL_IF(par)
        {
            Scratch sc;
            sc.locr.assign(static_cast<std::size_t>(n), -1); sc.locc.assign(static_cast<std::size_t>(n), -1);
            Stats st;
            FEMD_OMP(for schedule(dynamic, 1))
            for (int k = 0; k < nparts; ++k)
                for (int q = partptr[k]; q < partptr[k + 1]; ++q) process_front(nodes[q], Ap, Ai, Ax, Cp, Ci, Cx, sc, st, false);
            FEMD_OMP(critical)
            total_stats.merge(st);
        }
        {
            Scratch sc;
            sc.locr.assign(static_cast<std::size_t>(n), -1); sc.locc.assign(static_cast<std::size_t>(n), -1);
            Stats st;
            for (int s : top) process_front(s, Ap, Ai, Ax, Cp, Ci, Cx, sc, st, true);
            total_stats.merge(st);
        }
        delayed = total_stats.delayed; max_front = total_stats.max_front; offdiag = total_stats.offdiag;
        minpiv = total_stats.minpiv < std::numeric_limits<double>::infinity() ? total_stats.minpiv : 0.0;
        maxpiv = total_stats.maxpiv;
        std::vector<std::vector<int>>().swap(cb_r_); std::vector<std::vector<int>>().swap(cb_c_); std::vector<std::vector<T>>().swap(cb_);
        std::vector<std::vector<int>>().swap(delay_r_); std::vector<std::vector<int>>().swap(delay_c_);
    }

private:
    std::vector<std::vector<int>> cb_r_, cb_c_, delay_r_, delay_c_;   // contribution blocks and delayed variables, per supernode
    std::vector<std::vector<T>> cb_;

    void process_front(int s, const int *Ap, const int *Ai, const T *Ax, const int *Cp, const int *Ci, const T *Cx, Scratch &sc, Stats &st, bool par)
    {
        std::vector<int> &locr = sc.locr, &locc = sc.locc;
        std::vector<T> &F = sc.F, &col = sc.col;
        const int f = sn_first[s], l = sn_first[s + 1] - 1;
        Front &fr = fronts[s];
        std::vector<int> &ridx = fr.ridx, &cidx = fr.cidx;
        ridx.clear(); cidx.clear();
        for (int c : sn_children[s]) { ridx.insert(ridx.end(), delay_r_[c].begin(), delay_r_[c].end()); cidx.insert(cidx.end(), delay_c_[c].begin(), delay_c_[c].end()); }
        for (int j = f; j <= l; ++j) { ridx.push_back(j); cidx.push_back(j); }
        const int p = static_cast<int>(ridx.size());
        ridx.insert(ridx.end(), sn_rows[s].begin(), sn_rows[s].end());
        cidx.insert(cidx.end(), sn_rows[s].begin(), sn_rows[s].end());
        const int m = static_cast<int>(ridx.size());
        fr.m = m; fr.p = p;
        st.max_front = std::max(st.max_front, m);
        for (int t = 0; t < m; ++t) { locr[ridx[t]] = t; locc[cidx[t]] = t; }
        F.assign(static_cast<std::size_t>(m) * m, T(0));
        // the entries of A, each in the front of the earlier of its two variables: from the rows
        // of the supernode's variables the entries at variables of the supernode or beyond, from
        // their columns the entries at rows beyond the supernode
        for (int j = f; j <= l; ++j)
        {
            const int r = perm[j];
            for (int q = Ap[r]; q < Ap[r + 1]; ++q)
            {
                const int i = pinv[Ai[q]];
                if (i >= f) F[static_cast<std::size_t>(locc[i]) * m + locr[j]] += Ax[q];
            }
            for (int q = Cp[r]; q < Cp[r + 1]; ++q)
            {
                const int i = pinv[Ci[q]];
                if (i > l) F[static_cast<std::size_t>(locc[j]) * m + locr[i]] += Cx[q];
            }
        }
        // the children's contribution blocks
        for (int c : sn_children[s])
        {
            const std::vector<int> &cr = cb_r_[c], &ccol = cb_c_[c];
            const std::vector<T> &C = cb_[c];
            const int mc = static_cast<int>(cr.size());
            for (int b = 0; b < mc; ++b)
            {
                T *Fcol = &F[static_cast<std::size_t>(locc[ccol[b]]) * m];
                const T *Cb = &C[static_cast<std::size_t>(b) * mc];
                for (int a = 0; a < mc; ++a) Fcol[locr[cr[a]]] += Cb[a];
            }
            std::vector<T>().swap(cb_[c]);
            std::vector<int>().swap(cb_r_[c]); std::vector<int>().swap(cb_c_[c]);
        }
        // eliminate the fully summed variables 0 .. p-1 in panels
        const bool root = sn_parent[s] < 0;
        col.resize(static_cast<std::size_t>(m));
        int k = 0, ps = 0;
        auto candidate = [&](int j, int k0, int ps0) {        // column j with the panel's pivots applied, rows >= k0
            const T *Fj = &F[static_cast<std::size_t>(j) * m];
            for (int i = k0; i < m; ++i) col[i] = Fj[i];
            for (int t = ps0; t < k0; ++t)
            {
                const T utj = Fj[t];
                if (utj == T(0)) continue;
                const T *Lt = &F[static_cast<std::size_t>(t) * m];
                for (int i = k0; i < m; ++i) col[i] -= Lt[i] * utj;
            }
        };
        auto trailing_update = [&](int k0, int ps0) {        // F(i, j) -= sum_t L(i, t) U(t, j), i, j >= k0, t in [ps0, k0)
            if (k0 == ps0 || k0 >= m) return;
            const int nt = k0 - ps0, rows = m - k0;
            const bool big = par && static_cast<long>(rows) * rows * nt > 4 * FEMD_OMP_THRESHOLD;
            (void)big;
            FEMD_OMP_FOR_IF(big)
            for (int j = k0; j < m; ++j)
            {
                T *Fj = &F[static_cast<std::size_t>(j) * m];
                for (int t = ps0; t < k0; ++t)
                {
                    const T utj = Fj[t];
                    if (utj == T(0)) continue;
                    const T *Lt = &F[static_cast<std::size_t>(t) * m];
                    for (int i = k0; i < m; ++i) Fj[i] -= Lt[i] * utj;
                }
            }
        };
        auto swap_rows = [&](int a, int b) {
            if (a == b) return;
            for (int j = 0; j < m; ++j) std::swap(F[static_cast<std::size_t>(j) * m + a], F[static_cast<std::size_t>(j) * m + b]);
            std::swap(ridx[a], ridx[b]);
            std::swap(col[a], col[b]);
        };
        auto swap_cols = [&](int a, int b) {
            if (a == b) return;
            T *Fa = &F[static_cast<std::size_t>(a) * m], *Fb = &F[static_cast<std::size_t>(b) * m];
            for (int i = 0; i < m; ++i) std::swap(Fa[i], Fb[i]);
            std::swap(cidx[a], cidx[b]);
        };
        while (k < p)
        {
            int pj = -1, pi = -1, bj = -1, bi = -1;
            double best = 0.0;
            for (int j = k; j < p; ++j)
            {
                candidate(j, k, ps);
                double cmax = 0.0;
                for (int i = k; i < m; ++i) cmax = std::max(cmax, std::abs(col[i]));
                if (cmax <= 0.0) continue;
                int id = -1, imax = -1;
                double amax = -1.0;
                for (int i = k; i < p; ++i)
                {
                    const double a = std::abs(col[i]);
                    if (ridx[i] == cidx[j]) id = i;
                    if (a > amax) { amax = a; imax = i; }
                }
                if (id >= 0 && std::abs(col[id]) >= u * cmax) { pj = j; pi = id; break; }
                if (amax >= u * cmax) { pj = j; pi = imax; break; }
                if (amax > best) { best = amax; bj = j; bi = imax; }
            }
            if (pj < 0)
            {
                if (!root) break;                                    // delay the rest to the parent
                if (bj < 0) throw std::runtime_error("LU: the matrix is singular (no nonzero pivot left, " + std::to_string(p - k) + " variables)");
                pj = bj; pi = bi;                                    // the root has no parent to delay to
                candidate(pj, k, ps);
            }
            swap_cols(pj, k);
            swap_rows(pi, k);
            if (ridx[k] != cidx[k]) ++st.offdiag;
            T *Fk = &F[static_cast<std::size_t>(k) * m];
            const T pivot = col[k];
            const double ap = std::abs(pivot);
            st.minpiv = std::min(st.minpiv, ap);
            st.maxpiv = std::max(st.maxpiv, ap);
            Fk[k] = pivot;
            for (int i = k + 1; i < m; ++i) Fk[i] = col[i] / pivot;                      // the column of L
            for (int t = ps; t < k; ++t)                                                 // the row of U: the panel's pivots applied
            {
                const T lkt = F[static_cast<std::size_t>(t) * m + k];
                if (lkt == T(0)) continue;
                for (int j = k + 1; j < m; ++j) F[static_cast<std::size_t>(j) * m + k] -= lkt * F[static_cast<std::size_t>(j) * m + t];
            }
            ++k;
            if (k - ps == FEMD_LU_PANEL || k == p) { trailing_update(k, ps); ps = k; }
        }
        if (k != ps) { trailing_update(k, ps); ps = k; }
        fr.ne = k;
        delay_r_[s].assign(ridx.begin() + k, ridx.begin() + p);
        delay_c_[s].assign(cidx.begin() + k, cidx.begin() + p);
        st.delayed += p - k;
        // store L (m x ne) and U^T (ne x m), hand the rest of the front to the parent
        fr.L.assign(F.begin(), F.begin() + static_cast<std::size_t>(k) * m);
        fr.Ut.assign(static_cast<std::size_t>(k) * m, T(0));
        for (int c = 0; c < k; ++c)
            for (int j = c + 1; j < m; ++j) fr.Ut[static_cast<std::size_t>(c) * m + j] = F[static_cast<std::size_t>(j) * m + c];
        const int mc = m - k;
        if (mc > 0 && !root)
        {
            cb_r_[s].assign(ridx.begin() + k, ridx.end());
            cb_c_[s].assign(cidx.begin() + k, cidx.end());
            cb_[s].resize(static_cast<std::size_t>(mc) * mc);
            for (int b = 0; b < mc; ++b)
                for (int a = 0; a < mc; ++a) cb_[s][static_cast<std::size_t>(b) * mc + a] = F[static_cast<std::size_t>(k + b) * m + (k + a)];
        }
        for (int t = 0; t < m; ++t) { locr[ridx[t]] = -1; locc[cidx[t]] = -1; }
    }

public:
    /// The factors as lu.hpp keeps them: L and U by columns in the pivoted numbering (step k eliminates
    /// row pinv_out[row] = k and column q_out[k]), explicit zeros dropped, the rows of L sorted.
    void export_csc(std::vector<int> &q_out, std::vector<int> &pinv_out, std::vector<int> &Lp, std::vector<int> &Li, std::vector<T> &Lx,
                    std::vector<int> &Up, std::vector<int> &Ui, std::vector<T> &Ux) const
    {
        // the step of each variable as a row and as a column
        std::vector<int> rstep(static_cast<std::size_t>(n), -1), cstep(static_cast<std::size_t>(n), -1);
        int step = 0;
        for (const Front &fr : fronts)
            for (int c = 0; c < fr.ne; ++c) { rstep[fr.ridx[c]] = step; cstep[fr.cidx[c]] = step; ++step; }
        if (step != n) throw std::runtime_error("LU: the frontal factorization eliminated " + std::to_string(step) + " of " + std::to_string(n) + " variables");
        q_out.resize(static_cast<std::size_t>(n)); pinv_out.resize(static_cast<std::size_t>(n));
        for (int v = 0; v < n; ++v) { q_out[cstep[v]] = perm[v]; pinv_out[perm[v]] = rstep[v]; }
        // counts
        Lp.assign(static_cast<std::size_t>(n) + 1, 0); Up.assign(static_cast<std::size_t>(n) + 1, 0);
        for (const Front &fr : fronts)
        {
            const int m = fr.m;
            for (int c = 0; c < fr.ne; ++c)
            {
                const int k = cstep[fr.cidx[c]];
                int nl = 1, nu = 1;                                            // the unit diagonal of L, the pivot of U
                const T *Lc = &fr.L[static_cast<std::size_t>(c) * m];
                for (int i = c + 1; i < m; ++i) if (Lc[i] != T(0)) ++nl;
                const T *Uc = &fr.Ut[static_cast<std::size_t>(c) * m];
                for (int j = c + 1; j < m; ++j) if (Uc[j] != T(0)) ++Up[cstep[fr.cidx[j]] + 1];   // U(c, j) lies in column j
                Lp[k + 1] += nl; Up[k + 1] += nu;
            }
        }
        for (int k = 0; k < n; ++k) { Lp[k + 1] += Lp[k]; Up[k + 1] += Up[k]; }
        Li.resize(static_cast<std::size_t>(Lp[n])); Lx.resize(Li.size());
        Ui.resize(static_cast<std::size_t>(Up[n])); Ux.resize(Ui.size());
        std::vector<int> lpos(Lp.begin(), Lp.end() - 1), upos(Up.begin(), Up.end() - 1);
        for (const Front &fr : fronts)
        {
            const int m = fr.m;
            for (int c = 0; c < fr.ne; ++c)
            {
                const int k = cstep[fr.cidx[c]];
                const T *Lc = &fr.L[static_cast<std::size_t>(c) * m], *Uc = &fr.Ut[static_cast<std::size_t>(c) * m];
                Li[lpos[k]] = k; Lx[lpos[k]++] = T(1);
                for (int i = c + 1; i < m; ++i) if (Lc[i] != T(0)) { Li[lpos[k]] = rstep[fr.ridx[i]]; Lx[lpos[k]++] = Lc[i]; }
                for (int j = c + 1; j < m; ++j) if (Uc[j] != T(0)) { const int kj = cstep[fr.cidx[j]]; Ui[upos[kj]] = k; Ux[upos[kj]++] = Uc[j]; }
            }
        }
        // the pivots last in their columns of U (the rows k of column kj above were all < kj, in front order)
        for (const Front &fr : fronts)
            for (int c = 0; c < fr.ne; ++c)
            {
                const int k = cstep[fr.cidx[c]];
                Ui[upos[k]] = k; Ux[upos[k]++] = fr.L[static_cast<std::size_t>(c) * fr.m + c];
            }
        // sort the rows of each column of L after the unit diagonal
        std::vector<int> order; std::vector<int> ti; std::vector<T> tx;
        for (int k = 0; k < n; ++k)
        {
            const int p0 = Lp[k] + 1, len = Lp[k + 1] - p0;
            if (len < 2) continue;
            bool sorted = true;
            for (int p = p0 + 1; p < p0 + len && sorted; ++p) sorted = Li[p - 1] < Li[p];
            if (sorted) continue;
            order.resize(static_cast<std::size_t>(len));
            for (int t = 0; t < len; ++t) order[t] = p0 + t;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return Li[a] < Li[b]; });
            ti.resize(static_cast<std::size_t>(len)); tx.resize(static_cast<std::size_t>(len));
            for (int t = 0; t < len; ++t) { ti[t] = Li[order[t]]; tx[t] = Lx[order[t]]; }
            for (int t = 0; t < len; ++t) { Li[p0 + t] = ti[t]; Lx[p0 + t] = tx[t]; }
        }
    }
};

} // namespace frontal
} // namespace femd

#endif // FEMD_SPARSE_LU_FRONTAL_HPP
