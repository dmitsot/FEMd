//
//  lu_frontal.hpp  --  the multifrontal numeric phase of FEMd's sparse LU (lu.hpp):
//  Gaussian elimination with threshold partial pivoting inside dense frontal matrices,
//  after Duff and Reid and MA41, for matrices whose pattern is symmetric or nearly so
//  (finite element matrices: the Navier-Stokes Jacobian, saddle points, convection-diffusion).
//
//  The symbolic phase works on the pattern of A + A^T: it post-orders the elimination tree
//  of the ordered pattern, groups its chains into supernodes, each with the row structure of
//  its columns, and merges a supernode with the child just before it when the merged front
//  keeps few explicit zeros (relaxed amalgamation, CHOLMOD's rule).  The numeric phase walks
//  the supernodes children first.  A supernode's front is a dense square matrix on its
//  variables, the rows and columns below them, and the variables its children could not
//  eliminate; it is assembled from the entries of A and the children's contribution blocks.
//  Its fully summed variables are eliminated in panels of FEMD_LU_PANEL: a pivot for column j
//  is the entry of a fully summed row that is at least u times the largest entry of the whole
//  column (the diagonal when it passes, else the largest), the pivot row and column are
//  swapped into place, the column of L and the row of U in the fully summed columns are
//  computed with the panel's earlier pivots applied, and when the panel is full the other
//  columns receive the panel's rows of U (a small triangular solve) and the trailing update,
//  a dense product in register blocks, in parallel over the columns.  A column with no
//  acceptable pivot is delayed, with its row, to the parent front, where more of it is
//  summed.  Pivoting inside the fronts keeps the fill of the symmetric pattern (plus what the
//  delays add) and bounds the growth by 1/u per step.
//
//  The factors come out in the form lu.hpp uses, in the pivoted numbering: L by columns and U
//  by rows (a front holds whole columns of L and whole rows of U, so the export runs in
//  parallel over the fronts), with the tree of the fronts for the parallel solves of lu.hpp.
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

#ifndef FEMD_LU_RELAX
#define FEMD_LU_RELAX 1
#endif
#ifndef FEMD_LU_PANEL
#define FEMD_LU_PANEL 32
#endif

namespace femd {
namespace frontal {

// acc -= a b, and |x|^2, with complex products by components (no NaN recovery, so loops vectorize)
inline void msub(double &acc, double a, double b) { acc -= a * b; }
inline void msub(std::complex<double> &acc, const std::complex<double> &a, const std::complex<double> &b)
{
    acc = std::complex<double>(acc.real() - (a.real() * b.real() - a.imag() * b.imag()), acc.imag() - (a.real() * b.imag() + a.imag() * b.real()));
}
inline double absq(double x) { return x * x; }
inline double absq(const std::complex<double> &z) { return z.real() * z.real() + z.imag() * z.imag(); }
inline double inverse(double x) { return 1.0 / x; }
inline std::complex<double> inverse(const std::complex<double> &z) { const double d = absq(z); return std::complex<double>(z.real() / d, -z.imag() / d); }
inline double mul(double a, double b) { return a * b; }
inline std::complex<double> mul(const std::complex<double> &a, const std::complex<double> &b)
{
    return std::complex<double>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// The trailing update of a front, F(i, j) -= sum_t L(i, t) U(t, j) for rows i >= k0 and the columns
// j0 <= j < j1 (at most NC of them), t in [ps0, k0): F is m x m column-major, L(i, t) = F(i, t) and
// U(t, j) = F(t, j).  The rows ps0 .. k0-1 of the columns j >= jt are first solved for U (panel_solve);
// those of the columns before jt were updated pivot by pivot.  Register blocks of MR rows by the NC columns accumulate the sum over t in order
// and subtract it once, so each entry is read and written once per panel.  Complex values are
// multiplied by components (no NaN recovery of the library product, which keeps the loop vectorizable).
template <class T> struct Kernel;

// The rows ps0 .. k0-1 of column j of U from those of F: the forward substitution with the panel's unit
// lower triangle L(t2, t) = F(t2, t), t < t2 < k0 (column j of F is m long, column-major).
template <class T>
inline void panel_solve(T *F, int m, int k0, int ps0, int j)
{
    T *Fj = F + static_cast<std::size_t>(j) * m;
    for (int t = ps0; t < k0; ++t)
    {
        const T u = Fj[t];
        if (u == T(0)) continue;
        const T *Lt = F + static_cast<std::size_t>(t) * m;
        for (int t2 = t + 1; t2 < k0; ++t2) msub(Fj[t2], Lt[t2], u);
    }
}

template <>
struct Kernel<double> {
    static constexpr int NC = 4, MR = 4;
    template <int C>
    static void cols(double *F, int m, int k0, int ps0, int j0)
    {
        const int nt = k0 - ps0;
        double b[C][FEMD_LU_PANEL];
        double *Cc[C];
        for (int c = 0; c < C; ++c) { Cc[c] = F + static_cast<std::size_t>(j0 + c) * m; for (int t = 0; t < nt; ++t) b[c][t] = Cc[c][ps0 + t]; }
        const double *A0 = F + static_cast<std::size_t>(ps0) * m;
        int i = k0;
        for (; i + MR <= m; i += MR)
        {
            double acc[C][MR] = {};
            const double *A = A0 + i;
            for (int t = 0; t < nt; ++t, A += m)
                for (int c = 0; c < C; ++c)
                    for (int r = 0; r < MR; ++r) acc[c][r] += A[r] * b[c][t];
            for (int c = 0; c < C; ++c)
                for (int r = 0; r < MR; ++r) Cc[c][i + r] -= acc[c][r];
        }
        for (; i < m; ++i)
        {
            double acc[C] = {};
            const double *A = A0 + i;
            for (int t = 0; t < nt; ++t, A += m)
                for (int c = 0; c < C; ++c) acc[c] += A[0] * b[c][t];
            for (int c = 0; c < C; ++c) Cc[c][i] -= acc[c];
        }
    }
    static void update(double *F, int m, int k0, int ps0, int j0, int j1, int jt)
    {
        for (int j = std::max(j0, jt); j < j1; ++j) panel_solve(F, m, k0, ps0, j);
        switch (j1 - j0)
        {
            case 4: cols<4>(F, m, k0, ps0, j0); break;
            case 3: cols<3>(F, m, k0, ps0, j0); break;
            case 2: cols<2>(F, m, k0, ps0, j0); break;
            case 1: cols<1>(F, m, k0, ps0, j0); break;
            default: break;
        }
    }
};

template <>
struct Kernel<std::complex<double>> {
    using cd = std::complex<double>;
    static constexpr int NC = 2, MR = 4;
    template <int C>
    static void cols(cd *Fc, int m, int k0, int ps0, int j0)
    {
        const int nt = k0 - ps0;
        double *F = reinterpret_cast<double *>(Fc);                   // (re, im) pairs, as the standard guarantees
        double br[C][FEMD_LU_PANEL], bi[C][FEMD_LU_PANEL];
        double *Cc[C];
        for (int c = 0; c < C; ++c)
        {
            Cc[c] = F + 2 * static_cast<std::size_t>(j0 + c) * m;
            for (int t = 0; t < nt; ++t) { br[c][t] = Cc[c][2 * (ps0 + t)]; bi[c][t] = Cc[c][2 * (ps0 + t) + 1]; }
        }
        const double *A0 = F + 2 * static_cast<std::size_t>(ps0) * m;
        int i = k0;
        for (; i + MR <= m; i += MR)
        {
            double ar[C][MR] = {}, ai[C][MR] = {};
            const double *A = A0 + 2 * i;
            for (int t = 0; t < nt; ++t, A += 2 * m)
                for (int c = 0; c < C; ++c)
                    for (int r = 0; r < MR; ++r)
                    {
                        const double xr = A[2 * r], xi = A[2 * r + 1];
                        ar[c][r] += xr * br[c][t] - xi * bi[c][t];
                        ai[c][r] += xr * bi[c][t] + xi * br[c][t];
                    }
            for (int c = 0; c < C; ++c)
                for (int r = 0; r < MR; ++r) { Cc[c][2 * (i + r)] -= ar[c][r]; Cc[c][2 * (i + r) + 1] -= ai[c][r]; }
        }
        for (; i < m; ++i)
        {
            double ar[C] = {}, ai[C] = {};
            const double *A = A0 + 2 * i;
            for (int t = 0; t < nt; ++t, A += 2 * m)
                for (int c = 0; c < C; ++c)
                {
                    ar[c] += A[0] * br[c][t] - A[1] * bi[c][t];
                    ai[c] += A[0] * bi[c][t] + A[1] * br[c][t];
                }
            for (int c = 0; c < C; ++c) { Cc[c][2 * i] -= ar[c]; Cc[c][2 * i + 1] -= ai[c]; }
        }
    }
    static void update(cd *F, int m, int k0, int ps0, int j0, int j1, int jt)
    {
        for (int j = std::max(j0, jt); j < j1; ++j) panel_solve(F, m, k0, ps0, j);
        switch (j1 - j0)
        {
            case 2: cols<2>(F, m, k0, ps0, j0); break;
            case 1: cols<1>(F, m, k0, ps0, j0); break;
            default: break;
        }
    }
};

template <class T>
struct Factor {
    using kernel = Kernel<T>;
    struct Front {
        int m = 0, ne = 0, p = 0;
        std::vector<int> ridx, cidx;   // the m row and column variables; the first ne are eliminated in this order
        std::vector<T> L;              // m x ne, column-major: L(i, c) for i > c, the pivot at (c, c)
        std::vector<T> U;              // ne x m, column-major: U(c, j) at [j ne + c], for j > c
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

    /// Relaxed amalgamation (on by default): a supernode absorbs the child that ends just before it
    /// when the merged front keeps few explicit zeros.  Fewer, larger fronts: less assembly and
    /// copying per entry, longer panels, more room for pivots, for some arithmetic on zeros.
    bool relax = FEMD_LU_RELAX != 0;

    // Merge supernodes (sn_first, contiguous in the post-order) bottom up.  The node just before s in
    // the post-order is its last child when s has children; with nc and ns columns, rc and rs rows
    // below them, the merged node has the rows of s, nc + ns columns and (nc+ns)(nc+ns-1)/2 + (nc+ns) rs
    // entries in L, of which the zeros are those of the two plus what the child's columns gain.  The
    // test is CHOLMOD's: always for at most 4 columns, at most 80% zeros for 16, 10% for 48, 5% beyond.
    void amalgamate(const std::vector<int> &cc)
    {
        const int nsf = static_cast<int>(sn_first.size());
        std::vector<int> last(static_cast<std::size_t>(nsf));
        for (int t = 0; t < nsf; ++t) last[t] = (t + 1 < nsf ? sn_first[t + 1] : n) - 1;
        std::vector<int> first(sn_first), sn_of_col(static_cast<std::size_t>(n));
        for (int t = 0; t < nsf; ++t) for (int j = first[t]; j <= last[t]; ++j) sn_of_col[j] = t;
        std::vector<double> zeros(static_cast<std::size_t>(nsf), 0.0);
        std::vector<char> keep(static_cast<std::size_t>(nsf), 1);
        auto entries = [](double nc, double r) { return nc * (nc - 1.0) / 2.0 + nc * r; };
        for (int t = 1; t < nsf; ++t)
        {
            const int c = t - 1;                               // the node (maybe merged) that ends just before t
            const int pc = parent[last[c]];
            if (pc < 0 || sn_of_col[pc] != t) continue;        // not a child of t
            const double nc = last[c] - first[c] + 1, nt = last[t] - first[t] + 1;
            const double rc = cc[last[c]], rt = cc[last[t]];
            const double ncol = nc + nt;
            const double total = entries(ncol, rt);
            const double z = zeros[c] + zeros[t] + total - entries(nc, rc) - entries(nt, rt);
            const double frac = total > 0.0 ? z / total : 0.0;
            const bool merge = ncol <= 4 || (ncol <= 16 && frac < 0.8) || (ncol <= 48 && frac < 0.1) || frac < 0.05;
            if (!merge) continue;
            keep[c] = 0;
            first[t] = first[c];
            zeros[t] = z;
        }
        sn_first.clear();
        for (int t = 0; t < nsf; ++t) if (keep[t]) sn_first.push_back(first[t]);
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
        for (int j = 0; j < n; ++j)
        {
            const bool merge = j > 0 && parent[j - 1] == j && nchild[j] == 1 && cc[j - 1] == cc[j] + 1;
            if (!merge) sn_first.push_back(j);
        }
        if (relax) amalgamate(cc);
        std::vector<int> sn_of(static_cast<std::size_t>(n));
        for (std::size_t t = 0; t + 1 <= sn_first.size(); ++t)
            for (int j = sn_first[t]; j < (t + 1 < sn_first.size() ? sn_first[t + 1] : n); ++j) sn_of[j] = static_cast<int>(t);
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
        std::vector<int> locr, locc, rloc;
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
            std::vector<int> &rl = sc.rloc;
            rl.resize(static_cast<std::size_t>(mc));
            for (int a = 0; a < mc; ++a) rl[a] = locr[cr[a]];
            const int *rlv = rl.data();
            for (int b = 0; b < mc; ++b)
            {
                T *Fcol = &F[static_cast<std::size_t>(locc[ccol[b]]) * m];
                const T *Cb = &C[static_cast<std::size_t>(b) * mc];
                for (int a = 0; a < mc; ++a) Fcol[rlv[a]] += Cb[a];
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
                T *cv = col.data();
                for (int i = k0; i < m; ++i) msub(cv[i], Lt[i], utj);
            }
        };
        // F(i, j) -= sum_t L(i, t) U(t, j), i, j >= k0, t in [ps0, k0); the panel's rows of U in the
        // columns beyond the fully summed ones (j >= p) are solved here, in parallel over the columns
        auto trailing_update = [&](int k0, int ps0) {
            if (k0 == ps0 || k0 >= m) return;
            const int nt = k0 - ps0, rows = m - k0;
            const bool big = par && static_cast<long>(rows) * rows * nt > 4 * FEMD_OMP_THRESHOLD;
            (void)big;
            // blocks of kernel::NC columns from k0, each the same arithmetic whatever thread does it
            const int ng = (rows + kernel::NC - 1) / kernel::NC;
            T *F0 = F.data();
            FEMD_OMP_FOR_IF(big)
            for (int g = 0; g < ng; ++g)
            {
                const int j0 = k0 + g * kernel::NC;
                kernel::update(F0, m, k0, ps0, j0, std::min(m, j0 + kernel::NC), p);
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
                // squared magnitudes: |x| >= u |y| as |x|^2 >= u^2 |y|^2
                double cmax = 0.0;
                for (int i = k; i < m; ++i) cmax = std::max(cmax, absq(col[i]));
                if (cmax <= 0.0) continue;
                int id = -1, imax = -1;
                double amax = -1.0;
                for (int i = k; i < p; ++i)
                {
                    const double a = absq(col[i]);
                    if (ridx[i] == cidx[j]) id = i;
                    if (a > amax) { amax = a; imax = i; }
                }
                const double lim = u * u * cmax;
                if (id >= 0 && absq(col[id]) >= lim) { pj = j; pi = id; break; }
                if (amax >= lim) { pj = j; pi = imax; break; }
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
            const T rp = inverse(pivot);
            for (int i = k + 1; i < m; ++i) Fk[i] = mul(col[i], rp);                      // the column of L
            if (k > ps)                     // the row of U in the fully summed columns (the pivot candidates): the panel's pivots applied
            {
                T lk[FEMD_LU_PANEL];
                const int w = k - ps;
                for (int t = 0; t < w; ++t) lk[t] = F[static_cast<std::size_t>(ps + t) * m + k];
                for (int j = k + 1; j < p; ++j)
                {
                    T *Fj = &F[static_cast<std::size_t>(j) * m];
                    T v = Fj[k];
                    for (int t = 0; t < w; ++t) msub(v, lk[t], Fj[ps + t]);
                    Fj[k] = v;
                }
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
        fr.U.resize(static_cast<std::size_t>(k) * m);
        for (int j = 0; j < m; ++j)
        {
            const T *Fj = &F[static_cast<std::size_t>(j) * m];
            T *Uj = &fr.U[static_cast<std::size_t>(j) * k];
            for (int c = 0; c < k; ++c) Uj[c] = c < j ? Fj[c] : T(0);
        }
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
    /// The factors as lu.hpp keeps them, in the pivoted numbering (step k eliminates row pinv_out[row] = k
    /// and column q_out[k]): L by columns (the unit diagonal first, the other rows increasing) and U by
    /// rows (the pivot first, the other columns increasing), explicit zeros dropped; and the tree of the
    /// fronts on the steps, parent_out: the steps of a front form a chain, and its last step's parent is
    /// the first step that eliminates one of the variables the front passed on (its rows or columns
    /// beyond the eliminated ones).  Every entry of L(i, j) or U(j, i) has i an ancestor of j in it.
    /// The fronts own their columns of L and rows of U, so everything runs in parallel over them.
    void export_factors(std::vector<int> &q_out, std::vector<int> &pinv_out, std::vector<int> &Lp, std::vector<int> &Li, std::vector<T> &Lx,
                        std::vector<int> &Up, std::vector<int> &Ui, std::vector<T> &Ux, std::vector<int> &parent_out) const
    {
        const int nf = static_cast<int>(fronts.size());
        std::vector<int> fstart(static_cast<std::size_t>(nf) + 1, 0);       // the first step of each front
        for (int f = 0; f < nf; ++f) fstart[f + 1] = fstart[f] + fronts[f].ne;
        if (fstart[nf] != n) throw std::runtime_error("LU: the frontal factorization eliminated " + std::to_string(fstart[nf]) + " of " + std::to_string(n) + " variables");
        const bool par = n > 2 * FEMD_OMP_THRESHOLD;
        (void)par;
        // the step of each variable as a row and as a column
        std::vector<int> rstep(static_cast<std::size_t>(n), -1), cstep(static_cast<std::size_t>(n), -1);
        FEMD_OMP(parallel for schedule(dynamic, 64) if (par))
        for (int f = 0; f < nf; ++f)
        {
            const Front &fr = fronts[f];
            for (int c = 0; c < fr.ne; ++c) { rstep[fr.ridx[c]] = fstart[f] + c; cstep[fr.cidx[c]] = fstart[f] + c; }
        }
        q_out.resize(static_cast<std::size_t>(n)); pinv_out.resize(static_cast<std::size_t>(n)); parent_out.assign(static_cast<std::size_t>(n), -1);
        for (int v = 0; v < n; ++v) { q_out[cstep[v]] = perm[v]; pinv_out[perm[v]] = rstep[v]; }
        // counts: column k of L and row k of U, both of the front that takes step k
        Lp.assign(static_cast<std::size_t>(n) + 1, 0); Up.assign(static_cast<std::size_t>(n) + 1, 0);
        FEMD_OMP(parallel for schedule(dynamic, 64) if (par))
        for (int f = 0; f < nf; ++f)
        {
            const Front &fr = fronts[f];
            const int m = fr.m, ne = fr.ne;
            for (int c = 0; c < ne; ++c)
            {
                const int k = fstart[f] + c;
                int nl = 1, nu = 1;                                            // the unit diagonal of L, the pivot of U
                const T *Lc = &fr.L[static_cast<std::size_t>(c) * m];
                for (int i = c + 1; i < m; ++i) nl += Lc[i] != T(0);
                const T *Uc = &fr.U[c];
                for (int j = c + 1; j < m; ++j) nu += Uc[static_cast<std::size_t>(j) * ne] != T(0);
                Lp[k + 1] = nl; Up[k + 1] = nu;
            }
            // the tree: a chain through the front, then the first step of what the front passed on
            if (ne > 0)
            {
                for (int c = 0; c + 1 < ne; ++c) parent_out[fstart[f] + c] = fstart[f] + c + 1;
                int pa = n;
                for (int i = ne; i < m; ++i) pa = std::min(pa, std::min(rstep[fr.ridx[i]], cstep[fr.cidx[i]]));
                parent_out[fstart[f] + ne - 1] = pa < n ? pa : -1;
            }
        }
        for (int k = 0; k < n; ++k) { Lp[k + 1] += Lp[k]; Up[k + 1] += Up[k]; }
        Li.resize(static_cast<std::size_t>(Lp[n])); Lx.resize(Li.size());
        Ui.resize(static_cast<std::size_t>(Up[n])); Ux.resize(Ui.size());
        // fill, then sort each column of L and row of U after its diagonal
        FEMD_OMP(parallel if (par))
        {
            std::vector<std::pair<int, T>> tmp;
            FEMD_OMP(for schedule(dynamic, 64))
            for (int f = 0; f < nf; ++f)
            {
                const Front &fr = fronts[f];
                const int m = fr.m, ne = fr.ne;
                for (int c = 0; c < ne; ++c)
                {
                    const int k = fstart[f] + c;
                    const T *Lc = &fr.L[static_cast<std::size_t>(c) * m], *Uc = &fr.U[c];
                    int pl = Lp[k], pu = Up[k];
                    Li[pl] = k; Lx[pl++] = T(1);
                    for (int i = c + 1; i < m; ++i) if (Lc[i] != T(0)) { Li[pl] = rstep[fr.ridx[i]]; Lx[pl++] = Lc[i]; }
                    Ui[pu] = k; Ux[pu++] = Lc[c];                                   // the pivot
                    for (int j = c + 1; j < m; ++j)
                    {
                        const T v = Uc[static_cast<std::size_t>(j) * ne];
                        if (v != T(0)) { Ui[pu] = cstep[fr.cidx[j]]; Ux[pu++] = v; }
                    }
                    sort_tail(Li, Lx, Lp[k] + 1, Lp[k + 1], tmp);
                    sort_tail(Ui, Ux, Up[k] + 1, Up[k + 1], tmp);
                }
            }
        }
    }

private:
    // the entries [a, b) of a column or row in increasing index order (usually nearly sorted already)
    static void sort_tail(std::vector<int> &I, std::vector<T> &X, int a, int b, std::vector<std::pair<int, T>> &tmp)
    {
        bool sorted = true;
        for (int p = a + 1; p < b && sorted; ++p) sorted = I[p - 1] < I[p];
        if (sorted) return;
        tmp.resize(static_cast<std::size_t>(b - a));
        for (int p = a; p < b; ++p) tmp[p - a] = {I[p], X[p]};
        std::sort(tmp.begin(), tmp.end(), [](const std::pair<int, T> &x, const std::pair<int, T> &y) { return x.first < y.first; });
        for (int p = a; p < b; ++p) { I[p] = tmp[p - a].first; X[p] = tmp[p - a].second; }
    }
};

} // namespace frontal
} // namespace femd

#endif // FEMD_SPARSE_LU_FRONTAL_HPP
