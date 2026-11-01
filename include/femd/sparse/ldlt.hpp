//
//  ldlt.hpp  --  sparse symmetric indefinite LDL^T with pivoting, FEMd's direct solver for
//  symmetric matrices that are not positive definite: saddle points (Stokes, mixed methods),
//  Helmholtz-type operators, and whatever the Cholesky of cholesky.hpp refuses.
//
//      SparseLDLT F(A);                // order (AMD), analyse, factor: P A P^T = L D L^T
//      F.apply(b, x);                  // x = A^{-1} b
//      F.refactor(A2);                 // same pattern, new values: the numeric phase only
//
//  The method is multifrontal (Duff and Reid).  The symbolic phase orders the matrix (AMD) and
//  post-orders its elimination tree, finds the column counts of L (Gilbert, Ng and Peyton), the
//  fundamental supernodes and their rows, and merges a supernode with its child when few
//  explicit zeros result (CHOLMOD's rule, as in cholesky.hpp).  The numeric phase walks the
//  supernodes children first.  A supernode's front is a dense symmetric matrix on the variables
//  its children could not eliminate, its own columns and the rows below them; it is assembled
//  from the entries of A and the children's contribution blocks, and its fully summed variables
//  are eliminated with 1x1 and 2x2 pivots chosen by the threshold test of Duff and Reid (as in
//  MA27/MA57): a 1x1 pivot a_kk is taken when |a_kk| >= u max_i |a_ik|, and otherwise a 2x2
//  pivot P on (k, r), r the largest fully summed entry of column k, when
//  |P^{-1}| [g_k; g_r] <= [1/u; 1/u] with g the largest entries outside P.  A variable that
//  passes neither test is delayed to the parent front, where more of its column is summed.
//  This bounds the growth of the factors by 1/u per step (u = 0.01 by default), so the
//  factorization is stable for any symmetric matrix that is not singular, at the price of some
//  extra fill where pivots are delayed.
//
//  The pivots only touch the fully summed columns of a front (all their rows); the update of
//  the rest, F22 -= L21 W^T with W = L21 D the columns before scaling, is done once at the end
//  of the front by the register-blocked kernel of the frontal LU (lu_frontal.hpp), as in the
//  Cholesky.  The supernodal tree is cut into subtrees factored concurrently; the fronts above
//  them are factored one by one with their updates, assembly and copies shared between the
//  threads.  The solves go the same way: the subtrees concurrently (what they owe the variables
//  eliminated above them is gathered per subtree and added in a fixed order), the fronts above
//  them with their rows shared.  Each front's pivots and arithmetic do not depend on the thread
//  that does them, so the factor and the solution have the same bits for any number of threads.
//
//  The matrix must hold both triangles with a symmetric pattern (FEMd assembles full
//  symmetric patterns).  Stored entries that are exactly zero (the uncoupled blocks of a
//  ProductSpace pattern) are left out of the ordering and the fronts (prune.hpp); a refactor
//  with values that are nonzero there analyses the matrix again.  D is block diagonal with 1x1
//  and 2x2 blocks, which also gives the inertia (the numbers of positive, negative and zero
//  eigenvalues, by Sylvester's law).
//
#ifndef FEMD_SPARSE_LDLT_HPP
#define FEMD_SPARSE_LDLT_HPP

#include "femd/sparse/amd.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/sparse/lu_frontal.hpp"
#include "femd/sparse/precond.hpp"
#include "femd/sparse/prune.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// the cut of the supernodal tree: subtrees of at most 1/FEMD_LDLT_FPARTS of the work, factored and
// solved concurrently; FEMD_LDLT_RELAX = 0 turns the relaxed amalgamation off
#ifndef FEMD_LDLT_FPARTS
#define FEMD_LDLT_FPARTS 32.0
#endif
#ifndef FEMD_LDLT_RELAX
#define FEMD_LDLT_RELAX 1
#endif
// a front of the top with at least this many rows is assembled, updated and copied in parallel
#ifndef FEMD_LDLT_PARFRONT
#define FEMD_LDLT_PARFRONT 256
#endif

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
        tm_ = Timings();
        const double t0 = now();
        n_ = A.nrows();
        pattern_ = A.pattern;
        prune_.keep_input = false;
        prune_.build(n_, A.indptr(), A.indices(), A.data.data(), true);
        tm_.prune = now() - t0;
        analyse(A);
    }

    /// @brief New values on the same pattern: the numeric phase only (pivots are chosen afresh).
    void refactor(const CSRMatrix &A)
    {
        if (A.nrows() != n_ || A.ncols() != n_ || A.nnz() != pattern_->nnz() ||
            (A.pattern != pattern_ && (A.pattern->indptr != pattern_->indptr || A.pattern->indices != pattern_->indices)))
            throw std::invalid_argument("LDL^T::refactor: the matrix has a different pattern");
        if (!prune_.fits(A.data.data()))
        {
            // nonzero where the analysis dropped exact zeros: order and analyse again
            tm_ = Timings();
            const double t0 = now();
            prune_.build(n_, A.indptr(), A.indices(), A.data.data(), true, true);
            tm_.prune = now() - t0;
            analyse(A);
            return;
        }
        double t = now();
        if (prune_.active) { prune_.gather(A.data.data(), B_.data); numeric(B_); }
        else numeric(A);
        tm_.numeric = now() - t; t = now();
        solve_setup();
        tm_.partition = now() - t;
    }

    /// @brief z = A^{-1} r.
    void apply(const double *r, double *z) const override
    {
        const int n = n_;
        std::vector<double> &yv = work_;
        yv.resize(static_cast<std::size_t>(n));
        double *y = yv.data();
        const int *perm = perm_.data();
        const bool par = fparts_ > 1;
        (void)par;
        FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
        for (int k = 0; k < n; ++k) y[k] = r[perm[k]];
        // L y = b: the subtrees concurrently, what they owe the variables above them gathered per
        // subtree, then added subtree by subtree; then the top in order
        FEMD_OMP(parallel if (par))
        {
            FEMD_OMP(for schedule(dynamic, 1))
            for (int k = 0; k < fparts_; ++k)
            {
                double *buf = obuf_[k].data();
                std::fill(buf, buf + obuf_[k].size(), 0.0);
                for (int q = fpart_ptr_[k]; q < fpart_ptr_[k + 1]; ++q) forward(fronts_[fpart_node_[q]], y, buf, false);
            }
        }
        for (int k = 0; k < fparts_; ++k)
        {
            const std::vector<int> &ov = ovar_[k];
            const std::vector<double> &buf = obuf_[k];
            for (std::size_t t = 0; t < ov.size(); ++t) y[ov[t]] -= buf[t];
        }
        for (int s : ftop_) forward(fronts_[s], y, nullptr, true);
        // D
        const int ns = static_cast<int>(fronts_.size());
        FEMD_OMP(parallel for schedule(dynamic, 64) if (par))
        for (int s = 0; s < ns; ++s)
        {
            const Front &f = fronts_[s];
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
        }
        // L^T x = y: the top from its last front, then the subtrees concurrently from their roots down
        std::vector<double> &w = work2_;
        for (auto it = ftop_.rbegin(); it != ftop_.rend(); ++it) backward(fronts_[*it], y, w, true);
        FEMD_OMP(parallel if (par))
        {
            std::vector<double> wl;
            FEMD_OMP(for schedule(dynamic, 1))
            for (int k = 0; k < fparts_; ++k)
                for (int q = fpart_ptr_[k + 1] - 1; q >= fpart_ptr_[k]; --q) backward(fronts_[fpart_node_[q]], y, wl, false);
        }
        FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
        for (int k = 0; k < n; ++k) z[perm[k]] = y[k];
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
    int supernodes() const { return static_cast<int>(sn_ptr_.size()) - 1; }
    int max_front() const { return max_front_; }
    /// @brief Independent subtrees of the supernodal tree that the factorization and the solves run concurrently.
    int factor_subtrees() const { return fparts_; }
    double threshold() const { return u_; }
    std::string ordering_name() const { return ordering_ == Ordering::AMD ? "amd" : "natural"; }
    /// @brief Stored entries of A that are exactly zero and were left out of the ordering and the factor.
    long long dropped_zeros() const { return prune_.ndropped(); }
    const std::vector<int> &permutation() const { return perm_; }
    /// @brief Smallest |eigenvalue| of a pivot block, over all blocks.
    double min_pivot() const { return min_piv_; }
    /// @brief Seconds spent in each phase of the last factorization (a refactor: numeric and partition).
    struct Timings { double prune = 0, ordering = 0, symbolic = 0, numeric = 0, partition = 0; };
    const Timings &timings() const { return tm_; }

private:
    struct Front {
        int m = 0, ne = 0;
        std::vector<int> idx;          // m variables (permuted indices), the first ne eliminated in this order
        std::vector<double> L;         // m x ne, column-major; column c holds L(i, c) for i > c
        std::vector<int> bs;           // pivot block sizes, 1 or 2
        std::vector<double> D;         // per block: d, or (a, b, c) of [[a, b], [b, c]]
        std::vector<int> slot;         // for the rows ne .. m-1 of a front in a subtree: -1 when the variable is
                                       // eliminated in the subtree, else its place in the subtree's gather buffer
        // the pivots' statistics, summed over the fronts in order after the factorization
        int npos = 0, nneg = 0, n2 = 0, ndel = 0;
        double minpiv = std::numeric_limits<double>::infinity();
    };

    Ordering ordering_;
    double u_;
    Timings tm_;
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    std::shared_ptr<const SparsityPattern> pattern_;
    ZeroPruning<double> prune_;              // the exact zeros of A left out (prune.hpp)
    CSRMatrix B_;                            // A without them, when prune_.active
    std::vector<int> perm_, pinv_, parent_;
    std::vector<int> colp_, coli_, colq_;    // the lower triangle of P A P^T by columns: rows coli_ (increasing), positions colq_ in A's data
    std::vector<int> sn_ptr_;                // supernode s has the columns sn_ptr_[s] .. sn_ptr_[s+1]-1
    std::vector<int> sn_parent_;             // its parent supernode, -1 at a root
    std::vector<int> sn_cptr_, sn_child_;    // its children, increasing
    std::vector<int> sn_rptr_, sn_row_;      // its rows below the last column, increasing
    int fparts_ = 0;
    std::vector<int> fpart_ptr_, fpart_node_, ftop_;   // the cut of the supernodal tree: the supernodes of each subtree, increasing, and the top
    bool relax_ = FEMD_LDLT_RELAX != 0;
    std::vector<Front> fronts_;
    std::vector<std::vector<double>> cb_;    // the contribution blocks during numeric(), packed lower triangles
    std::vector<std::vector<int>> cbidx_, delay_;
    std::vector<std::vector<int>> ovar_;     // per subtree: the variables eliminated above it that its fronts update
    mutable std::vector<std::vector<double>> obuf_;
    std::array<int, 3> inertia_{{0, 0, 0}};
    int delayed_ = 0, n2x2_ = 0, max_front_ = 0;
    double min_piv_ = 0.0;
    mutable std::vector<double> work_, work2_;

    /// Analyse and factor: on A without its exact zeros when some were found.
    void analyse(const CSRMatrix &A)
    {
        double t = now();
        if (prune_.active)
        {
            auto P = std::make_shared<SparsityPattern>();
            P->nrows = P->ncols = n_;
            P->indptr.swap(prune_.Bp);
            P->indices.swap(prune_.Bi);
            B_ = CSRMatrix(P, A.symmetric);
            prune_.gather(A.data.data(), B_.data);
        }
        else B_ = CSRMatrix();
        const CSRMatrix &M = prune_.active ? B_ : A;
        tm_.prune += now() - t; t = now();
        order(M);
        tm_.ordering = now() - t; t = now();
        symbolic(M);
        tm_.symbolic = now() - t; t = now();
        numeric(M);
        tm_.numeric = now() - t; t = now();
        solve_setup();
        tm_.partition = now() - t;
    }

    // ---- ordering: AMD (or natural), then a post-order of the elimination tree, so that each
    // chain of the tree is a run of consecutive columns and its parent is the next node
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

    void order(const CSRMatrix &A)
    {
        const int n = n_;
        const std::size_t N = static_cast<std::size_t>(n);
        const int *Ap = A.indptr(), *Ai = A.indices();
        std::vector<int> p0 = (ordering_ == Ordering::AMD) ? ordering::amd_order(n, Ap, Ai) : ordering::natural_order(n);
        std::vector<int> pinv0(N);
        for (int k = 0; k < n; ++k) pinv0[p0[k]] = k;
        const std::vector<int> par0 = etree_of(n, Ap, Ai, p0, pinv0);
        std::vector<int> head(N, -1), next(N, -1), stack(N), post;
        post.reserve(N);
        for (int j = n - 1; j >= 0; --j)
            if (par0[j] >= 0) { next[j] = head[par0[j]]; head[par0[j]] = j; }
        for (int r = 0; r < n; ++r)
        {
            if (par0[r] != -1) continue;
            int top = 0;
            stack[0] = r;
            while (top >= 0)
            {
                const int v = stack[top], c = head[v];
                if (c == -1) { --top; post.push_back(v); }
                else { head[v] = next[c]; stack[++top] = c; }
            }
        }
        // the new order and the tree in it: a post-order keeps the tree, relabelled
        std::vector<int> newpos(N);
        for (int k = 0; k < n; ++k) newpos[post[k]] = k;
        perm_.resize(N);
        parent_.assign(N, -1);
        for (int k = 0; k < n; ++k)
        {
            perm_[k] = p0[post[k]];
            if (par0[post[k]] >= 0) parent_[k] = newpos[par0[post[k]]];
        }
        pinv_.assign(N, 0);
        for (int k = 0; k < n; ++k) pinv_[perm_[k]] = k;
    }

    // ---- symbolic: the lower triangle by columns, the column counts (Gilbert, Ng and Peyton; the
    // identity is a post-order), the fundamental supernodes and their rows, the relaxed
    // amalgamation, and the cut of the supernodal tree
    void symbolic(const CSRMatrix &A)
    {
        const int n = n_;
        const std::size_t N = static_cast<std::size_t>(n);
        const int *Ap = A.indptr(), *Ai = A.indices();
        const int *perm = perm_.data(), *pinv = pinv_.data(), *pa = parent_.data();
        const bool par = n > 2 * FEMD_OMP_THRESHOLD;
        (void)par;
        // the lower triangle of P A P^T by columns, rows increasing: a counting sort over chunks of
        // rows, the chunks in parallel and laid out in order
        {
            const int nch = ordering::detail::sort_chunks(par), chunk = (n + nch - 1) / nch;
            std::vector<int> cnt(static_cast<std::size_t>(nch) * N, 0);
            FEMD_OMP(parallel for schedule(static, 1) if (nch > 1))
            for (int c = 0; c < nch; ++c)
            {
                int *cc = cnt.data() + static_cast<std::size_t>(c) * N;
                for (int i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i)
                    for (int q = Ap[perm[i]]; q < Ap[perm[i] + 1]; ++q)
                    {
                        const int j = pinv[Ai[q]];
                        if (j <= i) ++cc[j];
                    }
            }
            colp_.assign(N + 1, 0);
            for (int j = 0; j < n; ++j)
            {
                int off = colp_[j];
                for (int c = 0; c < nch; ++c)
                {
                    int &x = cnt[static_cast<std::size_t>(c) * N + j];
                    const int t = x;
                    x = off;
                    off += t;
                }
                colp_[j + 1] = off;
            }
            coli_.resize(static_cast<std::size_t>(colp_[n]));
            colq_.resize(coli_.size());
            int *ci = coli_.data(), *cq = colq_.data();
            FEMD_OMP(parallel for schedule(static, 1) if (nch > 1))
            for (int c = 0; c < nch; ++c)
            {
                int *pos = cnt.data() + static_cast<std::size_t>(c) * N;
                for (int i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i)
                    for (int q = Ap[perm[i]]; q < Ap[perm[i] + 1]; ++q)
                    {
                        const int j = pinv[Ai[q]];
                        if (j <= i) { const int k = pos[j]++; ci[k] = i; cq[k] = q; }
                    }
            }
        }
        // the column counts of L below the diagonal (the identity is a post-order of the tree)
        std::vector<int> lnz(N);
        {
            std::vector<int> first(N, -1), maxfirst(N, -1), prevleaf(N, -1), anc(N);
            int *cc = lnz.data();
            for (int k = 0; k < n; ++k)
            {
                int j = k;
                cc[j] = first[j] == -1 ? 1 : 0;
                for (; j != -1 && first[j] == -1; j = pa[j]) first[j] = k;
            }
            for (int i = 0; i < n; ++i) anc[i] = i;
            const int *cp = colp_.data(), *ci = coli_.data();
            for (int j = 0; j < n; ++j)
            {
                const int fj = first[j];
                if (pa[j] != -1) --cc[pa[j]];
                for (int q = cp[j]; q < cp[j + 1]; ++q)
                {
                    const int i = ci[q];
                    if (i <= j || fj <= maxfirst[i]) continue;
                    maxfirst[i] = fj;
                    const int jprev = prevleaf[i];
                    prevleaf[i] = j;
                    ++cc[j];
                    if (jprev == -1) continue;
                    int a = jprev;
                    while (a != anc[a]) a = anc[a];
                    for (int v = jprev; v != a;) { const int w = anc[v]; anc[v] = a; v = w; }
                    --cc[a];
                }
                if (pa[j] != -1) anc[j] = pa[j];
            }
            for (int j = 0; j < n; ++j) if (pa[j] != -1) cc[pa[j]] += cc[j];
            for (int j = 0; j < n; ++j) --cc[j];
        }
        // fundamental supernodes: j joins j-1 when j-1's parent is j, its only child, and the column
        // of j-1 is that of j plus the row j
        std::vector<int> nchild(N, 0), ff;
        for (int j = 0; j < n; ++j) if (pa[j] >= 0) ++nchild[pa[j]];
        for (int j = 0; j < n; ++j)
            if (!(j > 0 && pa[j - 1] == j && nchild[j] == 1 && lnz[j - 1] == lnz[j] + 1)) ff.push_back(j);
        const int nf = static_cast<int>(ff.size());
        ff.push_back(n);
        std::vector<int> fof(N);
        for (int t = 0; t < nf; ++t) for (int j = ff[t]; j < ff[t + 1]; ++j) fof[j] = t;
        std::vector<int> fpar(static_cast<std::size_t>(nf), -1), fcp(static_cast<std::size_t>(nf) + 1, 0), fch,
            frp(static_cast<std::size_t>(nf) + 1, 0);
        for (int t = 0; t < nf; ++t)
        {
            const int p = pa[ff[t + 1] - 1];
            if (p >= 0) { fpar[t] = fof[p]; ++fcp[fpar[t] + 1]; }
            frp[t + 1] = frp[t] + lnz[ff[t + 1] - 1];
        }
        for (int t = 0; t < nf; ++t) fcp[t + 1] += fcp[t];
        fch.resize(static_cast<std::size_t>(fcp[nf]));
        {
            std::vector<int> pos(fcp.begin(), fcp.end() - 1);
            for (int t = 0; t < nf; ++t) if (fpar[t] >= 0) fch[pos[fpar[t]]++] = t;
        }
        // their rows below the last column (those of their columns of A and of their children), over
        // subtrees of the tree concurrently, each thread with its own marks, then the rest in order
        std::vector<int> fr(static_cast<std::size_t>(frp[nf]));
        {
            std::vector<int> part(static_cast<std::size_t>(nf), -1);
            std::vector<double> W(static_cast<std::size_t>(nf));
            const int *cp = colp_.data(), *ci = coli_.data();
            double work = 0.0;
            for (int t = 0; t < nf; ++t) W[t] = (frp[t + 1] - frp[t]) + (cp[ff[t + 1]] - cp[ff[t]]) + 1.0;
            for (int t = 0; t < nf; ++t) { if (fpar[t] >= 0) W[fpar[t]] += W[t]; else work += W[t]; }
            const double limit = par ? work / 64.0 : -1.0;
            int nparts = 0;
            for (int t = nf - 1; t >= 0; --t)
            {
                if (fpar[t] >= 0 && part[fpar[t]] >= 0) part[t] = part[fpar[t]];
                else if (W[t] <= limit) part[t] = nparts++;
            }
            std::vector<int> pptr(static_cast<std::size_t>(nparts) + 2, 0), order(static_cast<std::size_t>(nf));
            for (int t = 0; t < nf; ++t) ++pptr[(part[t] >= 0 ? part[t] : nparts) + 1];
            for (int k = 0; k <= nparts; ++k) pptr[k + 1] += pptr[k];
            {
                std::vector<int> pos(pptr.begin(), pptr.end() - 1);
                for (int t = 0; t < nf; ++t) order[pos[part[t] >= 0 ? part[t] : nparts]++] = t;
            }
            auto rows_of = [&](int t, int *mark) {
                const int f = ff[t], l = ff[t + 1] - 1, b = frp[t], e = frp[t + 1];
                int w = b;
                bool over = false;
                auto take = [&](int i) {
                    if (i > l && mark[i] != t)
                    {
                        mark[i] = t;
                        if (w < e) fr[w++] = i; else over = true;
                    }
                };
                for (int j = f; j <= l; ++j) for (int q = cp[j]; q < cp[j + 1]; ++q) take(ci[q]);
                for (int u = fcp[t]; u < fcp[t + 1]; ++u)
                    for (int q = frp[fch[u]]; q < frp[fch[u] + 1]; ++q) take(fr[q]);
                std::sort(fr.begin() + b, fr.begin() + w);
                return !over && w == e;
            };
            int bad = 0;
            FEMD_OMP(parallel if (nparts > 1))
            {
                std::vector<int> mark(N, -1);
                FEMD_OMP(for schedule(dynamic, 1) reduction(| : bad))
                for (int k = 0; k < nparts; ++k)
                    for (int q = pptr[k]; q < pptr[k + 1]; ++q)
                        if (!rows_of(order[q], mark.data())) bad |= 1;
            }
            {
                std::vector<int> mark(N, -1);
                for (int q = pptr[nparts]; q < pptr[nparts + 1]; ++q)
                    if (!rows_of(order[q], mark.data())) bad |= 1;
            }
            if (bad) throw std::logic_error("LDL^T: the supernodal structure disagrees with the column counts");
        }
        // relaxed amalgamation (CHOLMOD's rule): a supernode absorbs the one that ends just before it
        // when that one is its child and the merged front keeps few explicit zeros; the merged node
        // has the rows of the parent.  Larger fronts also give the pivoting more room.
        std::vector<int> first(ff.begin(), ff.end() - 1);
        std::vector<char> keep(static_cast<std::size_t>(nf), 1);
        if (relax_)
        {
            std::vector<double> zeros(static_cast<std::size_t>(nf), 0.0);
            auto entries = [](double nc, double r) { return nc * (nc - 1.0) / 2.0 + nc * r; };
            for (int t = 1; t < nf; ++t)
            {
                const int c = t - 1, lc = ff[t] - 1, lt = ff[t + 1] - 1;
                if (pa[lc] < 0 || fof[pa[lc]] != t) continue;
                const double nc = lc - first[c] + 1, nt = lt - first[t] + 1;
                const double rc = lnz[lc], rt = lnz[lt], ncol = nc + nt;
                const double all = entries(ncol, rt);
                const double z = zeros[c] + zeros[t] + all - entries(nc, rc) - entries(nt, rt);
                const double frac = all > 0.0 ? z / all : 0.0;
                if (!(ncol <= 4 || (ncol <= 16 && frac < 0.8) || (ncol <= 48 && frac < 0.1) || frac < 0.05)) continue;
                keep[c] = 0;
                first[t] = first[c];
                zeros[t] = z;
            }
        }
        sn_ptr_.clear(); sn_rptr_.assign(1, 0); sn_row_.clear();
        for (int t = 0; t < nf; ++t)
        {
            if (!keep[t]) continue;
            sn_ptr_.push_back(first[t]);
            sn_row_.insert(sn_row_.end(), fr.begin() + frp[t], fr.begin() + frp[t + 1]);
            sn_rptr_.push_back(static_cast<int>(sn_row_.size()));
        }
        const int ns = static_cast<int>(sn_ptr_.size());
        sn_ptr_.push_back(n);
        std::vector<int> sof(N);
        for (int s = 0; s < ns; ++s) for (int j = sn_ptr_[s]; j < sn_ptr_[s + 1]; ++j) sof[j] = s;
        sn_parent_.assign(static_cast<std::size_t>(ns), -1);
        sn_cptr_.assign(static_cast<std::size_t>(ns) + 1, 0);
        for (int s = 0; s < ns; ++s)
        {
            const int p = pa[sn_ptr_[s + 1] - 1];
            if (p >= 0) { sn_parent_[s] = sof[p]; ++sn_cptr_[sn_parent_[s] + 1]; }
        }
        for (int s = 0; s < ns; ++s) sn_cptr_[s + 1] += sn_cptr_[s];
        sn_child_.resize(static_cast<std::size_t>(sn_cptr_[ns]));
        {
            std::vector<int> pos(sn_cptr_.begin(), sn_cptr_.end() - 1);
            for (int s = 0; s < ns; ++s) if (sn_parent_[s] >= 0) sn_child_[pos[sn_parent_[s]]++] = s;
        }
        // the cut of the supernodal tree: subtrees of at most 1/FEMD_LDLT_FPARTS of the estimated work
        // (m^2 p per front, without delays), the biggest first; the rest is the top
        std::vector<double> W(static_cast<std::size_t>(ns));
        for (int s = 0; s < ns; ++s)
        {
            const double p = sn_ptr_[s + 1] - sn_ptr_[s], m = p + (sn_rptr_[s + 1] - sn_rptr_[s]);
            W[s] = m * m * p + m * m;
        }
        for (int s = 0; s < ns; ++s) if (sn_parent_[s] >= 0) W[sn_parent_[s]] += W[s];
        double work = 0.0;
        for (int s = 0; s < ns; ++s) if (sn_parent_[s] < 0) work += W[s];
        const double limit = par ? work / FEMD_LDLT_FPARTS : -1.0;
        std::vector<int> part(static_cast<std::size_t>(ns), -1), roots;
        for (int s = ns - 1; s >= 0; --s)
        {
            const int p = sn_parent_[s];
            if (p >= 0 && part[p] >= 0) { part[s] = part[p]; continue; }
            if (W[s] <= limit) { part[s] = static_cast<int>(roots.size()); roots.push_back(s); }
        }
        fparts_ = static_cast<int>(roots.size());
        std::vector<int> ord(static_cast<std::size_t>(fparts_)), newid(static_cast<std::size_t>(fparts_));
        for (int k = 0; k < fparts_; ++k) ord[k] = k;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return W[roots[a]] > W[roots[b]]; });
        for (int k = 0; k < fparts_; ++k) newid[ord[k]] = k;
        fpart_ptr_.assign(static_cast<std::size_t>(fparts_) + 1, 0);
        for (int s = 0; s < ns; ++s) if (part[s] >= 0) ++fpart_ptr_[newid[part[s]] + 1];
        for (int k = 0; k < fparts_; ++k) fpart_ptr_[k + 1] += fpart_ptr_[k];
        fpart_node_.resize(static_cast<std::size_t>(fpart_ptr_[fparts_]));
        ftop_.clear();
        {
            std::vector<int> pos(fpart_ptr_.begin(), fpart_ptr_.end() - 1);
            for (int s = 0; s < ns; ++s)
            {
                if (part[s] >= 0) fpart_node_[pos[newid[part[s]]]++] = s;
                else ftop_.push_back(s);
            }
        }
    }

    // ---- numeric ------------------------------------------------------------------------
    struct Scratch {
        std::vector<int> loc;
        std::vector<double> F, w1, w2;
    };

    static std::size_t packed(int b, int mc)
    {
        return static_cast<std::size_t>(b) * static_cast<std::size_t>(mc) - static_cast<std::size_t>(b) * static_cast<std::size_t>(b - 1) / 2;
    }

    template <class Body>
    static void loop(bool par, int count, Body &&body)
    {
        if (par)
        {
            FEMD_OMP(parallel for schedule(dynamic, 1))
            for (int g = 0; g < count; ++g) body(g);
        }
        else
            for (int g = 0; g < count; ++g) body(g);
    }

    /// The supernodes children first: the subtrees of the cut concurrently (one front at a time per
    /// thread, each with its own scratch), then the top in order, its fronts shared between threads.
    void numeric(const CSRMatrix &A)
    {
        const double *Ax = A.data.data();
        const int ns = static_cast<int>(sn_ptr_.size()) - 1;
        fronts_.assign(static_cast<std::size_t>(ns), Front());
        cb_.assign(static_cast<std::size_t>(ns), {});
        cbidx_.assign(static_cast<std::size_t>(ns), {});
        delay_.assign(static_cast<std::size_t>(ns), {});
        int fail = std::numeric_limits<int>::max();                    // the first supernode that failed
        std::string msg;
        const bool par = fparts_ > 1;
        (void)par;
        FEMD_OMP_PARALLEL_IF(par)
        {
            Scratch sc;
            sc.loc.assign(static_cast<std::size_t>(n_), -1);
            int myfail = std::numeric_limits<int>::max();
            std::string mymsg;
            FEMD_OMP(for schedule(dynamic, 1))
            for (int k = 0; k < fparts_; ++k)
                for (int q = fpart_ptr_[k]; q < fpart_ptr_[k + 1]; ++q)
                {
                    const int s = fpart_node_[q];
                    std::string e = front(s, Ax, sc, false);
                    if (!e.empty()) { if (s < myfail) { myfail = s; mymsg = e; } break; }
                }
            FEMD_OMP(critical)
            {
                if (myfail < fail) { fail = myfail; msg = mymsg; }
            }
        }
        if (fail == std::numeric_limits<int>::max())
        {
            Scratch sc;
            sc.loc.assign(static_cast<std::size_t>(n_), -1);
            for (int s : ftop_)
            {
                std::string e = front(s, Ax, sc, true);
                if (!e.empty()) { fail = s; msg = e; break; }
            }
        }
        std::vector<std::vector<double>>().swap(cb_);
        std::vector<std::vector<int>>().swap(cbidx_);
        std::vector<std::vector<int>>().swap(delay_);
        if (fail != std::numeric_limits<int>::max()) throw std::runtime_error(msg);
        inertia_ = {{0, 0, 0}};
        delayed_ = n2x2_ = max_front_ = 0;
        min_piv_ = std::numeric_limits<double>::infinity();
        for (const Front &f : fronts_)
        {
            inertia_[0] += f.npos; inertia_[1] += f.nneg;
            n2x2_ += f.n2; delayed_ += f.ndel;
            max_front_ = std::max(max_front_, f.m);
            min_piv_ = std::min(min_piv_, f.minpiv);
        }
        if (!(min_piv_ < std::numeric_limits<double>::infinity())) min_piv_ = 0.0;
    }

    /// One front: assembled from A and the children's contribution blocks; its fully summed variables
    /// eliminated with 1x1 and 2x2 threshold pivots, each pivot updating the fully summed columns only
    /// (all their rows) and leaving its unscaled column W in the upper part of the other columns; then
    /// the rest of the front updated at once, F22 -= L21 W^T, by the frontal LU's kernel in panels of
    /// 32 pivots.  Returns an error message for a singular matrix, else an empty string.
    std::string front(int s, const double *Ax, Scratch &sc, bool par)
    {
        int *loc = sc.loc.data();
        const int f = sn_ptr_[s], l = sn_ptr_[s + 1] - 1;
        Front &fr = fronts_[s];
        std::vector<int> &idx = fr.idx;
        idx.clear();
        for (int u = sn_cptr_[s]; u < sn_cptr_[s + 1]; ++u)
        {
            std::vector<int> &d = delay_[sn_child_[u]];
            idx.insert(idx.end(), d.begin(), d.end());
            std::vector<int>().swap(d);
        }
        for (int j = f; j <= l; ++j) idx.push_back(j);
        const int p = static_cast<int>(idx.size());
        idx.insert(idx.end(), sn_row_.begin() + sn_rptr_[s], sn_row_.begin() + sn_rptr_[s + 1]);
        const int m = static_cast<int>(idx.size());
        const std::size_t M = static_cast<std::size_t>(m);
        fr.m = m;
        for (int t = 0; t < m; ++t) loc[idx[t]] = t;
        if (sc.F.size() < M * M) sc.F.resize(M * M);
        double *F = sc.F.data();
        const bool big = par && m >= FEMD_LDLT_PARFRONT;
        const int p0 = p - (l - f + 1);                                  // the delayed variables come first
        // zero the lower triangle, and add the entries of A in the supernode's own columns (their rows
        // are its later columns and its rows, so they stay in the lower triangle)
        loop(big, m, [&](int c) {
            double *Fc = F + static_cast<std::size_t>(c) * M;
            std::fill(Fc + c, Fc + m, 0.0);
            if (c >= p0 && c < p)
            {
                const int j = f + (c - p0);
                for (int q = colp_[j]; q < colp_[j + 1]; ++q) Fc[loc[coli_[q]]] += Ax[colq_[q]];
            }
        });
        // the children's contribution blocks, in order (a child's columns go to distinct columns, and
        // its order maps to increasing positions here)
        for (int u = sn_cptr_[s]; u < sn_cptr_[s + 1]; ++u)
        {
            const int c = sn_child_[u];
            std::vector<double> &C = cb_[c];
            const std::vector<int> &ci = cbidx_[c];
            const int mc = static_cast<int>(ci.size());
            if (mc > 0 && !C.empty())
            {
                const double *Cv = C.data();
                loop(big && mc >= FEMD_LDLT_PARFRONT, mc, [&](int b) {
                    const double *Cb = Cv + packed(b, mc);
                    const int lb = loc[ci[b]];
                    for (int a = b; a < mc; ++a)
                    {
                        const int la = loc[ci[a]];
                        if (la >= lb) F[static_cast<std::size_t>(lb) * M + la] += Cb[a - b];
                        else F[static_cast<std::size_t>(la) * M + lb] += Cb[a - b];
                    }
                });
            }
            std::vector<double>().swap(C);
            std::vector<int>().swap(cbidx_[c]);
        }
        // eliminate the fully summed variables 0 .. p-1 with 1x1 and 2x2 pivots
        const bool root = sn_parent_[s] < 0;
        const double u = u_;
        auto get = [&](int i, int j) { return i >= j ? F[static_cast<std::size_t>(j) * M + i] : F[static_cast<std::size_t>(i) * M + j]; };
        auto colmax = [&](int j, int k, int skip1, int skip2) {
            double g = 0.0;
            for (int i = k; i < m; ++i) if (i != skip1 && i != skip2) g = std::max(g, std::abs(get(i, j)));
            return g;
        };
        auto swap = [&](int a, int b) {                                  // symmetric swap of positions a, b < p
            if (a == b) return;
            if (a > b) std::swap(a, b);
            auto L = [&](int i, int j) -> double & { return F[static_cast<std::size_t>(j) * M + i]; };   // i >= j
            std::swap(L(a, a), L(b, b));
            for (int j = 0; j < a; ++j) std::swap(L(a, j), L(b, j));
            for (int j = a + 1; j < b; ++j) std::swap(L(j, a), L(b, j));
            for (int i = b + 1; i < m; ++i) std::swap(L(i, a), L(i, b));
            std::swap(idx[a], idx[b]);
        };
        sc.w1.resize(M); sc.w2.resize(M);
        double *w1 = sc.w1.data(), *w2 = sc.w2.data();
        fr.bs.clear(); fr.D.clear();
        fr.npos = fr.nneg = fr.n2 = 0;
        fr.minpiv = std::numeric_limits<double>::infinity();
        // The candidates are taken from a window of NB fully summed columns, kept up to date pivot by
        // pivot (all their rows); a pivot leaves its unscaled column W in the upper part of the columns
        // after the window.  When the window has no acceptable pivot left, the columns after it get
        // the window's pivots at once (F -= L W^T by the kernel), and the next window starts with the
        // candidates that failed.  A window without any pivot grows until it covers all fully summed
        // columns; then the rest is delayed (or, at a root, the largest pivot is taken).
        using kernel = frontal::Kernel<double>;
        constexpr int NB = FEMD_LU_PANEL;
        // F(:, j0 .. j1) -= L(:, t0 .. t1) W(j0 .. j1, t0 .. t1)^T from each column's diagonal down, the
        // columns [c0, c1) in groups of 4, the pivots in panels of NB
        auto block_update = [&](int c0, int c1, int t0, int t1, bool parallel) {
            const int ng = (c1 - c0 + kernel::NC - 1) / kernel::NC;
            loop(parallel, ng, [&](int g) {
                const int j0 = c0 + g * kernel::NC, j1 = std::min(c1, j0 + kernel::NC);
                for (int ta = t0; ta < t1; ta += NB)
                {
                    const int tb = std::min(t1, ta + NB);
                    switch (j1 - j0)
                    {
                        case 4: kernel::cols<4>(F, m, tb, ta, j0, j0); break;
                        case 3: kernel::cols<3>(F, m, tb, ta, j0, j0); break;
                        case 2: kernel::cols<2>(F, m, tb, ta, j0, j0); break;
                        case 1: kernel::cols<1>(F, m, tb, ta, j0, j0); break;
                        default: break;
                    }
                }
            });
        };
        int k = 0;
        bool stop = false;
        while (k < p && !stop)
        {
            const int kstart = k;
            int kb = std::min(p, k + NB);
            while (k < kb)
            {
                int piv1 = -1, pa = -1, pb = -1;
                for (int j = k; j < kb && piv1 < 0 && pa < 0; ++j)
                {
                    const double ajj = std::abs(get(j, j));
                    const double g = colmax(j, k, j, -1);
                    if (ajj > 0.0 && ajj >= u * g) { piv1 = j; break; }
                    int r = -1; double bigv = 0.0;
                    for (int i = k; i < kb; ++i) if (i != j && std::abs(get(i, j)) > bigv) { bigv = std::abs(get(i, j)); r = i; }
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
                    if (k > kstart) break;                             // update the columns after the window
                    if (kb < p) { kb = std::min(p, kb + NB); continue; }   // no pivot yet: a wider window
                    if (!root) { stop = true; break; }                 // delay the rest to the parent
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
                    {
                        for (int t = 0; t < m; ++t) loc[idx[t]] = -1;
                        return "LDL^T: the matrix is singular (no nonzero pivot left, " + std::to_string(p - k) + " variables)";
                    }
                }
                const bool wide = par && static_cast<long>(kb - k) * (m - k) > 8L * FEMD_OMP_THRESHOLD;
                if (piv1 >= 0)
                {
                    swap(k, piv1);
                    double *Fk = F + static_cast<std::size_t>(k) * M;
                    const double d = Fk[k];
                    for (int i = k + 1; i < m; ++i) { w1[i] = Fk[i]; Fk[i] /= d; }
                    // the window's columns, all their rows; W for the columns after it
                    loop(wide, kb - k - 1, [&](int g) {
                        const int j = k + 1 + g;
                        const double wj = w1[j];
                        if (wj == 0.0) return;
                        double *Fj = F + static_cast<std::size_t>(j) * M;
                        for (int i = j; i < m; ++i) Fj[i] -= Fk[i] * wj;
                    });
                    for (int j = kb; j < m; ++j) F[static_cast<std::size_t>(j) * M + k] = w1[j];
                    fr.bs.push_back(1); fr.D.push_back(d);
                    if (d > 0) ++fr.npos; else ++fr.nneg;
                    fr.minpiv = std::min(fr.minpiv, std::abs(d));
                    k += 1;
                }
                else
                {
                    swap(k, pa);
                    const int r = (pb == k) ? pa : pb;
                    swap(k + 1, r);
                    double *F1 = F + static_cast<std::size_t>(k) * M, *F2 = F + static_cast<std::size_t>(k + 1) * M;
                    const double a = F1[k], b = F1[k + 1], c = F2[k + 1], det = a * c - b * b;
                    for (int i = k + 2; i < m; ++i)
                    {
                        w1[i] = F1[i]; w2[i] = F2[i];
                        F1[i] = (c * w1[i] - b * w2[i]) / det;
                        F2[i] = (a * w2[i] - b * w1[i]) / det;
                    }
                    F1[k + 1] = 0.0;
                    loop(wide, kb - k - 2, [&](int g) {
                        const int j = k + 2 + g;
                        const double v1 = w1[j], v2 = w2[j];
                        if (v1 == 0.0 && v2 == 0.0) return;
                        double *Fj = F + static_cast<std::size_t>(j) * M;
                        for (int i = j; i < m; ++i) Fj[i] -= F1[i] * v1 + F2[i] * v2;
                    });
                    for (int j = kb; j < m; ++j)
                    {
                        F[static_cast<std::size_t>(j) * M + k] = w1[j];
                        F[static_cast<std::size_t>(j) * M + k + 1] = w2[j];
                    }
                    fr.bs.push_back(2); fr.D.push_back(a); fr.D.push_back(b); fr.D.push_back(c);
                    ++fr.n2;
                    // eigenvalues of [[a, b], [b, c]]: signs from the determinant and the trace
                    if (det < 0) { ++fr.npos; ++fr.nneg; }
                    else if (a + c > 0) fr.npos += 2;
                    else fr.nneg += 2;
                    const double tr = 0.5 * (a + c), disc = std::sqrt(std::max(0.0, 0.25 * (a - c) * (a - c) + b * b));
                    fr.minpiv = std::min(fr.minpiv, std::min(std::abs(tr - disc), std::abs(tr + disc)));
                    k += 2;
                }
            }
            // the fully summed columns after the window get the window's pivots
            if (k > kstart && kb < p)
                block_update(kb, p, kstart, k, par && static_cast<long>(m - kb) * (p - kb) * (k - kstart) > 8L * FEMD_OMP_THRESHOLD);
        }
        const int ne = k;
        fr.ne = ne;
        fr.ndel = p - ne;
        // the rest of the front: F22 -= L21 W^T on the rows and columns p .. m-1, with all the pivots
        const int below = m - p;
        if (below > 0 && ne > 0 && !root)
            block_update(p, m, 0, ne, par && static_cast<long>(below) * below * ne > 8L * FEMD_OMP_THRESHOLD);
        // the panel of L, the delayed variables, and the contribution block for the parent
        fr.L.resize(static_cast<std::size_t>(ne) * M);
        loop(big, ne, [&](int c) { std::copy(F + static_cast<std::size_t>(c) * M, F + static_cast<std::size_t>(c + 1) * M, fr.L.data() + static_cast<std::size_t>(c) * M); });
        delay_[s].assign(idx.begin() + ne, idx.begin() + p);
        const int mc = m - ne;
        if (mc > 0 && !root)
        {
            cbidx_[s].assign(idx.begin() + ne, idx.end());
            std::vector<double> &C = cb_[s];
            C.resize(packed(mc, mc));
            double *Cv = C.data();
            loop(big, mc, [&](int b) {
                const double *Fc = F + static_cast<std::size_t>(ne + b) * M + (ne + b);
                std::copy(Fc, Fc + (mc - b), Cv + packed(b, mc));
            });
        }
        for (int t = 0; t < m; ++t) loc[idx[t]] = -1;
        return std::string();
    }

    // ---- solves -------------------------------------------------------------------------
    /// For the fronts in the subtrees: which of their rows below the pivots hold variables eliminated
    /// above the subtree (those go to a gather buffer per subtree, added in order after the subtrees).
    void solve_setup()
    {
        const int ns = static_cast<int>(fronts_.size());
        std::vector<int> spart(static_cast<std::size_t>(ns), -1), vpart(static_cast<std::size_t>(n_), -1);
        for (int k = 0; k < fparts_; ++k)
            for (int q = fpart_ptr_[k]; q < fpart_ptr_[k + 1]; ++q) spart[fpart_node_[q]] = k;
        FEMD_OMP(parallel for schedule(dynamic, 64) if (fparts_ > 1))
        for (int s = 0; s < ns; ++s)
        {
            const Front &f = fronts_[s];
            for (int c = 0; c < f.ne; ++c) vpart[f.idx[c]] = spart[s];
        }
        ovar_.assign(static_cast<std::size_t>(fparts_), {});
        obuf_.assign(static_cast<std::size_t>(fparts_), {});
        FEMD_OMP(parallel if (fparts_ > 1))
        {
            std::vector<int> mark(static_cast<std::size_t>(n_), -1);
            FEMD_OMP(for schedule(dynamic, 1))
            for (int k = 0; k < fparts_; ++k)
            {
                std::vector<int> &ov = ovar_[k];
                for (int q = fpart_ptr_[k]; q < fpart_ptr_[k + 1]; ++q)
                {
                    const Front &f = fronts_[fpart_node_[q]];
                    for (int i = f.ne; i < f.m; ++i)
                    {
                        const int v = f.idx[i];
                        if (vpart[v] != k && mark[v] != k) { mark[v] = k; ov.push_back(v); }
                    }
                }
                std::sort(ov.begin(), ov.end());
                for (int v : ov) mark[v] = -1;
                for (std::size_t t = 0; t < ov.size(); ++t) mark[ov[t]] = static_cast<int>(t);
                for (int q = fpart_ptr_[k]; q < fpart_ptr_[k + 1]; ++q)
                {
                    Front &f = fronts_[fpart_node_[q]];
                    f.slot.assign(static_cast<std::size_t>(f.m - f.ne), -1);
                    for (int i = f.ne; i < f.m; ++i)
                    {
                        const int v = f.idx[i];
                        if (vpart[v] != k) f.slot[i - f.ne] = mark[v];
                    }
                }
                for (int v : ov) mark[v] = -1;
                obuf_[k].assign(ov.size(), 0.0);
            }
        }
        for (int s : ftop_) std::vector<int>().swap(fronts_[s].slot);
    }

    /// Forward with one front: the triangle of its pivots column by column, then the rows below them
    /// one dot product each (in parallel when par and large).  buf: the gather buffer of its subtree.
    static void forward(const Front &f, double *y, double *buf, bool par)
    {
        const int m = f.m, ne = f.ne;
        if (ne == 0) return;
        const int *ix = f.idx.data();
        const double *L = f.L.data();
        const std::size_t M = static_cast<std::size_t>(m);
        for (int c = 0; c < ne; ++c)
        {
            const double yc = y[ix[c]];
            if (yc == 0.0) continue;
            const double *Lc = L + static_cast<std::size_t>(c) * M;
            for (int i = c + 1; i < ne; ++i) y[ix[i]] -= Lc[i] * yc;
        }
        const int below = m - ne;
        auto row = [&](int i) {
            double v = 0.0;
            for (int c = 0; c < ne; ++c) v += L[static_cast<std::size_t>(c) * M + i] * y[ix[c]];
            return v;
        };
        if (buf)
        {
            const int *slot = f.slot.data();
            for (int a = 0; a < below; ++a)
            {
                const double v = row(ne + a);
                if (slot[a] < 0) y[ix[ne + a]] -= v;
                else buf[slot[a]] += v;
            }
        }
        else loop(par && static_cast<long>(below) * ne > 8L * FEMD_OMP_THRESHOLD, below,
                  [&](int a) { y[ix[ne + a]] -= row(ne + a); });
    }

    /// Backward with one front: each pivot column's dot product with the rows below the pivots (in
    /// parallel when par and large), then the triangle from the last pivot.
    static void backward(const Front &f, double *y, std::vector<double> &w, bool par)
    {
        const int m = f.m, ne = f.ne;
        if (ne == 0) return;
        const int *ix = f.idx.data();
        const double *L = f.L.data();
        const std::size_t M = static_cast<std::size_t>(m);
        w.resize(static_cast<std::size_t>(ne));
        double *wv = w.data();
        loop(par && static_cast<long>(m - ne) * ne > 8L * FEMD_OMP_THRESHOLD, ne, [&](int c) {
            const double *Lc = L + static_cast<std::size_t>(c) * M;
            double v = 0.0;
            for (int i = ne; i < m; ++i) v += Lc[i] * y[ix[i]];
            wv[c] = v;
        });
        for (int c = ne - 1; c >= 0; --c)
        {
            const double *Lc = L + static_cast<std::size_t>(c) * M;
            double v = y[ix[c]] - wv[c];
            for (int i = c + 1; i < ne; ++i) v -= Lc[i] * y[ix[i]];
            y[ix[c]] = v;
        }
    }
};

} // namespace femd

#endif // FEMD_SPARSE_LDLT_HPP
