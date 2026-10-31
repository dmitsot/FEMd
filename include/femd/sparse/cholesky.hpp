//
//  cholesky.hpp  --  sparse LDL^T factorization of a symmetric matrix, FEMd's own
//  direct solver for the symmetric positive definite matrices of 2D problems
//  (mass, stiffness, mass plus dispersion), used by SparseMatrix.solver("cholesky")
//  and by default for such matrices.
//
//      SparseCholesky C(A);            // order (AMD), analyse, factor: P A P^T = L D L^T
//      C.apply(b, x);                  // x = A^{-1} b: permute, L y = b, D z = y, L^T w = z, permute back
//      C.refactor(A2);                 // same pattern, new values: the numeric phase only
//
//  The three phases are the classical ones.
//    ordering   a fill-reducing permutation P (amd.hpp), or the natural order;
//    symbolic   the elimination tree of P A P^T and the column counts of L, from
//               the row subtrees (each row k of L is the set of nodes reached by
//               walking up the tree from the entries of row k of A);
//    numeric    multifrontal, by supernodes (runs of columns with nested structure,
//               merged with a child when few explicit zeros result, CHOLMOD's rule).
//               A supernode's front is the dense lower triangle on its columns and
//               the rows below them, assembled from A and its children's contribution
//               blocks; its columns are eliminated in panels of 32, the trailing
//               update F22 -= L21 D1 L21^T by the register-blocked kernel of the
//               frontal LU (lu_frontal.hpp), and what is left on the rows below is
//               the contribution block for the parent.  The supernodal tree is cut
//               into subtrees factored concurrently; the fronts above them (the
//               separators, the largest) are factored one by one with their panels,
//               updates, assembly and copies shared between the threads.  Each entry
//               is formed by the same operations whatever thread does it, so the
//               factor has the same bits for any number of threads.
//  L is unit lower triangular, stored by columns at the rows of its symbolic
//  pattern (the explicit zeros of a merged front are left out), D is diagonal.  No square roots
//  are taken, and no pivoting is done: a symmetric positive definite matrix never
//  needs it.  With positive_definite = true a pivot d_k <= 0 raises (the matrix is
//  not SPD).  With false only a zero pivot raises, which admits symmetric
//  quasi-definite matrices, but an indefinite matrix may then be factored
//  inaccurately: use a pivoting LU for those.
//
//  The class is a SparsePreconditioner (an exact one), so it drops into CG,
//  GMRES and the explicit Runge-Kutta stage loop wherever z = M^{-1} r is taken.
//  The matrix must hold both triangles (FEMd assembles full symmetric patterns);
//  only the entries with row >= column in the new order are read.  Stored entries that
//  are exactly zero (the uncoupled blocks of a ProductSpace pattern) are left out of the
//  ordering and the factor (prune.hpp); a refactor with values that are nonzero there
//  orders and analyses the matrix again.
//
//  The triangular solves run in parallel with OpenMP over the elimination tree.
//  After the factorization the tree is cut into independent subtrees of about
//  1/64 of the work each (partition()); the nodes above them are the top.  In the
//  forward solve the subtrees are done concurrently, each a column-oriented sweep
//  whose updates stay inside the subtree; then every top node gathers what the
//  subtrees owe it from its row of L, in parallel over the nodes; then the top
//  nodes follow in order.  In the backward solve the top goes first, then the
//  subtrees concurrently.  The top holds the long columns (the separators), so it
//  is swept by fundamental supernodes, runs of columns with nested structure, cut
//  into chunks of at most FEMD_CHOL_SNCHUNK (32) columns: the triangle inside a
//  chunk is done by one thread, and the update of the rows below it (the rest of
//  the supernode included) is one dense block, done in parallel over its rows
//  (forward) or its columns (backward).  The whole solve is one parallel region.
//  Every entry of the solution is formed by one thread in a fixed order, so the
//  result has the same bits for any number of threads, and the serial build runs
//  the same sweeps.
//
#ifndef FEMD_SPARSE_CHOLESKY_HPP
#define FEMD_SPARSE_CHOLESKY_HPP

#include "femd/sparse/amd.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/sparse/lu_frontal.hpp"
#include "femd/sparse/precond.hpp"
#include "femd/sparse/prune.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// the cut of the elimination tree: subtrees of at most 1/FEMD_CHOL_PARTS of the work of the
// factor; a supernode update with more than FEMD_CHOL_BLOCK entries runs in parallel
#ifndef FEMD_CHOL_PARTS
#define FEMD_CHOL_PARTS 64.0
#endif
#ifndef FEMD_CHOL_BLOCK
#define FEMD_CHOL_BLOCK 4096
#endif
// the top supernodes are cut into chunks of at most this many columns, so that the dense
// triangle done by one thread stays small and the rest of the supernode is updated in parallel
#ifndef FEMD_CHOL_SNCHUNK
#define FEMD_CHOL_SNCHUNK 32
#endif
// the factorization: the supernodal tree is cut into subtrees of at most 1/FEMD_CHOL_FPARTS of the
// work, factored concurrently; the panel's rows below its diagonal block are solved in blocks of
// FEMD_CHOL_ROWBLOCK rows; a front of the top with at least FEMD_CHOL_PARFRONT rows is assembled and
// copied out in parallel; FEMD_CHOL_RELAX = 0 turns the relaxed amalgamation off
#ifndef FEMD_CHOL_FPARTS
#define FEMD_CHOL_FPARTS 32.0
#endif
#ifndef FEMD_CHOL_ROWBLOCK
#define FEMD_CHOL_ROWBLOCK 64
#endif
#ifndef FEMD_CHOL_PARFRONT
#define FEMD_CHOL_PARFRONT 256
#endif
#ifndef FEMD_CHOL_RELAX
#define FEMD_CHOL_RELAX 1
#endif

namespace femd {

class SparseCholesky : public SparsePreconditioner {
public:
    enum class Ordering { AMD, Natural };

    /**
     * @brief Order, analyse and factor A.
     * @param A                  a symmetric matrix with both triangles stored
     * @param ordering           Ordering::AMD (default) or Ordering::Natural
     * @param positive_definite  raise on a pivot <= 0 (true) or only on a zero pivot (false)
     */
    explicit SparseCholesky(const CSRMatrix &A, Ordering ordering = Ordering::AMD, bool positive_definite = true)
        : spd_(positive_definite), ordering_(ordering)
    {
        check_square(A, "Cholesky");
        n_ = A.nrows();
        pattern_ = A.pattern;
        prune_.build(n_, A.indptr(), A.indices(), A.data.data(), true);
        analyse(A);
    }

    /// @brief New values on the same pattern: the numeric phase only.
    void refactor(const CSRMatrix &A)
    {
        if (A.nrows() != n_ || A.ncols() != n_ || A.nnz() != pattern_->nnz())
            throw std::invalid_argument("Cholesky::refactor: the matrix has a different pattern");
        if (A.pattern != pattern_ &&
            (A.pattern->indptr != pattern_->indptr || A.pattern->indices != pattern_->indices))
            throw std::invalid_argument("Cholesky::refactor: the matrix has a different pattern");
        if (!prune_.fits(A.data.data()))
        {
            // nonzero where the analysis dropped exact zeros: order and analyse again
            prune_.build(n_, A.indptr(), A.indices(), A.data.data(), true, true);
            analyse(A);
            return;
        }
        if (prune_.active) { prune_.gather(A.data.data(), B_.data); numeric(B_); }
        else numeric(A);
    }

    /// @brief z = A^{-1} r.
    void apply(const double *r, double *z) const override
    {
        const int n = n_;
        std::vector<double> &y = work_;
        y.resize(static_cast<std::size_t>(n));
        const int *Lp = Lp_.data(), *Li = Li_.data(), *perm = perm_.data();
        const double *Lx = Lx_.data(), *D = D_.data();
        const int *nodes = nodes_.data(), *partptr = partptr_.data(), *root = root_.data();
        const int *top = top_.data(), *snp = topsn_ptr_.data();
        const int *trp = toprow_ptr_.data(), *trj = toprow_j_.data(), *trx = toprow_p_.data();
        const int nparts = nparts_, ntop = ntop_;
        const int nsn = static_cast<int>(topsn_ptr_.size()) - 1;
        double *yv = y.data();
        const bool par = n > FEMD_OMP_THRESHOLD && nparts > 1;
        (void)par;
        std::vector<double> &w = work2_;
        w.resize(static_cast<std::size_t>(std::max(FEMD_CHOL_SNCHUNK, 1)));
        double *wv = w.data();
        // One parallel region for the whole solve; every loop and single below ends in a barrier.
        // Which of omp for / omp single a block takes depends on shared sizes only, so every
        // thread takes the same branch.
        FEMD_OMP_PARALLEL_IF(par)
        {
            FEMD_OMP_FOR
            for (int k = 0; k < n; ++k) yv[k] = r[perm[k]];
            // L y = b.  The subtrees concurrently: a column sweep whose updates stay below the root
            // of the subtree (the entries of a column are sorted, the rows above the root come last).
            FEMD_OMP(for schedule(dynamic, 1))
            for (int s = 0; s < nparts; ++s)
            {
                const int rs = root[s];
                for (int q = partptr[s]; q < partptr[s + 1]; ++q)
                {
                    const int j = nodes[q];
                    const double yj = yv[j];
                    if (yj == 0.0) continue;
                    for (int p = Lp[j]; p < Lp[j + 1] && Li[p] <= rs; ++p) yv[Li[p]] -= Lx[p] * yj;
                }
            }
            // what the subtrees owe the top (their rows of L below the top): final once the
            // subtrees are done, so gathered for every top node at once, in parallel over the nodes
            FEMD_OMP_FOR
            for (int q = 0; q < ntop; ++q)
            {
                double v = 0.0;
                for (int p = trp[q]; p < trp[q + 1]; ++p) v += Lx[trx[p]] * yv[trj[p]];
                yv[top[q]] -= v;
            }
            // then the top by supernodes (cut into chunks of at most FEMD_CHOL_SNCHUNK columns, so
            // the rest of a long supernode is part of the block below a chunk): the dense triangle
            // inside the chunk, then the rows below it as one block, in parallel over the rows
            for (int u = 0; u < nsn; ++u)
            {
                const int a = snp[u], sz = snp[u + 1] - a, j1 = top[a];
                const int jl = j1 + sz - 1, below = Lp[jl + 1] - Lp[jl];
                const int *rows = Li + Lp[jl];
                const bool big = par && static_cast<long>(sz) * below > FEMD_CHOL_BLOCK;
                FEMD_OMP(single)
                {
                    for (int i = 0; i < sz; ++i)                            // the triangle
                    {
                        const int j = j1 + i;
                        const double yj = yv[j];
                        if (yj == 0.0) continue;
                        for (int k = 0; k < sz - 1 - i; ++k) yv[j + 1 + k] -= Lx[Lp[j] + k] * yj;
                    }
                    if (!big)
                        for (int k = 0; k < below; ++k)
                        {
                            double v = yv[rows[k]];
                            for (int i = 0; i < sz; ++i) v -= Lx[Lp[j1 + i] + (sz - 1 - i) + k] * yv[j1 + i];
                            yv[rows[k]] = v;
                        }
                }
                if (big)
                {
                    FEMD_OMP_FOR
                    for (int k = 0; k < below; ++k)
                    {
                        double v = yv[rows[k]];
                        for (int i = 0; i < sz; ++i) v -= Lx[Lp[j1 + i] + (sz - 1 - i) + k] * yv[j1 + i];
                        yv[rows[k]] = v;
                    }
                }
            }
            FEMD_OMP_FOR
            for (int j = 0; j < n; ++j) yv[j] /= D[j];                     // D
            // L^T x = y.  The top from its last chunk: each column gathers the rows below the chunk
            // (one dot product per column, in parallel over the columns), then the triangle; then
            // the subtrees concurrently, each from its root down (ancestors done, inside or top).
            for (int u = nsn - 1; u >= 0; --u)
            {
                const int a = snp[u], sz = snp[u + 1] - a, j1 = top[a];
                const int jl = j1 + sz - 1, below = Lp[jl + 1] - Lp[jl];
                const int *rows = Li + Lp[jl];
                const bool big = par && static_cast<long>(sz) * below > FEMD_CHOL_BLOCK;
                if (big)
                {
                    FEMD_OMP_FOR
                    for (int i = 0; i < sz; ++i)
                    {
                        const double *col = Lx + Lp[j1 + i] + (sz - 1 - i);
                        double v = 0.0;
                        for (int k = 0; k < below; ++k) v += col[k] * yv[rows[k]];
                        wv[i] = v;
                    }
                }
                FEMD_OMP(single)
                {
                    if (!big)
                        for (int i = 0; i < sz; ++i)
                        {
                            const double *col = Lx + Lp[j1 + i] + (sz - 1 - i);
                            double v = 0.0;
                            for (int k = 0; k < below; ++k) v += col[k] * yv[rows[k]];
                            wv[i] = v;
                        }
                    for (int i = sz - 1; i >= 0; --i)
                    {
                        const int j = j1 + i;
                        double v = yv[j];
                        for (int k = 0; k < sz - 1 - i; ++k) v -= Lx[Lp[j] + k] * yv[j + 1 + k];
                        yv[j] = v - wv[i];
                    }
                }
            }
            FEMD_OMP(for schedule(dynamic, 1))
            for (int s = 0; s < nparts; ++s)
                for (int q = partptr[s + 1] - 1; q >= partptr[s]; --q)
                {
                    const int j = nodes[q];
                    double v = yv[j];
                    for (int p = Lp[j]; p < Lp[j + 1]; ++p) v -= Lx[p] * yv[Li[p]];
                    yv[j] = v;
                }
            FEMD_OMP_FOR
            for (int k = 0; k < n; ++k) z[perm[k]] = yv[k];
        }
    }
    std::string name() const override { return "cholesky"; }

    std::vector<double> solve(const std::vector<double> &b) const
    {
        if (static_cast<int>(b.size()) != n_) throw std::invalid_argument("Cholesky::solve: length mismatch");
        std::vector<double> x(static_cast<std::size_t>(n_));
        if (n_ > 0) apply(b.data(), x.data());
        return x;
    }

    // ---- statistics ----------------------------------------------------------------
    /// @brief Entries of L below the diagonal.
    long long nnz_L() const { return static_cast<long long>(Lp_.empty() ? 0 : Lp_.back()); }
    /// @brief Multiply-adds of the numeric factorization, sum over columns of |L_j|^2 (about).
    double flops() const
    {
        double f = 0;
        for (int j = 0; j < n_; ++j) { const double c = Lp_[j + 1] - Lp_[j]; f += c * c + c; }
        return f;
    }
    double min_pivot() const { return D_.empty() ? 0.0 : *std::min_element(D_.begin(), D_.end()); }
    double max_pivot() const { return D_.empty() ? 0.0 : *std::max_element(D_.begin(), D_.end()); }
    const std::vector<int> &permutation() const { return perm_; }
    const std::vector<int> &etree() const { return parent_; }
    /// @brief Independent subtrees of the elimination tree that the solves run concurrently.
    int subtrees() const { return nparts_; }
    /// @brief Nodes above the subtrees, solved supernode by supernode.
    int top_nodes() const { return ntop_; }
    /// @brief Fundamental supernodes of the top.
    int top_supernodes() const { return static_cast<int>(topsn_ptr_.size()) - 1; }
    /// @brief The share of the entries of L touched in the sequential part of a solve.
    double top_fraction() const
    {
        if (nnz_L() == 0) return 0.0;
        double t = 0;
        for (int q = 0; q < ntop_; ++q) t += Lp_[top_[q] + 1] - Lp_[top_[q]];
        return (t + static_cast<double>(toprow_j_.size())) / static_cast<double>(nnz_L());
    }
    /// @brief Supernodes of the factorization (fronts), after the relaxed amalgamation.
    int supernodes() const { return static_cast<int>(sn_ptr_.size()) - 1; }
    /// @brief Order of the largest front (columns plus rows below them).
    int max_front() const { return max_front_; }
    /// @brief Independent subtrees of the supernodal tree that the factorization runs concurrently.
    int factor_subtrees() const { return fparts_; }
    std::string ordering_name() const { return ordering_ == Ordering::AMD ? "amd" : "natural"; }
    bool positive_definite() const { return spd_; }
    /// @brief Stored entries of A that are exactly zero and were left out of the ordering and the factor.
    long long dropped_zeros() const { return prune_.ndropped(); }

    /// @brief L (unit diagonal not stored) by columns, and D, in the permuted order.
    const std::vector<int> &Lp() const { return Lp_; }
    const std::vector<int> &Li() const { return Li_; }
    const std::vector<double> &Lx() const { return Lx_; }
    const std::vector<double> &D() const { return D_; }

private:
    bool spd_;
    Ordering ordering_;
    std::shared_ptr<const SparsityPattern> pattern_;
    ZeroPruning<double> prune_;              // the exact zeros of A left out (prune.hpp)
    CSRMatrix B_;                            // A without them, when prune_.active
    std::vector<int> perm_, pinv_, parent_, Lp_, Li_;
    std::vector<double> Lx_, D_;
    mutable std::vector<double> work_;
    // the cut of the elimination tree for the parallel solves (partition())
    int nparts_ = 0, ntop_ = 0;
    std::vector<int> part_;                  // subtree of each node, -1 in the top
    std::vector<int> nodes_, partptr_;       // the nodes of subtree s, increasing: nodes_[partptr_[s] .. partptr_[s+1])
    std::vector<int> root_;                  // the root of each subtree, its largest node
    std::vector<int> top_;                   // the top nodes, increasing
    std::vector<int> toprow_ptr_, toprow_j_, toprow_p_;   // for top node q: the entries L(t, j) with j in a subtree, (j, index into Lx)
    std::vector<int> topsn_ptr_;             // the top as fundamental supernodes: top_[topsn_ptr_[u] .. topsn_ptr_[u+1]) are consecutive columns with nested structure
    mutable std::vector<double> work2_;
    // the multifrontal numeric phase (symbolic(), numeric(), front())
    bool relax_ = FEMD_CHOL_RELAX != 0;
    std::vector<int> colp_, coli_, colq_;    // the lower triangle of P A P^T by columns: rows coli_ (increasing), positions colq_ in A's data
    std::vector<int> sn_ptr_;                // supernode s has the columns sn_ptr_[s] .. sn_ptr_[s+1]-1
    std::vector<int> sn_parent_;             // its parent supernode, -1 at a root
    std::vector<int> sn_cptr_, sn_child_;    // its children, increasing
    std::vector<int> sn_rptr_, sn_row_;      // its rows below the last column, increasing
    int fparts_ = 0, max_front_ = 0;
    std::vector<int> fpart_ptr_, fpart_node_, ftop_;   // the cut of the supernodal tree: the supernodes of each subtree, increasing, and the top
    std::vector<std::vector<double>> cb_;    // the contribution blocks during numeric(), packed lower triangles on the rows of the supernode

    /// Order, analyse and factor: on A without its exact zeros when some were found.
    void analyse(const CSRMatrix &A)
    {
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
        perm_ = (ordering_ == Ordering::AMD) ? ordering::amd_order(n_, M.indptr(), M.indices())
                                             : ordering::natural_order(n_);
        pinv_.assign(static_cast<std::size_t>(n_), 0);
        for (int k = 0; k < n_; ++k) pinv_[perm_[k]] = k;
        symbolic(M);
        numeric(M);
        partition();
    }

    /// Cut the elimination tree into subtrees of at most 1/64 of the work of the factor each,
    /// biggest first, and list the entries of L that join the subtrees to the top.
    void partition()
    {
        const int n = n_;
        nparts_ = ntop_ = 0;
        part_.assign(static_cast<std::size_t>(n), -1);
        nodes_.clear(); root_.clear(); top_.clear(); toprow_j_.clear(); toprow_p_.clear();
        partptr_.assign(1, 0); toprow_ptr_.assign(1, 0); topsn_ptr_.assign(1, 0);
        if (n == 0) return;
        // the work of a node is its column of L; of a subtree, the sum (children come before parents)
        std::vector<double> W(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) W[j] = static_cast<double>(Lp_[j + 1] - Lp_[j]) + 1.0;
        for (int j = 0; j < n; ++j) if (parent_[j] >= 0) W[parent_[j]] += W[j];
        double total = 0.0;
        for (int j = 0; j < n; ++j) if (parent_[j] < 0) total += W[j];
        const double limit = n > 2 * FEMD_OMP_THRESHOLD ? total / FEMD_CHOL_PARTS : -1.0;   // small: everything in the top
        // from the roots down: a subtree light enough is a part, a heavier node joins the top
        std::vector<int> roots;
        for (int j = n - 1; j >= 0; --j)
        {
            const int pa = parent_[j];
            if (pa >= 0 && part_[pa] >= 0) { part_[j] = part_[pa]; continue; }
            if (W[j] <= limit) { part_[j] = static_cast<int>(roots.size()); roots.push_back(j); }
        }
        // number the parts by decreasing work, so that the dynamic schedule starts with the big ones
        std::vector<int> order(roots.size());
        for (std::size_t s = 0; s < roots.size(); ++s) order[s] = static_cast<int>(s);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return W[roots[a]] > W[roots[b]]; });
        std::vector<int> newid(roots.size());
        for (std::size_t k = 0; k < order.size(); ++k) newid[order[k]] = static_cast<int>(k);
        nparts_ = static_cast<int>(roots.size());
        root_.resize(roots.size());
        for (std::size_t s = 0; s < roots.size(); ++s) root_[newid[s]] = roots[s];
        for (int j = 0; j < n; ++j) if (part_[j] >= 0) part_[j] = newid[part_[j]];
        // the nodes of each part in increasing order, and the top nodes
        partptr_.assign(static_cast<std::size_t>(nparts_) + 1, 0);
        for (int j = 0; j < n; ++j)
        {
            if (part_[j] >= 0) ++partptr_[part_[j] + 1];
            else top_.push_back(j);
        }
        for (int s = 0; s < nparts_; ++s) partptr_[s + 1] += partptr_[s];
        nodes_.resize(static_cast<std::size_t>(partptr_[nparts_]));
        std::vector<int> fill(partptr_.begin(), partptr_.end() - 1);
        for (int j = 0; j < n; ++j) if (part_[j] >= 0) nodes_[fill[part_[j]]++] = j;
        ntop_ = static_cast<int>(top_.size());
        // the top as fundamental supernodes: column j joins column j+1 when j+1 is its parent and
        // its structure is that of j+1 plus the row j+1 (then its entries are aligned with j+1's)
        for (int q = 0; q < ntop_; ++q)
        {
            const int t = top_[q];
            const bool chain = q + 1 < ntop_ && top_[q + 1] == t + 1 && parent_[t] == t + 1 &&
                               Lp_[t + 1] - Lp_[t] == Lp_[t + 2] - Lp_[t + 1] + 1;
            if (!chain || q + 1 - topsn_ptr_.back() >= FEMD_CHOL_SNCHUNK) topsn_ptr_.push_back(q + 1);
        }
        // the entries L(t, j) with t in the top and j in a part, grouped by t, j increasing within t
        std::vector<int> toppos(static_cast<std::size_t>(n), -1);
        for (int q = 0; q < ntop_; ++q) toppos[top_[q]] = q;
        toprow_ptr_.assign(static_cast<std::size_t>(ntop_) + 1, 0);
        for (int j = 0; j < n; ++j)
        {
            if (part_[j] < 0) continue;
            const int rs = root_[part_[j]];
            for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) if (Li_[p] > rs) ++toprow_ptr_[toppos[Li_[p]] + 1];
        }
        for (int q = 0; q < ntop_; ++q) toprow_ptr_[q + 1] += toprow_ptr_[q];
        toprow_j_.resize(static_cast<std::size_t>(toprow_ptr_[ntop_]));
        toprow_p_.resize(toprow_j_.size());
        std::vector<int> pos(toprow_ptr_.begin(), toprow_ptr_.end() - 1);
        for (int j = 0; j < n; ++j)
        {
            if (part_[j] < 0) continue;
            const int rs = root_[part_[j]];
            for (int p = Lp_[j]; p < Lp_[j + 1]; ++p)
                if (Li_[p] > rs) { const int k = pos[toppos[Li_[p]]]++; toprow_j_[k] = j; toprow_p_[k] = p; }
        }
    }

    // ---- symbolic phase -----------------------------------------------------------------
    // The elimination tree and the column counts of L from the row subtrees of P A P^T (row k
    // of its lower triangle: row perm[k] of A at the columns i = pinv[j] < k); the lower
    // triangle by columns (what each front assembles); the fundamental supernodes, their rows
    // and the row indices of L; then the supernodes of the fronts (relaxed amalgamation) and
    // the cut of their tree for the parallel factorization.
    void symbolic(const CSRMatrix &A)
    {
        const int n = n_;
        const int *Ap = A.indptr(), *Ai = A.indices();
        parent_.assign(static_cast<std::size_t>(n), -1);
        std::vector<int> flag(static_cast<std::size_t>(n), -1), lnz(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k)
        {
            flag[k] = k;
            const int row = perm_[k];
            for (int p = Ap[row]; p < Ap[row + 1]; ++p)
            {
                int i = pinv_[Ai[p]];
                if (i >= k) continue;
                for (; flag[i] != k; i = parent_[i])
                {
                    if (parent_[i] == -1) parent_[i] = k;
                    ++lnz[i];
                    flag[i] = k;
                }
            }
        }
        Lp_.assign(static_cast<std::size_t>(n) + 1, 0);
        long long total = 0;
        for (int j = 0; j < n; ++j)
        {
            total += lnz[j];
            if (total > std::numeric_limits<int>::max())
                throw std::length_error("Cholesky: the factor has more than 2^31 entries");
            Lp_[j + 1] = static_cast<int>(total);
        }
        Li_.assign(static_cast<std::size_t>(total), 0);
        Lx_.assign(static_cast<std::size_t>(total), 0.0);
        D_.assign(static_cast<std::size_t>(n), 0.0);

        // the lower triangle of P A P^T by columns, rows increasing: the entry (i, j), i >= j, is
        // A(perm[i], perm[j]), at position colq_ of A's data (read from row perm[i])
        colp_.assign(static_cast<std::size_t>(n) + 1, 0);
        for (int i = 0; i < n; ++i)
            for (int q = Ap[perm_[i]]; q < Ap[perm_[i] + 1]; ++q)
                if (pinv_[Ai[q]] <= i) ++colp_[pinv_[Ai[q]] + 1];
        for (int j = 0; j < n; ++j) colp_[j + 1] += colp_[j];
        coli_.resize(static_cast<std::size_t>(colp_[n]));
        colq_.resize(coli_.size());
        {
            std::vector<int> pos(colp_.begin(), colp_.end() - 1);
            for (int i = 0; i < n; ++i)
                for (int q = Ap[perm_[i]]; q < Ap[perm_[i] + 1]; ++q)
                {
                    const int j = pinv_[Ai[q]];
                    if (j <= i) { const int k = pos[j]++; coli_[k] = i; colq_[k] = q; }
                }
        }

        // fundamental supernodes: j joins j-1 when j-1's parent is j, its only child, and the
        // column of j-1 is that of j plus the row j
        std::vector<int> nchild(static_cast<std::size_t>(n), 0), ff;
        for (int j = 0; j < n; ++j) if (parent_[j] >= 0) ++nchild[parent_[j]];
        for (int j = 0; j < n; ++j)
            if (!(j > 0 && parent_[j - 1] == j && nchild[j] == 1 && lnz[j - 1] == lnz[j] + 1)) ff.push_back(j);
        const int nf = static_cast<int>(ff.size());
        ff.push_back(n);
        std::vector<int> fof(static_cast<std::size_t>(n));
        for (int t = 0; t < nf; ++t) for (int j = ff[t]; j < ff[t + 1]; ++j) fof[j] = t;
        std::vector<std::vector<int>> fchild(static_cast<std::size_t>(nf)), frows(static_cast<std::size_t>(nf));
        for (int t = 0; t < nf; ++t)
            if (parent_[ff[t + 1] - 1] >= 0) fchild[fof[parent_[ff[t + 1] - 1]]].push_back(t);
        // the rows below a supernode's last column: those of its columns of A and of its children
        {
            std::vector<int> mark(static_cast<std::size_t>(n), -1);
            for (int t = 0; t < nf; ++t)
            {
                const int f = ff[t], l = ff[t + 1] - 1;
                std::vector<int> &rows = frows[t];
                auto take = [&](int i) { if (i > l && mark[i] != t) { mark[i] = t; rows.push_back(i); } };
                for (int j = f; j <= l; ++j) for (int q = colp_[j]; q < colp_[j + 1]; ++q) take(coli_[q]);
                for (int c : fchild[t]) for (int i : frows[c]) take(i);
                std::sort(rows.begin(), rows.end());
                if (static_cast<int>(rows.size()) != lnz[l])
                    throw std::logic_error("Cholesky: the supernodal structure disagrees with the column counts");
            }
        }
        // the row indices of L: column j has the rows j+1 .. l of its supernode, then the supernode's rows
        FEMD_OMP(parallel for schedule(dynamic, 64) if (n > 2 * FEMD_OMP_THRESHOLD))
        for (int t = 0; t < nf; ++t)
        {
            const int f = ff[t], l = ff[t + 1] - 1;
            for (int j = f; j <= l; ++j)
            {
                int p = Lp_[j];
                for (int i = j + 1; i <= l; ++i) Li_[p++] = i;
                for (int i : frows[t]) Li_[p++] = i;
            }
        }

        // relaxed amalgamation (CHOLMOD's rule, as in lu_frontal.hpp): a supernode absorbs the one
        // that ends just before it when that one is its child and the merged front keeps few
        // explicit zeros.  The merged node has the rows of the parent: the child's rows are its
        // columns or its rows.  Fewer, larger fronts: less assembly per entry, longer panels.
        std::vector<int> first(ff.begin(), ff.end() - 1);
        std::vector<char> keep(static_cast<std::size_t>(nf), 1);
        if (relax_)
        {
            std::vector<double> zeros(static_cast<std::size_t>(nf), 0.0);
            auto entries = [](double nc, double r) { return nc * (nc - 1.0) / 2.0 + nc * r; };
            for (int t = 1; t < nf; ++t)
            {
                const int c = t - 1, lc = ff[t] - 1, lt = ff[t + 1] - 1;
                if (parent_[lc] < 0 || fof[parent_[lc]] != t) continue;
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
            sn_row_.insert(sn_row_.end(), frows[t].begin(), frows[t].end());
            sn_rptr_.push_back(static_cast<int>(sn_row_.size()));
        }
        const int ns = static_cast<int>(sn_ptr_.size());
        sn_ptr_.push_back(n);
        std::vector<int> sof(static_cast<std::size_t>(n));
        for (int s = 0; s < ns; ++s) for (int j = sn_ptr_[s]; j < sn_ptr_[s + 1]; ++j) sof[j] = s;
        sn_parent_.assign(static_cast<std::size_t>(ns), -1);
        sn_cptr_.assign(static_cast<std::size_t>(ns) + 1, 0);
        for (int s = 0; s < ns; ++s)
        {
            const int pa = parent_[sn_ptr_[s + 1] - 1];
            if (pa >= 0) { sn_parent_[s] = sof[pa]; ++sn_cptr_[sn_parent_[s] + 1]; }
        }
        for (int s = 0; s < ns; ++s) sn_cptr_[s + 1] += sn_cptr_[s];
        sn_child_.resize(static_cast<std::size_t>(sn_cptr_[ns]));
        {
            std::vector<int> pos(sn_cptr_.begin(), sn_cptr_.end() - 1);
            for (int s = 0; s < ns; ++s) if (sn_parent_[s] >= 0) sn_child_[pos[sn_parent_[s]]++] = s;
        }

        // the cut of the supernodal tree: subtrees of at most 1/FEMD_CHOL_FPARTS of the estimated
        // work (m^2 p per front, m rows and p columns), the biggest first; the rest is the top
        std::vector<double> W(static_cast<std::size_t>(ns));
        max_front_ = 0;
        for (int s = 0; s < ns; ++s)
        {
            const double p = sn_ptr_[s + 1] - sn_ptr_[s], m = p + (sn_rptr_[s + 1] - sn_rptr_[s]);
            max_front_ = std::max(max_front_, static_cast<int>(m));
            W[s] = m * m * p + m * m;
        }
        for (int s = 0; s < ns; ++s) if (sn_parent_[s] >= 0) W[sn_parent_[s]] += W[s];
        double work = 0.0;
        for (int s = 0; s < ns; ++s) if (sn_parent_[s] < 0) work += W[s];
        const double limit = n > 2 * FEMD_OMP_THRESHOLD ? work / FEMD_CHOL_FPARTS : -1.0;
        std::vector<int> part(static_cast<std::size_t>(ns), -1), roots;
        for (int s = ns - 1; s >= 0; --s)
        {
            const int pa = sn_parent_[s];
            if (pa >= 0 && part[pa] >= 0) { part[s] = part[pa]; continue; }
            if (W[s] <= limit) { part[s] = static_cast<int>(roots.size()); roots.push_back(s); }
        }
        fparts_ = static_cast<int>(roots.size());
        std::vector<int> order(static_cast<std::size_t>(fparts_)), newid(static_cast<std::size_t>(fparts_));
        for (int k = 0; k < fparts_; ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return W[roots[a]] > W[roots[b]]; });
        for (int k = 0; k < fparts_; ++k) newid[order[k]] = k;
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

    // ---- numeric phase: multifrontal --------------------------------------------------------
    struct Scratch {
        std::vector<int> loc, rl;
        std::vector<double> F;
    };
    struct Failure {
        int step = std::numeric_limits<int>::max();
        double d = 0.0;
    };

    // offset of column b of a packed lower triangle of order mc (column b holds mc - b entries)
    static std::size_t packed(int b, int mc)
    {
        return static_cast<std::size_t>(b) * static_cast<std::size_t>(mc) - static_cast<std::size_t>(b) * static_cast<std::size_t>(b - 1) / 2;
    }

    // Run body(g) for g < count: in parallel when par (a new team, dynamic schedule), else in order.
    // The arithmetic of each g does not depend on the thread that does it.
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

    /// The supernodes children first: the subtrees of the cut concurrently (one front at a time
    /// per thread, each thread with its own scratch), then the top in order, its fronts shared
    /// between the threads.  A front's arithmetic is the same whoever does it, so the factor has
    /// the same bits for any number of threads.
    void numeric(const CSRMatrix &A)
    {
        const double *Ax = A.data.data();
        const int ns = static_cast<int>(sn_ptr_.size()) - 1;
        cb_.assign(static_cast<std::size_t>(std::max(ns, 0)), {});
        Failure fail;
        const bool par = fparts_ > 1;
        (void)par;
        FEMD_OMP_PARALLEL_IF(par)
        {
            Scratch sc;
            sc.loc.assign(static_cast<std::size_t>(n_), -1);
            Failure mine;
            FEMD_OMP(for schedule(dynamic, 1))
            for (int k = 0; k < fparts_; ++k)
                for (int q = fpart_ptr_[k]; q < fpart_ptr_[k + 1]; ++q)
                    if (!front(fpart_node_[q], Ax, sc, false, mine)) break;
            FEMD_OMP(critical)
            {
                if (mine.step < fail.step) fail = mine;
            }
        }
        if (fail.step == std::numeric_limits<int>::max())
        {
            Scratch sc;
            sc.loc.assign(static_cast<std::size_t>(n_), -1);
            for (int s : ftop_) if (!front(s, Ax, sc, true, fail)) break;
        }
        std::vector<std::vector<double>>().swap(cb_);
        if (fail.step != std::numeric_limits<int>::max())
            throw std::runtime_error(std::string("Cholesky: ") +
                                     (spd_ ? "the matrix is not positive definite" : "zero pivot") +
                                     " (pivot " + std::to_string(fail.d) + " at step " + std::to_string(fail.step) +
                                     ", original row " + std::to_string(perm_[fail.step]) + ")");
    }

    /// One front: the dense lower triangle on the supernode's columns f..l and its rows R (an m x m
    /// column-major array, local index of a variable in loc), assembled from the lower triangle of
    /// P A P^T in the columns f..l and from the children's contribution blocks; its p = l-f+1
    /// columns eliminated in panels of NB = FEMD_LU_PANEL:
    ///   1. the panel's diagonal block, column by column, with the panel's earlier columns;
    ///   2. the rows below it, L21 = A21 L11^{-T} D1^{-1}, in fixed blocks of FEMD_CHOL_ROWBLOCK rows;
    ///   3. the trailing lower triangle, F22 -= L21 D1 L21^T, in groups of 4 columns: D1 L21^T is
    ///      written above the diagonal of the panel's rows, so the update is the register-blocked
    ///      kernel of the frontal LU (lu_frontal.hpp) from each group's diagonal down.
    /// Every entry is formed by the same operations in the same order whatever the blocks are run
    /// on.  Then column j of L and D(j) are copied out at the rows of the symbolic pattern (the
    /// other rows of a relaxed front hold exact zeros), and the trailing triangle on R is packed as
    /// the contribution block for the parent.  Returns false at a pivot that fails the test.
    bool front(int s, const double *Ax, Scratch &sc, bool par, Failure &fail)
    {
        int *loc = sc.loc.data();
        const int f = sn_ptr_[s], p = sn_ptr_[s + 1] - f;
        const int *R = sn_row_.data() + sn_rptr_[s];
        const int nr = sn_rptr_[s + 1] - sn_rptr_[s], m = p + nr;
        const std::size_t M = static_cast<std::size_t>(m);
        for (int c = 0; c < p; ++c) loc[f + c] = c;
        for (int a = 0; a < nr; ++a) loc[R[a]] = p + a;
        if (sc.F.size() < M * M) sc.F.resize(M * M);
        double *F = sc.F.data();
        const bool big = par && m >= FEMD_CHOL_PARFRONT;
        // zero the lower triangle, and add the entries of A in the columns of the supernode
        loop(big, m, [&](int c) {
            double *Fc = F + static_cast<std::size_t>(c) * M;
            std::fill(Fc + c, Fc + m, 0.0);
            if (c < p)
                for (int q = colp_[f + c]; q < colp_[f + c + 1]; ++q) Fc[loc[coli_[q]]] += Ax[colq_[q]];
        });
        // the children's contribution blocks, in order (a child's columns go to distinct columns)
        for (int k = sn_cptr_[s]; k < sn_cptr_[s + 1]; ++k)
        {
            const int c = sn_child_[k];
            std::vector<double> &C = cb_[c];
            const int mc = sn_rptr_[c + 1] - sn_rptr_[c];
            if (C.empty() || mc == 0) continue;
            const int *cr = sn_row_.data() + sn_rptr_[c];
            sc.rl.resize(static_cast<std::size_t>(mc));
            int *rl = sc.rl.data();
            for (int a = 0; a < mc; ++a) rl[a] = loc[cr[a]];
            const double *Cv = C.data();
            loop(big && mc >= FEMD_CHOL_PARFRONT, mc, [&](int b) {
                const double *Cb = Cv + packed(b, mc);
                double *Fc = F + static_cast<std::size_t>(rl[b]) * M;
                for (int a = b; a < mc; ++a) Fc[rl[a]] += Cb[a - b];
            });
            std::vector<double>().swap(C);
        }
        // eliminate the p columns in panels
        constexpr int NB = FEMD_LU_PANEL, RB = FEMD_CHOL_ROWBLOCK;
        double W[NB][NB], rd[NB];
        using kernel = frontal::Kernel<double>;
        for (int ps = 0; ps < p; ps += NB)
        {
            const int pe = std::min(p, ps + NB), nb = pe - ps;
            for (int k = ps; k < pe; ++k)                                   // 1. the diagonal block
            {
                double *Fk = F + static_cast<std::size_t>(k) * M;
                for (int t = ps; t < k; ++t)
                {
                    const double *Ft = F + static_cast<std::size_t>(t) * M;
                    const double w = Ft[k] * Ft[t];                          // D(t) L(k, t)
                    W[t - ps][k - ps] = w;
                    for (int i = k; i < pe; ++i) Fk[i] -= Ft[i] * w;
                }
                const double d = Fk[k];
                if (!(d > 0.0) && (spd_ || d == 0.0 || !std::isfinite(d)))
                {
                    if (f + k < fail.step) { fail.step = f + k; fail.d = d; }
                    for (int c = 0; c < p; ++c) loc[f + c] = -1;
                    for (int a = 0; a < nr; ++a) loc[R[a]] = -1;
                    return false;
                }
                rd[k - ps] = 1.0 / d;
                for (int i = k + 1; i < pe; ++i) Fk[i] *= rd[k - ps];
            }
            const int below = m - pe;
            if (below == 0) continue;
            const int nblk = (below + RB - 1) / RB;                          // 2. the rows below it
            loop(par && static_cast<long>(below) * nb * nb > 8L * FEMD_OMP_THRESHOLD, nblk, [&](int g) {
                const int i0 = pe + g * RB, i1 = std::min(m, i0 + RB);
                for (int k = ps; k < pe; ++k)
                {
                    double *Fk = F + static_cast<std::size_t>(k) * M;
                    for (int t = ps; t < k; ++t)
                    {
                        const double *Ft = F + static_cast<std::size_t>(t) * M;
                        const double w = W[t - ps][k - ps];
                        for (int i = i0; i < i1; ++i) Fk[i] -= Ft[i] * w;
                    }
                    const double r = rd[k - ps];
                    for (int i = i0; i < i1; ++i) Fk[i] *= r;
                }
            });
            const int ng = (below + kernel::NC - 1) / kernel::NC;           // 3. the trailing triangle
            loop(par && static_cast<long>(below) * below * nb > 8L * FEMD_OMP_THRESHOLD, ng, [&](int g) {
                const int j0 = pe + g * kernel::NC, j1 = std::min(m, j0 + kernel::NC);
                for (int j = j0; j < j1; ++j)
                {
                    double *Fj = F + static_cast<std::size_t>(j) * M;
                    for (int t = ps; t < pe; ++t) Fj[t] = F[static_cast<std::size_t>(t) * M + j] * F[static_cast<std::size_t>(t) * M + t];
                }
                switch (j1 - j0)
                {
                    case 4: kernel::cols<4>(F, m, pe, ps, j0, j0); break;
                    case 3: kernel::cols<3>(F, m, pe, ps, j0, j0); break;
                    case 2: kernel::cols<2>(F, m, pe, ps, j0, j0); break;
                    case 1: kernel::cols<1>(F, m, pe, ps, j0, j0); break;
                    default: break;
                }
            });
        }
        // D and the columns of L at the rows of the symbolic pattern
        loop(big, p, [&](int c) {
            const int j = f + c;
            const double *Fc = F + static_cast<std::size_t>(c) * M;
            D_[j] = Fc[c];
            for (int q = Lp_[j]; q < Lp_[j + 1]; ++q) Lx_[q] = Fc[loc[Li_[q]]];
        });
        // the contribution block: the trailing lower triangle on the rows R, packed by columns
        if (nr > 0 && sn_parent_[s] >= 0)
        {
            std::vector<double> &C = cb_[s];
            C.resize(packed(nr, nr));
            double *Cv = C.data();
            loop(big, nr, [&](int b) {
                const double *Fc = F + static_cast<std::size_t>(p + b) * M + (p + b);
                std::copy(Fc, Fc + (nr - b), Cv + packed(b, nr));
            });
        }
        for (int c = 0; c < p; ++c) loc[f + c] = -1;
        for (int a = 0; a < nr; ++a) loc[R[a]] = -1;
        return true;
    }
};

} // namespace femd

#endif // FEMD_SPARSE_CHOLESKY_HPP
