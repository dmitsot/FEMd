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
//    numeric    an up-looking factorization: row k of L is found by a sparse
//               triangular solve with the rows above, whose pattern is that row
//               subtree, and the entries are appended to the columns of L.
//  L is unit lower triangular, stored by columns, D is diagonal.  No square roots
//  are taken, and no pivoting is done: a symmetric positive definite matrix never
//  needs it.  With positive_definite = true a pivot d_k <= 0 raises (the matrix is
//  not SPD).  With false only a zero pivot raises, which admits symmetric
//  quasi-definite matrices, but an indefinite matrix may then be factored
//  inaccurately: use a pivoting LU for those.
//
//  The class is a SparsePreconditioner (an exact one), so it drops into CG,
//  GMRES and the explicit Runge-Kutta stage loop wherever z = M^{-1} r is taken.
//  The matrix must hold both triangles (FEMd assembles full symmetric patterns);
//  only the entries with row >= column in the new order are read.
//
//  The triangular solves run in parallel with OpenMP over the elimination tree.
//  After the factorization the tree is cut into independent subtrees of about
//  1/64 of the work each (partition()); the nodes above them are the top.  In the
//  forward solve the subtrees are done concurrently, each a column-oriented sweep
//  whose updates stay inside the subtree, and the top nodes follow in order, each
//  one gathering what the subtrees owe it from its row of L.  In the backward
//  solve the top goes first, then the subtrees concurrently.  The top holds the
//  long columns (the separators), so it is swept by fundamental supernodes, runs
//  of columns with nested structure: the update of the rows below a supernode is
//  one dense block, done in parallel over its rows (forward) or its columns
//  (backward).  Every entry of the solution is formed by one thread in a fixed
//  order, so the result has the same bits for any number of threads, and the
//  serial build runs the same sweeps.
//
#ifndef FEMD_SPARSE_CHOLESKY_HPP
#define FEMD_SPARSE_CHOLESKY_HPP

#include "femd/sparse/amd.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/sparse/precond.hpp"
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
        perm_ = (ordering == Ordering::AMD) ? ordering::amd_order(n_, A.indptr(), A.indices())
                                            : ordering::natural_order(n_);
        pinv_.assign(static_cast<std::size_t>(n_), 0);
        for (int k = 0; k < n_; ++k) pinv_[perm_[k]] = k;
        symbolic(A);
        numeric(A);
        partition();
    }

    /// @brief New values on the same pattern: the numeric phase only.
    void refactor(const CSRMatrix &A)
    {
        if (A.nrows() != n_ || A.ncols() != n_ || A.nnz() != pattern_->nnz())
            throw std::invalid_argument("Cholesky::refactor: the matrix has a different pattern");
        if (A.pattern != pattern_ &&
            (A.pattern->indptr != pattern_->indptr || A.pattern->indices != pattern_->indices))
            throw std::invalid_argument("Cholesky::refactor: the matrix has a different pattern");
        numeric(A);
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
        const int nparts = nparts_;
        double *yv = y.data();
        const bool par = n > FEMD_OMP_THRESHOLD && nparts > 1;
        (void)par;
        FEMD_OMP_FOR_IF(par)
        for (int k = 0; k < n; ++k) yv[k] = r[perm[k]];
        // L y = b.  The subtrees concurrently: a column sweep whose updates stay below the root of
        // the subtree (the entries of a column are sorted, the rows above the root come last).
        FEMD_OMP(parallel for schedule(dynamic, 1) if(par))
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
        // then the top, supernode by supernode: its nodes gather what the subtrees owe them (their
        // rows of L below the top), the dense triangle inside the supernode is solved, and the rows
        // below it are updated as one block, in parallel over the rows
        const int *top = top_.data(), *snp = topsn_ptr_.data();
        const int nsn = static_cast<int>(topsn_ptr_.size()) - 1;
        for (int u = 0; u < nsn; ++u)
        {
            const int a = snp[u], sz = snp[u + 1] - a, j1 = top[a];
            for (int q = a; q < a + sz; ++q)
            {
                const int t = top[q];
                double v = yv[t];
                for (int p = toprow_ptr_[q]; p < toprow_ptr_[q + 1]; ++p) v -= Lx[toprow_p_[p]] * yv[toprow_j_[p]];
                yv[t] = v;
            }
            for (int i = 0; i < sz; ++i)                                // the triangle: rows j+1 .. j1+sz-1 of column j
            {
                const int j = j1 + i;
                const double yj = yv[j];
                if (yj == 0.0) continue;
                for (int k = 0; k < sz - 1 - i; ++k) yv[j + 1 + k] -= Lx[Lp[j] + k] * yj;
            }
            const int jl = j1 + sz - 1, below = Lp[jl + 1] - Lp[jl];    // the rows below: those of the last column
            const int *rows = Li + Lp[jl];
            FEMD_OMP_FOR_IF(par && static_cast<long>(sz) * below > FEMD_CHOL_BLOCK)
            for (int k = 0; k < below; ++k)
            {
                double v = yv[rows[k]];
                for (int i = 0; i < sz; ++i) v -= Lx[Lp[j1 + i] + (sz - 1 - i) + k] * yv[j1 + i];
                yv[rows[k]] = v;
            }
        }
        FEMD_OMP_FOR_IF(par)
        for (int j = 0; j < n; ++j) yv[j] /= D[j];                    // D
        // L^T x = y.  The top from its last supernode: each column gathers the rows below the
        // supernode (one dot product per column, in parallel over the columns), then the triangle;
        // then the subtrees concurrently, each from its root down (ancestors done, inside or top).
        std::vector<double> &w = work2_;
        for (int u = nsn - 1; u >= 0; --u)
        {
            const int a = snp[u], sz = snp[u + 1] - a, j1 = top[a];
            const int jl = j1 + sz - 1, below = Lp[jl + 1] - Lp[jl];
            const int *rows = Li + Lp[jl];
            if (static_cast<int>(w.size()) < sz) w.resize(static_cast<std::size_t>(sz));
            double *wv = w.data();
            FEMD_OMP_FOR_IF(par && static_cast<long>(sz) * below > FEMD_CHOL_BLOCK)
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
        FEMD_OMP(parallel for schedule(dynamic, 1) if(par))
        for (int s = 0; s < nparts; ++s)
            for (int q = partptr[s + 1] - 1; q >= partptr[s]; --q)
            {
                const int j = nodes[q];
                double v = yv[j];
                for (int p = Lp[j]; p < Lp[j + 1]; ++p) v -= Lx[p] * yv[Li[p]];
                yv[j] = v;
            }
        FEMD_OMP_FOR_IF(par)
        for (int k = 0; k < n; ++k) z[perm[k]] = yv[k];
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
    std::string ordering_name() const { return ordering_ == Ordering::AMD ? "amd" : "natural"; }
    bool positive_definite() const { return spd_; }

    /// @brief L (unit diagonal not stored) by columns, and D, in the permuted order.
    const std::vector<int> &Lp() const { return Lp_; }
    const std::vector<int> &Li() const { return Li_; }
    const std::vector<double> &Lx() const { return Lx_; }
    const std::vector<double> &D() const { return D_; }

private:
    bool spd_;
    Ordering ordering_;
    std::shared_ptr<const SparsityPattern> pattern_;
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
            if (!chain) topsn_ptr_.push_back(q + 1);
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

    // entries of column k of the upper triangle of P A P^T: row perm[k] of A, columns i = pinv[j] <= k
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
    }

    void numeric(const CSRMatrix &A)
    {
        const int n = n_;
        const int *Ap = A.indptr(), *Ai = A.indices();
        const double *Ax = A.data.data();
        std::vector<double> Y(static_cast<std::size_t>(n), 0.0);
        std::vector<int> pattern(static_cast<std::size_t>(n)), flag(static_cast<std::size_t>(n), -1),
            lnz(static_cast<std::size_t>(n), 0);
        int *Li = Li_.data();
        double *Lx = Lx_.data();
        const int *Lp = Lp_.data();
        for (int k = 0; k < n; ++k)
        {
            // scatter column k of the upper triangle into Y, and find the row subtree of k
            int top = n;
            flag[k] = k;
            const int row = perm_[k];
            for (int p = Ap[row]; p < Ap[row + 1]; ++p)
            {
                int i = pinv_[Ai[p]];
                if (i > k) continue;
                Y[i] += Ax[p];
                int len = 0;
                for (; flag[i] != k; i = parent_[i])
                {
                    pattern[len++] = i;
                    flag[i] = k;
                }
                while (len > 0) pattern[--top] = pattern[--len];
            }
            // row k of L by a sparse triangular solve, and the pivot d_k
            double d = Y[k];
            Y[k] = 0.0;
            for (; top < n; ++top)
            {
                const int i = pattern[top];
                const double yi = Y[i];
                Y[i] = 0.0;
                const int p2 = Lp[i] + lnz[i];
                for (int p = Lp[i]; p < p2; ++p) Y[Li[p]] -= Lx[p] * yi;
                const double lki = yi / D_[i];
                d -= lki * yi;
                Li[p2] = k;
                Lx[p2] = lki;
                ++lnz[i];
            }
            if (!(d > 0.0) && (spd_ || d == 0.0 || !std::isfinite(d)))
                throw std::runtime_error(std::string("Cholesky: ") +
                                         (spd_ ? "the matrix is not positive definite" : "zero pivot") +
                                         " (pivot " + std::to_string(d) + " at step " + std::to_string(k) +
                                         ", original row " + std::to_string(row) + ")");
            D_[k] = d;
        }
    }
};

} // namespace femd

#endif // FEMD_SPARSE_CHOLESKY_HPP
