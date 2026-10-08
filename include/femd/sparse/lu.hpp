//
//  lu.hpp  --  sparse LU factorization with partial (threshold) pivoting, real or complex,
//  FEMd's direct solver for nonsymmetric matrices: convection-diffusion, the Navier-Stokes
//  Jacobian, and the complex stage systems  M - dt lambda J  of the implicit Runge-Kutta
//  methods.  It replaces SciPy's SuperLU in the library.
//
//      SparseLU<double> F(n, Ap, Ai, Ax);          // CSR input: order (AMD on A + A^T), factor
//      F.solve(b, x);                              // x = A^{-1} b
//      F.refactor(Ax2);                            // same pattern, new values
//      SparseLU<std::complex<double>> G(n, Ap, Ai, Az);
//
//  The method is the left-looking column algorithm of Gilbert and Peierls, as in CSparse's
//  cs_lu: the columns are taken in a fill-reducing order q, and for each one a sparse
//  triangular solve with the columns of L computed so far gives the new column of L and U,
//  whose nonzero pattern is found by a depth-first search, so the cost is proportional to
//  the arithmetic.  The pivot of column k is the diagonal entry (row q[k]) when it is at
//  least `pivot_tol` times the largest candidate, and the largest candidate otherwise:
//  threshold pivoting, which bounds the growth of the factors by 1 / pivot_tol per step.
//  pivot_tol = 1 is partial pivoting.
//
//  The column order matters under pivoting.  The default, Ordering::ATA, is the approximate
//  minimum degree ordering of the pattern of A^T A, the graph in which two columns are joined
//  when they share a row: by Gilbert and Ng, the structure of L + U lies within the Cholesky
//  factor of A^T A whatever rows the pivoting picks, so this order bounds the fill for any
//  pivot sequence (it is what COLAMD approximates without forming A^T A; here the pattern is
//  formed, about four times nnz(A) for finite element matrices, which is cheap).  On the
//  stage matrix of a Navier-Stokes Radau step, a saddle point whose pressure diagonal is
//  zero, the order of A + A^T gave three times the fill of SuperLU, this one slightly less.
//  Ordering::AMD, the order of A + A^T, suits matrices that need no pivoting, where it fills
//  less (half, on a convection-diffusion grid).  The default, Ordering::Auto, takes AMD when
//  every diagonal entry is present and nonzero and ATA otherwise (a saddle point).
//
//  L (unit lower, its diagonal stored first in each column, the other rows sorted) and U
//  (its diagonal last) are kept by columns in the pivoted numbering.  solve() permutes, does
//  the two triangular sweeps and permutes back.  The input is CSR (FEMd's store); it is
//  transposed once into CSC at construction and the transposition is kept for refactor().
//
//  Parallel solves.  After the factorization the symmetric pattern of L + U is given its
//  elimination tree (Liu's algorithm on the rows of L and the columns of U): L(i, j) != 0 or
//  U(j, i) != 0 only when i is an ancestor of j, whatever pivots were taken, so the tree holds
//  the dependencies of both sweeps.  The tree is cut into independent subtrees of about 1/64
//  of the work each (partition()), the nodes above them are the top.  The subtrees run
//  concurrently, column by column: L y = b scatters each column into the rows below the root
//  of its subtree, U z = y scatters each column into its subtree after gathering the row's
//  entries in the top columns.  The top is done row by row (gathers).  Its nodes form chains
//  (a node with a single child in the top continues the child's chain), the chains are
//  scheduled by their height in the tree, and each is cut into blocks of rows; a step of the
//  schedule takes one block of every chain at that height: first every row gathers its
//  entries left of the block, in parallel over the rows, then the blocks are finished in
//  order, in parallel over the chains.  Every entry of the solution is accumulated in one
//  fixed order, so the result is the same bits for any number of threads.  The containment in
//  the tree is verified on the factors; should it ever fail the solves fall back to the plain
//  sweeps (tree_contained() says which).
//
#ifndef FEMD_SPARSE_LU_HPP
#define FEMD_SPARSE_LU_HPP

#include "femd/sparse/amd.hpp"
#include "femd/sparse/lu_frontal.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

// the cut of the elimination tree of L + U: subtrees of at most 1/FEMD_LU_PARTS of the work of
// the factors; a block update with more than FEMD_LU_BLOCK entries runs in parallel
#ifndef FEMD_LU_PARTS
#define FEMD_LU_PARTS 64.0
#endif
#ifndef FEMD_LU_BLOCK
#define FEMD_LU_BLOCK 4096
#endif
#ifndef FEMD_LU_CHAIN_BLOCK
#define FEMD_LU_CHAIN_BLOCK 64
#endif

namespace femd {

template <class T>
class SparseLU {
public:
    enum class Ordering { ATA, AMD, Natural, Given, Auto };
    /// Frontal: the multifrontal elimination of lu_frontal.hpp on the pattern of A + A^T (the default,
    /// dense panels, pivots kept inside the fronts); Columns: the left-looking column algorithm.
    enum class Method { Frontal, Columns };

    /**
     * @brief Order and factor a square matrix given in CSR.
     * @param n          dimension
     * @param Ap, Ai     row pointers (n + 1) and column indices
     * @param Ax         values
     * @param ordering   Ordering::Auto (default: AMD with a full nonzero diagonal, ATA otherwise), ATA, AMD, Natural, or Given (q)
     * @param pivot_tol  threshold in (0, 1]: the diagonal is kept when it is at least this
     *                   fraction of the largest candidate (0.1 by default)
     */
    SparseLU(int n, const int *Ap, const int *Ai, const T *Ax, Ordering ordering = Ordering::Auto, double pivot_tol = 0.1,
             const std::vector<int> &q = std::vector<int>(), Method method = Method::Frontal)
        : n_(n), tol_(pivot_tol), ordering_(ordering), method_(method)
    {
        if (n < 0) throw std::invalid_argument("LU: negative dimension");
        if (!(pivot_tol > 0.0 && pivot_tol <= 1.0)) throw std::invalid_argument("LU: pivot_tol must lie in (0, 1]");
        transpose(Ap, Ai);
        if (method == Method::Frontal)
        {
            // the pattern of A + A^T, and the ordering on it: AMD, natural, or the given one
            std::vector<int> Sp(static_cast<std::size_t>(n) + 1, 0), Si;
            Si.reserve(static_cast<std::size_t>(2) * static_cast<std::size_t>(Ap[n]));
            {
                std::vector<int> mark(static_cast<std::size_t>(n), -1);
                for (int i = 0; i < n; ++i)
                {
                    for (int p = Ap[i]; p < Ap[i + 1]; ++p) if (Ai[p] != i && mark[Ai[p]] != i) { mark[Ai[p]] = i; Si.push_back(Ai[p]); }
                    for (int p = Cp_[i]; p < Cp_[i + 1]; ++p) if (Ci_[p] != i && mark[Ci_[p]] != i) { mark[Ci_[p]] = i; Si.push_back(Ci_[p]); }
                    Sp[i + 1] = static_cast<int>(Si.size());
                }
            }
            std::vector<int> p0;
            if (ordering == Ordering::Given)
            {
                if (static_cast<int>(q.size()) != n) throw std::invalid_argument("LU: the given order must have n entries");
                p0 = q;
            }
            else if (ordering == Ordering::Natural) p0 = ordering::natural_order(n);
            else { p0 = ordering::amd_order(n, Sp.data(), Si.data()); ordering_ = Ordering::AMD; }
            frontal_.u = pivot_tol;
            frontal_.symbolic(n, Sp, Si, p0);
            numeric_frontal(Ax);
            partition();
            return;
        }
        if (ordering == Ordering::Given)
        {
            if (static_cast<int>(q.size()) != n) throw std::invalid_argument("LU: the given column order must have n entries");
            q_ = q;
        }
        else
        {
            if (ordering == Ordering::Auto)
            {
                bool full = true;
                for (int i = 0; i < n && full; ++i)
                {
                    full = false;
                    for (int p = Ap[i]; p < Ap[i + 1]; ++p) if (Ai[p] == i) { full = Ax[p] != T(0); break; }
                }
                ordering_ = ordering = full ? Ordering::AMD : Ordering::ATA;
            }
            if (ordering == Ordering::ATA) q_ = ata_order(Ap, Ai);
            else q_ = (ordering == Ordering::AMD) ? ordering::amd_order(n, Ap, Ai) : ordering::natural_order(n);
        }
        numeric(Ax);
        partition();
    }

    /// @brief New values on the same pattern (same ordering, pivots chosen afresh).
    void refactor(const T *Ax) { if (method_ == Method::Frontal) numeric_frontal(Ax); else numeric(Ax); partition(); }

    /// @brief x = A^{-1} b (x and b of length n, may not alias).
    void solve(const T *b, T *x) const
    {
        if (!tree_ok_ || nparts_ == 0) { solve_serial(b, x); return; }
        const int n = n_;
        std::vector<T> &y = work_;
        y.resize(static_cast<std::size_t>(n));
        work2_.resize(static_cast<std::size_t>(std::max(ntop_, 1)));
        T *yv = y.data(), *wv = work2_.data();
        const int *Lp = Lp_.data(), *Li = Li_.data(), *Up = Up_.data(), *Ui = Ui_.data(), *pinv = pinv_.data();
        const T *Lx = Lx_.data(), *Ux = Ux_.data();
        const int *nodes = nodes_.data(), *partptr = partptr_.data(), *root = root_.data(), *top = top_.data();
        const int *gnodes = grpnodes_.data();
        const int nparts = nparts_;
        const bool par = n > FEMD_OMP_THRESHOLD && nparts > 1;
        const int nsteps = static_cast<int>(step_ptr_.size()) - 1;
        (void)par;
        // one parallel region for the whole solve; every loop below ends in a barrier
        FEMD_OMP_PARALLEL_IF(par)
        {
            FEMD_OMP_FOR
            for (int i = 0; i < n; ++i) yv[pinv[i]] = b[i];                  // P b
            // L y = P b.  The subtrees concurrently: a column sweep whose updates stay below the
            // root of the subtree (the rows of a column are sorted, those above the root come last).
            FEMD_OMP(for schedule(dynamic, 1))
            for (int s = 0; s < nparts; ++s)
            {
                const int rs = root[s];
                for (int q = partptr[s]; q < partptr[s + 1]; ++q)
                {
                    const int j = nodes[q];
                    const T yj = yv[j];
                    if (yj == T(0)) continue;
                    for (int p = Lp[j] + 1; p < Lp[j + 1] && Li[p] <= rs; ++p) yv[Li[p]] -= Lx[p] * yj;
                }
            }
            // then the top by rows, step by step: the rows of the step gather their entries left of
            // their block (final by now), in parallel over the rows; then each block is finished in
            // order, in parallel over the blocks (one per chain)
            for (int st = 0; st < nsteps; ++st)
            {
                FEMD_OMP_FOR
                for (int r = step_ptr_[st]; r < step_ptr_[st + 1]; ++r)
                {
                    const int c = step_c_[r], q = gnodes[c];
                    T v = T(0);
                    for (int p = lrow_ptr_[q]; p < lrow_blk_[q]; ++p) v += lrow_x_[p] * yv[lrow_j_[p]];
                    wv[c] = v;
                }
                FEMD_OMP(for schedule(dynamic, 1))
                for (int m = sseg_[st]; m < sseg_[st + 1]; ++m)
                    for (int r = segb_[m]; r < segb_[m + 1]; ++r)
                    {
                        const int c = step_c_[r], q = gnodes[c], t = top[q];
                        T v = yv[t] - wv[c];
                        for (int p = lrow_blk_[q]; p < lrow_ptr_[q + 1]; ++p) v -= lrow_x_[p] * yv[lrow_j_[p]];
                        yv[t] = v;
                    }
            }
            // U z = y.  The top by rows, the steps in reverse: the rows gather their entries right of
            // their block in the top columns (final by now), then each block is finished from its
            // last row down.
            for (int st = nsteps - 1; st >= 0; --st)
            {
                FEMD_OMP_FOR
                for (int r = step_ptr_[st]; r < step_ptr_[st + 1]; ++r)
                {
                    const int c = step_c_[r], q = gnodes[c];
                    T v = T(0);
                    for (int p = urow_blk_[q]; p < urow_ptr_[q + 1]; ++p) v += urow_x_[p] * yv[urow_k_[p]];
                    wv[c] = v;
                }
                FEMD_OMP(for schedule(dynamic, 1))
                for (int m = sseg_[st]; m < sseg_[st + 1]; ++m)
                    for (int r = segb_[m + 1] - 1; r >= segb_[m]; --r)
                    {
                        const int c = step_c_[r], q = gnodes[c], i = top[q];
                        T v = yv[i] - wv[c];
                        for (int p = urow_ptr_[q]; p < urow_blk_[q]; ++p) v -= urow_x_[p] * yv[urow_k_[p]];
                        yv[i] = v / Ux[Up[i + 1] - 1];
                    }
            }
            // then the subtrees concurrently, each node first gathering the entries of its row of U
            // in the top columns, then applying its column, which stays inside the subtree
            FEMD_OMP(for schedule(dynamic, 1))
            for (int s = 0; s < nparts; ++s)
                for (int q = partptr[s + 1] - 1; q >= partptr[s]; --q)
                {
                    const int i = nodes[q];
                    T v = yv[i];
                    for (int p = uprow_ptr_[q]; p < uprow_ptr_[q + 1]; ++p) v -= uprow_x_[p] * yv[uprow_i_[p]];
                    const int pd = Up[i + 1] - 1;
                    v /= Ux[pd];
                    yv[i] = v;
                    if (v == T(0)) continue;
                    for (int p = Up[i]; p < pd; ++p) yv[Ui[p]] -= Ux[p] * v;
                }
            FEMD_OMP_FOR
            for (int k = 0; k < n; ++k) x[q_[k]] = yv[k];                     // undo the column order
        }
    }

    int size() const { return n_; }
    long long nnz_L() const { return Lp_.empty() ? 0 : Lp_.back(); }
    long long nnz_U() const { return Up_.empty() ? 0 : Up_.back(); }
    /// @brief Pivots taken off the diagonal of the ordered matrix.
    int off_diagonal_pivots() const { return offdiag_; }
    double min_pivot() const { return minpiv_; }
    double max_pivot() const { return maxpiv_; }
    const std::vector<int> &column_order() const { return q_; }
    const std::vector<int> &row_pivots() const { return pinv_; }
    /// @brief The elimination tree of the pattern of L + U (parent of each column, -1 at a root).
    const std::vector<int> &etree() const { return parent_; }
    /// @brief Whether L and U lie within the tree (true by construction, verified); otherwise the solves are serial.
    bool tree_contained() const { return tree_ok_; }
    /// @brief Independent subtrees of the elimination tree that the solves run concurrently.
    int subtrees() const { return nparts_; }
    /// @brief Nodes above the subtrees, solved by rows.
    int top_nodes() const { return ntop_; }
    /// @brief The top as chains of nodes with a single child in the top, the units of its schedule.
    int top_chains() const { return static_cast<int>(grpptr_.size()) - 1; }
    /// @brief Heights of the tree of chains.
    int top_levels() const { return nlevels_; }
    /// @brief Steps of the top schedule (blocks of rows done together), each two parallel loops.
    int top_steps() const { return static_cast<int>(step_ptr_.size()) - 1; }
    /// @brief Share of the entries of L and U in the top columns.
    double top_fraction() const
    {
        long long t = 0;
        for (int q = 0; q < ntop_; ++q) t += (Lp_[top_[q] + 1] - Lp_[top_[q]]) + (Up_[top_[q] + 1] - Up_[top_[q]]);
        const long long all = nnz_L() + nnz_U();
        return all > 0 ? static_cast<double>(t) / static_cast<double>(all) : 0.0;
    }
    std::string method_name() const { return method_ == Method::Frontal ? "frontal" : "columns"; }
    /// @brief Frontal method: pivots delayed to a parent front, the largest front, the supernodes.
    int delayed_pivots() const { return method_ == Method::Frontal ? frontal_.delayed : 0; }
    int max_front() const { return method_ == Method::Frontal ? frontal_.max_front : 0; }
    int fronts() const { return method_ == Method::Frontal ? static_cast<int>(frontal_.sn_first.size()) - 1 : 0; }
    std::string ordering_name() const
    {
        switch (ordering_) { case Ordering::ATA: return "ata"; case Ordering::AMD: return "amd"; case Ordering::Given: return "given"; default: return "natural"; }
    }
    double pivot_tol() const { return tol_; }

private:
    int n_;
    double tol_;
    Ordering ordering_;
    Method method_ = Method::Columns;
    frontal::Factor<T> frontal_;
    std::vector<int> Ap_, Ai_;               // the pattern of A by rows (kept for the frontal method)
    std::vector<int> Cp_, Ci_, Cpos_;        // A by columns (CSC) and, for each CSR entry, its place in it
    std::vector<int> q_, pinv_;              // column order, row i -> its pivot step
    std::vector<int> Lp_, Li_, Up_, Ui_;
    std::vector<T> Lx_, Ux_;
    int offdiag_ = 0;
    double minpiv_ = 0.0, maxpiv_ = 0.0;
    mutable std::vector<T> work_, work2_;
    // the elimination tree of L + U and its cut for the parallel solves (partition())
    std::vector<int> parent_;
    bool tree_ok_ = true;
    int nparts_ = 0, ntop_ = 0;
    std::vector<int> part_;                  // subtree of each node, -1 in the top
    std::vector<int> nodes_, partptr_;       // the nodes of subtree s, increasing: nodes_[partptr_[s] .. partptr_[s+1])
    std::vector<int> root_;                  // the root of each subtree, its largest node
    std::vector<int> top_;                   // the top nodes, increasing
    std::vector<int> grpptr_, grpnodes_;     // the chains of the top: positions (in top_) of the nodes of chain g, increasing
    int nlevels_ = 0;                        // heights of the tree of chains
    std::vector<int> step_ptr_, step_c_;     // the steps of the top schedule: positions (in grpnodes_) of the rows of step s, by chain
    std::vector<int> sseg_, segb_;           // the blocks of step s: segments [sseg_[s], sseg_[s+1]) of step_c_, segment m = [segb_[m], segb_[m+1])
    std::vector<int> lrow_ptr_, lrow_j_, lrow_blk_;   // for top node q: the entries L(t, j) by rows, j increasing; from lrow_blk_ on, j lies in the block of t
    std::vector<int> urow_ptr_, urow_k_, urow_blk_;   // for top node q: the entries U(i, k) with k in the top, k increasing; from urow_blk_ on, k lies beyond the block of i
    std::vector<int> uprow_ptr_, uprow_i_;             // for subtree node q (position in nodes_): the entries U(j, i) with i in the top
    std::vector<T> lrow_x_, urow_x_, uprow_x_;         // their values, copied from L and U so that the gathers read them in order

    void transpose(const int *Ap, const int *Ai)
    {
        const int n = n_;
        const int nnz = Ap[n];
        Ap_.assign(Ap, Ap + n + 1);
        Ai_.assign(Ai, Ai + nnz);
        Cp_.assign(static_cast<std::size_t>(n) + 1, 0);
        for (int p = 0; p < nnz; ++p) ++Cp_[Ai[p] + 1];
        for (int j = 0; j < n; ++j) Cp_[j + 1] += Cp_[j];
        Ci_.assign(static_cast<std::size_t>(nnz), 0);
        Cpos_.assign(static_cast<std::size_t>(nnz), 0);
        std::vector<int> next(Cp_.begin(), Cp_.end() - 1);
        for (int i = 0; i < n; ++i)
            for (int p = Ap[i]; p < Ap[i + 1]; ++p)
            {
                const int dest = next[Ai[p]]++;
                Ci_[dest] = i;
                Cpos_[p] = dest;
            }
    }

    // AMD of the pattern of A^T A: column j is joined to every column of every row that holds j
    std::vector<int> ata_order(const int *Ap, const int *Ai) const
    {
        const int n = n_;
        std::vector<int> Bp(static_cast<std::size_t>(n) + 1, 0), Bi, mark(static_cast<std::size_t>(n), -1);
        Bi.reserve(static_cast<std::size_t>(4) * static_cast<std::size_t>(Cp_[n]));
        for (int j = 0; j < n; ++j)
        {
            for (int p = Cp_[j]; p < Cp_[j + 1]; ++p)
            {
                const int i = Ci_[p];
                for (int r = Ap[i]; r < Ap[i + 1]; ++r)
                {
                    const int k = Ai[r];
                    if (k == j || mark[k] == j) continue;
                    mark[k] = j;
                    Bi.push_back(k);
                }
            }
            Bp[j + 1] = static_cast<int>(Bi.size());
        }
        return ordering::amd_order(n, Bp.data(), Bi.data());
    }

    // The elimination tree of the symmetric pattern of L + U (Liu's algorithm with path
    // compression): node k joins, for every i < k with L(k, i) != 0 or U(i, k) != 0, the path
    // from i upwards.  The rows of L come from a transposition of its pattern.
    void factor_etree()
    {
        const int n = n_;
        parent_.assign(static_cast<std::size_t>(n), -1);
        std::vector<int> Rp(static_cast<std::size_t>(n) + 1, 0), Ri(static_cast<std::size_t>(Lp_[n]));
        for (int j = 0; j < n; ++j) for (int p = Lp_[j] + 1; p < Lp_[j + 1]; ++p) ++Rp[Li_[p] + 1];
        for (int k = 0; k < n; ++k) Rp[k + 1] += Rp[k];
        {
            std::vector<int> next(Rp.begin(), Rp.end() - 1);
            for (int j = 0; j < n; ++j) for (int p = Lp_[j] + 1; p < Lp_[j + 1]; ++p) Ri[next[Li_[p]]++] = j;
        }
        std::vector<int> ancestor(static_cast<std::size_t>(n), -1);
        auto join = [&](int i, int k) {
            for (; i != -1 && i < k;)
            {
                const int inext = ancestor[i];
                ancestor[i] = k;
                if (inext == -1) parent_[i] = k;
                i = inext;
            }
        };
        for (int k = 0; k < n; ++k)
        {
            for (int p = Rp[k]; p < Rp[k + 1]; ++p) join(Ri[p], k);
            for (int p = Up_[k]; p < Up_[k + 1] - 1; ++p) join(Ui_[p], k);
        }
    }

    static void grow(std::vector<int> &I, std::vector<T> &X, std::size_t need)
    {
        if (need <= I.size()) return;
        std::size_t cap = std::max(need, 2 * I.size());
        I.resize(cap);
        X.resize(cap);
    }

    // The nonzero pattern of x in L x = A(:, col): the rows reached from the entries of the
    // column through the columns of L of already pivoted rows (depth-first search with an
    // explicit stack; marks in `mark`, the result in topological order in xi[top .. n)).
    int reach(int col, std::vector<int> &xi, std::vector<int> &pstack, std::vector<char> &mark) const
    {
        const int n = n_;
        int top = n;
        for (int p = Cp_[col]; p < Cp_[col + 1]; ++p)
        {
            const int i = Ci_[p];
            if (mark[i]) continue;
            // DFS from row i: its L column is pinv[i] when it has been pivoted
            int head = 0;
            xi[0] = i;
            while (head >= 0)
            {
                const int j = xi[head];
                const int jcol = pinv_[j];
                if (!mark[j])
                {
                    mark[j] = 1;
                    pstack[head] = jcol < 0 ? 0 : Lp_[jcol] + 1;          // skip the unit diagonal
                }
                bool done = true;
                if (jcol >= 0)
                {
                    int p2 = pstack[head];
                    const int p3 = Lp_[jcol + 1];
                    for (; p2 < p3; ++p2)
                    {
                        const int r = Li_[p2];
                        if (mark[r]) continue;
                        pstack[head] = p2 + 1;
                        xi[++head] = r;
                        done = false;
                        break;
                    }
                }
                if (done)
                {
                    --head;
                    xi[--top] = j;
                }
            }
        }
        return top;
    }

    void numeric(const T *Ax)
    {
        const int n = n_;
        // the values by columns
        std::vector<T> Cx(static_cast<std::size_t>(Cp_[n]));
        for (int p = 0; p < Cp_[n]; ++p) Cx[Cpos_[p]] = Ax[p];
        std::size_t guess = static_cast<std::size_t>(4) * static_cast<std::size_t>(Cp_[n]) + static_cast<std::size_t>(n);
        Lp_.assign(static_cast<std::size_t>(n) + 1, 0);
        Up_.assign(static_cast<std::size_t>(n) + 1, 0);
        Li_.assign(guess, 0); Lx_.assign(guess, T(0));
        Ui_.assign(guess, 0); Ux_.assign(guess, T(0));
        pinv_.assign(static_cast<std::size_t>(n), -1);
        std::vector<T> x(static_cast<std::size_t>(n), T(0));
        std::vector<int> xi(static_cast<std::size_t>(n)), pstack(static_cast<std::size_t>(n));
        std::vector<char> mark(static_cast<std::size_t>(n), 0);
        int lnz = 0, unz = 0;
        offdiag_ = 0; minpiv_ = 0.0; maxpiv_ = 0.0;
        for (int k = 0; k < n; ++k)
        {
            Lp_[k] = lnz;
            Up_[k] = unz;
            const int col = q_[k];
            // x = L \ A(:, col), on the pattern found by the reach
            const int top = reach(col, xi, pstack, mark);
            for (int p = Cp_[col]; p < Cp_[col + 1]; ++p) x[Ci_[p]] = Cx[p];
            for (int t = top; t < n; ++t)
            {
                const int j = xi[t];                     // a row; its L column, when pivoted
                const int jcol = pinv_[j];
                if (jcol < 0) continue;
                const T xj = x[j];
                if (xj == T(0)) continue;
                for (int p = Lp_[jcol] + 1; p < Lp_[jcol + 1]; ++p) x[Li_[p]] -= Lx_[p] * xj;
            }
            // the pivot: the diagonal (row col) when large enough, else the largest unpivoted entry
            int ipiv = -1;
            double amax = -1.0;
            int nl = 0, nu = 0;
            for (int t = top; t < n; ++t)
            {
                const int i = xi[t];
                if (pinv_[i] < 0) { ++nl; const double a = std::abs(x[i]); if (a > amax) { amax = a; ipiv = i; } }
                else ++nu;
            }
            if (ipiv < 0 || amax <= 0.0)
                throw std::runtime_error("LU: the matrix is singular (no pivot in column " + std::to_string(col) + ", step " + std::to_string(k) + ")");
            if (pinv_[col] < 0 && std::abs(x[col]) >= tol_ * amax) ipiv = col;
            if (ipiv != col) ++offdiag_;
            const T pivot = x[ipiv];
            const double ap = std::abs(pivot);
            minpiv_ = k == 0 ? ap : std::min(minpiv_, ap);
            maxpiv_ = std::max(maxpiv_, ap);
            grow(Li_, Lx_, static_cast<std::size_t>(lnz + nl));
            grow(Ui_, Ux_, static_cast<std::size_t>(unz + nu + 1));
            // U(:, k): the pivoted rows, then the pivot last on the diagonal
            for (int t = top; t < n; ++t)
            {
                const int i = xi[t];
                if (pinv_[i] >= 0) { Ui_[unz] = pinv_[i]; Ux_[unz++] = x[i]; }
            }
            Ui_[unz] = k; Ux_[unz++] = pivot;
            // L(:, k): the unit diagonal first, then the rest divided by the pivot
            pinv_[ipiv] = k;
            Li_[lnz] = ipiv; Lx_[lnz++] = T(1);
            for (int t = top; t < n; ++t)
            {
                const int i = xi[t];
                if (pinv_[i] < 0) { Li_[lnz] = i; Lx_[lnz++] = x[i] / pivot; }
                x[i] = T(0);
                mark[i] = 0;
            }
        }
        Lp_[n] = lnz; Up_[n] = unz;
        Li_.resize(static_cast<std::size_t>(lnz)); Lx_.resize(static_cast<std::size_t>(lnz));
        Ui_.resize(static_cast<std::size_t>(unz)); Ux_.resize(static_cast<std::size_t>(unz));
        for (int p = 0; p < lnz; ++p) Li_[p] = pinv_[Li_[p]];           // L in the pivoted numbering
        // the rows of each column of L in increasing order (after the unit diagonal)
        std::vector<int> order;
        std::vector<int> ti; std::vector<T> tx;
        for (int j = 0; j < n; ++j)
        {
            const int p0 = Lp_[j] + 1, len = Lp_[j + 1] - p0;
            if (len < 2) continue;
            bool sorted = true;
            for (int p = p0 + 1; p < p0 + len && sorted; ++p) sorted = Li_[p - 1] < Li_[p];
            if (sorted) continue;
            order.resize(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k) order[k] = p0 + k;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return Li_[a] < Li_[b]; });
            ti.resize(static_cast<std::size_t>(len)); tx.resize(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k) { ti[k] = Li_[order[k]]; tx[k] = Lx_[order[k]]; }
            for (int k = 0; k < len; ++k) { Li_[p0 + k] = ti[k]; Lx_[p0 + k] = tx[k]; }
        }
    }

    void numeric_frontal(const T *Ax)
    {
        const int n = n_;
        std::vector<T> Cx(static_cast<std::size_t>(Cp_[n]));
        for (int p = 0; p < Cp_[n]; ++p) Cx[Cpos_[p]] = Ax[p];
        frontal_.numeric(Ap_.data(), Ai_.data(), Ax, Cp_.data(), Ci_.data(), Cx.data());
        frontal_.export_csc(q_, pinv_, Lp_, Li_, Lx_, Up_, Ui_, Ux_);
        offdiag_ = frontal_.offdiag; minpiv_ = frontal_.minpiv; maxpiv_ = frontal_.maxpiv;
        std::vector<typename frontal::Factor<T>::Front>().swap(frontal_.fronts);   // the dense panels are in the CSC factors now
    }

    // the plain sweeps, used when the factors do not lie in the elimination tree
    void solve_serial(const T *b, T *x) const
    {
        const int n = n_;
        std::vector<T> &y = work_;
        y.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) y[pinv_[i]] = b[i];
        const int *Lp = Lp_.data(), *Li = Li_.data(), *Up = Up_.data(), *Ui = Ui_.data();
        const T *Lx = Lx_.data(), *Ux = Ux_.data();
        for (int j = 0; j < n; ++j)
        {
            const T yj = y[j];
            if (yj == T(0)) continue;
            for (int p = Lp[j] + 1; p < Lp[j + 1]; ++p) y[Li[p]] -= Lx[p] * yj;
        }
        for (int j = n - 1; j >= 0; --j)
        {
            const int pd = Up[j + 1] - 1;
            y[j] /= Ux[pd];
            const T yj = y[j];
            if (yj == T(0)) continue;
            for (int p = Up[j]; p < pd; ++p) y[Ui[p]] -= Ux[p] * yj;
        }
        for (int k = 0; k < n; ++k) x[q_[k]] = y[k];
    }

    /// Build the elimination tree of L + U, check that the factors lie in it, cut the tree into
    /// subtrees of at most 1/64 of the work of the factors each, biggest first, group the top into
    /// chains by height cut into blocks, and list the rows of L and U that the top and the subtrees gather.
    void partition()
    {
        const int n = n_;
        factor_etree();
        nparts_ = ntop_ = 0;
        tree_ok_ = true;
        part_.assign(static_cast<std::size_t>(n), -1);
        nodes_.clear(); root_.clear(); top_.clear(); grpnodes_.clear(); step_c_.clear();
        lrow_j_.clear(); lrow_x_.clear(); lrow_blk_.clear(); urow_k_.clear(); urow_x_.clear(); urow_blk_.clear(); uprow_i_.clear(); uprow_x_.clear();
        partptr_.assign(1, 0); grpptr_.assign(1, 0); step_ptr_.assign(1, 0); sseg_.assign(1, 0); segb_.assign(1, 0); lrow_ptr_.assign(1, 0); urow_ptr_.assign(1, 0); uprow_ptr_.assign(1, 0);
        nlevels_ = 0;
        if (n == 0) return;
        // postorder of the tree: i is an ancestor of j (or j itself) iff first[i] <= post[j] <= post[i]
        std::vector<int> head(static_cast<std::size_t>(n), -1), next(static_cast<std::size_t>(n), -1);
        for (int j = n - 1; j >= 0; --j) if (parent_[j] >= 0) { next[j] = head[parent_[j]]; head[parent_[j]] = j; }
        std::vector<int> post(static_cast<std::size_t>(n)), first(static_cast<std::size_t>(n)), stack;
        stack.reserve(static_cast<std::size_t>(n));
        int count = 0;
        for (int r = 0; r < n; ++r)
        {
            if (parent_[r] >= 0) continue;
            stack.push_back(r);
            first[r] = count;
            while (!stack.empty())
            {
                const int j = stack.back();
                const int c = head[j];
                if (c >= 0) { head[j] = next[c]; first[c] = count; stack.push_back(c); }
                else { post[j] = count++; stack.pop_back(); }
            }
        }
        for (int j = 0; j < n && tree_ok_; ++j)
        {
            for (int p = Lp_[j] + 1; p < Lp_[j + 1]; ++p)                  // L(i, j): i an ancestor of j
            {
                const int i = Li_[p];
                if (!(first[i] <= post[j] && post[j] <= post[i])) { tree_ok_ = false; break; }
            }
            for (int p = Up_[j]; p < Up_[j + 1] - 1; ++p)                  // U(i, j): j an ancestor of i
            {
                const int i = Ui_[p];
                if (!(first[j] <= post[i] && post[i] <= post[j])) { tree_ok_ = false; break; }
            }
        }
        if (!tree_ok_) return;
        // the work of a node is its column of L and U; of a subtree, the sum (children before parents)
        std::vector<double> W(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) W[j] = static_cast<double>(Lp_[j + 1] - Lp_[j] + Up_[j + 1] - Up_[j]);
        for (int j = 0; j < n; ++j) if (parent_[j] >= 0) W[parent_[j]] += W[j];
        double total = 0.0;
        for (int j = 0; j < n; ++j) if (parent_[j] < 0) total += W[j];
        const double limit = n > 2 * FEMD_OMP_THRESHOLD ? total / FEMD_LU_PARTS : -1.0;   // small: everything in the top
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
        std::vector<int> nodepos(static_cast<std::size_t>(n), -1);
        {
            std::vector<int> fill(partptr_.begin(), partptr_.end() - 1);
            for (int j = 0; j < n; ++j) if (part_[j] >= 0) { nodepos[j] = fill[part_[j]]; nodes_[fill[part_[j]]++] = j; }
        }
        ntop_ = static_cast<int>(top_.size());
        std::vector<int> toppos(static_cast<std::size_t>(n), -1);
        for (int q = 0; q < ntop_; ++q) toppos[top_[q]] = q;
        // the chains: a top node joins the chain of its only child in the top
        std::vector<int> nchild(static_cast<std::size_t>(ntop_), 0), onlychild(static_cast<std::size_t>(ntop_), -1), grp(static_cast<std::size_t>(ntop_), -1);
        for (int q = 0; q < ntop_; ++q)
        {
            const int pa = parent_[top_[q]];
            if (pa >= 0) { ++nchild[toppos[pa]]; onlychild[toppos[pa]] = q; }
        }
        int ngrp = 0;
        for (int q = 0; q < ntop_; ++q) grp[q] = nchild[q] == 1 ? grp[onlychild[q]] : ngrp++;
        grpptr_.assign(static_cast<std::size_t>(ngrp) + 1, 0);
        for (int q = 0; q < ntop_; ++q) ++grpptr_[grp[q] + 1];
        for (int g = 0; g < ngrp; ++g) grpptr_[g + 1] += grpptr_[g];
        grpnodes_.resize(static_cast<std::size_t>(ntop_));
        {
            std::vector<int> fill(grpptr_.begin(), grpptr_.end() - 1);
            for (int q = 0; q < ntop_; ++q) grpnodes_[fill[grp[q]]++] = q;
        }
        // the height of each chain in the tree of chains (a chain's parent chain holds the parent
        // of its last node, and has a larger number)
        std::vector<int> level(static_cast<std::size_t>(ngrp), 0);
        nlevels_ = 0;
        for (int g = 0; g < ngrp; ++g)
        {
            const int last = top_[grpnodes_[grpptr_[g + 1] - 1]];
            const int pa = parent_[last];
            if (pa >= 0) level[grp[toppos[pa]]] = std::max(level[grp[toppos[pa]]], level[g] + 1);
            nlevels_ = std::max(nlevels_, level[g] + 1);
        }
        std::vector<int> lvlptr(static_cast<std::size_t>(nlevels_) + 1, 0), lvlgrp(static_cast<std::size_t>(ngrp));
        for (int g = 0; g < ngrp; ++g) ++lvlptr[level[g] + 1];
        for (int l = 0; l < nlevels_; ++l) lvlptr[l + 1] += lvlptr[l];
        {
            std::vector<int> fill(lvlptr.begin(), lvlptr.end() - 1);
            for (int g = 0; g < ngrp; ++g) lvlgrp[fill[level[g]]++] = g;
        }
        // the steps: at each height the chains are cut into blocks of at least FEMD_LU_CHAIN_BLOCK
        // rows (at most 32 per chain), and step k takes block k of every chain of that height;
        // blk0/blk1 hold the first and last node of the block of each top node
        std::vector<int> blk0(static_cast<std::size_t>(ntop_)), blk1(static_cast<std::size_t>(ntop_));
        step_ptr_.assign(1, 0); step_c_.clear(); sseg_.assign(1, 0); segb_.assign(1, 0);
        for (int l = 0; l < nlevels_; ++l)
        {
            int longest = 0;
            for (int c = lvlptr[l]; c < lvlptr[l + 1]; ++c) longest = std::max(longest, grpptr_[lvlgrp[c] + 1] - grpptr_[lvlgrp[c]]);
            const int bs = std::max(FEMD_LU_CHAIN_BLOCK, (longest + 31) / 32);
            for (int k = 0; k * bs < longest; ++k)
            {
                for (int c = lvlptr[l]; c < lvlptr[l + 1]; ++c)
                {
                    const int g = lvlgrp[c], a = grpptr_[g] + k * bs, e = std::min(grpptr_[g + 1], a + bs);
                    if (a >= e) continue;
                    for (int r = a; r < e; ++r) { step_c_.push_back(r); blk0[grpnodes_[r]] = top_[grpnodes_[a]]; blk1[grpnodes_[r]] = top_[grpnodes_[e - 1]]; }
                    segb_.push_back(static_cast<int>(step_c_.size()));
                }
                step_ptr_.push_back(static_cast<int>(step_c_.size()));
                sseg_.push_back(static_cast<int>(segb_.size()) - 1);
            }
        }
        // the rows of L of the top nodes, j increasing, split where j enters the block of the row
        lrow_ptr_.assign(static_cast<std::size_t>(ntop_) + 1, 0);
        for (int j = 0; j < n; ++j)
            for (int p = Lp_[j] + 1; p < Lp_[j + 1]; ++p) if (part_[Li_[p]] < 0) ++lrow_ptr_[toppos[Li_[p]] + 1];
        for (int q = 0; q < ntop_; ++q) lrow_ptr_[q + 1] += lrow_ptr_[q];
        lrow_j_.resize(static_cast<std::size_t>(lrow_ptr_[ntop_]));
        lrow_x_.resize(lrow_j_.size());
        {
            std::vector<int> pos(lrow_ptr_.begin(), lrow_ptr_.end() - 1);
            for (int j = 0; j < n; ++j)
                for (int p = Lp_[j] + 1; p < Lp_[j + 1]; ++p)
                {
                    const int t = Li_[p];
                    if (part_[t] >= 0) continue;
                    const int d = pos[toppos[t]]++;
                    lrow_j_[d] = j; lrow_x_[d] = Lx_[p];
                }
        }
        lrow_blk_.assign(static_cast<std::size_t>(ntop_), 0);
        for (int q = 0; q < ntop_; ++q)
        {
            int p = lrow_ptr_[q];
            while (p < lrow_ptr_[q + 1] && lrow_j_[p] < blk0[q]) ++p;
            lrow_blk_[q] = p;
        }
        // the rows of U of the top nodes within the top columns, k increasing, split where k leaves
        // the block of the row; and the entries of the top columns in subtree rows, listed by row
        urow_ptr_.assign(static_cast<std::size_t>(ntop_) + 1, 0);
        uprow_ptr_.assign(static_cast<std::size_t>(partptr_[nparts_]) + 1, 0);
        for (int q = 0; q < ntop_; ++q)
        {
            const int k = top_[q];
            for (int p = Up_[k]; p < Up_[k + 1] - 1; ++p)
            {
                const int r = Ui_[p];
                if (part_[r] < 0) ++urow_ptr_[toppos[r] + 1];
                else ++uprow_ptr_[nodepos[r] + 1];
            }
        }
        for (int q = 0; q < ntop_; ++q) urow_ptr_[q + 1] += urow_ptr_[q];
        for (int q = 0; q < partptr_[nparts_]; ++q) uprow_ptr_[q + 1] += uprow_ptr_[q];
        urow_k_.resize(static_cast<std::size_t>(urow_ptr_[ntop_]));  urow_x_.resize(urow_k_.size());
        uprow_i_.resize(static_cast<std::size_t>(uprow_ptr_[partptr_[nparts_]]));  uprow_x_.resize(uprow_i_.size());
        {
            std::vector<int> pos(urow_ptr_.begin(), urow_ptr_.end() - 1), ppos(uprow_ptr_.begin(), uprow_ptr_.end() - 1);
            for (int q = 0; q < ntop_; ++q)
            {
                const int k = top_[q];
                for (int p = Up_[k]; p < Up_[k + 1] - 1; ++p)
                {
                    const int r = Ui_[p];
                    if (part_[r] < 0) { const int d = pos[toppos[r]]++; urow_k_[d] = k; urow_x_[d] = Ux_[p]; }
                    else { const int d = ppos[nodepos[r]]++; uprow_i_[d] = k; uprow_x_[d] = Ux_[p]; }
                }
            }
        }
        urow_blk_.assign(static_cast<std::size_t>(ntop_), 0);
        for (int q = 0; q < ntop_; ++q)
        {
            int p = urow_ptr_[q];
            while (p < urow_ptr_[q + 1] && urow_k_[p] <= blk1[q]) ++p;
            urow_blk_[q] = p;
        }
    }
};

} // namespace femd

#endif // FEMD_SPARSE_LU_HPP
