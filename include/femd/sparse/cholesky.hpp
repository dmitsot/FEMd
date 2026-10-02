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
#ifndef FEMD_SPARSE_CHOLESKY_HPP
#define FEMD_SPARSE_CHOLESKY_HPP

#include "femd/sparse/amd.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/sparse/precond.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
        std::vector<double> &y = work_;
        y.resize(static_cast<std::size_t>(n_));
        for (int k = 0; k < n_; ++k) y[k] = r[perm_[k]];
        const int *Lp = Lp_.data(), *Li = Li_.data();
        const double *Lx = Lx_.data();
        for (int j = 0; j < n_; ++j)                                  // L y = b
        {
            const double yj = y[j];
            if (yj != 0.0)
                for (int p = Lp[j]; p < Lp[j + 1]; ++p) y[Li[p]] -= Lx[p] * yj;
        }
        for (int j = 0; j < n_; ++j) y[j] /= D_[j];                    // D
        for (int j = n_ - 1; j >= 0; --j)                             // L^T x = y
        {
            double s = y[j];
            for (int p = Lp[j]; p < Lp[j + 1]; ++p) s -= Lx[p] * y[Li[p]];
            y[j] = s;
        }
        for (int k = 0; k < n_; ++k) z[perm_[k]] = y[k];
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
