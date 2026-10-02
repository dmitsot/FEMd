//
//  precond.hpp  --  preconditioners on a CSRMatrix: Jacobi, SSOR and ILU(0).
//
//  Each is built once from the matrix and applied as z = M^{-1} r, the callable
//  shape the Krylov solvers take (krylov/cg.hpp and the csnewton GMRES through
//  krylov/adapters.hpp).  For a symmetric matrix SSOR and ILU(0) are symmetric
//  (ILU(0) is then IC(0) written as L D L^T), so either may precondition CG,
//  provided the factorization stays positive.  ILU(0) of a matrix that is not
//  an M-matrix (P2 and higher stiffness matrices are not) can meet a small or
//  negative pivot; the constructor reports the smallest one it saw.
//
//  SSOR and ILU(0) take an ordering: "natural" (the numbering of the space) or "rcm"
//  (reverse Cuthill-McKee, amd.hpp).  With "rcm" they are built on P A P^T and applied
//  as P^T M^{-1} P, which on an unstructured mesh usually makes them stronger, since
//  the entries they drop sit closer to the diagonal.
//
#ifndef FEMD_SPARSE_PRECOND_HPP
#define FEMD_SPARSE_PRECOND_HPP

#include "femd/sparse/csr.hpp"
#include "femd/sparse/amd.hpp"
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class SparsePreconditioner {
public:
    virtual ~SparsePreconditioner() = default;
    virtual void apply(const double *r, double *z) const = 0;
    virtual std::string name() const = 0;
    int size() const { return n_; }
    void operator()(const std::vector<double> &r, std::vector<double> &z) const
    {
        z.resize(static_cast<std::size_t>(n_));
        apply(r.data(), z.data());
    }
    /// @brief "natural", or "rcm" when the preconditioner works on the reordered matrix.
    std::string ordering() const { return perm_.empty() ? "natural" : "rcm"; }
    const std::vector<int> &permutation() const { return perm_; }
protected:
    int n_ = 0;
    std::vector<int> perm_;                       // empty: natural order
    mutable std::vector<double> rp_, zp_;
    /// The order named by `ordering` for A ("natural": none, "rcm": reverse Cuthill-McKee).
    static std::vector<int> order_for(const CSRMatrix &A, const std::string &ordering, const char *what)
    {
        if (ordering == "natural") return {};
        if (ordering == "rcm") return ordering::rcm_order(A.nrows(), A.indptr(), A.indices());
        throw std::invalid_argument(std::string(what) + ": ordering is 'natural' or 'rcm', got '" + ordering + "'");
    }
    /// z = P^T core(P r) when a permutation is set, core(r) otherwise.
    template <class F> void permuted_apply(const double *r, double *z, F core) const
    {
        if (perm_.empty()) { core(r, z); return; }
        rp_.resize(static_cast<std::size_t>(n_)); zp_.resize(static_cast<std::size_t>(n_));
        for (int k = 0; k < n_; ++k) rp_[k] = r[perm_[k]];
        core(rp_.data(), zp_.data());
        for (int k = 0; k < n_; ++k) z[perm_[k]] = zp_[k];
    }
    static void check_square(const CSRMatrix &A, const char *what)
    {
        if (A.nrows() != A.ncols()) throw std::invalid_argument(std::string(what) + ": the matrix must be square");
    }
    static std::vector<int> diagonal_positions(const CSRMatrix &A, const char *what)
    {
        std::vector<int> d(static_cast<std::size_t>(A.nrows()));
        for (int i = 0; i < A.nrows(); ++i)
        {
            d[i] = A.pattern->find(i, i);
            if (d[i] < 0 || A.data[d[i]] == 0.0)
                throw std::invalid_argument(std::string(what) + ": zero diagonal entry in row " + std::to_string(i));
        }
        return d;
    }
};

/// z_i = r_i / a_ii.
class JacobiPreconditioner : public SparsePreconditioner {
public:
    explicit JacobiPreconditioner(const CSRMatrix &A)
    {
        check_square(A, "Jacobi");
        n_ = A.nrows();
        std::vector<int> d = diagonal_positions(A, "Jacobi");
        inv_.resize(static_cast<std::size_t>(n_));
        for (int i = 0; i < n_; ++i) inv_[i] = 1.0 / A.data[d[i]];
    }
    void apply(const double *r, double *z) const override
    {
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) z[i] = inv_[i] * r[i];
    }
    std::string name() const override { return "jacobi"; }
private:
    std::vector<double> inv_;
};

/// Symmetric successive over-relaxation, M = w/(2-w) (D/w + L) (D/w)^{-1} (D/w + U).
class SSORPreconditioner : public SparsePreconditioner {
public:
    SSORPreconditioner(const CSRMatrix &A, double omega = 1.0, const std::string &ordering = "natural") : omega_(omega)
    {
        check_square(A, "SSOR");
        if (!(omega > 0.0 && omega < 2.0)) throw std::invalid_argument("SSOR: omega must lie in (0, 2)");
        n_ = A.nrows();
        perm_ = order_for(A, ordering, "SSOR");
        A_ = perm_.empty() ? A : permuted(A, perm_);
        diag_ = diagonal_positions(A_, "SSOR");
    }
    void apply(const double *r, double *z) const override
    {
        permuted_apply(r, z, [this](const double *a, double *b) { core(a, b); });
    }
    std::string name() const override { return "ssor"; }
private:
    void core(const double *r, double *z) const
    {
        const int *ip = A_.indptr(), *ix = A_.indices();
        const double *v = A_.data.data();
        const double w = omega_;
        // forward: (D/w + L) y = r
        for (int i = 0; i < n_; ++i)
        {
            double s = r[i];
            for (int p = ip[i]; p < diag_[i]; ++p) s -= v[p] * z[ix[p]];
            z[i] = s * w / v[diag_[i]];
        }
        // scale by (D/w) and (2-w)/w
        const double c = (2.0 - w) / w;
        for (int i = 0; i < n_; ++i) z[i] *= c * v[diag_[i]] / w;
        // backward: (D/w + U) z = y
        for (int i = n_ - 1; i >= 0; --i)
        {
            double s = z[i];
            for (int p = diag_[i] + 1; p < ip[i + 1]; ++p) s -= v[p] * z[ix[p]];
            z[i] = s * w / v[diag_[i]];
        }
    }
    CSRMatrix A_;                 // a copy: the preconditioner outlives any Python temporary
    double omega_;
    std::vector<int> diag_;
};

/// Incomplete LU with no fill, on the pattern of A.
class ILU0Preconditioner : public SparsePreconditioner {
public:
    explicit ILU0Preconditioner(const CSRMatrix &A, const std::string &ordering = "natural")
    {
        check_square(A, "ILU(0)");
        n_ = A.nrows();
        perm_ = order_for(A, ordering, "ILU(0)");
        LU_ = perm_.empty() ? A : permuted(A, perm_);
        diag_ = diagonal_positions(LU_, "ILU(0)");
        const int *ip = LU_.indptr(), *ix = LU_.indices();
        double *v = LU_.data.data();
        std::vector<int> pos(static_cast<std::size_t>(n_), -1);
        min_pivot_ = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n_; ++i)
        {
            for (int p = ip[i]; p < ip[i + 1]; ++p) pos[ix[p]] = p;
            for (int p = ip[i]; p < diag_[i]; ++p)
            {
                const int k = ix[p];
                const double piv = v[diag_[k]];
                v[p] /= piv;
                const double lik = v[p];
                for (int q = diag_[k] + 1; q < ip[k + 1]; ++q)
                {
                    const int t = pos[ix[q]];
                    if (t >= 0) v[t] -= lik * v[q];
                }
            }
            for (int p = ip[i]; p < ip[i + 1]; ++p) pos[ix[p]] = -1;
            const double d = v[diag_[i]];
            if (d == 0.0) throw std::runtime_error("ILU(0): zero pivot in row " + std::to_string(i));
            min_pivot_ = std::min(min_pivot_, d);
        }
    }
    void apply(const double *r, double *z) const override
    {
        permuted_apply(r, z, [this](const double *a, double *b) { core(a, b); });
    }
    void core(const double *r, double *z) const
    {
        const int *ip = LU_.indptr(), *ix = LU_.indices();
        const double *v = LU_.data.data();
        for (int i = 0; i < n_; ++i)
        {
            double s = r[i];
            for (int p = ip[i]; p < diag_[i]; ++p) s -= v[p] * z[ix[p]];
            z[i] = s;
        }
        for (int i = n_ - 1; i >= 0; --i)
        {
            double s = z[i];
            for (int p = diag_[i] + 1; p < ip[i + 1]; ++p) s -= v[p] * z[ix[p]];
            z[i] = s / v[diag_[i]];
        }
    }
    std::string name() const override { return "ilu0"; }
    /// @brief The smallest pivot met (negative: the factorization is indefinite, not for CG).
    double min_pivot() const { return min_pivot_; }
private:
    CSRMatrix LU_;
    std::vector<int> diag_;
    double min_pivot_ = 0.0;
};

} // namespace femd

#endif // FEMD_SPARSE_PRECOND_HPP
