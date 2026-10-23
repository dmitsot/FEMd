//
//  stage_solver.hpp  --  the simplified-Newton matrix of an implicit Runge-Kutta step,
//
//      I (x) M - dt A (x) J,     A the s x s Butcher matrix, M and J n x n sparse,
//
//  solved through the eigen decomposition A = T diag(lambda) T^{-1} (Butcher's transformation,
//  as in Hairer and Wanner's RADAU5): one system  M - dt lambda_k J  of size n per real
//  eigenvalue and one complex one per conjugate pair, instead of one real system of size s n,
//  whose sparse factorization would cost about s^3 times more.  The real systems go to the
//  sparse Cholesky, the pivoting LDL^T or the LU by the matrix (symmetric with a positive
//  diagonal, symmetric, nonsymmetric), the complex ones to the complex LU, M - dt lambda J
//  being formed as two real matrices on one merged pattern.
//
//      StageSystemSolver S(M, &J, A, s, dt);       // factor
//      S.solve(r, d);                              // d = (I (x) M - dt A (x) J)^{-1} r, stages stacked
//      S.refactor(M, &J2, dt);                     // new values on the same patterns: the orderings and
//                                                  // symbolic analyses are kept, only the numeric phases run
//
#ifndef FEMD_TIMESTEP_STAGE_SOLVER_HPP
#define FEMD_TIMESTEP_STAGE_SOLVER_HPP

#include "femd/sparse/cholesky.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/sparse/ldlt.hpp"
#include "femd/sparse/lu.hpp"
#include "femd/util/small_eig.hpp"
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class StageSystemSolver {
public:
    using cd = std::complex<double>;

    /**
     * @param M, J      the mass matrix and the Jacobian f'(t_n, u^n) (J may be null: f linear and
     *                  already in M's space is not a case, so J = 0 then means f' = 0)
     * @param A         s x s Butcher matrix, row-major
     * @param backend   "auto" (by the matrix), "cholesky", "ldlt" or "lu" for the real systems
     */
    StageSystemSolver(const CSRMatrix &M, const CSRMatrix *J, const std::vector<double> &A, int s, double dt,
                      const std::string &backend = "auto")
        : s_(s), n_(M.nrows()), E_(SmallEigen::of(A, s)), has_J_(J != nullptr), backend_(backend)
    {
        if (M.nrows() != M.ncols()) throw std::invalid_argument("stage solver: M must be square");
        if (J && (J->nrows() != n_ || J->ncols() != n_)) throw std::invalid_argument("stage solver: J must have the size of M");
        if (E_.condition > 1e10) throw std::invalid_argument("stage solver: the Butcher matrix is not (well) diagonalizable");
        sys_.resize(static_cast<std::size_t>(s));
        for (int k = 0; k < s; ++k)
        {
            if (E_.partner[k] >= 0) continue;
            const cd lam = E_.values[k];
            Sys &S = sys_[k];
            if (lam.imag() == 0.0)
            {
                CSRMatrix B = J ? combine(M, *J, 1.0, -dt * lam.real()) : scaled(M, 1.0);
                factor_real(S, B, backend);
                S.pattern = B.pattern;
            }
            else
            {
                CSRMatrix re = J ? combine(M, *J, 1.0, -dt * lam.real()) : scaled(M, 1.0);
                CSRMatrix im = J ? combine(M, *J, 0.0, -dt * lam.imag()) : scaled(M, 0.0);
                if (re.pattern->indices != im.pattern->indices || re.pattern->indptr != im.pattern->indptr)
                    throw std::runtime_error("stage solver: the real and imaginary parts do not share a pattern");
                std::vector<cd> z(re.data.size());
                for (std::size_t p = 0; p < z.size(); ++p) z[p] = cd(re.data[p], im.data[p]);
                S.clu = std::make_unique<SparseLU<cd>>(n_, re.indptr(), re.indices(), z.data());
                S.kind = "lu (complex)";
                S.pattern = re.pattern;
            }
            ++factorizations_;
        }
    }

    /**
     * @brief New M, J and dt on the patterns of the factorization: each system keeps its ordering and
     *        symbolic analysis and runs only its numeric phase (a Cholesky that fails falls back to LDL^T).
     * @return false, with nothing changed, when a pattern differs (or J is given where it was not, or the
     *         other way round): build a new solver then.
     */
    bool refactor(const CSRMatrix &M, const CSRMatrix *J, double dt)
    {
        if (M.nrows() != n_ || M.ncols() != n_ || (J != nullptr) != has_J_) return false;
        if (J && (J->nrows() != n_ || J->ncols() != n_)) return false;
        auto same = [](const SparsityPattern &a, const SparsityPattern &b) {
            return &a == &b || (a.indptr == b.indptr && a.indices == b.indices);
        };
        // form every system first, so that a pattern that differs changes nothing
        std::vector<CSRMatrix> re(static_cast<std::size_t>(s_)), im(static_cast<std::size_t>(s_));
        for (int k = 0; k < s_; ++k)
        {
            if (E_.partner[k] >= 0) continue;
            const cd lam = E_.values[k];
            re[k] = J ? combine(M, *J, 1.0, -dt * lam.real()) : scaled(M, 1.0);
            if (!sys_[k].pattern || !same(*re[k].pattern, *sys_[k].pattern)) return false;
            if (lam.imag() != 0.0) im[k] = J ? combine(M, *J, 0.0, -dt * lam.imag()) : scaled(M, 0.0);
            if (lam.imag() != 0.0 && !same(*im[k].pattern, *re[k].pattern)) return false;
        }
        for (int k = 0; k < s_; ++k)
        {
            if (E_.partner[k] >= 0) continue;
            Sys &S = sys_[k];
            if (S.clu)
            {
                std::vector<cd> z(re[k].data.size());
                for (std::size_t p = 0; p < z.size(); ++p) z[p] = cd(re[k].data[p], im[k].data[p]);
                S.clu->refactor(z.data());
            }
            else if (S.lu) S.lu->refactor(re[k].data.data());
            else if (S.ldlt) S.ldlt->refactor(re[k]);
            else
            {
                try { S.chol->refactor(re[k]); }
                catch (const std::runtime_error &) { Sys T; factor_real(T, re[k], "ldlt"); T.pattern = S.pattern; S = std::move(T); }
            }
            ++factorizations_;
        }
        return true;
    }

    StageSystemSolver(const StageSystemSolver &) = delete;
    StageSystemSolver &operator=(const StageSystemSolver &) = delete;
    StageSystemSolver(StageSystemSolver &&) = default;

    int size() const { return s_ * n_; }
    int stages() const { return s_; }
    int factorizations() const { return factorizations_; }
    /// @brief The kind of factorization of each distinct system, in order of the eigenvalues.
    std::vector<std::string> kinds() const
    {
        std::vector<std::string> k;
        for (int i = 0; i < s_; ++i) if (E_.partner[i] < 0) k.push_back(sys_[i].kind);
        return k;
    }
    const SmallEigen &eigen() const { return E_; }

    /// @brief d = (I (x) M - dt A (x) J)^{-1} r, r and d of length s n, the stages stacked.
    void solve(const double *r, double *d) const
    {
        const int s = s_, n = n_;
        std::vector<cd> &Z = Z_, &W = W_;
        Z.assign(static_cast<std::size_t>(s) * n, cd(0.0));
        W.assign(static_cast<std::size_t>(s) * n, cd(0.0));
        for (int k = 0; k < s; ++k)                                 // Z = T^{-1} R
            for (int j = 0; j < s; ++j)
            {
                const cd t = E_.Tinv[static_cast<std::size_t>(k) * s + j];
                if (t == cd(0.0)) continue;
                const double *rj = r + static_cast<std::size_t>(j) * n;
                cd *zk = Z.data() + static_cast<std::size_t>(k) * n;
                for (int i = 0; i < n; ++i) zk[i] += t * rj[i];
            }
        std::vector<double> &re = re_, &im = im_, &xr = xr_, &xi = xi_;
        re.resize(static_cast<std::size_t>(n)); im.resize(static_cast<std::size_t>(n));
        xr.resize(static_cast<std::size_t>(n)); xi.resize(static_cast<std::size_t>(n));
        for (int k = 0; k < s; ++k)
        {
            if (E_.partner[k] >= 0) continue;
            const Sys &S = sys_[k];
            const cd *zk = Z.data() + static_cast<std::size_t>(k) * n;
            cd *wk = W.data() + static_cast<std::size_t>(k) * n;
            if (S.clu) S.clu->solve(zk, wk);
            else
            {
                for (int i = 0; i < n; ++i) { re[i] = zk[i].real(); im[i] = zk[i].imag(); }
                S.apply(re.data(), xr.data());
                S.apply(im.data(), xi.data());
                for (int i = 0; i < n; ++i) wk[i] = cd(xr[i], xi[i]);
            }
        }
        for (int k = 0; k < s; ++k)
            if (E_.partner[k] >= 0)
            {
                const cd *wp = W.data() + static_cast<std::size_t>(E_.partner[k]) * n;
                cd *wk = W.data() + static_cast<std::size_t>(k) * n;
                for (int i = 0; i < n; ++i) wk[i] = std::conj(wp[i]);
            }
        for (int j = 0; j < s; ++j)                                 // D = Re(T W)
        {
            double *dj = d + static_cast<std::size_t>(j) * n;
            for (int i = 0; i < n; ++i) dj[i] = 0.0;
            for (int k = 0; k < s; ++k)
            {
                const cd t = E_.T[static_cast<std::size_t>(j) * s + k];
                if (t == cd(0.0)) continue;
                const cd *wk = W.data() + static_cast<std::size_t>(k) * n;
                for (int i = 0; i < n; ++i) dj[i] += (t * wk[i]).real();
            }
        }
    }

private:
    struct Sys {
        std::unique_ptr<SparseCholesky> chol;
        std::unique_ptr<SparseLDLT> ldlt;
        std::unique_ptr<SparseLU<double>> lu;
        std::unique_ptr<SparseLU<cd>> clu;
        std::string kind;
        std::shared_ptr<const SparsityPattern> pattern;   // of the system's matrix, for refactor()
        void apply(const double *b, double *x) const
        {
            if (chol) chol->apply(b, x);
            else if (ldlt) ldlt->apply(b, x);
            else lu->solve(b, x);
        }
    };
    int s_, n_;
    SmallEigen E_;
    std::vector<Sys> sys_;
    int factorizations_ = 0;
    bool has_J_ = false;
    std::string backend_;
    mutable std::vector<cd> Z_, W_;
    mutable std::vector<double> re_, im_, xr_, xi_;

    static void factor_real(Sys &S, const CSRMatrix &B, const std::string &backend)
    {
        std::string b = backend;
        if (b == "auto" || b == "Auto" || b.empty())
        {
            const bool sym = asymmetry(B) <= 1e-12;
            if (!sym) b = "lu";
            else
            {
                bool posdiag = true;
                for (double v : B.diagonal()) if (!(v > 0.0)) { posdiag = false; break; }
                b = posdiag ? "cholesky" : "ldlt";
            }
        }
        if (b == "cholesky")
        {
            try { S.chol = std::make_unique<SparseCholesky>(B); S.kind = "cholesky"; return; }
            catch (const std::runtime_error &) { b = "ldlt"; }                  // not positive definite after all
        }
        if (b == "ldlt") { S.ldlt = std::make_unique<SparseLDLT>(B); S.kind = "ldlt"; return; }
        S.lu = std::make_unique<SparseLU<double>>(B.nrows(), B.indptr(), B.indices(), B.data.data());
        S.kind = "lu";
    }
};

} // namespace femd

#endif // FEMD_TIMESTEP_STAGE_SOLVER_HPP
