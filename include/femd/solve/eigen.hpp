//
//  eigen.hpp  --  a few eigenvalues of largest magnitude of a linear operator by Arnoldi's method
//  with full reorthogonalization, explicit restarts and locking, the engine behind fd.eigs.
//
//  The caller gives the operator y = OP x (for the eigenvalues of A x = lambda M x nearest a
//  shift sigma, OP = (A - sigma M)^{-1} M, whose eigenvalues theta = 1 / (lambda - sigma) are
//  largest for the lambda nearest sigma) and, optionally, a matrix B defining the inner product
//  in which OP is self-adjoint (B = M in the symmetric case, so Arnoldi reduces to Lanczos and
//  the Ritz values are real).  Each restart builds an m-step Arnoldi basis orthogonal to the
//  locked vectors, takes the Ritz pairs of the small Hessenberg matrix (util/small_eig.hpp),
//  locks the wanted pairs whose residual estimate |h_{m+1,m} y_m| <= tol |theta| has been
//  met (a complex pair as the two real vectors spanning its invariant subspace), and restarts
//  from the sum of the unconverged wanted Ritz vectors.  Not ARPACK's implicit restart, but
//  with shift-invert the wanted values dominate and a few restarts do.
//
#ifndef FEMD_SOLVE_EIGEN_HPP
#define FEMD_SOLVE_EIGEN_HPP

#include "femd/util/small_eig.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

struct ArnoldiOptions {
    int k = 6;                       ///< eigenvalues wanted
    int ncv = 0;                     ///< Arnoldi vectors per restart (0: min(n, max(2k + 1, 20)))
    double tol = 0.0;                ///< relative residual tolerance (0: 1e-10)
    int maxiter = 0;                 ///< restarts (0: 300)
    bool symmetric = false;          ///< OP self-adjoint in the B inner product: real Ritz values
};

struct ArnoldiResult {
    using cd = std::complex<double>;
    std::vector<cd> values;          ///< the k Ritz values of OP, largest magnitude first
    std::vector<cd> vectors;         ///< n x k, column j at [j * n, (j + 1) n)
    std::vector<double> residuals;   ///< the residual estimates at locking
    int restarts = 0;
    int operator_applications = 0;
    bool converged = false;
    std::string message;
};

/**
 * @param n     dimension
 * @param op    y = OP x (vectors of length n)
 * @param bmat  y = B x for the inner product, or empty for the standard one
 * @param v0    the starting vector (length n)
 */
inline ArnoldiResult arnoldi_largest(int n, const std::function<void(const std::vector<double> &, std::vector<double> &)> &op,
                                     const std::function<void(const std::vector<double> &, std::vector<double> &)> &bmat,
                                     const ArnoldiOptions &o, const std::vector<double> &v0)
{
    using cd = std::complex<double>;
    using vec = std::vector<double>;
    ArnoldiResult R;
    const int k = o.k;
    if (k < 1 || k > n - 1) throw std::invalid_argument("arnoldi: k must be between 1 and n - 1");
    const int m = o.ncv > 0 ? std::min(n, o.ncv) : std::min(n, std::max(2 * k + 1, 20));
    if (m <= k) throw std::invalid_argument("arnoldi: ncv must exceed k");
    const double tol = o.tol > 0.0 ? o.tol : 1e-10;
    const int maxiter = o.maxiter > 0 ? o.maxiter : 300;
    const bool haveB = static_cast<bool>(bmat);

    std::vector<vec> locked, lockedB;                 // B-orthonormal locked vectors and their B images
    struct Pair { cd theta; std::vector<cd> x; double res; };
    std::vector<Pair> found;                           // the locked Ritz pairs

    auto Bapply = [&](const vec &x, vec &y) { if (haveB) bmat(x, y); else y = x; };
    auto dot = [&](const vec &a, const vec &b) { double s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s; };
    // orthogonalize v against the locked set and the basis vectors (B inner products through the B images), twice
    std::vector<vec> V, W;                             // Arnoldi vectors and their B images
    std::vector<double> H;                             // (m + 1) x m, column-major
    auto h = [&](int i, int j) -> double & { return H[static_cast<std::size_t>(j) * (m + 1) + i]; };
    auto orthogonalize = [&](vec &v, vec &bv, int jmax, bool record) {
        for (int pass = 0; pass < 2; ++pass)
        {
            for (std::size_t l = 0; l < locked.size(); ++l)
            {
                const double c = dot(lockedB[l], v);
                for (int i = 0; i < n; ++i) v[i] -= c * locked[l][i];
            }
            for (int j = 0; j < jmax; ++j)
            {
                const double c = dot(W[j], v);
                if (record) h(j, jmax - 1) += c;
                for (int i = 0; i < n; ++i) v[i] -= c * V[j][i];
            }
        }
        Bapply(v, bv);
        return std::sqrt(std::max(dot(v, bv), 0.0));
    };

    vec start = v0;
    if (static_cast<int>(start.size()) != n) throw std::invalid_argument("arnoldi: v0 must have length n");
    for (int restart = 0; restart < maxiter; ++restart)
    {
        R.restarts = restart + 1;
        const int locked_n = static_cast<int>(locked.size());
        const int wanted = k - static_cast<int>(found.size());
        if (wanted <= 0) break;
        const int steps = std::min(m, n - locked_n);
        if (steps < 1) { R.message = "the locked vectors span the space"; break; }
        V.assign(static_cast<std::size_t>(steps) + 1, vec(static_cast<std::size_t>(n)));
        W.assign(static_cast<std::size_t>(steps) + 1, vec(static_cast<std::size_t>(n)));
        H.assign(static_cast<std::size_t>(m + 1) * m, 0.0);
        // the first vector: start, B-normalized after orthogonalization against the locked set
        vec v = start, bv;
        double nv = orthogonalize(v, bv, 0, false);
        if (nv <= 1e-300)
        {
            for (int i = 0; i < n; ++i) v[i] = std::sin(0.37 * i + 1.0 + restart);
            nv = orthogonalize(v, bv, 0, false);
        }
        for (int i = 0; i < n; ++i) { V[0][i] = v[i] / nv; W[0][i] = bv[i] / nv; }
        int j = 0;
        bool breakdown = false;
        for (j = 0; j < steps; ++j)
        {
            op(V[j], v);
            ++R.operator_applications;
            const double nb = orthogonalize(v, bv, j + 1, true);
            h(j + 1, j) = nb;
            if (nb <= 1e-14 * (std::abs(h(j, j)) + 1.0)) { breakdown = true; ++j; break; }
            if (j + 1 <= steps) for (int i = 0; i < n; ++i) { V[j + 1][i] = v[i] / nb; W[j + 1][i] = bv[i] / nb; }
        }
        const int mj = j;                               // Arnoldi steps done
        // Ritz pairs of the leading mj x mj block
        std::vector<double> Hm(static_cast<std::size_t>(mj) * mj);
        for (int r = 0; r < mj; ++r) for (int c = 0; c < mj; ++c) Hm[static_cast<std::size_t>(r) * mj + c] = h(r, c);
        if (o.symmetric) for (int r = 0; r < mj; ++r) for (int c = 0; c < r; ++c) { const double a = 0.5 * (Hm[r * mj + c] + Hm[c * mj + r]); Hm[r * mj + c] = Hm[c * mj + r] = a; }
        const SmallEigen E = SmallEigen::of(Hm, mj);
        std::vector<int> order(static_cast<std::size_t>(mj));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return std::abs(E.values[a]) > std::abs(E.values[b]); });
        const double hlast = mj < steps || breakdown ? 0.0 : h(mj, mj - 1);
        // the candidates: the pairs kept so far and the Ritz pairs of this space; the k of largest
        // magnitude are kept (a converged pair that falls out of the k is dropped again, so an exact
        // invariant direction of small magnitude, a kernel vector say, cannot take a slot for good)
        struct Cand { cd theta; std::vector<cd> x; double res; int partner; bool conv; };
        std::vector<Cand> cands;
        for (const Pair &P : found) cands.push_back({P.theta, P.x, P.res, -1, true});
        std::vector<char> seen(static_cast<std::size_t>(mj), 0);
        for (int idx = 0; idx < mj; ++idx)
        {
            const int r = order[idx];
            if (seen[r]) continue;
            seen[r] = 1;
            const cd theta = E.values[r];
            std::vector<cd> x(static_cast<std::size_t>(n), cd(0.0));
            for (int c = 0; c < mj; ++c)
            {
                const cd y = E.T[static_cast<std::size_t>(c) * mj + r];
                if (y == cd(0.0)) continue;
                for (int i = 0; i < n; ++i) x[i] += y * V[c][i];
            }
            const double res = std::abs(hlast * E.T[static_cast<std::size_t>(mj - 1) * mj + r]) / std::max(std::abs(theta), 1e-300);
            int rp = -1;
            if (theta.imag() != 0.0)
                for (int c = 0; c < mj; ++c) if (E.partner[c] == r || E.partner[r] == c) rp = c;
            const bool conv = res <= tol || breakdown;
            cands.push_back({theta, x, res, rp >= 0 ? static_cast<int>(cands.size()) + 1 : -1, conv});
            if (rp >= 0)
            {
                seen[rp] = 1;
                std::vector<cd> xc(x.size());
                for (int i = 0; i < n; ++i) xc[i] = std::conj(x[i]);
                cands.push_back({std::conj(theta), xc, res, static_cast<int>(cands.size()) - 1, conv});
            }
        }
        std::vector<int> cord(cands.size());
        std::iota(cord.begin(), cord.end(), 0);
        std::stable_sort(cord.begin(), cord.end(), [&](int a, int b) { return std::abs(cands[a].theta) > std::abs(cands[b].theta); });
        std::vector<char> chosen(cands.size(), 0);
        int slots = 0;
        for (int c : cord)
        {
            if (slots >= k) break;
            if (chosen[c]) continue;
            chosen[c] = 1; ++slots;
            if (cands[c].partner >= 0 && !chosen[cands[c].partner]) { chosen[cands[c].partner] = 1; ++slots; }
        }
        found.clear();
        locked.clear(); lockedB.clear();
        vec newstart(static_cast<std::size_t>(n), 0.0);
        int unconverged = 0;
        for (std::size_t c = 0; c < cands.size(); ++c)
        {
            if (!chosen[c]) continue;
            const Cand &C = cands[c];
            if (C.conv)
            {
                found.push_back({C.theta, C.x, C.res});
                // its real and imaginary parts join the deflated set (the partner shares them)
                const bool second_of_pair = C.partner >= 0 && C.partner < static_cast<int>(c);
                if (!second_of_pair)
                    for (int part = 0; part < (C.theta.imag() != 0.0 ? 2 : 1); ++part)
                    {
                        vec lv(static_cast<std::size_t>(n)), lb;
                        for (int i = 0; i < n; ++i) lv[i] = part == 0 ? C.x[i].real() : C.x[i].imag();
                        const double nl = orthogonalize(lv, lb, 0, false);
                        if (nl > 1e-12)
                        {
                            for (int i = 0; i < n; ++i) { lv[i] /= nl; lb[i] /= nl; }
                            locked.push_back(lv); lockedB.push_back(lb);
                        }
                    }
            }
            else
            {
                ++unconverged;
                for (int i = 0; i < n; ++i) newstart[i] += C.x[i].real() + C.x[i].imag();
            }
        }
        if (unconverged == 0 && static_cast<int>(found.size()) >= k) { R.converged = true; break; }
        if (unconverged == 0)
        {
            // everything chosen has converged but the slots are not full (an invariant subspace):
            // restart from a fresh direction
            for (int i = 0; i < n; ++i) newstart[i] = std::cos(0.53 * i + restart);
        }
        start = newstart;
    }
    if (!R.converged && static_cast<int>(found.size()) >= k) R.converged = true;
    if (!R.converged && R.message.empty()) R.message = "the restarts ran out before all the wanted pairs converged";
    // the result: the k found pairs (or what there is), largest |theta| first
    std::vector<int> ord(found.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return std::abs(found[a].theta) > std::abs(found[b].theta); });
    const int kk = std::min<int>(k, static_cast<int>(found.size()));
    R.values.resize(static_cast<std::size_t>(kk));
    R.residuals.resize(static_cast<std::size_t>(kk));
    R.vectors.resize(static_cast<std::size_t>(kk) * n);
    for (int c = 0; c < kk; ++c)
    {
        const Pair &P = found[ord[c]];
        R.values[c] = P.theta;
        R.residuals[c] = P.res;
        std::copy(P.x.begin(), P.x.end(), R.vectors.begin() + static_cast<std::ptrdiff_t>(c) * n);
    }
    return R;
}

} // namespace femd

#endif // FEMD_SOLVE_EIGEN_HPP
