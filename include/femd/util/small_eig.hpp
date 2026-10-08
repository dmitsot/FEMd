//
//  small_eig.hpp  --  eigenvalues and eigenvectors of a small real matrix (the Butcher matrix of
//  an implicit Runge-Kutta method, a few stages), in complex arithmetic: Hessenberg reduction,
//  the shifted QR iteration with Wilkinson shifts and deflation, then one eigenvector per
//  eigenvalue by solving (A - lambda I) v = 0 with a pivoted elimination.  Conjugate pairs are
//  matched, so a real system can be solved once per pair.  Not for large matrices: O(s^3) per
//  QR sweep is nothing for s <= 10 and far too much beyond.
//
#ifndef FEMD_UTIL_SMALL_EIG_HPP
#define FEMD_UTIL_SMALL_EIG_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace femd {

struct SmallEigen {
    using cd = std::complex<double>;
    int s = 0;
    std::vector<cd> values;           ///< eigenvalues
    std::vector<cd> T, Tinv;          ///< s x s row-major: A = T diag(values) T^{-1}
    std::vector<int> partner;         ///< partner[k] = j when values[k] = conj(values[j]) with j < k, -1 otherwise
    double condition = 0.0;           ///< an estimate of ||T|| ||T^{-1}|| (Frobenius)

    /// @param A  s x s row-major, real
    static SmallEigen of(const std::vector<double> &A, int s)
    {
        if (s < 1 || static_cast<int>(A.size()) != s * s) throw std::invalid_argument("small_eig: A must be s x s with s >= 1");
        SmallEigen E;
        E.s = s;
        std::vector<cd> H(A.begin(), A.end());
        E.values = eigenvalues(H, s);
        // conjugate pairs: pair each eigenvalue with an earlier conjugate, and make the pair exact
        E.partner.assign(static_cast<std::size_t>(s), -1);
        for (int k = 0; k < s; ++k)
        {
            const double scale = std::max(1.0, std::abs(E.values[k]));
            if (std::abs(E.values[k].imag()) <= 1e-12 * scale) { E.values[k] = cd(E.values[k].real(), 0.0); continue; }
            for (int j = 0; j < k; ++j)
                if (E.partner[j] < 0 && E.values[j].imag() != 0.0 && std::abs(E.values[j] - std::conj(E.values[k])) <= 1e-8 * scale)
                {
                    E.partner[k] = j;
                    E.values[k] = std::conj(E.values[j]);
                    break;
                }
        }
        // eigenvectors, the partner's the conjugate of its partner's
        E.T.assign(static_cast<std::size_t>(s) * s, cd(0.0));
        for (int k = 0; k < s; ++k)
        {
            std::vector<cd> v;
            if (E.partner[k] >= 0)
            {
                v.resize(static_cast<std::size_t>(s));
                for (int i = 0; i < s; ++i) v[i] = std::conj(E.T[static_cast<std::size_t>(i) * s + E.partner[k]]);
            }
            else v = eigenvector(A, s, E.values[k]);
            for (int i = 0; i < s; ++i) E.T[static_cast<std::size_t>(i) * s + k] = v[i];
        }
        E.Tinv = inverse(E.T, s);
        double nt = 0, ni = 0;
        for (const cd &z : E.T) nt += std::norm(z);
        for (const cd &z : E.Tinv) ni += std::norm(z);
        E.condition = std::sqrt(nt * ni);
        return E;
    }

private:
    // QR iteration on the full (small) matrix in complex arithmetic, deflating from the bottom
    static std::vector<cd> eigenvalues(std::vector<cd> H, int s)
    {
        std::vector<cd> lam;
        lam.reserve(static_cast<std::size_t>(s));
        int n = s;
        auto at = [&](int i, int j) -> cd & { return H[static_cast<std::size_t>(i) * s + j]; };
        hessenberg(H, s);
        int iter = 0;
        while (n > 0)
        {
            if (n == 1) { lam.push_back(at(0, 0)); break; }
            // deflate when the subdiagonal entry is negligible
            const double off = std::abs(at(n - 1, n - 2));
            const double diag = std::abs(at(n - 1, n - 1)) + std::abs(at(n - 2, n - 2));
            if (off <= 1e-15 * std::max(diag, 1e-300) || ++iter > 500 * s)
            {
                lam.push_back(at(n - 1, n - 1));
                --n;
                iter = 0;
                continue;
            }
            // Wilkinson shift: the eigenvalue of the trailing 2 x 2 block nearer to a_nn
            const cd a = at(n - 2, n - 2), b = at(n - 2, n - 1), c = at(n - 1, n - 2), d = at(n - 1, n - 1);
            const cd tr = a + d, det = a * d - b * c;
            const cd disc = std::sqrt(tr * tr - 4.0 * det);
            cd mu1 = 0.5 * (tr + disc), mu2 = 0.5 * (tr - disc);
            cd mu = std::abs(mu1 - d) < std::abs(mu2 - d) ? mu1 : mu2;
            if (iter % 11 == 10) mu += cd(off, 0.0);                   // an exceptional shift against cycling
            // QR step on the leading n x n block: H - mu I = Q R, H <- R Q + mu I, by Givens rotations
            for (int i = 0; i < n; ++i) at(i, i) -= mu;
            std::vector<cd> cs(static_cast<std::size_t>(n)), sn(static_cast<std::size_t>(n));
            for (int k = 0; k + 1 < n; ++k)
            {
                const cd x = at(k, k), y = at(k + 1, k);
                const double r = std::sqrt(std::norm(x) + std::norm(y));
                cd cc = r > 0 ? x / r : cd(1.0), ss = r > 0 ? y / r : cd(0.0);
                cs[k] = cc; sn[k] = ss;
                for (int j = k; j < n; ++j)
                {
                    const cd u = at(k, j), w = at(k + 1, j);
                    at(k, j) = std::conj(cc) * u + std::conj(ss) * w;
                    at(k + 1, j) = -ss * u + cc * w;
                }
            }
            for (int k = 0; k + 1 < n; ++k)
            {
                const cd cc = cs[k], ss = sn[k];
                for (int i = 0; i <= std::min(k + 1, n - 1); ++i)
                {
                    const cd u = at(i, k), w = at(i, k + 1);
                    at(i, k) = u * cc + w * ss;
                    at(i, k + 1) = -u * std::conj(ss) + w * std::conj(cc);
                }
            }
            for (int i = 0; i < n; ++i) at(i, i) += mu;
        }
        return lam;
    }

    static void hessenberg(std::vector<cd> &H, int s)
    {
        auto at = [&](int i, int j) -> cd & { return H[static_cast<std::size_t>(i) * s + j]; };
        for (int k = 0; k + 2 < s; ++k)
            for (int i = k + 2; i < s; ++i)
            {
                if (at(i, k) == cd(0.0)) continue;
                const cd x = at(k + 1, k), y = at(i, k);
                const double r = std::sqrt(std::norm(x) + std::norm(y));
                const cd cc = x / r, ss = y / r;
                for (int j = 0; j < s; ++j)                       // rows k+1 and i
                {
                    const cd u = at(k + 1, j), w = at(i, j);
                    at(k + 1, j) = std::conj(cc) * u + std::conj(ss) * w;
                    at(i, j) = -ss * u + cc * w;
                }
                for (int j = 0; j < s; ++j)                       // columns k+1 and i
                {
                    const cd u = at(j, k + 1), w = at(j, i);
                    at(j, k + 1) = u * cc + w * ss;
                    at(j, i) = -u * std::conj(ss) + w * std::conj(cc);
                }
                at(i, k) = cd(0.0);
            }
    }

    // an eigenvector for lambda by inverse iteration on A - (lambda + delta) I, delta a tiny
    // shift that keeps the elimination nonsingular (two steps from a fixed start, pivoted LU)
    static std::vector<cd> eigenvector(const std::vector<double> &A, int s, cd lambda)
    {
        double scale = 0.0;
        for (double a : A) scale = std::max(scale, std::abs(a));
        scale = std::max(scale, std::abs(lambda));
        const cd delta = cd(4e-16 * std::max(scale, 1e-300), 0.0);
        std::vector<cd> B(static_cast<std::size_t>(s) * s);
        for (int i = 0; i < s; ++i)
            for (int j = 0; j < s; ++j)
                B[static_cast<std::size_t>(i) * s + j] = cd(A[static_cast<std::size_t>(i) * s + j]) - (i == j ? lambda + delta : cd(0.0));
        // LU with partial pivoting of B
        std::vector<int> piv(static_cast<std::size_t>(s));
        auto at = [&](int i, int j) -> cd & { return B[static_cast<std::size_t>(i) * s + j]; };
        for (int k = 0; k < s; ++k)
        {
            int p = k;
            for (int i = k + 1; i < s; ++i) if (std::abs(at(i, k)) > std::abs(at(p, k))) p = i;
            piv[k] = p;
            if (p != k) for (int c = 0; c < s; ++c) std::swap(at(k, c), at(p, c));
            if (std::abs(at(k, k)) < 1e-300) at(k, k) = cd(1e-300, 0.0);
            for (int i = k + 1; i < s; ++i)
            {
                at(i, k) /= at(k, k);
                for (int c = k + 1; c < s; ++c) at(i, c) -= at(i, k) * at(k, c);
            }
        }
        std::vector<cd> v(static_cast<std::size_t>(s));
        for (int i = 0; i < s; ++i) v[i] = cd(1.0 + 0.1 * i, 0.05 * i);
        for (int step = 0; step < 3; ++step)
        {
            for (int k = 0; k < s; ++k) { if (piv[k] != k) std::swap(v[k], v[piv[k]]); for (int i = k + 1; i < s; ++i) v[i] -= at(i, k) * v[k]; }
            for (int i = s - 1; i >= 0; --i) { for (int c = i + 1; c < s; ++c) v[i] -= at(i, c) * v[c]; v[i] /= at(i, i); }
            double nv = 0.0;
            for (const cd &z : v) nv += std::norm(z);
            nv = std::sqrt(nv);
            for (cd &z : v) z /= nv;
        }
        // a real eigenvalue of a real matrix has a real eigenvector: rotate the phase away
        if (lambda.imag() == 0.0)
        {
            int big = 0;
            for (int i = 1; i < s; ++i) if (std::abs(v[i]) > std::abs(v[big])) big = i;
            const cd ph = std::abs(v[big]) > 0 ? std::conj(v[big]) / std::abs(v[big]) : cd(1.0);
            for (cd &z : v) z = cd((z * ph).real(), 0.0);
        }
        return v;
    }

    static std::vector<cd> inverse(std::vector<cd> M, int s)
    {
        std::vector<cd> I(static_cast<std::size_t>(s) * s, cd(0.0));
        for (int i = 0; i < s; ++i) I[static_cast<std::size_t>(i) * s + i] = cd(1.0);
        auto at = [&](std::vector<cd> &X, int i, int j) -> cd & { return X[static_cast<std::size_t>(i) * s + j]; };
        for (int k = 0; k < s; ++k)
        {
            int p = k;
            for (int i = k + 1; i < s; ++i) if (std::abs(at(M, i, k)) > std::abs(at(M, p, k))) p = i;
            if (std::abs(at(M, p, k)) == 0.0) throw std::runtime_error("small_eig: the eigenvector matrix is singular (A is not diagonalizable)");
            if (p != k) for (int c = 0; c < s; ++c) { std::swap(at(M, k, c), at(M, p, c)); std::swap(at(I, k, c), at(I, p, c)); }
            const cd d = at(M, k, k);
            for (int c = 0; c < s; ++c) { at(M, k, c) /= d; at(I, k, c) /= d; }
            for (int i = 0; i < s; ++i)
            {
                if (i == k) continue;
                const cd f = at(M, i, k);
                if (f == cd(0.0)) continue;
                for (int c = 0; c < s; ++c) { at(M, i, c) -= f * at(M, k, c); at(I, i, c) -= f * at(I, k, c); }
            }
        }
        return I;
    }
};

} // namespace femd

#endif // FEMD_UTIL_SMALL_EIG_HPP
