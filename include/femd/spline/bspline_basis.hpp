//
//  bspline_basis.hpp  --  stable evaluation of B-spline values and derivatives
//  (Piegl & Tiller, The NURBS Book, algorithms A2.2 / A2.3).
//
//  No naive Cox-de Boor recursion: the triangular scheme below never divides
//  by a zero knot difference as long as the point lies in a span of nonzero
//  measure and no multiplicity exceeds p+1, which KnotVector guarantees.
//  Cost O(p^2 + n p) for derivatives up to order n.  Derivatives of order > p
//  are zero.
//
#ifndef FEMD_SPLINE_BSPLINE_BASIS_HPP
#define FEMD_SPLINE_BSPLINE_BASIS_HPP

#include <algorithm>
#include <vector>

namespace femd {

/**
 * @brief Values and derivatives of the p+1 B-splines nonzero on span i at u.
 * @param U     knot vector
 * @param i     span index, U[i] <= u < U[i+1]  (u == U[i+1] allowed for the last span)
 * @param u     evaluation point
 * @param p     degree
 * @param n     highest derivative order wanted
 * @param ders  [out] row-major (n+1) x (p+1):  ders[k*(p+1) + r] = d^k B_{i-p+r} / du^k (u)
 */
inline void ders_basis_funs(const std::vector<double> &U, int i, double u, int p, int n, double *ders)
{
    const int P1 = p + 1;
    std::vector<double> ndu(static_cast<std::size_t>(P1) * P1), left(P1), right(P1);
    std::vector<double> a(2 * static_cast<std::size_t>(P1));
    auto NDU = [&](int r, int c) -> double & { return ndu[r * P1 + c]; };
    auto A   = [&](int s, int c) -> double & { return a[s * P1 + c]; };

    NDU(0, 0) = 1.0;
    for (int j = 1; j <= p; ++j)
    {
        left[j]  = u - U[i + 1 - j];
        right[j] = U[i + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            NDU(j, r) = right[r + 1] + left[j - r];          // knot difference, > 0 on a nonempty span
            double temp = NDU(r, j - 1) / NDU(j, r);
            NDU(r, j) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        NDU(j, j) = saved;
    }
    for (int k = 0; k <= n; ++k) for (int r = 0; r <= p; ++r) ders[k * P1 + r] = 0.0;
    for (int j = 0; j <= p; ++j) ders[j] = NDU(j, p);

    int nn = std::min(n, p);
    for (int r = 0; r <= p; ++r)
    {
        int s1 = 0, s2 = 1;
        A(0, 0) = 1.0;
        for (int k = 1; k <= nn; ++k)
        {
            double d = 0.0;
            int rk = r - k, pk = p - k;
            if (r >= k) { A(s2, 0) = A(s1, 0) / NDU(pk + 1, rk); d = A(s2, 0) * NDU(rk, pk); }
            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : p - r;
            for (int j = j1; j <= j2; ++j)
            {
                A(s2, j) = (A(s1, j) - A(s1, j - 1)) / NDU(pk + 1, rk + j);
                d += A(s2, j) * NDU(rk + j, pk);
            }
            if (r <= pk) { A(s2, k) = -A(s1, k - 1) / NDU(pk + 1, r); d += A(s2, k) * NDU(r, pk); }
            ders[k * P1 + r] = d;
            std::swap(s1, s2);
        }
    }
    double fac = p;
    for (int k = 1; k <= nn; ++k)
    {
        for (int r = 0; r <= p; ++r) ders[k * P1 + r] *= fac;
        fac *= (p - k);
    }
}

/// @brief Values only (A2.2), out[r] = B_{i-p+r}(u).
inline void basis_funs(const std::vector<double> &U, int i, double u, int p, double *out)
{
    ders_basis_funs(U, i, u, p, 0, out);
}

} // namespace femd

#endif // FEMD_SPLINE_BSPLINE_BASIS_HPP
