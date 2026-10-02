//
//  cyclicmatrix.hpp  --  general periodic (cyclic) banded matrix
//
//  Lifted and cleaned from old_poseidon/matrix_h/{cyclic.h,cyclic_code.h}
//  (D. Mitsotakis, 2012).  BUG FIX applied during lift:
//    * the `sized` member was never initialised in the (int,int) ctor, so
//      bandm.pt = new T[sized] allocated a garbage-sized block.  Now set
//      sized = band*N.
//  Plus C++17 include/namespace cleanup and non-interactive print.
//
//  Periodic banded solve via a banded core (bandmatrix::LU, WITHOUT pivoting) plus a
//  low-rank corner correction (Sherman-Morrison-Woodbury).  1-based indexing.
//  Stable for diagonally dominant and positive definite matrices.  LinearSolver's Auto
//  checks every solve of this kind and falls back to detail::PivotedCyclicSolver.
//
#ifndef FEMD_LINALG_CYCLICMATRIX_HPP
#define FEMD_LINALG_CYCLICMATRIX_HPP

#include "matrix.hpp"
#include "bandmatrix.hpp"

namespace femd {

template <typename T> class cyclicmatrix;
template <typename T> cyclicmatrix<T> operator+(const cyclicmatrix<T> &, const cyclicmatrix<T> &);
template <typename T> cyclicmatrix<T> operator-(const cyclicmatrix<T> &, const cyclicmatrix<T> &);
template <typename T> cyclicmatrix<T> operator*(const T &, const cyclicmatrix<T> &);
template <typename T> cyclicmatrix<T> operator*(const cyclicmatrix<T> &, const T &);
template <typename T> matrix<T> operator*(const cyclicmatrix<T> &, const matrix<T> &);

template <typename T> class cyclicmatrix {
public:
    /// @brief Default 1x1 matrix.
    cyclicmatrix();
    /// @brief Copy constructor.
    cyclicmatrix(const cyclicmatrix &);
    /// @brief Allocate a zero general periodic banded matrix. @param rows N. @param upperband half-bandwidth (lband=uband; band wraps at the corners).
    cyclicmatrix(int, int);
    virtual ~cyclicmatrix() {}

    inline int rows()       const { return N; }
    inline int columns()    const { return N; }
    inline int ubandwidth() const { return uband; }
    inline int lbandwidth() const { return lband; }
    inline int bandwidth()  const { return band; }
    inline int size()       const { return sized; }
    void print() const;
    /// @brief Factorise via Sherman-Morrison-Woodbury (banded LU core + low-rank corner correction).
    void LU();
    /// @brief Solve A x = b. @param b right-hand side. @param x [out] solution. Auto-factorises on first use.
    void solve(matrix<T> &, matrix<T> &);
    /// @brief Solve A x = b in place. @param b [in,out] overwritten with the solution.
    void solve(matrix<T> &);

    inline       T &operator()(int, int);
    inline const T  operator()(int, int) const;
    cyclicmatrix<T> &operator=(T);

    friend cyclicmatrix<T> operator+ <>(const cyclicmatrix<T> &, const cyclicmatrix<T> &);
    friend cyclicmatrix<T> operator- <>(const cyclicmatrix<T> &, const cyclicmatrix<T> &);
    friend cyclicmatrix<T> operator* <>(const T &, const cyclicmatrix<T> &);
    friend cyclicmatrix<T> operator* <>(const cyclicmatrix<T> &, const T &);
    friend matrix<T> operator* <>(const cyclicmatrix<T> &, const matrix<T> &);

private:
    int N;
    int uband, lband;
    int band;
    int sized;
    bandmatrix<T> bandm;
    matrix<T> ublock;
    matrix<T> lblock;
    matrix<T> C;
    matrix<T> EPS;
    bool factor;
};

// ---------------------------------------------------------------------------

template <typename T>
cyclicmatrix<T>::cyclicmatrix() : N(1), uband(0), lband(0), factor(false)
{
    band = uband + lband + 1;
    sized = band * N;
}

template <typename T>
cyclicmatrix<T>::cyclicmatrix(const cyclicmatrix<T> &A)
    : N(A.N), uband(A.uband), lband(A.lband), band(A.band), sized(A.sized),
      bandm(A.bandm), ublock(A.ublock), lblock(A.lblock), C(A.C), EPS(A.EPS),
      factor(A.factor) {}

template <typename T>
cyclicmatrix<T>::cyclicmatrix(int rows, int upperband) : N(rows), uband(upperband)
{
    lband = uband;
    band = uband + lband + 1;
    sized = band * N;                 // FIX: was uninitialised
    factor = false;

    // banded core (general)
    bandm.N = N; bandm.M = N;
    bandm.uband = uband; bandm.lband = uband; bandm.band = band;
    bandm.sized = band * N;
    bandm.pt = new T[bandm.sized];
    for (int i = 0; i < bandm.sized; i++) bandm.pt[i] = T();
    bandm.index = new int[N];
    for (int i = 0; i < N; i++) bandm.index[i] = i;
    bandm.factor = false;

    ublock.N = uband; ublock.M = uband;
    ublock.pt = new T[uband * uband];
    for (int i = 0; i < uband * uband; i++) ublock.pt[i] = T();
    ublock.index = new int[uband];
    for (int i = 0; i < uband; i++) ublock.index[i] = i;
    ublock.symmetric = false; ublock.factor = false;

    lblock.N = uband; lblock.M = uband;
    lblock.pt = new T[uband * uband];
    for (int i = 0; i < uband * uband; i++) lblock.pt[i] = T();
    lblock.index = new int[uband];
    for (int i = 0; i < uband; i++) lblock.index[i] = i;
    lblock.symmetric = false; lblock.factor = false;

    C.N = 2 * uband; C.M = 2 * uband;
    C.pt = new T[4 * uband * uband];
    for (int i = 0; i < 4 * uband * uband; i++) C.pt[i] = T();
    C.index = new int[2 * uband];
    for (int i = 0; i < 2 * uband; i++) C.index[i] = i;
    C.symmetric = false; C.factor = false;

    EPS.N = 2 * uband; EPS.M = N;
    EPS.pt = new T[2 * uband * N];
    for (int i = 0; i < 2 * uband * N; i++) EPS.pt[i] = T();
    EPS.index = new int[2 * uband];
    for (int i = 0; i < 2 * uband; i++) EPS.index[i] = i;
    EPS.symmetric = false; EPS.factor = false;
}

template <typename T>
void cyclicmatrix<T>::print() const
{
    std::cout << "Size = " << N << ", " << N << "  (cyclic, uband=" << uband
              << ", lband=" << lband << ")" << std::endl;
    int mx = min(8, N);
    for (int i = 1; i <= mx; i++)
    {
        for (int j = 1; j <= mx; j++) std::cout << (*this)(i, j) << " ";
        if (N > mx) std::cout << "...";
        std::cout << std::endl;
    }
    if (N > mx) std::cout << "..." << std::endl;
}

template <typename T>
void cyclicmatrix<T>::LU()
{
    int i, j, k;
    matrix<T> E(N);
    T sum;
    T z1 = (T) 1.;

    C = T();
    bandm.LU();

    for (j = 1; j <= 2 * uband; j++)
    {
        E = T();
        if ((j >= 1) && (j <= uband))             E(j) = z1;
        if ((j >= uband + 1) && (j <= 2 * uband)) E(j + N - 2 * uband) = z1;

        bandm.solve(E);

        for (i = 1; i <= N; i++) EPS(j, i) = E(i);

        for (i = 1; i <= uband; i++)
        {
            sum = T();
            for (k = 1; k <= i; k++) sum += lblock(i, k) * E(k);
            if ((uband + i) == j) sum += z1;
            C(uband + i, j) = sum;
        }
        for (i = 1; i <= uband; i++)
        {
            sum = T();
            for (k = i; k <= uband; k++) sum += ublock(i, k) * E(N - uband + k);
            if (i == j) sum += z1;
            C(i, j) = sum;
        }
    }

    C.LU();
    factor = true;
}

template <typename T>
void cyclicmatrix<T>::solve(matrix<T> &b, matrix<T> &x)
{
    x = b;
    solve(x);
}

template <typename T>
void cyclicmatrix<T>::solve(matrix<T> &b)
{
    int i, j, k;
    T sum;
    matrix<T> G(2 * uband);
    G = T();

    if (!factor) LU();   // build the banded factor + capacitance matrix first

    bandm.solve(b);

    for (i = 1; i <= uband; i++)
    {
        sum = T();
        for (k = 1; k <= i; k++) sum += lblock(i, k) * b(k);
        G(uband + i) = sum;
    }
    for (i = 1; i <= uband; i++)
    {
        sum = T();
        for (k = i; k <= uband; k++) sum += ublock(i, k) * b(N - uband + k);
        G(i) = sum;
    }

    C.solve(G);

    for (j = 1; j <= 2 * uband; j++)
    {
        sum = G(j);
        for (i = 1; i <= N; i++) b(i) -= sum * EPS(j, i);
    }
}

template <typename T>
inline T &cyclicmatrix<T>::operator()(int i, int j)
{
    assert(i >= 1 && i <= N && j >= 1 && j <= N);
    static T zero;
    zero = T();
    if ((i >= max(1, j - uband)) && (i <= min(N, j + lband)))
        return bandm(i, j);
    else if ((i <= uband) && (j >= N - uband + 1))
        return ublock(i, j + uband - N);
    else if ((i >= N - uband + 1) && (j <= uband))
        return lblock(i + uband - N, j);
    return zero;
}

template <typename T>
inline const T cyclicmatrix<T>::operator()(int i, int j) const
{
    assert(i >= 1 && i <= N && j >= 1 && j <= N);
    if ((i >= max(1, j - uband)) && (i <= min(N, j + lband)))
        return bandm(i, j);
    else if ((i <= uband) && (j >= N - uband + 1))
        return ublock(i, j + uband - N);
    else if ((i >= N - uband + 1) && (j <= uband))
        return lblock(i + uband - N, j);
    return T();
}

template <typename T>
cyclicmatrix<T> &cyclicmatrix<T>::operator=(T a)
{
    factor = false;
    for (int i = 0; i < sized; i++)         bandm.pt[i] = a;
    for (int i = 0; i < uband * uband; i++) ublock.pt[i] = a;
    for (int i = 0; i < uband * uband; i++) lblock.pt[i] = a;
    return *this;
}

// ---------------------------------------------------------------------------
//  Algebra:  A + B,  A - B,  a * A,  A * a,  A * x
//
//  The stored pattern is the cyclic band: the banded core, plus the upper-right
//  corner block (upper triangular in block coordinates) and the lower-left
//  corner block (lower triangular).  These routines touch exactly the entries
//  LU() and solve() use, so A * x and A.solve(...) always describe the same
//  operator.  Like the SMW solve, they assume a non-degenerate band, N > 2*uband.
//
//  A * x must be formed BEFORE the first factorisation: LU() overwrites the
//  banded core in place.  Copy the matrix if both are needed.
// ---------------------------------------------------------------------------

template <typename T>
cyclicmatrix<T> operator+(const cyclicmatrix<T> &A, const cyclicmatrix<T> &B)
{
    assert(A.N == B.N);
    const int n = A.N, ub = max(A.uband, B.uband);
    cyclicmatrix<T> R(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = max(1, i - ub); j <= min(n, i + ub); j++)
            R(i, j) = A(i, j) + B(i, j);
    for (int i = 1; i <= ub; i++)
    {
        for (int k = i; k <= ub; k++) R(i, n - ub + k) = A(i, n - ub + k) + B(i, n - ub + k);
        for (int k = 1; k <= i; k++)  R(n - ub + i, k) = A(n - ub + i, k) + B(n - ub + i, k);
    }
    return R;
}

template <typename T>
cyclicmatrix<T> operator-(const cyclicmatrix<T> &A, const cyclicmatrix<T> &B)
{
    assert(A.N == B.N);
    const int n = A.N, ub = max(A.uband, B.uband);
    cyclicmatrix<T> R(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = max(1, i - ub); j <= min(n, i + ub); j++)
            R(i, j) = A(i, j) - B(i, j);
    for (int i = 1; i <= ub; i++)
    {
        for (int k = i; k <= ub; k++) R(i, n - ub + k) = A(i, n - ub + k) - B(i, n - ub + k);
        for (int k = 1; k <= i; k++)  R(n - ub + i, k) = A(n - ub + i, k) - B(n - ub + i, k);
    }
    return R;
}

template <typename T>
cyclicmatrix<T> operator*(const T &a, const cyclicmatrix<T> &A)
{
    const int n = A.N, ub = A.uband;
    cyclicmatrix<T> R(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = max(1, i - ub); j <= min(n, i + ub); j++)
            R(i, j) = a * A(i, j);
    for (int i = 1; i <= ub; i++)
    {
        for (int k = i; k <= ub; k++) R(i, n - ub + k) = a * A(i, n - ub + k);
        for (int k = 1; k <= i; k++)  R(n - ub + i, k) = a * A(n - ub + i, k);
    }
    return R;
}

template <typename T>
cyclicmatrix<T> operator*(const cyclicmatrix<T> &A, const T &a) { return a * A; }

/// @brief Periodic banded matrix-vector product d = A b (banded core + corner blocks).
template <typename T>
matrix<T> operator*(const cyclicmatrix<T> &A, const matrix<T> &b)
{
    assert(A.N == b.rows());
    assert(b.columns() == 1);
    assert(!A.factor && "cyclicmatrix: multiply before LU()/solve(), the band is overwritten in place");

    const int n = A.N, ub = A.uband;
    matrix<T> d = A.bandm * b;               // banded core

    for (int i = 1; i <= ub; i++)            // upper-right corner (rows 1..ub)
    {
        T s = T();
        for (int k = i; k <= ub; k++) s += A.ublock(i, k) * b(n - ub + k);
        d(i) += s;
    }
    for (int i = 1; i <= ub; i++)            // lower-left corner (rows n-ub+1..n)
    {
        T s = T();
        for (int k = 1; k <= i; k++) s += A.lblock(i, k) * b(k);
        d(n - ub + i) += s;
    }
    return d;
}

} // namespace femd

#endif // FEMD_LINALG_CYCLICMATRIX_HPP
