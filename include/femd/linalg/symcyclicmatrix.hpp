//
//  symcyclicmatrix.hpp  --  symmetric periodic (cyclic) banded matrix
//
//  Lifted and cleaned from old_poseidon/matrix_h/{scyclic.h,scyclic_code.h}
//  (D. Mitsotakis, 2012).  BUG FIXES applied during lift:
//    * the `sized` member was never initialised in the (int,int) ctor, so
//      bandm.pt = new T[sized] allocated a garbage-sized block.  Now set
//      sized = (uband+1)*N.
//    * an EPS index init loop wrote C.index instead of EPS.index.
//  Plus the usual C++17 include/namespace cleanup and non-interactive print.
//
//  A periodic banded SPD system is solved as a banded core plus a low-rank
//  corner correction (Sherman-Morrison-Woodbury): factor the band, solve the
//  2*uband correction columns, assemble and factor the small capacitance
//  matrix C, then correct.  1-based indexing.
//
#ifndef FEMD_LINALG_SYMCYCLICMATRIX_HPP
#define FEMD_LINALG_SYMCYCLICMATRIX_HPP

#include "matrix.hpp"
#include "symbandmatrix.hpp"

namespace femd {

template <typename T> class symcyclicmatrix;
template <typename T> symcyclicmatrix<T> operator+(const symcyclicmatrix<T> &, const symcyclicmatrix<T> &);
template <typename T> symcyclicmatrix<T> operator-(const symcyclicmatrix<T> &, const symcyclicmatrix<T> &);
template <typename T> symcyclicmatrix<T> operator*(const T &, const symcyclicmatrix<T> &);
template <typename T> symcyclicmatrix<T> operator*(const symcyclicmatrix<T> &, const T &);
template <typename T> matrix<T> operator*(const symcyclicmatrix<T> &, const matrix<T> &);

template <typename T> class symcyclicmatrix {
public:
    /// @brief Default 1x1 matrix.
    symcyclicmatrix();
    /// @brief Copy constructor.
    symcyclicmatrix(const symcyclicmatrix &);
    /// @brief Allocate a zero symmetric periodic banded matrix. @param rows N. @param upperband half-bandwidth (band wraps at the corners).
    symcyclicmatrix(int, int);
    ~symcyclicmatrix() {}

    inline int rows()       const { return N; }
    inline int columns()    const { return N; }
    inline int ubandwidth() const { return uband; }
    inline int lbandwidth() const { return uband; }
    inline int bandwidth()  const { return band; }
    inline int size()       const { return sized; }
    void print() const;
    /// @brief Factorise via Sherman-Morrison-Woodbury (Cholesky banded core + low-rank corner correction).
    void LU();
    /// @brief Solve A x = b. @param b right-hand side. @param x [out] solution. Auto-factorises on first use.
    void solve(matrix<T> &, matrix<T> &);
    /// @brief Solve A x = b in place. @param b [in,out] overwritten with the solution.
    void solve(matrix<T> &);

    inline       T &operator()(int, int);
    inline const T  operator()(int, int) const;
    symcyclicmatrix<T> &operator=(T);

    friend symcyclicmatrix<T> operator+ <>(const symcyclicmatrix<T> &, const symcyclicmatrix<T> &);
    friend symcyclicmatrix<T> operator- <>(const symcyclicmatrix<T> &, const symcyclicmatrix<T> &);
    friend symcyclicmatrix<T> operator* <>(const T &, const symcyclicmatrix<T> &);
    friend symcyclicmatrix<T> operator* <>(const symcyclicmatrix<T> &, const T &);
    friend matrix<T> operator* <>(const symcyclicmatrix<T> &, const matrix<T> &);

private:
    int N;
    int uband;
    int band;
    int sized;
    symbandmatrix<T> bandm;
    matrix<T> ublock;
    matrix<T> lblock;
    matrix<T> C;
    matrix<T> EPS;
    bool factor;
};

// ---------------------------------------------------------------------------

template <typename T>
symcyclicmatrix<T>::symcyclicmatrix() : N(1), uband(0), factor(false)
{
    band = 2 * uband + 1;
    sized = (uband + 1) * N;
}

template <typename T>
symcyclicmatrix<T>::symcyclicmatrix(const symcyclicmatrix<T> &A)
    : N(A.N), uband(A.uband), band(A.band), sized(A.sized),
      bandm(A.bandm), ublock(A.ublock), lblock(A.lblock), C(A.C), EPS(A.EPS),
      factor(A.factor) {}

template <typename T>
symcyclicmatrix<T>::symcyclicmatrix(int rows, int upperband)
    : N(rows), uband(upperband)
{
    band = 2 * uband + 1;
    sized = (uband + 1) * N;          // FIX: was uninitialised
    factor = false;

    // banded core (symmetric)
    bandm.N = N;
    bandm.uband = uband;
    bandm.band = band;
    bandm.sized = (uband + 1) * N;
    bandm.pt = new T[bandm.sized];
    for (int i = 0; i < bandm.sized; i++) bandm.pt[i] = T();
    bandm.index = new int[N];
    for (int i = 0; i < N; i++) bandm.index[i] = i;
    bandm.factor = false;
    bandm.definite = true;

    // upper-right corner block
    ublock.N = uband; ublock.M = uband;
    ublock.pt = new T[uband * uband];
    for (int i = 0; i < uband * uband; i++) ublock.pt[i] = T();
    ublock.index = new int[uband];
    for (int i = 0; i < uband; i++) ublock.index[i] = i;
    ublock.symmetric = false; ublock.factor = false;

    // lower-left corner block
    lblock.N = uband; lblock.M = uband;
    lblock.pt = new T[uband * uband];
    for (int i = 0; i < uband * uband; i++) lblock.pt[i] = T();
    lblock.index = new int[uband];
    for (int i = 0; i < uband; i++) lblock.index[i] = i;
    lblock.symmetric = false; lblock.factor = false;

    // capacitance matrix C (2*uband square)
    C.N = 2 * uband; C.M = 2 * uband;
    C.pt = new T[4 * uband * uband];
    for (int i = 0; i < 4 * uband * uband; i++) C.pt[i] = T();
    C.index = new int[2 * uband];
    for (int i = 0; i < 2 * uband; i++) C.index[i] = i;
    C.symmetric = false; C.factor = false;

    // correction columns EPS (2*uband x N)
    EPS.N = 2 * uband; EPS.M = N;
    EPS.pt = new T[2 * uband * N];
    for (int i = 0; i < 2 * uband * N; i++) EPS.pt[i] = T();
    EPS.index = new int[2 * uband];
    for (int i = 0; i < 2 * uband; i++) EPS.index[i] = i;   // FIX: was C.index
    EPS.symmetric = false; EPS.factor = false;
}

template <typename T>
void symcyclicmatrix<T>::print() const
{
    std::cout << "Size = " << N << ", " << N << "  (cyclic, uband=" << uband << ")" << std::endl;
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
void symcyclicmatrix<T>::LU()
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
void symcyclicmatrix<T>::solve(matrix<T> &b, matrix<T> &x)
{
    x = b;
    solve(x);
}

template <typename T>
void symcyclicmatrix<T>::solve(matrix<T> &b)
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
inline T &symcyclicmatrix<T>::operator()(int i, int j)
{
    assert(i >= 1 && i <= N && j >= 1 && j <= N);
    static T zero;
    zero = T();
    if ((i >= max(1, j - uband)) && (i <= min(N, j + uband)))
        return bandm(i, j);
    else if ((i <= uband) && (j >= N - uband + 1))
        return ublock(i, j + uband - N);
    else if ((i >= N - uband + 1) && (j <= uband))
        return lblock(i + uband - N, j);
    return zero;
}

template <typename T>
inline const T symcyclicmatrix<T>::operator()(int i, int j) const
{
    assert(i >= 1 && i <= N && j >= 1 && j <= N);
    if ((i >= max(1, j - uband)) && (i <= min(N, j + uband)))
        return bandm(i, j);
    else if ((i <= uband) && (j >= N - uband + 1))
        return ublock(i, j + uband - N);
    else if ((i >= N - uband + 1) && (j <= uband))
        return lblock(i + uband - N, j);
    return T();
}

template <typename T>
symcyclicmatrix<T> &symcyclicmatrix<T>::operator=(T a)
{
    factor = false;
    for (int i = 0; i < sized; i++)            bandm.pt[i] = a;
    for (int i = 0; i < uband * uband; i++)    ublock.pt[i] = a;
    for (int i = 0; i < uband * uband; i++)    lblock.pt[i] = a;
    return *this;
}

// ---------------------------------------------------------------------------
//  Algebra:  A + B,  A - B,  a * A,  A * a,  A * x
//
//  The stored pattern is the symmetric cyclic band: the symmetric banded core
//  (upper triangle stored), plus the upper-right corner block (upper triangular
//  in block coordinates) and the lower-left corner block (lower triangular).
//  As elsewhere in this class the two corner blocks are stored independently,
//  so a symmetric matrix is built by setting BOTH A(i,j) and A(j,i) in the
//  corners, exactly as LU()/solve() expect.  These routines touch only the
//  entries the solve uses, so A * x and A.solve(...) describe the same
//  operator.  Like the SMW solve they assume N > 2*uband.
//
//  A * x must be formed BEFORE the first factorisation: LU() overwrites the
//  banded core in place.  Copy the matrix if both are needed.
// ---------------------------------------------------------------------------

template <typename T>
symcyclicmatrix<T> operator+(const symcyclicmatrix<T> &A, const symcyclicmatrix<T> &B)
{
    assert(A.N == B.N);
    const int n = A.N, ub = max(A.uband, B.uband);
    symcyclicmatrix<T> R(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= min(n, i + ub); j++)
            R(i, j) = A(i, j) + B(i, j);
    for (int i = 1; i <= ub; i++)
    {
        for (int k = i; k <= ub; k++) R(i, n - ub + k) = A(i, n - ub + k) + B(i, n - ub + k);
        for (int k = 1; k <= i; k++)  R(n - ub + i, k) = A(n - ub + i, k) + B(n - ub + i, k);
    }
    return R;
}

template <typename T>
symcyclicmatrix<T> operator-(const symcyclicmatrix<T> &A, const symcyclicmatrix<T> &B)
{
    assert(A.N == B.N);
    const int n = A.N, ub = max(A.uband, B.uband);
    symcyclicmatrix<T> R(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= min(n, i + ub); j++)
            R(i, j) = A(i, j) - B(i, j);
    for (int i = 1; i <= ub; i++)
    {
        for (int k = i; k <= ub; k++) R(i, n - ub + k) = A(i, n - ub + k) - B(i, n - ub + k);
        for (int k = 1; k <= i; k++)  R(n - ub + i, k) = A(n - ub + i, k) - B(n - ub + i, k);
    }
    return R;
}

template <typename T>
symcyclicmatrix<T> operator*(const T &a, const symcyclicmatrix<T> &A)
{
    const int n = A.N, ub = A.uband;
    symcyclicmatrix<T> R(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= min(n, i + ub); j++)
            R(i, j) = a * A(i, j);
    for (int i = 1; i <= ub; i++)
    {
        for (int k = i; k <= ub; k++) R(i, n - ub + k) = a * A(i, n - ub + k);
        for (int k = 1; k <= i; k++)  R(n - ub + i, k) = a * A(n - ub + i, k);
    }
    return R;
}

template <typename T>
symcyclicmatrix<T> operator*(const symcyclicmatrix<T> &A, const T &a) { return a * A; }

/// @brief Symmetric periodic banded matrix-vector product d = A b.
template <typename T>
matrix<T> operator*(const symcyclicmatrix<T> &A, const matrix<T> &b)
{
    assert(A.N == b.rows());
    assert(b.columns() == 1);
    assert(!A.factor && "symcyclicmatrix: multiply before LU()/solve(), the band is overwritten in place");

    const int n = A.N, ub = A.uband;
    matrix<T> d = A.bandm * b;               // symmetric banded core

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

#endif // FEMD_LINALG_SYMCYCLICMATRIX_HPP
