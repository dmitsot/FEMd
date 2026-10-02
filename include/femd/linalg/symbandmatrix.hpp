//
//  symbandmatrix.hpp  --  symmetric banded matrix, Cholesky solve
//
//  Lifted and cleaned from old_poseidon/matrix_h/{sbndmatrix.h,sbndmatrix_code.h}
//  (D. Mitsotakis, 2012).  Changes for C++17 / clang:
//    * gathered includes; wrapped in namespace femd
//    * `definite` made mutable so indefinite() can stay const
//    * simplified non-interactive print()
//
//  Storage: by diagonals.  uband = number of super-diagonals (= bandwidth).
//  Cholesky factorisation; requires symmetric positive definite (mass,
//  stiffness, Helmholtz-positive operators all qualify).  LU() throws
//  std::domain_error at a pivot that is not positive (the matrix is then
//  partly overwritten), so an indefinite matrix is refused, not solved to NaN.
//  1-based indexing A(i,j) like MATLAB.
//
#ifndef FEMD_LINALG_SYMBANDMATRIX_HPP
#define FEMD_LINALG_SYMBANDMATRIX_HPP

#include "matrix.hpp"
#include <stdexcept>
#include <string>

namespace femd {

template <typename T> class symbandmatrix;
template <typename T> symbandmatrix<T> operator+(const symbandmatrix<T> &, const symbandmatrix<T> &);
template <typename T> symbandmatrix<T> operator-(const symbandmatrix<T> &, const symbandmatrix<T> &);
template <typename T> symbandmatrix<T> operator*(const T &, const symbandmatrix<T> &);
template <typename T> symbandmatrix<T> operator*(const symbandmatrix<T> &, const T &);
template <typename T> matrix<T> operator*(const symbandmatrix<T> &, const matrix<T> &);

template <typename T> class symbandmatrix {
public:
    /// @brief Default 1x1 matrix.
    symbandmatrix();
    /// @brief Copy constructor.
    symbandmatrix(const symbandmatrix &);
    /// @brief Allocate an n x n zero symmetric banded matrix. @param rows N. @param upperband number of super-diagonals (= half-bandwidth).
    symbandmatrix(int, int);

    virtual ~symbandmatrix()
    {
        delete[] pt;
        delete[] index;
    }

    inline int rows()       const { return N; }
    inline int columns()    const { return N; }
    inline int ubandwidth() const { return uband; }
    inline int lbandwidth() const { return uband; }
    inline int bandwidth()  const { return band; }
    inline int size()       const { return sized; }
    inline void indefinite() const { definite = false; }
    void print() const;
    /// @brief In-place banded Cholesky factorisation (requires SPD).
    void LU();
    /// @brief Solve A x = b. @param b right-hand side. @param x [out] solution. Factorises on first use.
    void solve(matrix<T> &, matrix<T> &);
    /// @brief Solve A x = b in place. @param b [in,out] overwritten with the solution.
    void solve(matrix<T> &);

    inline       T &operator()(int, int);
    inline const T  operator()(int, int) const;
    symbandmatrix<T> &operator=(T);

    friend symbandmatrix<T> operator+ <>(const symbandmatrix<T> &, const symbandmatrix<T> &);
    friend symbandmatrix<T> operator- <>(const symbandmatrix<T> &, const symbandmatrix<T> &);
    friend symbandmatrix<T> operator* <>(const T &, const symbandmatrix<T> &);
    friend symbandmatrix<T> operator* <>(const symbandmatrix<T> &, const T &);
    friend matrix<T> operator* <>(const symbandmatrix<T> &, const matrix<T> &);

    template <typename> friend class symcyclicmatrix;

private:
    int N;
    int uband;
    int band;
    int sized;
    T *pt;
    int *index;
    bool factor;
    mutable bool definite;   // true => positive definite
};

// ---------------------------------------------------------------------------

template <typename T>
symbandmatrix<T>::symbandmatrix()
{
    N = 1; uband = 1; band = 2 * uband + 1; sized = (uband + 1) * N;
    pt = new T[sized];
    for (int i = 0; i < sized; i++) pt[i] = T();
    index = new int[N];
    for (int i = 0; i < N; i++) index[i] = i;
    factor = false; definite = true;
}

template <typename T>
symbandmatrix<T>::symbandmatrix(const symbandmatrix<T> &A)
{
    N = A.N; uband = A.uband; band = A.band; sized = A.sized;
    pt = new T[sized];
    index = new int[N];
    memcpy(pt, A.pt, sized * sizeof(T));
    memcpy(index, A.index, N * sizeof(int));
    factor = A.factor; definite = A.definite;
}

template <typename T>
symbandmatrix<T>::symbandmatrix(int rows, int upperband)
{
    N = rows; uband = upperband; band = 2 * uband + 1; sized = (uband + 1) * N;
    pt = new T[sized];
    for (int i = 0; i < sized; i++) pt[i] = T();
    index = new int[rows];
    for (int i = 0; i < rows; i++) index[i] = i;
    factor = false; definite = true;
}

template <typename T>
void symbandmatrix<T>::print() const
{
    std::cout << "Size = " << N << ", " << N << "  (uband=" << uband << ")" << std::endl;
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
inline T &symbandmatrix<T>::operator()(int i, int j)
{
    assert(i >= 1 && i <= N && j >= 1 && j <= N);
    if (i < j) { int t = i; i = j; j = t; }
    static T zero;
    zero = T();
    if ((i <= min(N, j + uband)) && (i >= j))
        return pt[(i - j) * N + j - 1];
    return zero;
}

template <typename T>
inline const T symbandmatrix<T>::operator()(int i, int j) const
{
    assert(i >= 1 && i <= N && j >= 1 && j <= N);
    if (i < j) { int t = i; i = j; j = t; }
    if ((i <= min(N, j + uband)) && (i >= j))
        return pt[(i - j) * N + j - 1];
    return T();
}

template <typename T>
void symbandmatrix<T>::LU()
{
    int i, j, k, lambda;
    T temp;
    for (j = 1; j <= N; j++)
    {
        for (k = max(1, j - uband); k <= j - 1; k++)
        {
            lambda = min(k + uband, N);
            temp = pt[(j - k) * N + k - 1];
            for (i = j; i <= lambda; i++)
                pt[(i - j) * N + j - 1] -= temp * pt[(i - k) * N + k - 1];
        }
        lambda = min(j + uband, N);
        if (definite && !(pt[j - 1] > T()))            // a zero, negative or NaN pivot: not positive definite
        {
            factor = false;
            throw std::domain_error("symbandmatrix::LU: the matrix is not positive definite (pivot " + std::to_string(j) +
                                    " of " + std::to_string(N) + "); Auto falls back to a pivoted LU, or choose Band (Cyclic when periodic)");
        }
        pt[j - 1] = sqrt(pt[j - 1]);
        temp = pt[j - 1];
        for (i = j + 1; i <= lambda; i++)
            pt[(i - j) * N + j - 1] /= temp;
    }
    factor = true;
}

template <typename T>
void symbandmatrix<T>::solve(matrix<T> &b, matrix<T> &x)
{
    int i, j;
    T tmp;
    x = b;
    if (!factor) LU();
    for (j = 1; j <= N; j++)
    {
        x(j) /= pt[j - 1];
        tmp = x(j);
        for (i = j + 1; i <= min(j + uband, N); i++)
            x(i) -= pt[(i - j) * N + j - 1] * tmp;
    }
    for (j = N; j >= 1; j--)
    {
        x(j) = x(j) / pt[j - 1];
        for (i = max(1, j - uband); i <= j - 1; i++)
            x(i) -= pt[(j - i) * N + i - 1] * x(j);
    }
}

template <typename T>
void symbandmatrix<T>::solve(matrix<T> &b)
{
    int i, j;
    T tmp;
    if (!factor) LU();
    for (j = 1; j <= N; j++)
    {
        b(j) /= pt[j - 1];
        tmp = b(j);
        for (i = j + 1; i <= min(j + uband, N); i++)
            b(i) -= pt[(i - j) * N + j - 1] * tmp;
    }
    for (j = N; j >= 1; j--)
    {
        b(j) = b(j) / pt[j - 1];
        for (i = max(1, j - uband); i <= j - 1; i++)
            b(i) -= pt[(j - i) * N + i - 1] * b(j);
    }
}

template <typename T>
symbandmatrix<T> &symbandmatrix<T>::operator=(T a)
{
    factor = false;
    for (int i = 0; i < sized; i++) pt[i] = a;
    for (int i = 0; i < N; i++) index[i] = i;
    return *this;
}

// ---------------------------------------------------------------------------

template <typename T>
symbandmatrix<T> operator+(const symbandmatrix<T> &A, const symbandmatrix<T> &B)
{
    assert(A.N == B.N);
    int n = A.N, ub = max(A.uband, B.uband);
    symbandmatrix<T> C(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = 0; j <= ub; j++)
            if (i + j <= n) C(i, i + j) = A(i, i + j) + B(i, i + j);
    return C;
}

template <typename T>
symbandmatrix<T> operator-(const symbandmatrix<T> &A, const symbandmatrix<T> &B)
{
    assert(A.N == B.N);
    int n = A.N, ub = max(A.uband, B.uband);
    symbandmatrix<T> C(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = 0; j <= ub; j++)
            if (i + j <= n) C(i, i + j) = A(i, i + j) - B(i, i + j);
    return C;
}

template <typename T>
symbandmatrix<T> operator*(const T &a, const symbandmatrix<T> &A)
{
    int n = A.N, ub = A.uband;
    symbandmatrix<T> C(n, ub);
    for (int i = 1; i <= n; i++)
        for (int j = 0; j <= ub; j++)
            if (i + j <= n) C(i, i + j) = a * A(i, i + j);
    return C;
}

template <typename T>
symbandmatrix<T> operator*(const symbandmatrix<T> &A, const T &a)
{
    return a * A;
}

template <typename T>
matrix<T> operator*(const symbandmatrix<T> &A, const matrix<T> &b)
{
    assert(A.N == b.rows());
    assert(b.columns() == 1);
    int n = A.N, ub = A.uband, mm = A.N;
    matrix<T> d(n, 1);
    T temp1, temp2;
    for (int j = 1; j <= n; j++)
    {
        temp1 = b(j);
        temp2 = T();
        d(j) += temp1 * A.pt[j - 1];
        int k = 1 - j;
        for (int i = j + 1; i <= min(n, j + ub); i++)
        {
            d(i) += temp1 * A.pt[(k + i - 1) * mm + j - 1];
            temp2 += A.pt[(k + i - 1) * mm + j - 1] * b(i);
        }
        d(j) += temp2;
    }
    return d;
}

} // namespace femd

#endif // FEMD_LINALG_SYMBANDMATRIX_HPP
