//
//  bandmatrix.hpp  --  general (nonsymmetric) banded matrix, pivoted LU
//
//  Lifted and cleaned from old_poseidon/matrix_h/{bndmatrix.h,bndmatrix_code.h}
//  (D. Mitsotakis, 2012).  Changes for C++17 / clang:
//    * gathered includes; wrapped in namespace femd
//    * removed a stray debug cout inside operator()
//    * out-of-band access returns a shared static sentinel (was a dangling
//      reference to a local)
//
//  Storage: by diagonals, uband super- and lband sub-diagonals.
//  1-based indexing A(i,j) like MATLAB.  Intended for square systems.
//
#ifndef FEMD_LINALG_BANDMATRIX_HPP
#define FEMD_LINALG_BANDMATRIX_HPP

#include "matrix.hpp"

namespace femd {

template <typename T> class bandmatrix;
template <typename T> bandmatrix<T> operator+(const bandmatrix<T> &, const bandmatrix<T> &);
template <typename T> bandmatrix<T> operator-(const bandmatrix<T> &, const bandmatrix<T> &);
template <typename T> bandmatrix<T> operator*(const T &, const bandmatrix<T> &);
template <typename T> bandmatrix<T> operator*(const bandmatrix<T> &, const T &);
template <typename T> matrix<T> operator*(const bandmatrix<T> &, const matrix<T> &);

template <typename T> class bandmatrix {
public:
    /// @brief Default 1x1 matrix.
    bandmatrix();
    /// @brief Copy constructor.
    bandmatrix(const bandmatrix &);
    /// @brief Allocate a zero banded matrix. @param rows N. @param cols M. @param upperband super-diagonals. @param lowerband sub-diagonals.
    bandmatrix(int, int, int, int);

    virtual ~bandmatrix()
    {
        delete[] pt;
        delete[] index;
    }

    inline int rows()       const { return N; }
    inline int columns()    const { return M; }
    inline int ubandwidth() const { return uband; }
    inline int lbandwidth() const { return lband; }
    inline int bandwidth()  const { return band; }
    inline int size()       const { return sized; }
    void print() const;
    /// @brief In-place banded LU WITHOUT pivoting (exact for diagonally dominant / SPD; use dense otherwise).
    void LU();
    /// @brief LU(), then whether the factors can be trusted: false at a zero or non-finite pivot, or when
    ///        an entry of the factors exceeds `growth` times the largest entry of the matrix.
    bool LU_checked(double growth);
    /// @brief Solve A x = b. @param b right-hand side. @param x [out] solution. Factorises on first use.
    void solve(matrix<T> &, matrix<T> &);
    /// @brief Solve A x = b in place. @param b [in,out] overwritten with the solution.
    void solve(matrix<T> &);

    inline       T &operator()(int, int);
    inline const T  operator()(int, int) const;
    bandmatrix<T> &operator=(T);

    friend bandmatrix<T> operator+ <>(const bandmatrix<T> &, const bandmatrix<T> &);
    friend bandmatrix<T> operator- <>(const bandmatrix<T> &, const bandmatrix<T> &);
    friend bandmatrix<T> operator* <>(const T &, const bandmatrix<T> &);
    friend bandmatrix<T> operator* <>(const bandmatrix<T> &, const T &);
    friend matrix<T> operator* <>(const bandmatrix<T> &, const matrix<T> &);

    template <typename> friend class cyclicmatrix;

private:
    int N, M;
    int uband, lband;
    int band;
    int sized;
    T *pt;
    int *index;
    bool factor;
};

// ---------------------------------------------------------------------------

template <typename T>
bandmatrix<T>::bandmatrix()
{
    N = 1; M = 1; uband = 0; lband = 0; band = 1;
    sized = band * min(M, N);
    pt = new T[sized];
    for (int i = 0; i < sized; i++) pt[i] = T();
    index = new int[N];
    for (int i = 0; i < N; i++) index[i] = i;
    factor = false;
}

template <typename T>
bandmatrix<T>::bandmatrix(const bandmatrix<T> &A)
{
    N = A.N; M = A.M; uband = A.uband; lband = A.lband; band = A.band; sized = A.sized;
    pt = new T[sized];
    index = new int[N];
    memcpy(pt, A.pt, sized * sizeof(T));
    memcpy(index, A.index, N * sizeof(int));
    factor = A.factor;
}

template <typename T>
bandmatrix<T>::bandmatrix(int rows, int columns, int upperband, int lowerband)
{
    N = rows; M = columns; uband = upperband; lband = lowerband;
    band = uband + lband + 1;
    sized = band * min(M, N);
    pt = new T[sized];
    for (int i = 0; i < sized; i++) pt[i] = T();
    index = new int[rows];
    for (int i = 0; i < rows; i++) index[i] = i;
    factor = false;
}

template <typename T>
void bandmatrix<T>::print() const
{
    std::cout << "Size = " << N << ", " << M
              << "  (uband=" << uband << ", lband=" << lband << ")" << std::endl;
    int mr = min(8, N), mc = min(8, M);
    for (int i = 1; i <= mr; i++)
    {
        for (int j = 1; j <= mc; j++) std::cout << (*this)(i, j) << " ";
        if (M > mc) std::cout << "...";
        std::cout << std::endl;
    }
    if (N > mr) std::cout << "..." << std::endl;
}

template <typename T>
inline T &bandmatrix<T>::operator()(int i, int j)
{
    assert(i >= 1 && i <= N && j >= 1 && j <= M);
    int mm = min(N, M);
    static T zero;
    zero = T();
    if ((i >= max(1, j - uband)) && (i <= min(mm, j + lband)))
        return pt[(uband + i - j) * mm + j - 1];
    return zero;
}

template <typename T>
inline const T bandmatrix<T>::operator()(int i, int j) const
{
    assert(i >= 1 && i <= N && j >= 1 && j <= M);
    int mm = min(N, M);
    if ((i >= max(1, j - uband)) && (i <= min(mm, j + lband)))
        return pt[(uband + i - j) * mm + j - 1];
    return T();
}

template <typename T>
void bandmatrix<T>::LU()
{
    // Banded LU WITHOUT row interchanges.  The diagonal-banded storage has no
    // room for the extra ku fill-in that partial pivoting would create, so the
    // original pivoting version corrupted any matrix that needed a swap (even
    // SPD ones, where it pivoted unnecessarily).  Pivot-free banded LU is exact
    // for diagonally dominant / positive-definite systems -- which is what the
    // FE assembly produces for diffusion-type problems.  For indefinite or
    // advection-dominated operators that genuinely need pivoting, use the dense
    // solver instead.
    T tmp, m;
    int j, k, l;

    for (k = 1; k < N; k++)
    {
        tmp = pt[uband * M + k - 1];
        if (tmp != (T) 0.0)
            for (j = k + 1; j <= min(M, k + lband); j++)
            {
                m = pt[(uband + j - k) * M + k - 1] / tmp;
                pt[(uband + j - k) * M + k - 1] = m;
                for (l = k + 1; l <= min(M, k + uband); l++)
                    pt[(uband + j - l) * M + l - 1] -= m * pt[(uband + k - l) * M + l - 1];
            }
    }
    factor = true;
}

template <typename T>
bool bandmatrix<T>::LU_checked(double growth)
{
    double amax = 0.0;
    for (int k = 0; k < sized; k++) amax = std::max(amax, static_cast<double>(std::abs(pt[k])));
    LU();
    if (!(amax > 0.0)) return false;
    double fmax = 0.0;
    for (int k = 0; k < sized; k++)
    {
        const double v = static_cast<double>(std::abs(pt[k]));
        if (!std::isfinite(v)) return false;
        fmax = std::max(fmax, v);
    }
    for (int k = 1; k <= N; k++)
        if (pt[uband * M + k - 1] == (T) 0.0) return false;
    return fmax <= growth * amax;
}

template <typename T>
void bandmatrix<T>::solve(matrix<T> &b, matrix<T> &x)
{
    int j, k;
    T s;
    if (!factor) LU();

    for (k = 1; k <= N; k++)
        for (j = k + 1; j <= min(k + lband, N); j++)
            b(index[j - 1] + 1) -= pt[(uband + j - k) * M + k - 1] * b(index[k - 1] + 1);

    for (k = N; k > 0; k--)
    {
        s = b(index[k - 1] + 1);
        for (j = k + 1; j <= min(k + lband, N); j++)
            s -= pt[(uband + k - j) * M + j - 1] * x(j);
        x(k) = s / pt[uband * M + k - 1];
    }
}

template <typename T>
void bandmatrix<T>::solve(matrix<T> &b)
{
    int j, k;
    T s;
    if (!factor) LU();

    for (k = 1; k <= N; k++)
        for (j = k + 1; j <= min(k + lband, N); j++)
            b(index[j - 1] + 1) -= pt[(uband + j - k) * M + k - 1] * b(index[k - 1] + 1);

    for (k = N; k > 0; k--)
    {
        s = b(index[k - 1] + 1);
        for (j = k + 1; j <= min(k + lband, N); j++)
            s -= pt[(uband + k - j) * M + j - 1] * b(j);
        b(k) = s / pt[uband * M + k - 1];
    }
}

// ---------------------------------------------------------------------------

template <typename T>
bandmatrix<T> operator+(const bandmatrix<T> &A, const bandmatrix<T> &B)
{
    assert(A.N == B.N && A.M == B.M);
    int m = A.M, ub = max(A.uband, B.uband), lb = max(A.lband, B.lband);
    int mm = min(A.N, A.M);
    bandmatrix<T> C(A.N, m, ub, lb);
    for (int i = 1; i <= mm; i++)
    {
        for (int j = 0; j <= ub; j++) if (i + j <= m) C(i, i + j) = A(i, i + j) + B(i, i + j);
        for (int j = 1; j <= lb; j++) if (i - j > 0)  C(i, i - j) = A(i, i - j) + B(i, i - j);
    }
    return C;
}

template <typename T>
bandmatrix<T> operator-(const bandmatrix<T> &A, const bandmatrix<T> &B)
{
    assert(A.N == B.N && A.M == B.M);
    int m = A.M, ub = max(A.uband, B.uband), lb = max(A.lband, B.lband);
    int mm = min(A.N, A.M);
    bandmatrix<T> C(A.N, m, ub, lb);
    for (int i = 1; i <= mm; i++)
    {
        for (int j = 0; j <= ub; j++) if (i + j <= m) C(i, i + j) = A(i, i + j) - B(i, i + j);
        for (int j = 1; j <= lb; j++) if (i - j > 0)  C(i, i - j) = A(i, i - j) - B(i, i - j);
    }
    return C;
}

template <typename T>
bandmatrix<T> operator*(const T &a, const bandmatrix<T> &A)
{
    int m = A.M, ub = A.uband, lb = A.lband, mm = min(A.N, A.M);
    bandmatrix<T> C(A.N, m, ub, lb);
    for (int i = 1; i <= mm; i++)
    {
        for (int j = 0; j <= ub; j++) if (i + j <= m) C(i, i + j) = a * A(i, i + j);
        for (int j = 1; j <= lb; j++) if (i - j > 0)  C(i, i - j) = a * A(i, i - j);
    }
    return C;
}

template <typename T>
bandmatrix<T> operator*(const bandmatrix<T> &A, const T &a) { return a * A; }

template <typename T>
matrix<T> operator*(const bandmatrix<T> &A, const matrix<T> &b)
{
    assert(A.M == b.rows());
    assert(b.columns() == 1);
    int n = A.N, m = A.M, ub = A.uband, lb = A.lband, mm = min(A.N, A.M);
    matrix<T> d(n, 1);
    int ubp1 = ub + 1, l = 1;
    for (int j = 1; j <= m; j++)
    {
        if (b(l) != (T) 0.0)
        {
            int k = ubp1 - j;
            T temp = b(l);
            for (int i = max(1, j - ub); i <= min(n, j + lb); i++)
                d(i) += temp * A.pt[(k + i - 1) * mm + j - 1];
        }
        l++;
    }
    return d;
}

template <typename T>
bandmatrix<T> &bandmatrix<T>::operator=(T a)
{
    factor = false;
    for (int i = 0; i < sized; i++) pt[i] = a;
    return *this;
}

} // namespace femd

#endif // FEMD_LINALG_BANDMATRIX_HPP
