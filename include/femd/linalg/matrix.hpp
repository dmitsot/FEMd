//
//  matrix.hpp  --  dense general matrix / column vector
//
//  Lifted and cleaned from old_poseidon/matrix_h/{matrix.h,matrix_code.h}
//  (D. Mitsotakis, 2012-2015).  Changes for C++17 / clang:
//    * gathered the includes the old umbrella header used to provide
//    * removed the (now ill-formed) `register` keyword
//    * dropped the two VLA-based reindexing operator() overloads
//    * everything now lives in namespace femd
//
//  MATLAB-like conventions are preserved: 1-based indexing A(i,j), A(i).
//
#ifndef FEMD_LINALG_MATRIX_HPP
#define FEMD_LINALG_MATRIX_HPP

#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>   // memcpy
#include <cstdlib>   // rand, srand
#include <ctime>     // time
#include <cmath>     // sqrt, abs
#include <algorithm> // min, max
#include <cassert>
#include <type_traits> // is_floating_point_v (guard complex out of OMP reductions)
#include "femd/util/omp.hpp"

namespace femd {

using std::complex;
using std::min;
using std::max;
using std::abs;
using std::sqrt;
using std::conj;
using std::ostream;

template <typename T> class matrix;

template <typename T> T ddot(const matrix<T> &, const matrix<T> &);
template <typename T> matrix<T> operator*(const matrix<T> &, const matrix<T> &);
template <typename T> matrix<T> operator*(const T &, const matrix<T> &);
template <typename T> matrix<T> operator*(const matrix<T> &, const T &);
template <typename T> matrix<T> operator+(const matrix<T> &, const matrix<T> &);
template <typename T> matrix<T> operator+(const matrix<T> &, const T &);
template <typename T> matrix<T> operator-(const matrix<T> &, const matrix<T> &);
template <typename T> matrix<T> operator-(const matrix<T> &);
template <typename T> T maxentry(const matrix<T> &);

template <typename T> class matrix {
public:
    using value_type = T;   ///< scalar type (std-like; lets generic code deduce T)

    // constructors
    /// @brief Empty 0x0 matrix.
    matrix();
    /// @brief Copy constructor.
    matrix(const matrix &);
    /// @brief Allocate an n x m zero matrix. @param rows N. @param cols M.
    matrix(int, int);
    /// @brief Allocate an n x 1 zero column vector. @param rows N.
    matrix(int);

    virtual ~matrix()
    {
        delete[] pt;
        delete[] index;
    }

    // member functions
    inline int rows()    const { return N; }
    inline int columns() const { return M; }
    inline int size()    const { return N * M; }
    inline void setsymmetry(bool sym) { symmetric = sym; }
    inline bool symmetry() const { return symmetric; }
    void print() const;
    void save(const char *) const;
    void resize(int, int = 1);
    /// @brief In-place LU factorisation (Cholesky if flagged symmetric via setsymmetry).
    void LU();
    /// @brief Solve A x = b. @param b right-hand side (column vector). @param x [out] solution. Factorises on first use.
    void solve(matrix<T> &, matrix<T> &);
    /// @brief Solve A x = b in place. @param b [in,out] right-hand side, overwritten with the solution.
    void solve(matrix<T> &);
    /// @brief Set to the n x m identity. @param rows N. @param cols M.
    void eye(int, int);
    /// @brief Set to the n x m zero matrix. @param rows N. @param cols M.
    void zeros(int, int);
    /// @brief Fill with uniform random entries in [0,1). @param rows N. @param cols M.
    void random(int, int);

    // raw data access (contiguous, column-vector friendly for Python wrapping)
    inline       T *data()       { return pt; }
    inline const T *data() const { return pt; }

    // operators (1-based)
    inline       T &operator()(int, int);
    inline       T &operator()(int);
    inline const T  operator()(int, int) const;
    inline const T  operator()(int) const;

    matrix<T> &operator=(T);
    matrix<T> &operator=(const matrix<T> &);
    matrix<T> &operator+=(const matrix<T> &);
    matrix<T> &operator-=(const matrix<T> &);

    friend matrix<T> operator* <>(const matrix<T> &, const matrix<T> &);
    friend matrix<T> operator* <>(const T &, const matrix<T> &);
    friend matrix<T> operator* <>(const matrix<T> &, const T &);
    friend matrix<T> operator+ <>(const matrix<T> &, const matrix<T> &);
    friend matrix<T> operator+ <>(const matrix<T> &, const T &);
    friend matrix<T> operator- <>(const matrix<T> &, const matrix<T> &);
    friend matrix<T> operator- <>(const matrix<T> &);
    friend T ddot <>(const matrix<T> &, const matrix<T> &);
    friend T maxentry <>(const matrix<T> &);

    template <typename> friend class bandmatrix;
    template <typename> friend class symbandmatrix;
    template <typename> friend class cyclicmatrix;
    template <typename> friend class symcyclicmatrix;

private:
    int N, M;
    T *pt;
    int *index;
    bool factor;
    bool symmetric;
};

// ---------------------------------------------------------------------------
// constructors
// ---------------------------------------------------------------------------

template <typename T>
matrix<T>::matrix() : N(0), M(0)
{
    pt = new T[0];
    index = new int[0];
    factor = false;
    symmetric = false;
}

template <typename T>
matrix<T>::matrix(const matrix<T> &A)
{
    N = A.N; M = A.M;
    pt = new T[M * N];
    index = new int[N];
    memcpy(pt, A.pt, M * N * sizeof(T));
    memcpy(index, A.index, N * sizeof(int));
    factor = A.factor;
    symmetric = A.symmetric;
}

template <typename T>
matrix<T>::matrix(int rows, int columns) : N(rows), M(columns)
{
    pt = new T[rows * columns];
    for (int i = 0; i < rows * columns; i++) pt[i] = T();
    index = new int[rows];
    for (int i = 0; i < rows; i++) index[i] = i;
    factor = false;
    symmetric = false;
}

template <typename T>
matrix<T>::matrix(int rows) : N(rows), M(1)
{
    pt = new T[rows];
    for (int i = 0; i < rows; i++) pt[i] = T();
    index = new int[rows];
    for (int i = 0; i < rows; i++) index[i] = i;
    factor = false;
    symmetric = false;
}

// ---------------------------------------------------------------------------
// member functions
// ---------------------------------------------------------------------------

template <typename T>
void matrix<T>::print() const
{
    std::cout << "Size = " << N << ", " << M << std::endl;
    int maxrow = min(8, N), maxcol = min(8, M);
    for (int i = 0; i < maxrow; i++)
    {
        for (int j = 0; j < maxcol; j++)
            std::cout << pt[i * M + j] << " ";
        if (M > maxcol) std::cout << "...";
        std::cout << std::endl;
    }
    if (N > maxrow) std::cout << "..." << std::endl;
}

template <typename T>
void matrix<T>::save(const char *filename) const
{
    std::ofstream file(filename, std::ios::out);
    file.setf(std::ios_base::scientific, std::ios_base::floatfield);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
            file << std::setprecision(20) << pt[i * M + j] << " ";
        file << std::endl;
    }
    file.close();
}

template <typename T>
void matrix<T>::resize(int rows, int columns)
{
    delete[] pt;
    delete[] index;
    N = rows; M = columns;
    pt = new T[M * N];
    for (int i = 0; i < M * N; i++) pt[i] = T();
    index = new int[N];
    for (int i = 0; i < N; i++) index[i] = i;
    factor = false;
}

template <typename T>
void matrix<T>::LU()
{
    double maxval;
    T tmp, m;
    int itmp;
    int j, k, l;

    if (symmetric)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k <= j - 1; k++)
            {
                tmp = pt[j * M + k];
                for (l = j; l < N; l++)
                    pt[l * M + j] -= tmp * pt[l * M + k];
            }
            pt[j * M + j] = sqrt(pt[j * M + j]);
            tmp = pt[j * M + j];
            for (k = j + 1; k < N; k++)
                pt[k * M + j] /= tmp;
        }
    }
    else
    {
        for (k = 0; k < N - 1; k++)
        {
            maxval = abs(pt[k * M + k]);
            l = k;
            for (j = k + 1; j < N; j++)
                if (abs(pt[j * M + k]) > maxval) { maxval = abs(pt[j * M + k]); l = j; }

            itmp = index[l]; index[l] = index[k]; index[k] = itmp;
            for (j = 0; j < N; j++)
            {
                tmp = pt[l * M + j];
                pt[l * M + j] = pt[k * M + j];
                pt[k * M + j] = tmp;
            }
            tmp = pt[k * M + k];
            if (tmp != (T) 0.0)
                for (j = k + 1; j < N; j++)
                {
                    m = pt[j * M + k] / tmp;
                    pt[j * M + k] = m;
                    for (l = k + 1; l < N; l++)
                        pt[j * M + l] -= m * pt[k * M + l];
                }
        }
    }
    factor = true;
}

template <typename T>
void matrix<T>::solve(matrix<T> &b, matrix<T> &x)
{
    int i, j, k;
    T s;
    if (!factor) LU();

    if (symmetric)
    {
        x = b;
        for (j = 0; j < N - 1; j++)
        {
            x.pt[j] /= pt[j * M + j];
            s = x.pt[j];
            for (i = j + 1; i < N; i++) x.pt[i] -= pt[i * M + j] * s;
        }
        x.pt[N - 1] = x.pt[N - 1] / pt[(N - 1) * M + (N - 1)];
        for (j = N - 1; j > 0; j--)
        {
            x.pt[j] = x.pt[j] / pt[j * M + j];
            s = x.pt[j];
            for (i = 0; i <= j - 1; i++) x.pt[i] -= pt[j * N + i] * s;
        }
        x.pt[0] = x.pt[0] / pt[0];
    }
    else
    {
        for (k = 0; k < N; k++)
            for (j = k + 1; j < N; j++)
                b.pt[index[j]] -= pt[j * M + k] * b.pt[index[k]];
        for (k = N - 1; k >= 0; k--)
        {
            s = b.pt[index[k]];
            for (j = k + 1; j < N; j++) s -= pt[k * M + j] * x.pt[j];
            x.pt[k] = s / pt[k * M + k];
        }
    }
}

template <typename T>
void matrix<T>::solve(matrix<T> &b)
{
    // The in-place general back-substitution in the original code mixed
    // permuted (index[k]) and unpermuted (k) accesses and gave wrong results
    // whenever pivoting reordered rows.  Delegate to the correct two-argument
    // solve and copy the result back.
    matrix<T> x(N);
    solve(b, x);
    b = x;
}

template <typename T>
void matrix<T>::eye(int rows, int columns)
{
    delete[] pt; delete[] index;
    N = rows; M = columns;
    pt = new T[M * N];
    index = new int[N];
    factor = true;
    for (int i = 0; i < N * M; i++) pt[i] = (T) 0.0;
    for (int i = 0; i < min(N, M); i++) pt[i * M + i] = (T) 1.0;
}

template <typename T>
void matrix<T>::zeros(int rows, int columns)
{
    delete[] pt; delete[] index;
    N = rows; M = columns;
    pt = new T[M * N];
    index = new int[N];
    factor = true;
    for (int i = 0; i < N * M; i++) pt[i] = (T) 0.0;
}

template <typename T>
void matrix<T>::random(int rows, int columns)
{
    delete[] pt; delete[] index;
    N = rows; M = columns;
    pt = new T[M * N];
    index = new int[N];
    factor = true;
    srand((unsigned) time(NULL));
    for (int i = 0; i < N * M; i++) pt[i] = (T) ((double) rand() / RAND_MAX);
}

// ---------------------------------------------------------------------------
// operators
// ---------------------------------------------------------------------------

template <typename T>
inline T &matrix<T>::operator()(int i, int j)
{
    assert(i >= 1 && i <= N && j >= 1 && j <= M);
    return pt[(i - 1) * M + j - 1];
}

template <typename T>
inline T &matrix<T>::operator()(int i)
{
    assert(i >= 1 && i <= N * M);
    return pt[i - 1];
}

template <typename T>
inline const T matrix<T>::operator()(int i, int j) const
{
    assert(i >= 1 && i <= N && j >= 1 && j <= M);
    return pt[(i - 1) * M + j - 1];
}

template <typename T>
inline const T matrix<T>::operator()(int i) const
{
    assert(i >= 1 && i <= N * M);
    return pt[i - 1];
}

template <typename T>
matrix<T> &matrix<T>::operator=(T a)
{
    FEMD_OMP_FOR_IF((long) N * M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < N * M; i++) pt[i] = a;
    return *this;
}

template <typename T>
matrix<T> &matrix<T>::operator=(const matrix<T> &B)
{
    if (this == &B) return *this;
    if (N * M != B.N * B.M)
    {
        delete[] pt;
        pt = new T[B.N * B.M];
    }
    if (N != B.N)
    {
        delete[] index;
        index = new int[B.N];
    }
    N = B.N; M = B.M;
    FEMD_OMP_FOR_IF((long) N * M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < N * M; i++) pt[i] = B.pt[i];
    for (int i = 0; i < N; i++) index[i] = B.index[i];
    factor = B.factor;
    symmetric = B.symmetric;
    return *this;
}

template <typename T>
matrix<T> &matrix<T>::operator+=(const matrix<T> &B)
{
    FEMD_OMP_FOR_IF((long) B.N * B.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < B.N * B.M; i++) pt[i] += B.pt[i];
    return *this;
}

template <typename T>
matrix<T> &matrix<T>::operator-=(const matrix<T> &B)
{
    FEMD_OMP_FOR_IF((long) B.N * B.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < B.N * B.M; i++) pt[i] -= B.pt[i];
    return *this;
}

// ---------------------------------------------------------------------------
// friend / free operators
// ---------------------------------------------------------------------------

template <typename T>
matrix<T> operator*(const matrix<T> &A, const matrix<T> &B)
{
    assert(A.M == B.N);
    matrix<T> C(A.N, B.M);
    FEMD_OMP_FOR_IF((long) A.N * B.M > FEMD_OMP_THRESHOLD)  // distinct rows -> race-free
    for (int i = 1; i <= A.N; i++)
        for (int k = 1; k <= B.N; k++)
        {
            T sum = A(i, k);   // private per-iteration
            if (sum != (T) 0.)
                for (int j = 1; j <= B.M; j++)
                    C(i, j) += sum * B(k, j);
        }
    return C;
}

template <typename T>
matrix<T> operator*(const T &a, const matrix<T> &A)
{
    matrix<T> C(A.N, A.M);
    FEMD_OMP_FOR_IF((long) A.N * A.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < A.N * A.M; i++) C.pt[i] = a * A.pt[i];
    return C;
}

template <typename T>
matrix<T> operator*(const matrix<T> &A, const T &a)
{
    matrix<T> C(A.N, A.M);
    FEMD_OMP_FOR_IF((long) A.N * A.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < A.N * A.M; i++) C.pt[i] = A.pt[i] * a;
    return C;
}

template <typename T>
matrix<T> operator+(const matrix<T> &A, const matrix<T> &B)
{
    assert(A.N == B.N && A.M == B.M);
    matrix<T> C(A.N, A.M);
    FEMD_OMP_FOR_IF((long) A.N * A.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < A.N * A.M; i++) C.pt[i] = A.pt[i] + B.pt[i];
    return C;
}

template <typename T>
matrix<T> operator+(const matrix<T> &A, const T &a)
{
    matrix<T> C(A.N, A.M);
    FEMD_OMP_FOR_IF((long) A.N * A.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < A.N * A.M; i++) C.pt[i] = A.pt[i] + a;
    return C;
}

template <typename T>
matrix<T> operator-(const matrix<T> &A, const matrix<T> &B)
{
    assert(A.N == B.N && A.M == B.M);
    matrix<T> C(A.N, A.M);
    FEMD_OMP_FOR_IF((long) A.N * A.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < A.N * A.M; i++) C.pt[i] = A.pt[i] - B.pt[i];
    return C;
}

template <typename T>
matrix<T> operator-(const matrix<T> &A)
{
    matrix<T> C(A.N, A.M);
    FEMD_OMP_FOR_IF((long) A.N * A.M > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < A.N * A.M; i++) C.pt[i] = -A.pt[i];
    return C;
}

template <typename T>
T ddot(const matrix<T> &A, const matrix<T> &B)
{
    assert(A.N == B.N && A.M == B.M);
    int n = A.N * A.M;
    T sum = T();
    // OpenMP's reduction(+) is only defined for built-in arithmetic types, so we
    // parallelise the floating-point case and keep complex on the serial path.
    // if constexpr discards the unused branch, so the pragma never sees complex.
    if constexpr (std::is_floating_point_v<T>)
    {
        FEMD_OMP_FOR_REDUCE_IF(+, sum, (long) n > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n; i++) sum += A.pt[i] * B.pt[i];
    }
    else
        for (int i = 0; i < n; i++) sum += A.pt[i] * B.pt[i];
    return sum;
}

template <typename T>
T maxentry(const matrix<T> &A)
{
    // abs() returns a real magnitude, so this only instantiates for real T;
    // reduction(max) is well-defined there.
    int n = A.N * A.M;
    T dmax = T();
    FEMD_OMP_FOR_REDUCE_IF(max, dmax, (long) n > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < n; i++) dmax = max(dmax, abs(A.pt[i]));
    return dmax;
}

} // namespace femd

#endif // FEMD_LINALG_MATRIX_HPP
