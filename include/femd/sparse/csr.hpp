//
//  csr.hpp  --  compressed sparse row storage for 2D (and any unstructured) operators.
//
//  A SparsityPattern (row pointers, sorted column indices) is built once per pair
//  of spaces and shared, through a shared_ptr, by every matrix assembled on it, so
//  mass and stiffness matrices of one space hold one copy of the indices and
//  A + B on a shared pattern is a sum of the value arrays.
//
//  CSRMatrix is FEMd's own store: the Python side views its three arrays as a
//  scipy.sparse.csr_matrix with no copy, and every solver and preconditioner here
//  (krylov/cg.hpp, sparse/precond.hpp, the csnewton GMRES) works on it directly.
//  Indices are 0-based and int32, as SciPy expects.
//
#ifndef FEMD_SPARSE_CSR_HPP
#define FEMD_SPARSE_CSR_HPP

#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace femd {

struct SparsityPattern {
    int nrows = 0, ncols = 0;
    std::vector<int> indptr;      // nrows + 1
    std::vector<int> indices;     // nnz, sorted and unique within each row

    int nnz() const { return indptr.empty() ? 0 : indptr.back(); }

    /// @brief Position of (i, j) in the value array, or -1 when it is not in the pattern.
    int find(int i, int j) const
    {
        const int *b = indices.data() + indptr[i], *e = indices.data() + indptr[i + 1];
        const int *p = std::lower_bound(b, e, j);
        return (p != e && *p == j) ? static_cast<int>(p - indices.data()) : -1;
    }

    /**
     * @brief Pattern of sum_e R_e^T S_e: row i couples to column j when some cell holds both.
     * @param ncells  number of cells
     * @param rows    rows[c*nr + k]: row of local test function k on cell c, -1 to skip
     * @param cols    cols[c*nc + k]: column of local trial function k on cell c, -1 to skip
     */
    static std::shared_ptr<SparsityPattern> from_cells(int nrows, int ncols, int ncells,
                                                       const std::vector<int> &rows, int nr,
                                                       const std::vector<int> &cols, int nc)
    {
        auto P = std::make_shared<SparsityPattern>();
        P->nrows = nrows; P->ncols = ncols;
        // row -> cells, by counting
        std::vector<int> cnt(static_cast<std::size_t>(nrows) + 1, 0);
        for (int c = 0; c < ncells; ++c)
            for (int k = 0; k < nr; ++k) { int i = rows[static_cast<std::size_t>(c) * nr + k]; if (i >= 0) ++cnt[i + 1]; }
        for (int i = 0; i < nrows; ++i) cnt[i + 1] += cnt[i];
        std::vector<int> cellof(static_cast<std::size_t>(cnt[nrows]));
        std::vector<int> fill(cnt.begin(), cnt.end() - 1);
        for (int c = 0; c < ncells; ++c)
            for (int k = 0; k < nr; ++k) { int i = rows[static_cast<std::size_t>(c) * nr + k]; if (i >= 0) cellof[fill[i]++] = c; }
        P->indptr.assign(static_cast<std::size_t>(nrows) + 1, 0);
        std::vector<std::vector<int>> rowcols(static_cast<std::size_t>(nrows));
        const bool par = nrows > FEMD_OMP_THRESHOLD;
        (void) par;                                          // unused in the serial build
        FEMD_OMP_PARALLEL_IF(par)
        {
        std::vector<int> buf;
        FEMD_OMP_FOR
        for (int i = 0; i < nrows; ++i)
        {
            buf.clear();
            for (int t = cnt[i]; t < cnt[i + 1]; ++t)
            {
                int c = cellof[t];
                for (int k = 0; k < nc; ++k) { int j = cols[static_cast<std::size_t>(c) * nc + k]; if (j >= 0) buf.push_back(j); }
            }
            std::sort(buf.begin(), buf.end());
            buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
            rowcols[i] = buf;
        }
        }
        for (int i = 0; i < nrows; ++i) P->indptr[i + 1] = P->indptr[i] + static_cast<int>(rowcols[i].size());
        P->indices.resize(static_cast<std::size_t>(P->indptr[nrows]));
        FEMD_OMP_FOR_IF(par)
        for (int i = 0; i < nrows; ++i) std::copy(rowcols[i].begin(), rowcols[i].end(), P->indices.begin() + P->indptr[i]);
        return P;
    }

    /// @brief Pattern from arbitrary (i, j) pairs (duplicates merged).
    static std::shared_ptr<SparsityPattern> from_pairs(int nrows, int ncols, const std::vector<int> &I, const std::vector<int> &J)
    {
        auto P = std::make_shared<SparsityPattern>();
        P->nrows = nrows; P->ncols = ncols;
        std::vector<std::vector<int>> rc(static_cast<std::size_t>(nrows));
        for (std::size_t t = 0; t < I.size(); ++t) rc[I[t]].push_back(J[t]);
        P->indptr.assign(static_cast<std::size_t>(nrows) + 1, 0);
        for (int i = 0; i < nrows; ++i)
        {
            std::sort(rc[i].begin(), rc[i].end());
            rc[i].erase(std::unique(rc[i].begin(), rc[i].end()), rc[i].end());
            P->indptr[i + 1] = P->indptr[i] + static_cast<int>(rc[i].size());
        }
        for (int i = 0; i < nrows; ++i) P->indices.insert(P->indices.end(), rc[i].begin(), rc[i].end());
        return P;
    }

    /// @brief The union of two patterns of the same shape.
    static std::shared_ptr<SparsityPattern> merge(const SparsityPattern &A, const SparsityPattern &B)
    {
        auto P = std::make_shared<SparsityPattern>();
        P->nrows = A.nrows; P->ncols = A.ncols;
        const int n = A.nrows;
        P->indptr.assign(static_cast<std::size_t>(n) + 1, 0);
        const bool par = n > FEMD_OMP_THRESHOLD;
    (void) par;
        (void) par;
        FEMD_OMP_FOR_IF(par)                               // pass 1: the length of each union
        for (int i = 0; i < n; ++i)
        {
            int a = A.indptr[i], b = B.indptr[i], len = 0;
            const int ae = A.indptr[i + 1], be = B.indptr[i + 1];
            while (a < ae && b < be)
            {
                const int x = A.indices[a], y = B.indices[b];
                a += (x <= y); b += (y <= x); ++len;
            }
            P->indptr[i + 1] = len + (ae - a) + (be - b);
        }
        for (int i = 0; i < n; ++i) P->indptr[i + 1] += P->indptr[i];
        P->indices.resize(static_cast<std::size_t>(P->indptr[n]));
        FEMD_OMP_FOR_IF(par)                               // pass 2: the unions themselves
        for (int i = 0; i < n; ++i)
            std::set_union(A.indices.begin() + A.indptr[i], A.indices.begin() + A.indptr[i + 1],
                           B.indices.begin() + B.indptr[i], B.indices.begin() + B.indptr[i + 1],
                           P->indices.begin() + P->indptr[i]);
        return P;
    }
};

struct CSRMatrix {
    std::shared_ptr<const SparsityPattern> pattern;
    std::vector<double> data;
    bool symmetric = false;          // a declaration by the assembler, as for AssembledMatrix

    CSRMatrix() : pattern(std::make_shared<SparsityPattern>()) {}
    explicit CSRMatrix(std::shared_ptr<const SparsityPattern> P, bool sym = false)
        : pattern(std::move(P)), data(static_cast<std::size_t>(pattern->nnz()), 0.0), symmetric(sym) {}

    int nrows() const { return pattern->nrows; }
    int ncols() const { return pattern->ncols; }
    int nnz()   const { return pattern->nnz(); }
    const int *indptr()  const { return pattern->indptr.data(); }
    const int *indices() const { return pattern->indices.data(); }

    /// @brief K(i,j) += v; (i,j) must be in the pattern.
    void add(int i, int j, double v)
    {
        int p = pattern->find(i, j);
        if (p < 0) throw std::out_of_range("CSRMatrix::add: (" + std::to_string(i) + ", " + std::to_string(j) + ") is not in the pattern");
        data[p] += v;
    }
    double at(int i, int j) const { int p = pattern->find(i, j); return p < 0 ? 0.0 : data[p]; }

    /// @brief y = K x (overwrites y).
    void apply(const double *x, double *y) const
    {
        const int n = nrows();
        const int *ip = indptr(), *ix = indices();
        const double *v = data.data();
        FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n; ++i)
        {
            double s = 0.0;
            for (int p = ip[i]; p < ip[i + 1]; ++p) s += v[p] * x[ix[p]];
            y[i] = s;
        }
    }
    /// @brief y^T K x without forming K x: each row's product is multiplied by y[i] as it is
    ///        computed, and the rows are added in the ordered chunked sum (the same bits for any
    ///        number of threads).  x has ncols entries and y has nrows.
    double inner(const double *x, const double *y) const
    {
        const int *ip = indptr(), *ix = indices();
        const double *v = data.data();
        return detail::ordered_sum(static_cast<std::size_t>(nrows()), [&](std::size_t i) {
            double s = 0.0;
            for (int p = ip[i]; p < ip[i + 1]; ++p) s += v[p] * x[ix[p]];
            return s * y[i];
        });
    }
    std::vector<double> apply(const std::vector<double> &x) const
    {
        if (static_cast<int>(x.size()) != ncols()) throw std::invalid_argument("CSRMatrix::apply: length mismatch");
        std::vector<double> y(static_cast<std::size_t>(nrows()));
        if (nrows() > 0) apply(x.data(), y.data());
        return y;
    }
    /// @brief y = K^T x (overwrites y), serial scatter.
    void apply_transpose(const double *x, double *y) const
    {
        std::fill(y, y + ncols(), 0.0);
        const int *ip = indptr(), *ix = indices();
        for (int i = 0; i < nrows(); ++i)
            for (int p = ip[i]; p < ip[i + 1]; ++p) y[ix[p]] += data[p] * x[i];
    }

    std::vector<double> diagonal() const
    {
        const int n = std::min(nrows(), ncols());
        std::vector<double> d(static_cast<std::size_t>(n), 0.0);
        FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n; ++i) d[i] = at(i, i);
        return d;
    }

    /// @brief Dense row-major copy.  Tests and oracles only.
    std::vector<double> dense() const
    {
        std::vector<double> D(static_cast<std::size_t>(nrows()) * ncols(), 0.0);
        for (int i = 0; i < nrows(); ++i)
            for (int p = indptr()[i]; p < indptr()[i + 1]; ++p) D[static_cast<std::size_t>(i) * ncols() + indices()[p]] += data[p];
        return D;
    }

    void clear() { std::fill(data.begin(), data.end(), 0.0); }

    /// @brief Remove entries with |v| <= tol from the pattern (a new pattern is made).
    CSRMatrix pruned(double tol = 0.0) const
    {
        auto P = std::make_shared<SparsityPattern>();
        P->nrows = nrows(); P->ncols = ncols();
        P->indptr.assign(static_cast<std::size_t>(nrows()) + 1, 0);
        std::vector<double> v;
        for (int i = 0; i < nrows(); ++i)
        {
            for (int p = indptr()[i]; p < indptr()[i + 1]; ++p)
                if (std::abs(data[p]) > tol) { P->indices.push_back(indices()[p]); v.push_back(data[p]); }
            P->indptr[i + 1] = static_cast<int>(P->indices.size());
        }
        CSRMatrix C(P, symmetric);
        C.data = std::move(v);
        return C;
    }
};

/// @brief C = alpha A + beta B.  Shares A's pattern when the two patterns are one object.
/// @brief B = P A P^T for a square A: B(k, l) = A(p[k], p[l]), with sorted rows.
inline CSRMatrix permuted(const CSRMatrix &A, const std::vector<int> &p)
{
    const int n = A.nrows();
    if (A.ncols() != n || static_cast<int>(p.size()) != n) throw std::invalid_argument("permuted: square matrix and a full permutation needed");
    std::vector<int> pinv(static_cast<std::size_t>(n), -1);
    for (int k = 0; k < n; ++k)
    {
        if (p[k] < 0 || p[k] >= n || pinv[p[k]] >= 0) throw std::invalid_argument("permuted: not a permutation");
        pinv[p[k]] = k;
    }
    auto P = std::make_shared<SparsityPattern>();
    P->nrows = P->ncols = n;
    P->indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int k = 0; k < n; ++k) P->indptr[k + 1] = P->indptr[k] + (A.indptr()[p[k] + 1] - A.indptr()[p[k]]);
    P->indices.resize(static_cast<std::size_t>(P->indptr[n]));
    std::vector<double> v(P->indices.size());
    std::vector<std::pair<int, double>> row;
    for (int k = 0; k < n; ++k)
    {
        row.clear();
        for (int q = A.indptr()[p[k]]; q < A.indptr()[p[k] + 1]; ++q) row.emplace_back(pinv[A.indices()[q]], A.data[q]);
        std::sort(row.begin(), row.end());
        for (std::size_t t = 0; t < row.size(); ++t) { P->indices[P->indptr[k] + t] = row[t].first; v[P->indptr[k] + t] = row[t].second; }
    }
    CSRMatrix B(P, A.symmetric);
    B.data = std::move(v);
    return B;
}

inline CSRMatrix combine(const CSRMatrix &A, const CSRMatrix &B, double alpha, double beta)
{
    if (A.nrows() != B.nrows() || A.ncols() != B.ncols())
        throw std::invalid_argument("CSRMatrix: shape mismatch in a sum");
    if (A.pattern == B.pattern)
    {
        CSRMatrix C(A.pattern, A.symmetric && B.symmetric);
        const long nz = static_cast<long>(C.data.size());
        FEMD_OMP_FOR_IF(nz > FEMD_OMP_THRESHOLD)
        for (long k = 0; k < nz; ++k) C.data[k] = alpha * A.data[k] + beta * B.data[k];
        return C;
    }
    auto P = SparsityPattern::merge(*A.pattern, *B.pattern);
    CSRMatrix C(P, A.symmetric && B.symmetric);
    auto pour = [&](const CSRMatrix &M, double c, int i) {
        if (c == 0.0) return;
        int q = P->indptr[i];
        for (int p = M.indptr()[i]; p < M.indptr()[i + 1]; ++p)
        {
            const int j = M.indices()[p];
            while (P->indices[q] < j) ++q;               // both rows sorted, M's columns are a subset
            C.data[q] += c * M.data[p];
        }
    };
    const int n = A.nrows();
    FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)        // rows are independent; A then B within each
    for (int i = 0; i < n; ++i) { pour(A, alpha, i); pour(B, beta, i); }
    return C;
}

inline CSRMatrix scaled(const CSRMatrix &A, double a)
{
    CSRMatrix C(A.pattern, A.symmetric);
    const long nz = static_cast<long>(C.data.size());
    FEMD_OMP_FOR_IF(nz > FEMD_OMP_THRESHOLD)
    for (long k = 0; k < nz; ++k) C.data[k] = a * A.data[k];
    return C;
}

/// @brief A^T, a new pattern (columns become rows, still sorted).
inline CSRMatrix transposed(const CSRMatrix &A)
{
    auto P = std::make_shared<SparsityPattern>();
    P->nrows = A.ncols(); P->ncols = A.nrows();
    P->indptr.assign(static_cast<std::size_t>(P->nrows) + 1, 0);
    for (int p = 0; p < A.nnz(); ++p) ++P->indptr[A.indices()[p] + 1];
    for (int i = 0; i < P->nrows; ++i) P->indptr[i + 1] += P->indptr[i];
    P->indices.resize(static_cast<std::size_t>(A.nnz()));
    std::vector<double> v(static_cast<std::size_t>(A.nnz()));
    std::vector<int> fill(P->indptr.begin(), P->indptr.end() - 1);
    for (int i = 0; i < A.nrows(); ++i)                 // row order: each new row fills in increasing column
        for (int p = A.indptr()[i]; p < A.indptr()[i + 1]; ++p)
        {
            int q = fill[A.indices()[p]]++;
            P->indices[q] = i; v[q] = A.data[p];
        }
    CSRMatrix C(P, A.symmetric);
    C.data = std::move(v);
    return C;
}

/// @brief C = A B (Gustavson, row by row with a dense accumulator).
inline CSRMatrix multiply(const CSRMatrix &A, const CSRMatrix &B)
{
    if (A.ncols() != B.nrows()) throw std::invalid_argument("CSRMatrix: inner dimensions differ in a product");
    const int n = A.nrows(), m = B.ncols();
    auto P = std::make_shared<SparsityPattern>();
    P->nrows = n; P->ncols = m;
    P->indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    const bool par = n > FEMD_OMP_THRESHOLD;
    (void) par;
    // pass 1: the number of entries of each row
    FEMD_OMP_PARALLEL_IF(par)
    {
    std::vector<int> mark(static_cast<std::size_t>(m), -1);
    FEMD_OMP_FOR
    for (int i = 0; i < n; ++i)
    {
        int len = 0;
        for (int p = A.indptr()[i]; p < A.indptr()[i + 1]; ++p)
        {
            const int k = A.indices()[p];
            for (int q = B.indptr()[k]; q < B.indptr()[k + 1]; ++q)
                if (mark[B.indices()[q]] != i) { mark[B.indices()[q]] = i; ++len; }
        }
        P->indptr[i + 1] = len;
    }
    }
    for (int i = 0; i < n; ++i) P->indptr[i + 1] += P->indptr[i];
    P->indices.resize(static_cast<std::size_t>(P->indptr[n]));
    CSRMatrix C(P, false);
    // pass 2: each row with a dense accumulator, the same operations in the same order as in serial
    FEMD_OMP_PARALLEL_IF(par)
    {
    std::vector<double> acc(static_cast<std::size_t>(m), 0.0);
    std::vector<int> mark(static_cast<std::size_t>(m), -1), cols;
    FEMD_OMP_FOR
    for (int i = 0; i < n; ++i)
    {
        cols.clear();
        for (int p = A.indptr()[i]; p < A.indptr()[i + 1]; ++p)
        {
            const int k = A.indices()[p];
            const double a = A.data[p];
            for (int q = B.indptr()[k]; q < B.indptr()[k + 1]; ++q)
            {
                const int j = B.indices()[q];
                if (mark[j] != i) { mark[j] = i; acc[j] = 0.0; cols.push_back(j); }
                acc[j] += a * B.data[q];
            }
        }
        std::sort(cols.begin(), cols.end());
        int o = P->indptr[i];
        for (int j : cols) { P->indices[o] = j; C.data[o] = acc[j]; ++o; }
    }
    }
    return C;
}

/// @brief Relative asymmetry max |a_ij - a_ji| / max |a_ij| (0 for an exactly symmetric matrix).
inline double asymmetry(const CSRMatrix &A)
{
    if (A.nrows() != A.ncols()) return 1.0;
    double big = 0.0, d = 0.0;             // maxima: exact, so any order gives the same bits
    const int n = A.nrows();
    FEMD_OMP_FOR_REDUCE_IF(max, big, A.nnz() > FEMD_OMP_THRESHOLD)
    for (int p = 0; p < A.nnz(); ++p) big = std::max(big, std::abs(A.data[p]));
    FEMD_OMP_FOR_REDUCE_IF(max, d, n > FEMD_OMP_THRESHOLD)
    for (int i = 0; i < n; ++i)
        for (int p = A.indptr()[i]; p < A.indptr()[i + 1]; ++p)
            d = std::max(d, std::abs(A.data[p] - A.at(A.indices()[p], i)));
    return big > 0.0 ? d / big : 0.0;
}

} // namespace femd

#endif // FEMD_SPARSE_CSR_HPP
