//
//  assembly.hpp  --  the three kernels of design doc Section 15.1, reading the
//  QuadratureCache, plus the banded accumulation buffer they scatter into.
//
//      rank 0:  integral c
//      rank 1:  f_i   += integral c (D^a N_i)
//      rank 2:  K_ij  += integral c (D^a N_i)(D^b N_j)      i = test (row), j = trial (column)
//
//  Indices are 0-based here.  The AssembledMatrix keeps a dense band
//  (|i-j| <= uband) plus an overflow list for the periodic corners, and
//  converts to the 1-based solver storage in fill() (solve/linear_solver.hpp).
//
#ifndef FEMD_FORMS_ASSEMBLY_HPP
#define FEMD_FORMS_ASSEMBLY_HPP

#include "femd/fe/quadrature_cache.hpp"
#include "femd/forms/coefficient.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace femd {

namespace detail {
/// Rows [lo, hi) of y = K x with a compile-time band width, so the inner loop is
/// unrolled into 2P+1 fused multiply-adds with no bounds.  Same summation order as
/// the generic loop, so the result is bit-identical.
template <int P>
inline void band_apply_interior(int lo, int hi, const double *band, const double *x, double *y)
{
    constexpr int W = 2 * P + 1;
    FEMD_OMP_FOR_IF(hi - lo > FEMD_OMP_THRESHOLD)
    for (int i = lo; i < hi; ++i)
    {
        const double *row = band + static_cast<std::size_t>(i) * W;
        const double *xi = x + (i - P);
        double s = 0.0;
        for (int j = 0; j < W; ++j) s += row[j] * xi[j];
        y[i] = s;
    }
}
} // namespace detail

struct AssembledMatrix {
    int n = 0, p = 0, w = 0;                 // size, half-bandwidth, band width 2p+1
    std::vector<double> band;                // n * w row-major, band[i*w + (j-i+p)]
    std::vector<int> oi, oj;                 // out-of-band entries (periodic corners)
    std::vector<double> ov;
    bool symmetric = true;                   // set false by any (a != b) term with a non-symmetric role

    AssembledMatrix() = default;
    AssembledMatrix(int size, int half) : n(size), p(half), w(2 * half + 1), band(static_cast<std::size_t>(size) * (2 * half + 1), 0.0) {}

    inline void add(int i, int j, double v)
    {
        int off = j - i;
        if (off >= -p && off <= p) band[static_cast<std::size_t>(i) * w + (off + p)] += v;
        else { oi.push_back(i); oj.push_back(j); ov.push_back(v); }
    }
    /// @brief Entry (i,j), including overflow.  O(overflow) in the corner case; tests and oracles only.
    double at(int i, int j) const
    {
        int off = j - i;
        if (off >= -p && off <= p) return band[static_cast<std::size_t>(i) * w + (off + p)];
        double s = 0.0;
        for (std::size_t k = 0; k < ov.size(); ++k) if (oi[k] == i && oj[k] == j) s += ov[k];
        return s;
    }
    /// @brief Dense row-major copy (n x n).  Tests and oracles only.
    std::vector<double> dense() const
    {
        std::vector<double> D(static_cast<std::size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int off = -p; off <= p; ++off)
            {
                int j = i + off;
                if (j >= 0 && j < n) D[static_cast<std::size_t>(i) * n + j] += band[static_cast<std::size_t>(i) * w + (off + p)];
            }
        for (std::size_t k = 0; k < ov.size(); ++k) D[static_cast<std::size_t>(oi[k]) * n + oj[k]] += ov[k];
        return D;
    }
    /**
     * @brief Matrix-vector product y = K x, straight off the banded store.
     *
     * O(n(2p+1) + |ov|), gathering 2p+1 contiguous doubles per row against
     * x[i-p .. i+p].  No index arrays, no sparse conversion: this is what the
     * Python `Matrix @ x` routes through, and it replaces rebuilding a CSR on
     * every product.  The band holds both triangles regardless of `symmetric`,
     * which is only a hint for solver selection, so there is no symmetric case.
     *
     * @param x input vector, length n.
     * @param y [out] result, length n, overwritten (not accumulated).
     */
    void apply(const double *x, double *y) const
    {
        // Rows p .. n-p-1 see the whole band; the first and last p rows are clipped.
        const int a = std::min(p, n), e = std::max(n - p, a);
        const double *b = band.data();
        bool fixed = true;
        switch (p)          // the band widths degrees 0..7 give; anything wider takes the generic loop
        {
        case 0: detail::band_apply_interior<0>(a, e, b, x, y); break;
        case 1: detail::band_apply_interior<1>(a, e, b, x, y); break;
        case 2: detail::band_apply_interior<2>(a, e, b, x, y); break;
        case 3: detail::band_apply_interior<3>(a, e, b, x, y); break;
        case 4: detail::band_apply_interior<4>(a, e, b, x, y); break;
        case 5: detail::band_apply_interior<5>(a, e, b, x, y); break;
        case 6: detail::band_apply_interior<6>(a, e, b, x, y); break;
        case 7: detail::band_apply_interior<7>(a, e, b, x, y); break;
        default: fixed = false;
        }
        if (fixed)
        {
            for (int i = 0; i < a; ++i) y[i] = row_dot(i, x);
            for (int i = e; i < n; ++i) y[i] = row_dot(i, x);
        }
        else
        {
            FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
            for (int i = 0; i < n; ++i) y[i] = row_dot(i, x);
        }
        // periodic corners: a handful of entries, scattered, so left serial
        for (std::size_t k = 0; k < ov.size(); ++k) y[oi[k]] += ov[k] * x[oj[k]];
    }

    /// @brief y^T K x: the band rows times x, each multiplied by y[i], added in the ordered
    ///        chunked sum, plus the periodic corners.  Nothing is allocated.
    double inner(const double *x, const double *y) const
    {
        double s = detail::ordered_sum(static_cast<std::size_t>(n), [&](std::size_t i) {
            return row_dot(static_cast<int>(i), x) * y[i]; });
        for (std::size_t k = 0; k < ov.size(); ++k) s += ov[k] * x[oj[k]] * y[oi[k]];
        return s;
    }

    /// @brief Row i of the band times x, clipped to the columns 0..n-1 (the generic loop).
    double row_dot(int i, const double *x) const
    {
        const double *row = &band[static_cast<std::size_t>(i) * w];
        const int lo = (i - p < 0) ? -i : -p;
        const int hi = (i + p >= n) ? n - 1 - i : p;
        double s = 0.0;
        for (int off = lo; off <= hi; ++off) s += row[off + p] * x[i + off];
        return s;
    }

    /// @brief y = K x. @param x input vector of length n. @return K x.
    std::vector<double> apply(const std::vector<double> &x) const
    {
        if (static_cast<int>(x.size()) != n)
            throw std::invalid_argument("AssembledMatrix::apply: length mismatch");
        std::vector<double> y(static_cast<std::size_t>(n), 0.0);
        if (n > 0) apply(x.data(), y.data());
        return y;
    }

    void clear() { std::fill(band.begin(), band.end(), 0.0); oi.clear(); oj.clear(); ov.clear(); }
};

// ---------------------------------------------------------------------------
//  Algebra on the assembled store.
//
//  These are the operators behind Python's `Matrix` arithmetic.  They work on
//  the band buffer directly and return a new AssembledMatrix, so nothing goes
//  through a sparse format and nothing touches the solver storage (which LU()
//  overwrites in place anyway).
//
//  `symmetric` is a declaration the assembler makes, not a measurement, so it
//  is propagated conservatively: a sum is symmetric only when both terms are,
//  and a product is never declared symmetric, since A B is generally not
//  symmetric even when A and B both are.  Measure with detail::structure().
// ---------------------------------------------------------------------------

/// @brief C = alpha A + beta B. Result half-bandwidth is max(A.p, B.p).
/// @param A first matrix. @param B second, same dimension. @param alpha scale on A. @param beta scale on B.
inline AssembledMatrix combine(const AssembledMatrix &A, const AssembledMatrix &B, double alpha, double beta)
{
    if (A.n != B.n)
        throw std::invalid_argument("AssembledMatrix: dimension mismatch, " + std::to_string(A.n) +
                                    " vs " + std::to_string(B.n));
    AssembledMatrix C(A.n, std::max(A.p, B.p));
    C.symmetric = A.symmetric && B.symmetric;
    // the band of M lies inside the band of C, so rows are independent (parallel); the few
    // overflow entries are added afterwards, in the same order as before
    auto pour = [&](const AssembledMatrix &M, double c)
    {
        if (c == 0.0) return;
        FEMD_OMP_FOR_IF(M.n > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < M.n; ++i)
            for (int off = -M.p; off <= M.p; ++off)
            {
                int j = i + off;
                if (j < 0 || j >= M.n) continue;
                double v = M.band[static_cast<std::size_t>(i) * M.w + (off + M.p)];
                if (v != 0.0) C.band[static_cast<std::size_t>(i) * C.w + (off + C.p)] += c * v;
            }
        for (std::size_t q = 0; q < M.ov.size(); ++q)
            if (M.ov[q] != 0.0) C.add(M.oi[q], M.oj[q], c * M.ov[q]);
    };
    pour(A, alpha);
    pour(B, beta);
    return C;
}

/// @brief A += beta B, in place. Requires A.p >= B.p. @param A [in,out] target. @param B addend. @param beta scale on B.
inline void axpy(AssembledMatrix &A, const AssembledMatrix &B, double beta)
{
    if (A.n != B.n)
        throw std::invalid_argument("AssembledMatrix::axpy: dimension mismatch");
    if (A.p < B.p)
        throw std::invalid_argument("AssembledMatrix::axpy: target half-bandwidth " + std::to_string(A.p) +
                                    " is narrower than " + std::to_string(B.p) + "; combine() instead");
    if (beta == 0.0) return;
    A.symmetric = A.symmetric && B.symmetric;
    FEMD_OMP_FOR_IF(B.n > FEMD_OMP_THRESHOLD)        // B's band lies inside A's: rows are independent
    for (int i = 0; i < B.n; ++i)
        for (int off = -B.p; off <= B.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= B.n) continue;
            double v = B.band[static_cast<std::size_t>(i) * B.w + (off + B.p)];
            if (v != 0.0) A.band[static_cast<std::size_t>(i) * A.w + (off + A.p)] += beta * v;
        }
    for (std::size_t q = 0; q < B.ov.size(); ++q)
        if (B.ov[q] != 0.0) A.add(B.oi[q], B.oj[q], beta * B.ov[q]);
}

/// @brief C = a A, same bandwidth and the same `symmetric` declaration. @param A matrix. @param a scalar.
inline AssembledMatrix scaled(const AssembledMatrix &A, double a)
{
    AssembledMatrix C(A.n, A.p);
    C.symmetric = A.symmetric;
    const long nb = static_cast<long>(A.band.size());
    FEMD_OMP_FOR_IF(nb > FEMD_OMP_THRESHOLD)
    for (long k = 0; k < nb; ++k) C.band[k] = a * A.band[k];
    C.oi = A.oi; C.oj = A.oj;
    C.ov.resize(A.ov.size());
    for (std::size_t k = 0; k < A.ov.size(); ++k) C.ov[k] = a * A.ov[k];
    return C;
}

/// @brief C = A^T. Same bandwidth, O(nnz). @param A matrix.
inline AssembledMatrix transposed(const AssembledMatrix &A)
{
    AssembledMatrix C(A.n, A.p);
    C.symmetric = A.symmetric;
    // gathered by rows of C: C(j, j - off) = A(j - off, j), one write per entry, rows in parallel
    FEMD_OMP_FOR_IF(A.n > FEMD_OMP_THRESHOLD)
    for (int j = 0; j < A.n; ++j)
        for (int off = -A.p; off <= A.p; ++off)
        {
            int i = j - off;                                    // A(i, j) sits at offset off in row i
            if (i < 0 || i >= A.n) continue;
            double v = A.band[static_cast<std::size_t>(i) * A.w + (off + A.p)];
            if (v != 0.0) C.band[static_cast<std::size_t>(j) * C.w + (-off + C.p)] += v;
        }
    for (std::size_t q = 0; q < A.ov.size(); ++q)
        if (A.ov[q] != 0.0) C.add(A.oj[q], A.oi[q], A.ov[q]);
    return C;
}

/// @brief C = A B. Half-bandwidth min(A.p + B.p, n-1). O(nnz(A) * rowwidth(B)).
/// @param A left factor. @param B right factor, same dimension.
inline AssembledMatrix multiply(const AssembledMatrix &A, const AssembledMatrix &B)
{
    if (A.n != B.n)
        throw std::invalid_argument("AssembledMatrix: dimension mismatch, " + std::to_string(A.n) +
                                    " vs " + std::to_string(B.n));
    const int n = A.n;
    AssembledMatrix C(n, n > 0 ? std::min(A.p + B.p, n - 1) : 0);
    C.symmetric = false;                       // A B is not symmetric in general, even when both factors are
    if (n == 0) return C;

    std::vector<std::vector<std::pair<int, double> > > Brow(n);
    FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)          // each row of B on its own; overflow appended after
    for (int j = 0; j < n; ++j)
        for (int off = -B.p; off <= B.p; ++off)
        {
            int k = j + off;
            if (k < 0 || k >= n) continue;
            double v = B.band[static_cast<std::size_t>(j) * B.w + (off + B.p)];
            if (v != 0.0) Brow[j].push_back(std::make_pair(k, v));
        }
    for (std::size_t q = 0; q < B.ov.size(); ++q)
        if (B.ov[q] != 0.0) Brow[B.oi[q]].push_back(std::make_pair(B.oj[q], B.ov[q]));

    // out-of-band contributions are summed here, so the overflow list stays short
    std::map<std::pair<int, int>, double> corner;
    double *const Cb = C.band.data();                    // raw copies: no reloads through C in the loops
    const int Cp = C.p, Cw = C.w;
    auto emit = [Cb, Cp, Cw](std::map<std::pair<int, int>, double> &cm, int i, int k, double v)
    {
        int off = k - i;
        if (off >= -Cp && off <= Cp) Cb[static_cast<std::size_t>(i) * Cw + (off + Cp)] += v;
        else cm[std::make_pair(i, k)] += v;
    };
    auto rowdot = [&Brow, &emit](std::map<std::pair<int, int>, double> &cm, int i, int j, double a)
    {
        const std::vector<std::pair<int, double> > &Bj = Brow[j];
        for (std::size_t t = 0; t < Bj.size(); ++t) emit(cm, i, Bj[t].first, a * Bj[t].second);
    };
    // the band of A: row i of C is written only by row i, so rows run in parallel; a thread's
    // corner entries all lie in its own rows, so merging the per-thread maps adds nothing twice
    FEMD_OMP_PARALLEL_IF(n > FEMD_OMP_THRESHOLD)
    {
    std::map<std::pair<int, int>, double> mine;
    FEMD_OMP_FOR
    for (int i = 0; i < n; ++i)
        for (int off = -A.p; off <= A.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= n) continue;
            double a = A.band[static_cast<std::size_t>(i) * A.w + (off + A.p)];
            if (a != 0.0) rowdot(mine, i, j, a);
        }
    if (!mine.empty())
    {
        FEMD_PRAGMA_CRITICAL
        corner.insert(mine.begin(), mine.end());
    }
    }
    for (std::size_t q = 0; q < A.ov.size(); ++q)
        if (A.ov[q] != 0.0) rowdot(corner, A.oi[q], A.oj[q], A.ov[q]);

    for (std::map<std::pair<int, int>, double>::const_iterator it = corner.begin(); it != corner.end(); ++it)
        if (it->second != 0.0) { C.oi.push_back(it->first.first); C.oj.push_back(it->first.second); C.ov.push_back(it->second); }
    return C;
}

// ---------------------------------------------------------------------------
//  The kernels below compute every element's contribution first, in parallel
//  when OpenMP is on and the problem is large (detail::parallel_elements), into
//  one buffer, and then add the buffer into the global object in element order.
//  Each entry is therefore summed in the same order whatever the number of
//  threads: the result is bit-for-bit independent of it, which matters for
//  schemes whose invariants are sensitive to rounding (manual, Section 7.7).
// ---------------------------------------------------------------------------
namespace detail {
/// elem(e, buf + e*stride) for every element (in parallel), then scatter(e, buf + e*stride) in order.
template <class Elem, class Scatter>
inline void by_elements(int ne, int nq, std::size_t stride, Elem &&elem, Scatter &&scatter)
{
    std::vector<double> buf(static_cast<std::size_t>(ne) * stride, 0.0);
    (void) nq;                                               // unused in the serial build
    FEMD_OMP_FOR_IF(parallel_elements(ne, nq))
    for (int e = 0; e < ne; ++e) elem(e, buf.data() + static_cast<std::size_t>(e) * stride);
    for (int e = 0; e < ne; ++e) scatter(e, buf.data() + static_cast<std::size_t>(e) * stride);
}

/// Element matrix  Ke[ki*nj + kj] = sum_q w c (D^a T_ki)(D^b S_kj)  on element e.
inline void element_matrix(const QuadratureCache &QT, const QuadratureCache &QS, const std::vector<double> &cq,
                           int e, int a, int b, double *Ke)
{
    const int ni = QT.nloc(e), nj = QS.nloc(e), nq = QT.nq();
    for (int q = 0; q < nq; ++q)
    {
        const double s = QT.weight(e, q) * cq[static_cast<std::size_t>(e) * nq + q];
        const double *ra = QT.basis_row(e, q, a);
        const double *rb = QS.basis_row(e, q, b);
        for (int ki = 0; ki < ni; ++ki)
        {
            const double sa = s * ra[ki];
            double *row = Ke + static_cast<std::size_t>(ki) * nj;
            for (int kj = 0; kj < nj; ++kj) row[kj] += sa * rb[kj];
        }
    }
}

/// Element vector  fe[k] = sum_q w c (D^a N_k)  on element e.
inline void element_vector(const QuadratureCache &Q, const std::vector<double> &cq, int e, int a, double *fe)
{
    const int nl = Q.nloc(e), nq = Q.nq();
    for (int q = 0; q < nq; ++q)
    {
        const double s = Q.weight(e, q) * cq[static_cast<std::size_t>(e) * nq + q];
        const double *row = Q.basis_row(e, q, a);
        for (int k = 0; k < nl; ++k) fe[k] += s * row[k];
    }
}
} // namespace detail

// ---- rank 0 ----------------------------------------------------------------
inline double assemble_scalar(const QuadratureCache &Q, const Coefficient &c)
{
    std::vector<double> cq = c.at_quad(Q);
    const int nq = Q.nq();
    double s = 0.0;
    detail::by_elements(Q.nelem(), nq, 1,
        [&](int e, double *se) { for (int q = 0; q < nq; ++q) *se += Q.weight(e, q) * cq[static_cast<std::size_t>(e) * nq + q]; },
        [&](int, const double *se) { s += *se; });
    return s;
}

// ---- rank 1 ----------------------------------------------------------------
/// @brief f_i += integral c (D^a N_i).  f must have length V.dim(); accumulates.
inline void assemble_vector(const QuadratureCache &Q, int a, const Coefficient &c, std::vector<double> &f)
{
    if (a > Q.nder()) throw std::invalid_argument("assemble_vector: cache holds fewer derivatives than requested");
    std::vector<double> cq = c.at_quad(Q);
    detail::by_elements(Q.nelem(), Q.nq(), static_cast<std::size_t>(Q.nloc_max()),
        [&](int e, double *fe) { detail::element_vector(Q, cq, e, a, fe); },
        [&](int e, const double *fe) { for (int k = 0; k < Q.nloc(e); ++k) f[Q.dof(e, k)] += fe[k]; });
}

// ---- rank 2 ----------------------------------------------------------------
/// @brief K_ij += integral c (D^a N_i)(D^b N_j), i = test, j = trial.  Accumulates into K.
inline void assemble_matrix(const QuadratureCache &Q, int a, int b, const Coefficient &c, AssembledMatrix &K)
{
    if (a > Q.nder() || b > Q.nder()) throw std::invalid_argument("assemble_matrix: cache holds fewer derivatives than requested");
    if (a != b) K.symmetric = false;
    std::vector<double> cq = c.at_quad(Q);
    const std::size_t nlm = static_cast<std::size_t>(Q.nloc_max());
    detail::by_elements(Q.nelem(), Q.nq(), nlm * nlm,
        [&](int e, double *Ke) { detail::element_matrix(Q, Q, cq, e, a, b, Ke); },
        [&](int e, const double *Ke) {
            const int nl = Q.nloc(e);
            for (int ki = 0; ki < nl; ++ki)
            {
                const int I = Q.dof(e, ki);
                for (int kj = 0; kj < nl; ++kj) K.add(I, Q.dof(e, kj), Ke[static_cast<std::size_t>(ki) * nl + kj]);
            }
        });
}

/**
 * @brief Right-hand side of the inhomogeneous lift:  f_i -= sum_terms integral c (D^a N_i)(D^b u_g),
 *        with u_g given at the quadrature nodes and its derivatives (ug_der[b]).
 *        One rank-1 call per term (design doc 5.6).  Accumulates into f.
 */
inline void assemble_lift_rhs(const QuadratureCache &Q, int a, int b, const Coefficient &c,
                              const std::vector<std::vector<double>> &ug_der, std::vector<double> &f)
{
    std::vector<double> cq = c.at_quad(Q);
    const std::vector<double> &ug = ug_der.at(b);
    for (std::size_t i = 0; i < cq.size(); ++i) cq[i] *= -ug[i];
    assemble_vector(Q, a, Coefficient(cq), f);
}

} // namespace femd

#endif // FEMD_FORMS_ASSEMBLY_HPP
