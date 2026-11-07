//
//  assembly_2d.hpp  --  the three kernels on triangles, reading a Cache2D (cells or
//  boundary facets), scattering into a CSRMatrix or a vector.
//
//      rank 0:  sum_e sum_q w c
//      rank 1:  f_i  += sum_t integral c_t (D^{a_t} N_i)
//      rank 2:  K_ij += sum_t integral c_t (D^{a_t} T_i)(D^{b_t} S_j)
//
//  A whole list of terms goes in one call, so a Laplacian (d/dx d/dx + d/dy d/dy)
//  reads the cache once.  Derivative codes: 0 value, 1 d/dx, 2 d/dy, and for a vector
//  family (Raviart-Thomas, Nedelec) 3 c + d for component c, so div u = (code 1) + (code 5)
//  and rot u = (code 4) - (code 2).  Coefficients are arrays at the cache points, [e*nq + q].
//
//  As in 1D, element contributions are computed in parallel (by chunks of cells, so
//  the buffer stays small).  They are then added in element order, also in parallel:
//  each row belongs to one thread (detail::owns_row), which adds that row's entries
//  in the serial order.  The result does not depend on the number of threads.
//
#ifndef FEMD_FORMS_ASSEMBLY_2D_HPP
#define FEMD_FORMS_ASSEMBLY_2D_HPP

#include "femd/fe/cache_2d.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace femd {

/// @brief CSR pattern of every coupling test T_i -- trial S_j through a common cell.
inline std::shared_ptr<SparsityPattern> cell_pattern(const Space2D &T, const Space2D &S)
{
    if (T.ncells() != S.ncells() || T.nverts() != S.nverts())
        throw std::invalid_argument("cell_pattern: the two spaces are on different meshes");
    return SparsityPattern::from_cells(T.dim(), S.dim(), T.ncells(), T.cell_dofs(), T.nloc(), S.cell_dofs(), S.nloc());
}

/**
 * @brief CSR pattern of a system: test fields T[f] numbered from toff[f], trial fields S[g] from
 *        soff[g], every field on the same mesh.  Every pair of fields is coupled through the cells
 *        (a superset of the blocks a form fills; the others hold zeros).
 */
inline std::shared_ptr<SparsityPattern> product_pattern(const std::vector<const Space2D *> &T, const std::vector<int> &toff,
                                                        int nrows, const std::vector<const Space2D *> &S,
                                                        const std::vector<int> &soff, int ncols)
{
    if (T.empty() || S.empty() || T.size() != toff.size() || S.size() != soff.size())
        throw std::invalid_argument("product_pattern: one offset per field");
    const int nc = T[0]->ncells();
    auto gather = [nc](const std::vector<const Space2D *> &F, const std::vector<int> &off, int &width) {
        width = 0;
        for (const auto *f : F)
        {
            if (f->ncells() != nc) throw std::invalid_argument("product_pattern: the fields are on different meshes");
            width += f->nloc();
        }
        std::vector<int> d(static_cast<std::size_t>(nc) * width, -1);
        for (int c = 0; c < nc; ++c)
        {
            int k = 0;
            for (std::size_t q = 0; q < F.size(); ++q)
                for (int l = 0; l < F[q]->nloc(); ++l, ++k)
                {
                    const int a = F[q]->dof(c, l);
                    d[static_cast<std::size_t>(c) * width + k] = a < 0 ? -1 : a + off[q];
                }
        }
        return d;
    };
    int wr = 0, wc = 0;
    std::vector<int> R = gather(T, toff, wr), C = gather(S, soff, wc);
    return SparsityPattern::from_cells(nrows, ncols, nc, R, wr, C, wc);
}

namespace detail {
inline void check_codes(const std::vector<int> &codes, int ncodes = 3)
{
    for (int m : codes)
        if (m < 0 || m >= ncodes)
            throw std::invalid_argument("assembly_2d: derivative code " + std::to_string(m) + " is not in 0.." + std::to_string(ncodes - 1));
}
constexpr int max_codes_2d = 6;
inline void check_coeffs(const Cache2D &Q, const std::vector<std::vector<double>> &c, std::size_t nterms)
{
    if (c.size() != nterms) throw std::invalid_argument("assembly_2d: one coefficient array per term");
    const std::size_t np = static_cast<std::size_t>(Q.nent()) * Q.nq();
    for (const auto &v : c)
        if (v.size() != np) throw std::invalid_argument("assembly_2d: coefficient length " + std::to_string(v.size()) +
                                                        " != entries x points = " + std::to_string(np));
}
constexpr int chunk_2d = 2048;

/// The element matrix Ke (zero on entry) += sum_t sum_q w c_t (D^{a_t} T_i)(D^{b_t} S_j): the terms
/// in order, then the points, then the rows i and columns j, each product skipped when it is zero.
/// NI, NJ > 0: the sizes known at compile time (unrolled, vectorized); 0: the general loop.  Every
/// entry is formed by the same operations in the same order either way.
template <int NI, int NJ>
inline void element_matrix_2d(double *Ke, int ni_, int nj_, std::size_t nt, int nq, std::size_t e, const int *a, const int *b,
                              const double *w, const std::vector<std::vector<double>> &c, const double *RT, const double *RS)
{
    const int ni = NI > 0 ? NI : ni_, nj = NJ > 0 ? NJ : nj_;
    for (std::size_t t = 0; t < nt; ++t)
    {
        const double *ct = c[t].data();
        for (int q = 0; q < nq; ++q)
        {
            const std::size_t k = e * static_cast<std::size_t>(nq) + q;
            const double s = w[k] * ct[k];
            if (s == 0.0) continue;
            const double *ra = RT + (static_cast<std::size_t>(a[t]) * nq + q) * ni;
            const double *rb = RS + (static_cast<std::size_t>(b[t]) * nq + q) * nj;
            for (int i = 0; i < ni; ++i)
            {
                const double sa = s * ra[i];
                if (sa == 0.0) continue;
                double *row = Ke + static_cast<std::size_t>(i) * nj;
                for (int j = 0; j < nj; ++j) row[j] += sa * rb[j];
            }
        }
    }
}
} // namespace detail

/**
 * @brief The scatter map of assemble_matrix_2d: for every entry (e, i, j) of the element matrices of
 *        QT and QS, its position in the data of a matrix on pattern P (rows from row_offset, columns
 *        from col_offset), or -1 where the test or trial function is constrained.  It depends on the
 *        caches and the pattern only, so it is built once and reused (FormAssembler2D).  Throws when
 *        an entry is outside the pattern.
 */
inline std::vector<int> scatter_map_2d(const Cache2D &QT, const Cache2D &QS, const SparsityPattern &P, int row_offset, int col_offset)
{
    const int ni = QT.nloc(), nj = QS.nloc(), ne = QT.nent();
    const std::size_t bs = static_cast<std::size_t>(ni) * nj;
    std::vector<int> pos(static_cast<std::size_t>(ne) * bs, -1);
    int outside = 0;
    FEMD_OMP(parallel for schedule(static) reduction(| : outside) if (ne > FEMD_OMP_THRESHOLD / 8))
    for (int e = 0; e < ne; ++e)
    {
        const int *dT = QT.dofs(e), *dS = QS.dofs(e);
        int *pe = pos.data() + static_cast<std::size_t>(e) * bs;
        for (int i = 0; i < ni; ++i)
        {
            if (dT[i] < 0) continue;
            const int I = dT[i] + row_offset;
            const int *rb = P.indices.data() + P.indptr[I], *re = P.indices.data() + P.indptr[I + 1];
            for (int j = 0; j < nj; ++j)
            {
                if (dS[j] < 0) continue;
                const int J = dS[j] + col_offset;
                const int *p = std::lower_bound(rb, re, J);
                if (p == re || *p != J) { outside |= 1; continue; }
                pe[static_cast<std::size_t>(i) * nj + j] = static_cast<int>(p - P.indices.data());
            }
        }
    }
    if (outside) throw std::logic_error("assemble_matrix_2d: entry outside the pattern");
    return pos;
}

inline double assemble_scalar_2d(const Cache2D &Q, const std::vector<double> &c)
{
    detail::check_coeffs(Q, std::vector<std::vector<double>>{c}, 1);
    const std::vector<double> &w = Q.weights();
    return detail::ordered_sum(w.size(), [&](std::size_t k) { return w[k] * c[k]; });
}

inline void assemble_vector_2d(const Cache2D &Q, const std::vector<int> &a, const std::vector<std::vector<double>> &c,
                               std::vector<double> &f)
{
    detail::check_codes(a, Q.ncodes());
    detail::check_coeffs(Q, c, a.size());
    if (static_cast<int>(f.size()) != Q.space().dim()) throw std::invalid_argument("assemble_vector_2d: f has the wrong length");
    const int n = Q.nloc(), nq = Q.nq(), ne = Q.nent();
    const std::size_t nt = a.size();
    for (int e0 = 0; e0 < ne; e0 += detail::chunk_2d)
    {
        const int e1 = std::min(ne, e0 + detail::chunk_2d);
        std::vector<double> buf(static_cast<std::size_t>(e1 - e0) * n, 0.0);
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(e1 - e0, nq))
        {
        std::vector<double> r(static_cast<std::size_t>(n));
        FEMD_OMP_FOR
        for (int e = e0; e < e1; ++e)
        {
            double *fe = &buf[static_cast<std::size_t>(e - e0) * n];
            for (std::size_t t = 0; t < nt; ++t)
                for (int q = 0; q < nq; ++q)
                {
                    const std::size_t k = static_cast<std::size_t>(e) * nq + q;
                    const double s = Q.weights()[k] * c[t][k];
                    if (s == 0.0) continue;
                    Q.row(e, q, a[t], r.data());
                    for (int l = 0; l < n; ++l) fe[l] += s * r[l];
                }
        }
        const int nth = detail::thread_count(), me = detail::thread_id();
        for (int e = e0; e < e1; ++e)
        {
            const int *d = Q.dofs(e);
            const double *fe = &buf[static_cast<std::size_t>(e - e0) * n];
            for (int l = 0; l < n; ++l) if (d[l] >= 0 && detail::owns_row(d[l], me, nth)) f[d[l]] += fe[l];
        }
        }
    }
}

/**
 * @brief K += sum_t integral c_t (D^{a_t} T_i)(D^{b_t} S_j) over the entries of QT and QS,
 *        which must be caches of the same kind and rule on the same mesh (test and trial space).
 *        Row i of the test space goes to row row_offset + i of K and column j of the trial space
 *        to column col_offset + j: the block of a system (product_pattern).
 */
inline void assemble_matrix_2d(const Cache2D &QT, const Cache2D &QS, const std::vector<int> &a, const std::vector<int> &b,
                               const std::vector<std::vector<double>> &c, CSRMatrix &K, int row_offset = 0, int col_offset = 0,
                               const int *map = nullptr)
{
    detail::check_codes(a, QT.ncodes()); detail::check_codes(b, QS.ncodes());
    if (a.size() != b.size()) throw std::invalid_argument("assemble_matrix_2d: a and b differ in length");
    detail::check_coeffs(QT, c, a.size());
    if (QT.nent() != QS.nent() || QT.nq() != QS.nq() || QT.on_facets() != QS.on_facets())
        throw std::invalid_argument("assemble_matrix_2d: test and trial caches differ (entries, points or kind)");
    if (row_offset < 0 || col_offset < 0 || K.nrows() < row_offset + QT.space().dim() || K.ncols() < col_offset + QS.space().dim())
        throw std::invalid_argument("assemble_matrix_2d: the spaces (with their offsets) do not fit in the matrix");
    const int ni = QT.nloc(), nj = QS.nloc(), nq = QT.nq(), ne = QT.nent();
    const std::size_t nt = a.size(), bs = static_cast<std::size_t>(ni) * nj;
    const int ncT = QT.ncodes(), ncS = QS.ncodes();
    bool needT[detail::max_codes_2d] = {}, needS[detail::max_codes_2d] = {};
    for (std::size_t t = 0; t < nt; ++t) { needT[a[t]] = true; needS[b[t]] = true; }
    const SparsityPattern &P = *K.pattern;
    std::atomic<bool> outside{false};        // set by any thread that meets one
    for (int e0 = 0; e0 < ne; e0 += detail::chunk_2d)
    {
        const int e1 = std::min(ne, e0 + detail::chunk_2d);
        std::vector<double> buf(static_cast<std::size_t>(e1 - e0) * bs, 0.0);
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(e1 - e0, nq))
        {
        std::vector<double> RT(static_cast<std::size_t>(ncT) * nq * ni), RS(static_cast<std::size_t>(ncS) * nq * nj);
        FEMD_OMP_FOR
        for (int e = e0; e < e1; ++e)
        {
            for (int m = 0; m < std::max(ncT, ncS); ++m)
                for (int q = 0; q < nq; ++q)
                {
                    if (m < ncT && needT[m]) QT.row(e, q, m, &RT[(static_cast<std::size_t>(m) * nq + q) * ni]);
                    if (m < ncS && needS[m]) QS.row(e, q, m, &RS[(static_cast<std::size_t>(m) * nq + q) * nj]);
                }
            double *Ke = &buf[static_cast<std::size_t>(e - e0) * bs];
            const double *wq = QT.weights().data();
            const std::size_t ee = static_cast<std::size_t>(e);
            if (ni == 6 && nj == 6) detail::element_matrix_2d<6, 6>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else if (ni == 6 && nj == 3) detail::element_matrix_2d<6, 3>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else if (ni == 3 && nj == 6) detail::element_matrix_2d<3, 6>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else if (ni == 3 && nj == 3) detail::element_matrix_2d<3, 3>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else if (ni == 10 && nj == 10) detail::element_matrix_2d<10, 10>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else if (ni == 9 && nj == 9) detail::element_matrix_2d<9, 9>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else if (ni == 4 && nj == 4) detail::element_matrix_2d<4, 4>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
            else detail::element_matrix_2d<0, 0>(Ke, ni, nj, nt, nq, ee, a.data(), b.data(), wq, c, RT.data(), RS.data());
        }
        // scatter in element order, each row by its owning thread: the same bits for any team size
        {
        const int nth = detail::thread_count(), me = detail::thread_id();
        for (int e = e0; e < e1; ++e)
        {
            const int *dT = QT.dofs(e), *dS = QS.dofs(e);
            const double *Ke = &buf[static_cast<std::size_t>(e - e0) * bs];
            if (map)
            {
                const int *pe = map + static_cast<std::size_t>(e) * bs;
                for (int i = 0; i < ni; ++i)
                {
                    if (dT[i] < 0 || !detail::owns_row(dT[i] + row_offset, me, nth)) continue;
                    const int *pi = pe + static_cast<std::size_t>(i) * nj;
                    const double *ki = Ke + static_cast<std::size_t>(i) * nj;
                    for (int j = 0; j < nj; ++j)
                        if (pi[j] >= 0) K.data[static_cast<std::size_t>(pi[j])] += ki[j];
                }
                continue;
            }
            for (int i = 0; i < ni; ++i)
            {
                if (dT[i] < 0) continue;
                const int I = dT[i] + row_offset;
                if (!detail::owns_row(I, me, nth)) continue;
                const int *rb = P.indices.data() + P.indptr[I], *re = P.indices.data() + P.indptr[I + 1];
                for (int j = 0; j < nj; ++j)
                {
                    if (dS[j] < 0) continue;
                    const int J = dS[j] + col_offset;
                    const int *p = std::lower_bound(rb, re, J);
                    if (p == re || *p != J) { outside.store(true, std::memory_order_relaxed); continue; }
                    K.data[static_cast<std::size_t>(p - P.indices.data())] += Ke[static_cast<std::size_t>(i) * nj + j];
                }
            }
        }
        }
        }
        if (outside.load()) throw std::logic_error("assemble_matrix_2d: entry outside the pattern");
    }
}

} // namespace femd

#endif // FEMD_FORMS_ASSEMBLY_2D_HPP
