//
//  facet_assembly_2d.hpp  --  the dS kernels on a 2D mesh, reading InteriorFacetCache2D, and
//  the sparsity patterns that couple the two cells of every interior facet.
//
//      rank 0:  sum_f sum_q w c
//      rank 1:  f_i  += sum_t integral c_t (D^{a_t} N_i)|_{s_t}
//      rank 2:  K_ij += sum_t integral c_t (D^{a_t} T_i)|_{s_t} (D^{b_t} S_j)|_{r_t}
//
//  Each term t carries the side of its test function (s_t) and of its trial function (r_t),
//  0 for '-' and 1 for '+', so jump(u) * jump(v) is four terms.  As for the cell kernels, the
//  facet contributions are computed in parallel and added in facet order by the thread that
//  owns each row, so the result does not depend on the number of threads.
//
#ifndef FEMD_FORMS_FACET_ASSEMBLY_2D_HPP
#define FEMD_FORMS_FACET_ASSEMBLY_2D_HPP

#include "femd/fe/facet_cache_2d.hpp"
#include "femd/forms/assembly_2d.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>

namespace femd {

namespace detail {
inline void check_sides(const std::vector<int> &s)
{
    for (int v : s)
        if (v != 0 && v != 1) throw std::invalid_argument("facet assembly: a side is 0 ('-') or 1 ('+')");
}
inline void check_facet_coeffs(const InteriorFacetCache2D &F, const std::vector<std::vector<double>> &c, std::size_t nt)
{
    if (c.size() != nt) throw std::invalid_argument("facet assembly: one coefficient array per term");
    const std::size_t np = static_cast<std::size_t>(F.nf()) * F.nq();
    for (const auto &v : c)
        if (v.size() != np) throw std::invalid_argument("facet assembly: coefficient length != facets x points");
}
} // namespace detail

/**
 * @brief The CSR pattern of a system whose test fields T[f] (numbered from toff[f]) and trial fields
 *        S[g] (from soff[g]) live on one mesh, coupling every pair of fields through each cell and
 *        through the two cells of each interior facet (the pattern of a form with dS terms).
 *        T and S must have the same interior facets (same mesh, same periodic pairing).
 */
inline std::shared_ptr<SparsityPattern> facet_pattern(const std::vector<const Space2D *> &T, const std::vector<int> &toff, int nrows,
                                                      const std::vector<const Space2D *> &S, const std::vector<int> &soff, int ncols)
{
    if (T.empty() || S.empty() || T.size() != toff.size() || S.size() != soff.size())
        throw std::invalid_argument("facet_pattern: one offset per field");
    const int nc = T[0]->ncells();
    const auto &F0 = T[0]->interior_facets();
    const int nf = static_cast<int>(F0.size());
    auto check = [&](const Space2D *f) {
        if (f->ncells() != nc) throw std::invalid_argument("facet_pattern: the fields are on different meshes");
        const auto &F = f->interior_facets();
        if (static_cast<int>(F.size()) != nf) throw std::invalid_argument("facet_pattern: the fields have different interior facets "
                                                                          "(one is periodic and the other is not)");
        for (int i = 0; i < nf; ++i)
            if (F[i].cell[0] != F0[i].cell[0] || F[i].cell[1] != F0[i].cell[1])
                throw std::invalid_argument("facet_pattern: the fields have different interior facets");
    };
    auto gather = [&](const std::vector<const Space2D *> &Fs, const std::vector<int> &off, int &width) {
        int w1 = 0;
        for (const auto *f : Fs) { check(f); w1 += f->nloc(); }
        width = 2 * w1;
        std::vector<int> d(static_cast<std::size_t>(nc + nf) * width, -1);
        auto put = [&](int ent, int c, int base) {
            int k = base;
            for (std::size_t q = 0; q < Fs.size(); ++q)
                for (int l = 0; l < Fs[q]->nloc(); ++l, ++k)
                {
                    const int a = Fs[q]->dof(c, l);
                    d[static_cast<std::size_t>(ent) * width + k] = a < 0 ? -1 : a + off[q];
                }
        };
        for (int c = 0; c < nc; ++c) put(c, c, 0);
        for (int i = 0; i < nf; ++i) { put(nc + i, F0[i].cell[0], 0); put(nc + i, F0[i].cell[1], w1); }
        return d;
    };
    int wr = 0, wc = 0;
    std::vector<int> R = gather(T, toff, wr), C = gather(S, soff, wc);
    return SparsityPattern::from_cells(nrows, ncols, nc + nf, R, wr, C, wc);
}

inline double assemble_facet_scalar_2d(const InteriorFacetCache2D &F, const std::vector<double> &c)
{
    detail::check_facet_coeffs(F, std::vector<std::vector<double>>{c}, 1);
    const std::vector<double> &w = F.weights();
    return detail::ordered_sum(w.size(), [&](std::size_t k) { return w[k] * c[k]; });
}

inline void assemble_facet_vector_2d(const InteriorFacetCache2D &F, const std::vector<int> &side, const std::vector<int> &a,
                                     const std::vector<std::vector<double>> &c, std::vector<double> &f)
{
    detail::check_codes(a, F.ncodes());
    detail::check_sides(side);
    if (side.size() != a.size()) throw std::invalid_argument("assemble_facet_vector_2d: one side per term");
    detail::check_facet_coeffs(F, c, a.size());
    if (static_cast<int>(f.size()) != F.space().dim()) throw std::invalid_argument("assemble_facet_vector_2d: f has the wrong length");
    const int n = F.nloc(), nq = F.nq(), nf = F.nf();
    const std::size_t nt = a.size();
    for (int f0 = 0; f0 < nf; f0 += detail::chunk_2d)
    {
        const int f1 = std::min(nf, f0 + detail::chunk_2d);
        std::vector<double> buf(static_cast<std::size_t>(f1 - f0) * 2 * n, 0.0);
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(f1 - f0, nq))
        {
        std::vector<double> r(static_cast<std::size_t>(n));
        FEMD_OMP_FOR
        for (int e = f0; e < f1; ++e)
        {
            double *fe = &buf[static_cast<std::size_t>(e - f0) * 2 * n];
            for (std::size_t t = 0; t < nt; ++t)
                for (int q = 0; q < nq; ++q)
                {
                    const std::size_t k = static_cast<std::size_t>(e) * nq + q;
                    const double s = F.weights()[k] * c[t][k];
                    if (s == 0.0) continue;
                    F.row(e, q, side[t], a[t], r.data());
                    double *dst = fe + side[t] * n;
                    for (int l = 0; l < n; ++l) dst[l] += s * r[l];
                }
        }
        const int nth = detail::thread_count(), me = detail::thread_id();
        for (int e = f0; e < f1; ++e)
            for (int s = 0; s < 2; ++s)
            {
                const int *d = F.dofs(e, s);
                const double *fe = &buf[static_cast<std::size_t>(e - f0) * 2 * n + s * n];
                for (int l = 0; l < n; ++l) if (d[l] >= 0 && detail::owns_row(d[l], me, nth)) f[d[l]] += fe[l];
            }
        }
    }
}

/**
 * @brief K += the dS terms between test space (cache FT) and trial space (cache FS), each with its
 *        sides; rows offset by row_offset and columns by col_offset (a block of a system).  K's
 *        pattern must hold the facet couplings (facet_pattern).
 */
inline void assemble_facet_matrix_2d(const InteriorFacetCache2D &FT, const InteriorFacetCache2D &FS, const std::vector<int> &sa,
                                     const std::vector<int> &sb, const std::vector<int> &a, const std::vector<int> &b,
                                     const std::vector<std::vector<double>> &c, CSRMatrix &K, int row_offset = 0, int col_offset = 0)
{
    detail::check_codes(a, FT.ncodes()); detail::check_codes(b, FS.ncodes());
    detail::check_sides(sa); detail::check_sides(sb);
    const std::size_t nt = a.size();
    if (b.size() != nt || sa.size() != nt || sb.size() != nt) throw std::invalid_argument("assemble_facet_matrix_2d: lengths differ");
    detail::check_facet_coeffs(FT, c, nt);
    if (FT.nf() != FS.nf() || FT.nq() != FS.nq()) throw std::invalid_argument("assemble_facet_matrix_2d: test and trial caches differ");
    if (row_offset < 0 || col_offset < 0 || K.nrows() < row_offset + FT.space().dim() || K.ncols() < col_offset + FS.space().dim())
        throw std::invalid_argument("assemble_facet_matrix_2d: the spaces (with their offsets) do not fit in the matrix");
    const int ni = FT.nloc(), nj = FS.nloc(), nq = FT.nq(), nf = FT.nf();
    const std::size_t blk = static_cast<std::size_t>(ni) * nj, per = 4 * blk;     // four side pairs
    const SparsityPattern &P = *K.pattern;
    std::atomic<bool> outside{false};
    for (int f0 = 0; f0 < nf; f0 += detail::chunk_2d)
    {
        const int f1 = std::min(nf, f0 + detail::chunk_2d);
        std::vector<double> buf(static_cast<std::size_t>(f1 - f0) * per, 0.0);
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(f1 - f0, nq))
        {
        std::vector<double> ra(static_cast<std::size_t>(ni)), rb(static_cast<std::size_t>(nj));
        FEMD_OMP_FOR
        for (int e = f0; e < f1; ++e)
        {
            double *Ke = &buf[static_cast<std::size_t>(e - f0) * per];
            for (std::size_t t = 0; t < nt; ++t)
            {
                double *B = Ke + (2 * sa[t] + sb[t]) * blk;
                for (int q = 0; q < nq; ++q)
                {
                    const std::size_t k = static_cast<std::size_t>(e) * nq + q;
                    const double s = FT.weights()[k] * c[t][k];
                    if (s == 0.0) continue;
                    FT.row(e, q, sa[t], a[t], ra.data());
                    FS.row(e, q, sb[t], b[t], rb.data());
                    for (int i = 0; i < ni; ++i)
                    {
                        const double si = s * ra[i];
                        if (si == 0.0) continue;
                        double *row = B + static_cast<std::size_t>(i) * nj;
                        for (int j = 0; j < nj; ++j) row[j] += si * rb[j];
                    }
                }
            }
        }
        const int nth = detail::thread_count(), me = detail::thread_id();
        for (int e = f0; e < f1; ++e)
            for (int st = 0; st < 2; ++st)
                for (int ss = 0; ss < 2; ++ss)
                {
                    const int *dT = FT.dofs(e, st), *dS = FS.dofs(e, ss);
                    const double *B = &buf[static_cast<std::size_t>(e - f0) * per + (2 * st + ss) * blk];
                    for (int i = 0; i < ni; ++i)
                    {
                        if (dT[i] < 0) continue;
                        const int I = dT[i] + row_offset;
                        if (!detail::owns_row(I, me, nth)) continue;
                        const int *rb0 = P.indices.data() + P.indptr[I], *re = P.indices.data() + P.indptr[I + 1];
                        for (int j = 0; j < nj; ++j)
                        {
                            const double v = B[static_cast<std::size_t>(i) * nj + j];
                            if (dS[j] < 0 || v == 0.0) continue;
                            const int J = dS[j] + col_offset;
                            const int *p = std::lower_bound(rb0, re, J);
                            if (p == re || *p != J) { outside.store(true, std::memory_order_relaxed); continue; }
                            K.data[static_cast<std::size_t>(p - P.indices.data())] += v;
                        }
                    }
                }
        }
    }
    if (outside.load()) throw std::logic_error("assemble_facet_matrix_2d: entry outside the pattern (build it with facet_pattern)");
}

} // namespace femd

#endif // FEMD_FORMS_FACET_ASSEMBLY_2D_HPP
