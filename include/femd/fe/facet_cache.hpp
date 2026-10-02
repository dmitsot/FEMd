//
//  facet_cache.hpp  --  the traces of a space on the interior facets of a 1D mesh,
//  and the kernels that assemble facet terms from them.
//
//  A facet of a 1D mesh is a vertex.  Facet f of the interior list joins two elements:
//
//      side 0, '-'   the element on the LEFT,  its outward normal is +1
//      side 1, '+'   the element on the RIGHT, its outward normal is -1
//
//  Interior vertices x_1 .. x_{ne-1} give facets 0 .. ne-2 (minus = e-1, plus = e).
//  A periodic space adds the seam as facet ne-1: minus = the last element at b,
//  plus = the first element at a.  The two ends of a non-periodic mesh are boundary
//  facets and are handled by ds, not here.
//
//  For each facet and side the cache holds the adapted basis functions of that
//  side's element and their derivatives at the facet point, so a facet term
//
//      sum_f  c_f  D^a N_i^{s}(x_f)  D^b N_j^{t}(x_f)
//
//  is a short double loop per facet.  The jump and average of DG methods are sums of
//  such terms over the four side pairs, which the form compiler writes out.
//
#ifndef FEMD_FE_FACET_CACHE_HPP
#define FEMD_FE_FACET_CACHE_HPP

#include "femd/fe/function_space.hpp"
#include "femd/forms/assembly.hpp"

#include <stdexcept>
#include <vector>

namespace femd {

class FacetCache {
public:
    FacetCache() = default;

    /// @brief Traces of V (values and derivatives 0..nder) on both sides of every interior facet.
    FacetCache(const FunctionSpace &V, int nder) : nder_(nder), nloc_max_(V.nloc_max())
    {
        const Mesh1D &m = V.mesh();
        const int ne = V.nelem();
        for (int e = 1; e < ne; ++e) add(e - 1, m.vertices()[e], e, m.vertices()[e]);
        if (V.bc().periodic && ne >= 1) add(ne - 1, m.b(), 0, m.a());
        tab_.assign(static_cast<std::size_t>(nf()) * 2 * (nder_ + 1) * nloc_max_, 0.0);
        dofs_.assign(static_cast<std::size_t>(nf()) * 2 * nloc_max_, -1);
        nloc_.assign(static_cast<std::size_t>(nf()) * 2, 0);
        const int nfac = nf();
        FEMD_OMP_PARALLEL_IF(nfac > FEMD_OMP_THRESHOLD)       // each facet writes its own slots
        {
        std::vector<double> vals, raw;
        FEMD_OMP_FOR
        for (int f = 0; f < nfac; ++f)
            for (int s = 0; s < 2; ++s)
            {
                const int e = elem_[2 * f + s];
                const std::vector<int> &d = V.element_dofs(e);
                const int na = static_cast<int>(d.size());
                nloc_[2 * f + s] = na;
                for (int k = 0; k < na; ++k) dofs_[(2 * f + s) * nloc_max_ + k] = d[k];
                // side 0 is the element on the left, traced at its right end (xi = +1); side 1 at xi = -1
                V.eval_on_element_ref(e, s == 0 ? 1.0 : -1.0, nder_, vals, raw);
                for (int mm = 0; mm <= nder_; ++mm)
                    for (int k = 0; k < na; ++k)
                        tab_[((static_cast<std::size_t>(2 * f + s)) * (nder_ + 1) + mm) * nloc_max_ + k] = vals[mm * na + k];
            }
        }
    }

    int nf() const { return static_cast<int>(elem_.size() / 2); }
    int nder() const { return nder_; }
    int nloc_max() const { return nloc_max_; }
    /// @brief Element on side s (0 = '-', 1 = '+') of facet f.
    int element(int f, int s) const { return elem_[2 * f + s]; }
    /// @brief The facet point as seen from side s (they differ only at the periodic seam).
    double x(int f, int s) const { return xs_[2 * f + s]; }
    int nloc(int f, int s) const { return nloc_[2 * f + s]; }
    const int *dofs(int f, int s) const { return &dofs_[static_cast<std::size_t>(2 * f + s) * nloc_max_]; }
    /// @brief d^m N_{dofs(f,s)[k]} / dx^m at the facet point, k = 0..nloc(f,s)-1.
    const double *values(int f, int s, int m) const
    {
        return &tab_[(static_cast<std::size_t>(2 * f + s) * (nder_ + 1) + m) * nloc_max_];
    }

private:
    void add(int em, double xm, int ep, double xp)
    {
        elem_.push_back(em); elem_.push_back(ep);
        xs_.push_back(xm); xs_.push_back(xp);
    }

    int nder_ = 0, nloc_max_ = 0;
    std::vector<int> elem_, dofs_, nloc_;
    std::vector<double> xs_, tab_;
};

namespace detail {
inline void check_facet_args(const FacetCache &F, int s, int m)
{
    if (s != 0 && s != 1) throw std::invalid_argument("facet side must be 0 ('-') or 1 ('+')");
    if (m < 0 || m > F.nder()) throw std::invalid_argument("facet derivative order exceeds the cache's nder");
}
} // namespace detail

/**
 * @brief K_ij += sum_f c_f (D^a N_i^{st})(x_f) (D^b N_j^{ss})(x_f): test function i from side st
 *        of cache FT, trial function j from side ss of cache FS (the same space for a square K).
 *        The symmetric flag is cleared unless the term is its own transpose (a == b, st == ss).
 */
inline void assemble_facet_matrix(AssembledMatrix &K, const FacetCache &FT, const FacetCache &FS,
                                  int st, int ss, int a, int b, const double *c)
{
    detail::check_facet_args(FT, st, a);
    detail::check_facet_args(FS, ss, b);
    if (FT.nf() != FS.nf()) throw std::invalid_argument("assemble_facet_matrix: caches of different meshes");
    for (int f = 0; f < FT.nf(); ++f)
    {
        if (c[f] == 0.0) continue;
        const int *ti = FT.dofs(f, st), *sj = FS.dofs(f, ss);
        const double *tv = FT.values(f, st, a), *sv = FS.values(f, ss, b);
        for (int i = 0; i < FT.nloc(f, st); ++i)
        {
            const double ci = c[f] * tv[i];
            if (ci == 0.0) continue;
            for (int j = 0; j < FS.nloc(f, ss); ++j) K.add(ti[i], sj[j], ci * sv[j]);
        }
    }
    if (a != b || st != ss) K.symmetric = false;
}

/// @brief The same term as COO triplets, for a rectangular block (test and trial in different spaces).
inline void facet_triplets(const FacetCache &FT, const FacetCache &FS, int st, int ss, int a, int b, const double *c,
                           std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals)
{
    detail::check_facet_args(FT, st, a);
    detail::check_facet_args(FS, ss, b);
    for (int f = 0; f < FT.nf(); ++f)
    {
        if (c[f] == 0.0) continue;
        const int *ti = FT.dofs(f, st), *sj = FS.dofs(f, ss);
        const double *tv = FT.values(f, st, a), *sv = FS.values(f, ss, b);
        for (int i = 0; i < FT.nloc(f, st); ++i)
            for (int j = 0; j < FS.nloc(f, ss); ++j)
            {
                rows.push_back(ti[i]); cols.push_back(sj[j]); vals.push_back(c[f] * tv[i] * sv[j]);
            }
    }
}

/// @brief r_i += sum_f c_f (D^a N_i^{s})(x_f), a vector of length n.
inline std::vector<double> assemble_facet_vector(const FacetCache &F, int s, int a, const double *c, int n)
{
    detail::check_facet_args(F, s, a);
    std::vector<double> r(n, 0.0);
    for (int f = 0; f < F.nf(); ++f)
    {
        const int *d = F.dofs(f, s);
        const double *v = F.values(f, s, a);
        for (int k = 0; k < F.nloc(f, s); ++k) r[d[k]] += c[f] * v[k];
    }
    return r;
}

/// @brief D^m u on side s of every facet, u = sum_j coeffs[j] N_j.
inline std::vector<double> facet_values(const FacetCache &F, const std::vector<double> &coeffs, int s, int m)
{
    detail::check_facet_args(F, s, m);
    std::vector<double> out(F.nf(), 0.0);
    FEMD_OMP_FOR_IF(F.nf() > FEMD_OMP_THRESHOLD)
    for (int f = 0; f < F.nf(); ++f)
    {
        const int *d = F.dofs(f, s);
        const double *v = F.values(f, s, m);
        double acc = 0.0;
        for (int k = 0; k < F.nloc(f, s); ++k) acc += coeffs[d[k]] * v[k];
        out[f] = acc;
    }
    return out;
}

} // namespace femd

#endif // FEMD_FE_FACET_CACHE_HPP
