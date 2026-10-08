//
//  assembler_1d.hpp  --  a whole 1D form assembled in C++: the known fields evaluated at the
//  points of each term, the coefficient computed by its Integrand program, and the kernels of
//  assembly.hpp, block_assembly.hpp and facet_cache.hpp called term by term, with no Python in
//  the loop.
//
//  The Python Form compiles itself once into a FormAssembler1D (Form._compiled_1d): it names
//  the point sets (the quadrature nodes of a QuadratureCache for dx, Gauss or Lobatto; the
//  interior facets of a FacetCache for dS; one end point of the interval for ds), the fields
//  read at each of them (which coefficient vector, which entries of it for a field of a
//  ProductSpace, which derivative, which side of a facet), and the terms in the order of the
//  form, each with its program and its kernel.  assemble_* then takes the current
//  coefficient vectors and constants.  The object is immutable once built and assemble_* is
//  const, so several threads may assemble one form with different data at once.
//
//  The arithmetic is that of the Python driver it replaces, term after term: each rank-1 term
//  is assembled into a vector of its own and added to the result, rank-2 terms go straight
//  into the AssembledMatrix through the same kernels, and the ds terms add c (D^a N_i)(x) and
//  (c D^a N_i)(D^b N_j) at the end point.  The fields are evaluated by the same routines
//  (Field::at_quad, facet_values, the element evaluation at points).  Only the coefficient
//  itself changes, from NumPy to the Integrand, which agrees with it to rounding.
//
#ifndef FEMD_FORMS_ASSEMBLER_1D_HPP
#define FEMD_FORMS_ASSEMBLER_1D_HPP

#include "femd/fe/facet_cache.hpp"
#include "femd/fe/field.hpp"
#include "femd/fe/function_space.hpp"
#include "femd/fe/product_space.hpp"
#include "femd/fe/quadrature_cache.hpp"
#include "femd/forms/assembly.hpp"
#include "femd/forms/block_assembly.hpp"
#include "femd/forms/coefficient.hpp"
#include "femd/forms/integrand.hpp"
#include "femd/sparse/csr.hpp"
#include "femd/util/omp.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class FormAssembler1D {
public:
    /// @brief A coefficient vector handed to assemble_*: n values at p.
    struct VectorRef { const double *p; std::size_t n; };

    /// @param n      the length of the result of a rank-1 form, the number of rows of a rank-2 one
    /// @param ncols  -1 for a square rank-2 form (an AssembledMatrix), or the dimension of the trial
    ///               space of a rectangular one (test and trial in two spaces on one grid), which
    ///               assemble_rect returns as a CSRMatrix
    FormAssembler1D(int rank, int nvectors, int nconsts, int n, int ncols = -1)
        : rank_(rank), nvec_(nvectors), ncon_(nconsts), n_(n), ncols_(ncols < 0 ? n : ncols), rect_(ncols >= 0)
    {
        if (rank < 0 || rank > 2) throw std::invalid_argument("FormAssembler1D: rank is 0, 1 or 2");
        if (nvectors < 0 || nconsts < 0 || n < 0) throw std::invalid_argument("FormAssembler1D: negative count");
        if (rect_ && rank != 2) throw std::invalid_argument("FormAssembler1D: a rectangular form has rank 2");
    }
    bool rectangular() const { return rect_; }
    int columns() const { return ncols_; }

    int rank() const { return rank_; }
    int vectors() const { return nvec_; }
    int constants() const { return ncon_; }
    int values() const { return static_cast<int>(fields_.size()); }
    int terms() const { return static_cast<int>(terms_.size()); }
    int size() const { return n_; }

    // ---- point sets ---------------------------------------------------------------------
    /// @brief The quadrature nodes of Q (a dx term, Gauss or Lobatto).
    int add_cells(const QuadratureCache &Q)
    {
        PointSet s;
        s.kind = Cells; s.Q = &Q; s.x = Q.nodes();
        sets_.push_back(std::move(s));
        return static_cast<int>(sets_.size()) - 1;
    }
    /// @brief The interior facets of F (a dS term); x is the facet point seen from side '-'.
    int add_facets(const FacetCache &F)
    {
        PointSet s;
        s.kind = Facets; s.F = &F; s.x.resize(static_cast<std::size_t>(F.nf()));
        for (int f = 0; f < F.nf(); ++f) s.x[f] = F.x(f, 0);
        sets_.push_back(std::move(s));
        return static_cast<int>(sets_.size()) - 1;
    }
    /// @brief One end point of the interval (a ds term).
    int add_point(double x)
    {
        PointSet s;
        s.kind = Point; s.x.assign(1, x);
        sets_.push_back(std::move(s));
        return static_cast<int>(sets_.size()) - 1;
    }

    // ---- fields -------------------------------------------------------------------------
    //  The coefficients of a field are vectors[vec] itself (map empty) or its entries map[j]
    //  (a field of a ProductSpace).  Each add_* returns the index of the value array.

    /// @brief D^k u at the nodes of the cells set ps, read from Qf, a cache of the field's space V
    ///        with the same rule on the same grid (Field::at_quad).
    int add_quad_field(int ps, const FunctionSpace &V, const QuadratureCache &Qf, int vec, const std::vector<int> &map, int k)
    {
        const PointSet &s = set(ps);
        if (s.kind != Cells) throw std::invalid_argument("FormAssembler1D::add_quad_field: not a cells point set");
        if (Qf.nelem() != s.Q->nelem() || Qf.nq() != s.Q->nq()) throw std::invalid_argument("FormAssembler1D::add_quad_field: the cache differs from the point set's");
        if (k < 0 || k > Qf.nder()) throw std::invalid_argument("FormAssembler1D::add_quad_field: the cache holds fewer derivatives");
        FieldSpec f = spec(ps, V, vec, map, k);
        f.kind = Quad; f.Q = &Qf;
        fields_.push_back(std::move(f));
        return static_cast<int>(fields_.size()) - 1;
    }
    /// @brief D^k u evaluated at the points of ps by locating each point (a field on another grid,
    ///        or at an end point).
    int add_point_field(int ps, const FunctionSpace &V, int vec, const std::vector<int> &map, int k)
    {
        if (set(ps).kind == Facets) throw std::invalid_argument("FormAssembler1D::add_point_field: use add_facet_field on facets");
        if (k < 0) throw std::invalid_argument("FormAssembler1D::add_point_field: negative derivative");
        FieldSpec f = spec(ps, V, vec, map, k);
        f.kind = Points;
        fields_.push_back(std::move(f));
        return static_cast<int>(fields_.size()) - 1;
    }
    /// @brief D^k u on side s (0 '-', 1 '+') of the facets of ps, read from Ff, the facet cache of
    ///        the field's space V (facet_values).
    int add_facet_field(int ps, const FunctionSpace &V, const FacetCache &Ff, int vec, const std::vector<int> &map, int side, int k)
    {
        const PointSet &s = set(ps);
        if (s.kind != Facets) throw std::invalid_argument("FormAssembler1D::add_facet_field: not a facet point set");
        if (Ff.nf() != s.F->nf()) throw std::invalid_argument("FormAssembler1D::add_facet_field: the facet caches differ");
        detail::check_facet_args(Ff, side, k);
        FieldSpec f = spec(ps, V, vec, map, k);
        f.kind = FacetVals; f.Ff = &Ff; f.side = side;
        fields_.push_back(std::move(f));
        return static_cast<int>(fields_.size()) - 1;
    }

    // ---- terms --------------------------------------------------------------------------
    /**
     * @brief A dx term on the cells set ps (whose cache is the test cache, or the node cache of a
     *        rank-0 form).  Rank 1: f += integral c D^a N_i, through the product numbering of P
     *        (field I) when P is given.  Rank 2: K += integral c (D^a N_i)(D^b N_j) on the cache
     *        (P null), or block (I, J) with the trial cache QS.
     */
    void add_cell_term(int ps, const Integrand &prog, const std::vector<int> &inputs, const std::vector<int> &consts,
                       int a, int b, const ProductSpace *P, int I, int J, const QuadratureCache *QS)
    {
        const PointSet &s = set(ps);
        if (s.kind != Cells) throw std::invalid_argument("FormAssembler1D::add_cell_term: not a cells point set");
        if (rank_ >= 1 && (a < 0 || a > s.Q->nder())) throw std::invalid_argument("FormAssembler1D::add_cell_term: test derivative not in the cache");
        if (rank_ == 2)
        {
            if ((P || rect_) && QS == nullptr) throw std::invalid_argument("FormAssembler1D::add_cell_term: a block or a rectangular form needs the trial cache");
            if (rect_ && P) throw std::invalid_argument("FormAssembler1D::add_cell_term: a rectangular form has no product space");
            const QuadratureCache &Qb = (P || rect_) ? *QS : *s.Q;
            if (b < 0 || b > Qb.nder()) throw std::invalid_argument("FormAssembler1D::add_cell_term: trial derivative not in the cache");
            if (rect_ && (QS->nelem() != s.Q->nelem() || QS->nq() != s.Q->nq())) throw std::invalid_argument("FormAssembler1D::add_cell_term: the trial cache has another grid or rule");
        }
        if (P && (I < 0 || I >= P->nfields() || (rank_ == 2 && (J < 0 || J >= P->nfields()))))
            throw std::invalid_argument("FormAssembler1D::add_cell_term: field out of range");
        if (rank_ >= 1 && (P ? P->dim() : static_cast<int>(n_)) != n_) throw std::invalid_argument("FormAssembler1D::add_cell_term: the product space's dimension != n");
        Term t = term(ps, prog, inputs, consts);
        t.a = a; t.b = b; t.P = P; t.I = I; t.J = J; t.QS = QS;
        terms_.push_back(std::move(t));
    }
    /**
     * @brief A dS term on the facets set ps: test side st of FT, trial side ss of FS (rank 2).
     *        Rank 1: the facet vector of length nt (the test space's dimension), added at f[map[i]]
     *        (map empty: at f[i]).  Rank 2: into K (P null), or block (I, J) through P, which clears
     *        K.symmetric when clear_symmetric.
     */
    void add_facet_term(int ps, const Integrand &prog, const std::vector<int> &inputs, const std::vector<int> &consts,
                        int st, int ss, int a, int b, int nt, const std::vector<int> &map,
                        const FacetCache *FS, const ProductSpace *P, int I, int J, bool clear_symmetric)
    {
        const PointSet &s = set(ps);
        if (s.kind != Facets) throw std::invalid_argument("FormAssembler1D::add_facet_term: not a facet point set");
        if (rank_ >= 1) detail::check_facet_args(*s.F, st, a);
        if (rank_ == 2)
        {
            if (FS == nullptr) throw std::invalid_argument("FormAssembler1D::add_facet_term: a rank-2 term needs the trial facet cache");
            detail::check_facet_args(*FS, ss, b);
            if (FS->nf() != s.F->nf()) throw std::invalid_argument("FormAssembler1D::add_facet_term: the facet caches differ");
        }
        if (rank_ == 1)
        {
            if (!map.empty() && static_cast<int>(map.size()) != nt) throw std::invalid_argument("FormAssembler1D::add_facet_term: map length != nt");
            for (int g : map) if (g < 0 || g >= n_) throw std::invalid_argument("FormAssembler1D::add_facet_term: map entry out of range");
            if (map.empty() && nt != n_) throw std::invalid_argument("FormAssembler1D::add_facet_term: nt != n");
        }
        Term t = term(ps, prog, inputs, consts);
        t.sa = st; t.sb = ss; t.a = a; t.b = b; t.nt = nt; t.map = map; t.FS = FS; t.P = P; t.I = I; t.J = J; t.clear_sym = clear_symmetric;
        terms_.push_back(std::move(t));
    }
    /**
     * @brief A ds term at the point of ps: rank 1 f[mapT[i]] += c D^a T_i(x); rank 2
     *        K[mapT[i], mapS[j]] += (c D^a T_i(x)) D^b S_j(x), T and S the test and trial spaces
     *        (maps empty: the indices themselves).
     */
    void add_point_term(int ps, const Integrand &prog, const std::vector<int> &inputs, const std::vector<int> &consts,
                        const FunctionSpace *T, int a, const std::vector<int> &mapT,
                        const FunctionSpace *S, int b, const std::vector<int> &mapS)
    {
        const PointSet &s = set(ps);
        if (s.kind != Point) throw std::invalid_argument("FormAssembler1D::add_point_term: not an end point");
        if (rank_ >= 1 && (T == nullptr || a < 0)) throw std::invalid_argument("FormAssembler1D::add_point_term: the test space and derivative");
        if (rank_ == 2 && (S == nullptr || b < 0)) throw std::invalid_argument("FormAssembler1D::add_point_term: the trial space and derivative");
        auto check_map = [](const std::vector<int> &m, const FunctionSpace *V, int n) {
            if (V == nullptr) return;
            if (!m.empty() && static_cast<int>(m.size()) != V->dim()) throw std::invalid_argument("FormAssembler1D::add_point_term: map length != dim");
            if (m.empty() && V->dim() > n) throw std::invalid_argument("FormAssembler1D::add_point_term: the space is larger than the matrix");
            for (int g : m) if (g < 0 || g >= n) throw std::invalid_argument("FormAssembler1D::add_point_term: map entry out of range");
        };
        if (rank_ >= 1) check_map(mapT, T, n_);
        if (rank_ == 2) check_map(mapS, S, ncols_);
        Term t = term(ps, prog, inputs, consts);
        t.T = T; t.a = a; t.map = mapT; t.S = S; t.b = b; t.mapS = mapS;
        terms_.push_back(std::move(t));
    }

    // ---- assembly -----------------------------------------------------------------------
    double assemble_scalar(const std::vector<VectorRef> &vectors, const std::vector<double> &consts) const
    {
        if (rank_ != 0) throw std::invalid_argument("FormAssembler1D::assemble_scalar: not a rank-0 form");
        Values V = evaluate_fields(vectors, consts);
        double out = 0.0;
        for (const Term &t : terms_)
        {
            const PointSet &s = sets_[static_cast<std::size_t>(t.ps)];
            std::vector<double> c = coefficient(t, V, consts);
            if (s.kind == Cells) out += femd::assemble_scalar(*s.Q, Coefficient(std::move(c)));
            else
            {
                double sum = 0.0;
                for (double v : c) sum += v;
                out += sum;
            }
        }
        return out;
    }

    /// @brief f (n_ values, zeroed here) = the vector of the form.
    void assemble_vector(const std::vector<VectorRef> &vectors, const std::vector<double> &consts, double *f, std::size_t n) const
    {
        if (rank_ != 1) throw std::invalid_argument("FormAssembler1D::assemble_vector: not a rank-1 form");
        if (n != static_cast<std::size_t>(n_)) throw std::invalid_argument("FormAssembler1D::assemble_vector: f has the wrong length");
        Values V = evaluate_fields(vectors, consts);
        for (std::size_t i = 0; i < n; ++i) f[i] = 0.0;
        std::vector<double> vals, raw;
        for (const Term &t : terms_)
        {
            const PointSet &s = sets_[static_cast<std::size_t>(t.ps)];
            std::vector<double> c = coefficient(t, V, consts);
            if (s.kind == Cells)
            {
                std::vector<double> tmp(n, 0.0);           // each term on its own, then added (as the Python driver did)
                if (t.P) femd::assemble_vector(*t.P, t.I, *s.Q, t.a, Coefficient(std::move(c)), tmp);
                else femd::assemble_vector(*s.Q, t.a, Coefficient(std::move(c)), tmp);
                for (std::size_t i = 0; i < n; ++i) f[i] += tmp[i];
            }
            else if (s.kind == Facets)
            {
                std::vector<double> tmp = assemble_facet_vector(*s.F, t.sa, t.a, c.data(), t.nt);
                if (t.map.empty()) for (int i = 0; i < t.nt; ++i) f[i] += tmp[static_cast<std::size_t>(i)];
                else for (int i = 0; i < t.nt; ++i) f[t.map[static_cast<std::size_t>(i)]] += tmp[static_cast<std::size_t>(i)];
            }
            else
            {
                const int e = t.T->mesh().element_of(s.x[0]);
                t.T->eval_on_element(e, s.x[0], t.a, vals, raw);
                const std::vector<int> &d = t.T->element_dofs(e);
                const int na = static_cast<int>(d.size());
                for (int k = 0; k < na; ++k)
                {
                    const int g = t.map.empty() ? d[k] : t.map[static_cast<std::size_t>(d[k])];
                    f[g] += c[0] * vals[static_cast<std::size_t>(t.a) * na + k];
                }
            }
        }
    }

    /**
     * @brief Rank 0 with complex values (the complex-step derivative): vectors and constants as
     *        real and imaginary parts (an imaginary vector may be null: a real field).  Fields and
     *        kernels are real and linear and act on the two parts separately; the coefficients are
     *        complex.  Term by term as assemble_scalar.
     */
    void assemble_scalar_complex(const std::vector<VectorRef> &vre, const std::vector<VectorRef> &vim,
                                 const std::vector<double> &cre, const std::vector<double> &cim, double &out_re, double &out_im) const
    {
        if (rank_ != 0) throw std::invalid_argument("FormAssembler1D::assemble_scalar_complex: not a rank-0 form");
        if (cim.size() != cre.size()) throw std::invalid_argument("FormAssembler1D: the constants' parts differ in length");
        Values Vr = evaluate_fields(vre, cre), Vi = evaluate_fields(vim, cre, true);
        out_re = 0.0; out_im = 0.0;
        std::vector<double> cr, ci;
        for (const Term &t : terms_)
        {
            const PointSet &s = sets_[static_cast<std::size_t>(t.ps)];
            coefficient_complex(t, Vr, Vi, cre, cim, cr, ci);
            if (s.kind == Cells)
            {
                out_re += femd::assemble_scalar(*s.Q, Coefficient(cr));
                out_im += femd::assemble_scalar(*s.Q, Coefficient(ci));
            }
            else
            {
                double sr = 0.0, si = 0.0;
                for (std::size_t q = 0; q < cr.size(); ++q) { sr += cr[q]; si += ci[q]; }
                out_re += sr; out_im += si;
            }
        }
    }

    /// @brief Rank 1 with complex values: f_re + i f_im (n values each, zeroed here), term by term.
    void assemble_vector_complex(const std::vector<VectorRef> &vre, const std::vector<VectorRef> &vim,
                                 const std::vector<double> &cre, const std::vector<double> &cim,
                                 double *f_re, double *f_im, std::size_t n) const
    {
        if (rank_ != 1) throw std::invalid_argument("FormAssembler1D::assemble_vector_complex: not a rank-1 form");
        if (n != static_cast<std::size_t>(n_)) throw std::invalid_argument("FormAssembler1D::assemble_vector_complex: f has the wrong length");
        if (cim.size() != cre.size()) throw std::invalid_argument("FormAssembler1D: the constants' parts differ in length");
        Values Vr = evaluate_fields(vre, cre), Vi = evaluate_fields(vim, cre, true);
        for (std::size_t i = 0; i < n; ++i) { f_re[i] = 0.0; f_im[i] = 0.0; }
        std::vector<double> cr, ci, vals, raw;
        for (const Term &t : terms_)
        {
            const PointSet &s = sets_[static_cast<std::size_t>(t.ps)];
            coefficient_complex(t, Vr, Vi, cre, cim, cr, ci);
            for (int part = 0; part < 2; ++part)
            {
                const std::vector<double> &c = part ? ci : cr;
                double *f = part ? f_im : f_re;
                if (s.kind == Cells)
                {
                    std::vector<double> tmp(n, 0.0);
                    if (t.P) femd::assemble_vector(*t.P, t.I, *s.Q, t.a, Coefficient(c), tmp);
                    else femd::assemble_vector(*s.Q, t.a, Coefficient(c), tmp);
                    for (std::size_t i = 0; i < n; ++i) f[i] += tmp[i];
                }
                else if (s.kind == Facets)
                {
                    std::vector<double> tmp = assemble_facet_vector(*s.F, t.sa, t.a, c.data(), t.nt);
                    if (t.map.empty()) for (int i = 0; i < t.nt; ++i) f[i] += tmp[static_cast<std::size_t>(i)];
                    else for (int i = 0; i < t.nt; ++i) f[t.map[static_cast<std::size_t>(i)]] += tmp[static_cast<std::size_t>(i)];
                }
                else
                {
                    const int e = t.T->mesh().element_of(s.x[0]);
                    t.T->eval_on_element(e, s.x[0], t.a, vals, raw);
                    const std::vector<int> &d = t.T->element_dofs(e);
                    const int na = static_cast<int>(d.size());
                    for (int k = 0; k < na; ++k)
                    {
                        const int g = t.map.empty() ? d[k] : t.map[static_cast<std::size_t>(d[k])];
                        f[g] += c[0] * vals[static_cast<std::size_t>(t.a) * na + k];
                    }
                }
            }
        }
    }

    /**
     * @brief The rectangular form as a CSRMatrix (n rows, the test space; ncols columns, the trial
     *        space): the triplets of every term in the order of the form (assemble_rect on the
     *        cells, facet_triplets on the facets, (c T_i) S_j at an end point), summed by
     *        from_triplets.
     */
    CSRMatrix assemble_rect(const std::vector<VectorRef> &vectors, const std::vector<double> &consts) const
    {
        if (!rect_) throw std::invalid_argument("FormAssembler1D::assemble_rect: not a rectangular form");
        Values V = evaluate_fields(vectors, consts);
        std::vector<int> I, J;
        std::vector<double> X, vt, vs, raw;
        for (const Term &t : terms_)
        {
            const PointSet &s = sets_[static_cast<std::size_t>(t.ps)];
            std::vector<double> c = coefficient(t, V, consts);
            if (s.kind == Cells) femd::assemble_rect(*s.Q, *t.QS, t.a, t.b, Coefficient(std::move(c)), I, J, X);
            else if (s.kind == Facets) facet_triplets(*s.F, *t.FS, t.sa, t.sb, t.a, t.b, c.data(), I, J, X);
            else
            {
                const double x = s.x[0];
                const int eT = t.T->mesh().element_of(x), eS = t.S->mesh().element_of(x);
                t.T->eval_on_element(eT, x, t.a, vt, raw);
                t.S->eval_on_element(eS, x, t.b, vs, raw);
                const std::vector<int> &dT = t.T->element_dofs(eT), &dS = t.S->element_dofs(eS);
                const int nT = static_cast<int>(dT.size()), nS = static_cast<int>(dS.size());
                for (int i = 0; i < nT; ++i)
                {
                    const double ci = c[0] * vt[static_cast<std::size_t>(t.a) * nT + i];
                    for (int j = 0; j < nS; ++j) { I.push_back(dT[i]); J.push_back(dS[j]); X.push_back(ci * vs[static_cast<std::size_t>(t.b) * nS + j]); }
                }
            }
        }
        return from_triplets(n_, ncols_, I, J, X);
    }

    /// @brief K += the matrix of the form (K of order n, created by the caller).
    void assemble_matrix(const std::vector<VectorRef> &vectors, const std::vector<double> &consts, AssembledMatrix &K) const
    {
        if (rank_ != 2) throw std::invalid_argument("FormAssembler1D::assemble_matrix: not a rank-2 form");
        if (rect_) throw std::invalid_argument("FormAssembler1D::assemble_matrix: a rectangular form; use assemble_rect");
        if (K.n != n_) throw std::invalid_argument("FormAssembler1D::assemble_matrix: K has the wrong order");
        Values V = evaluate_fields(vectors, consts);
        std::vector<double> vt, vs, raw;
        for (const Term &t : terms_)
        {
            const PointSet &s = sets_[static_cast<std::size_t>(t.ps)];
            std::vector<double> c = coefficient(t, V, consts);
            if (s.kind == Cells)
            {
                if (t.P) femd::assemble_matrix(*t.P, t.I, t.J, *s.Q, *t.QS, t.a, t.b, Coefficient(std::move(c)), K);
                else femd::assemble_matrix(*s.Q, t.a, t.b, Coefficient(std::move(c)), K);
            }
            else if (s.kind == Facets)
            {
                if (!t.P) assemble_facet_matrix(K, *s.F, *t.FS, t.sa, t.sb, t.a, t.b, c.data());
                else
                {
                    std::vector<int> r, cc;
                    std::vector<double> v;
                    facet_triplets(*s.F, *t.FS, t.sa, t.sb, t.a, t.b, c.data(), r, cc, v);
                    for (std::size_t k = 0; k < v.size(); ++k) K.add(t.P->global(t.I, r[k]), t.P->global(t.J, cc[k]), v[k]);
                    if (t.clear_sym) K.symmetric = false;
                }
            }
            else
            {
                const double x = s.x[0];
                const int eT = t.T->mesh().element_of(x), eS = t.S->mesh().element_of(x);
                t.T->eval_on_element(eT, x, t.a, vt, raw);
                t.S->eval_on_element(eS, x, t.b, vs, raw);
                const std::vector<int> &dT = t.T->element_dofs(eT), &dS = t.S->element_dofs(eS);
                const int nT = static_cast<int>(dT.size()), nS = static_cast<int>(dS.size());
                for (int i = 0; i < nT; ++i)
                {
                    const int gi = t.map.empty() ? dT[i] : t.map[static_cast<std::size_t>(dT[i])];
                    const double ci = c[0] * vt[static_cast<std::size_t>(t.a) * nT + i];
                    for (int j = 0; j < nS; ++j)
                    {
                        const int gj = t.mapS.empty() ? dS[j] : t.mapS[static_cast<std::size_t>(dS[j])];
                        K.add(gi, gj, ci * vs[static_cast<std::size_t>(t.b) * nS + j]);
                    }
                }
            }
        }
    }

private:
    enum SetKind { Cells, Facets, Point };
    enum FieldKind { Quad, Points, FacetVals };
    struct PointSet {
        SetKind kind = Cells;
        const QuadratureCache *Q = nullptr;
        const FacetCache *F = nullptr;
        std::vector<double> x;
    };
    struct FieldSpec {
        FieldKind kind = Quad;
        int ps = 0, vec = 0, k = 0, side = 0;
        const FunctionSpace *V = nullptr;
        const QuadratureCache *Q = nullptr;
        const FacetCache *Ff = nullptr;
        std::vector<int> map;
    };
    struct Term {
        int ps = 0;
        Integrand prog{std::vector<int>{0, 0, 0, 0}, 1, 1, 0};
        std::vector<int> inputs, consts;
        int a = 0, b = 0, sa = 0, sb = 0, I = 0, J = 0, nt = 0;
        bool clear_sym = false;
        const ProductSpace *P = nullptr;
        const QuadratureCache *QS = nullptr;
        const FacetCache *FS = nullptr;
        const FunctionSpace *T = nullptr, *S = nullptr;
        std::vector<int> map, mapS;
    };
    using Values = std::vector<std::vector<double>>;

    const PointSet &set(int ps) const
    {
        if (ps < 0 || ps >= static_cast<int>(sets_.size())) throw std::invalid_argument("FormAssembler1D: no such point set");
        return sets_[static_cast<std::size_t>(ps)];
    }
    FieldSpec spec(int ps, const FunctionSpace &V, int vec, const std::vector<int> &map, int k) const
    {
        if (vec < 0 || vec >= nvec_) throw std::invalid_argument("FormAssembler1D: vector index out of range");
        if (!map.empty() && static_cast<int>(map.size()) != V.dim()) throw std::invalid_argument("FormAssembler1D: map length != the field's dimension");
        for (int g : map) if (g < 0) throw std::invalid_argument("FormAssembler1D: negative map entry");
        FieldSpec f;
        f.ps = ps; f.V = &V; f.vec = vec; f.map = map; f.k = k;
        return f;
    }
    Term term(int ps, const Integrand &prog, const std::vector<int> &inputs, const std::vector<int> &consts) const
    {
        if (prog.needs_normal()) throw std::invalid_argument("FormAssembler1D: no normal on a 1D mesh");
        if (static_cast<int>(inputs.size()) != prog.inputs()) throw std::invalid_argument("FormAssembler1D: the program takes " + std::to_string(prog.inputs()) + " inputs");
        if (static_cast<int>(consts.size()) != prog.constants()) throw std::invalid_argument("FormAssembler1D: the program takes " + std::to_string(prog.constants()) + " constants");
        for (int i : inputs)
        {
            if (i < 0 || i >= static_cast<int>(fields_.size())) throw std::invalid_argument("FormAssembler1D: input out of range");
            if (fields_[static_cast<std::size_t>(i)].ps != ps) throw std::invalid_argument("FormAssembler1D: an input lives on another point set");
        }
        for (int k : consts) if (k < 0 || k >= ncon_) throw std::invalid_argument("FormAssembler1D: constant out of range");
        Term t;
        t.ps = ps; t.prog = prog; t.inputs = inputs; t.consts = consts;
        return t;
    }

    /// The field values; with allow_null a null vector (the imaginary part of a real field) leaves
    /// its value arrays empty, read as zeros.
    Values evaluate_fields(const std::vector<VectorRef> &vectors, const std::vector<double> &consts, bool allow_null = false) const
    {
        if (static_cast<int>(vectors.size()) != nvec_) throw std::invalid_argument("FormAssembler1D: " + std::to_string(nvec_) + " coefficient vectors expected");
        if (static_cast<int>(consts.size()) != ncon_) throw std::invalid_argument("FormAssembler1D: " + std::to_string(ncon_) + " constants expected");
        Values out(fields_.size());
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            const FieldSpec &f = fields_[i];
            const VectorRef &v = vectors[static_cast<std::size_t>(f.vec)];
            if (allow_null && v.p == nullptr) continue;
            const std::size_t dim = static_cast<std::size_t>(f.V->dim());
            std::vector<double> c(dim);
            if (f.map.empty())
            {
                if (v.p == nullptr || v.n != dim)
                    throw std::invalid_argument("FormAssembler1D: coefficient vector " + std::to_string(f.vec) + " has " + std::to_string(v.n) +
                                                " entries, the field's space has " + std::to_string(dim));
                std::copy(v.p, v.p + dim, c.begin());
            }
            else
                for (std::size_t j = 0; j < dim; ++j)
                {
                    const std::size_t g = static_cast<std::size_t>(f.map[j]);
                    if (v.p == nullptr || g >= v.n) throw std::invalid_argument("FormAssembler1D: coefficient vector " + std::to_string(f.vec) + " is too short");
                    c[j] = v.p[g];
                }
            const PointSet &s = sets_[static_cast<std::size_t>(f.ps)];
            if (f.kind == Quad) out[i] = Field<double>(*f.V, std::move(c)).at_quad(*f.Q, f.k);
            else if (f.kind == FacetVals) out[i] = facet_values(*f.Ff, c, f.side, f.k);
            else out[i] = at_points(*f.V, c, s.x, f.k);
        }
        return out;
    }

    /// d^k u at the points x (FunctionSpace.evaluate)
    static std::vector<double> at_points(const FunctionSpace &V, const std::vector<double> &c, const std::vector<double> &x, int k)
    {
        const long np = static_cast<long>(x.size());
        std::vector<double> out(x.size());
        FEMD_OMP_PARALLEL_IF(np > FEMD_OMP_THRESHOLD)
        {
            std::vector<double> vals, raw;
            FEMD_OMP_FOR
            for (long i = 0; i < np; ++i)
            {
                const int e = V.mesh().element_of(x[static_cast<std::size_t>(i)]);
                V.eval_on_element(e, x[static_cast<std::size_t>(i)], k, vals, raw);
                const std::vector<int> &d = V.element_dofs(e);
                const int na = static_cast<int>(d.size());
                double sum = 0.0;
                for (int kk = 0; kk < na; ++kk) sum += c[d[kk]] * vals[static_cast<std::size_t>(k) * na + kk];
                out[static_cast<std::size_t>(i)] = sum;
            }
        }
        return out;
    }

    void coefficient_complex(const Term &t, const Values &Vr, const Values &Vi, const std::vector<double> &cr,
                             const std::vector<double> &ci, std::vector<double> &out_re, std::vector<double> &out_im) const
    {
        const std::vector<double> &x = sets_[static_cast<std::size_t>(t.ps)].x;
        std::vector<const double *> ir(t.inputs.size()), ii(t.inputs.size());
        for (std::size_t i = 0; i < t.inputs.size(); ++i)
        {
            const std::size_t f = static_cast<std::size_t>(t.inputs[i]);
            ir[i] = Vr[f].data();
            ii[i] = Vi[f].empty() ? nullptr : Vi[f].data();
        }
        std::vector<double> kr(t.consts.size()), ki(t.consts.size());
        for (std::size_t k = 0; k < t.consts.size(); ++k) { kr[k] = cr[static_cast<std::size_t>(t.consts[k])]; ki[k] = ci[static_cast<std::size_t>(t.consts[k])]; }
        out_re.assign(x.size(), 0.0); out_im.assign(x.size(), 0.0);
        t.prog.evaluate_complex(x.size(), x.data(), x.data(), nullptr, nullptr, ir, ii, kr, ki, out_re.data(), out_im.data());
    }

    std::vector<double> coefficient(const Term &t, const Values &V, const std::vector<double> &consts) const
    {
        const std::vector<double> &x = sets_[static_cast<std::size_t>(t.ps)].x;
        std::vector<const double *> in(t.inputs.size());
        for (std::size_t i = 0; i < t.inputs.size(); ++i) in[i] = V[static_cast<std::size_t>(t.inputs[i])].data();
        std::vector<double> cv(t.consts.size());
        for (std::size_t k = 0; k < t.consts.size(); ++k) cv[k] = consts[static_cast<std::size_t>(t.consts[k])];
        std::vector<double> c(x.size(), 0.0);
        t.prog.evaluate(x.size(), x.data(), x.data(), nullptr, nullptr, in, cv, c.data());
        return c;
    }

    int rank_, nvec_, ncon_, n_, ncols_;
    bool rect_;
    std::vector<PointSet> sets_;
    std::vector<FieldSpec> fields_;
    std::vector<Term> terms_;
};

} // namespace femd

#endif // FEMD_FORMS_ASSEMBLER_1D_HPP
