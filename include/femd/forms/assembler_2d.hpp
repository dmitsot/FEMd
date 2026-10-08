//
//  assembler_2d.hpp  --  a whole 2D form assembled in C++: the known fields evaluated at the
//  quadrature points, the coefficient of every term computed by its Integrand program, and
//  the cell, boundary-facet and interior-facet kernels called, with no Python in the loop.
//
//  The Python Form compiles itself once into a FormAssembler2D (Form._compile_2d): it names
//  the point sets (a Cache2D of cells or boundary facets, or an InteriorFacetCache2D for dS),
//  the fields read at each of them (which coefficient vector, which block of it, which side,
//  which derivative codes) and, per group of terms sharing a point set and a test / trial
//  block, the terms (program, inputs, constants, derivative codes and sides).  Each call of
//  assemble_* then takes the current coefficient vectors and constants and fills a scalar,
//  a vector or a CSRMatrix.  The object is immutable once built and assemble_* is const and
//  keeps every buffer on the stack, so several threads may assemble the same form with
//  different data at once (the stage residuals of an implicit Runge-Kutta step).
//
//  The arithmetic is that of the former Python driver, term by term and group by group, so
//  results agree bit for bit with it; the kernels themselves are unchanged (assembly_2d.hpp,
//  facet_assembly_2d.hpp).
//
#ifndef FEMD_FORMS_ASSEMBLER_2D_HPP
#define FEMD_FORMS_ASSEMBLER_2D_HPP

#include "femd/fe/cache_2d.hpp"
#include "femd/fe/facet_cache_2d.hpp"
#include "femd/forms/assembly_2d.hpp"
#include "femd/forms/facet_assembly_2d.hpp"
#include "femd/forms/integrand.hpp"
#include "femd/sparse/csr.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class FormAssembler2D {
public:
    /// @brief A coefficient vector handed to assemble_*: n values at p.
    struct VectorRef { const double *p; std::size_t n; };

    FormAssembler2D(int rank, int nvectors, int nconsts) : rank_(rank), nvec_(nvectors), ncon_(nconsts)
    {
        if (rank < 0 || rank > 2) throw std::invalid_argument("FormAssembler2D: rank is 0, 1 or 2");
        if (nvectors < 0 || nconsts < 0) throw std::invalid_argument("FormAssembler2D: negative count");
    }

    int rank() const { return rank_; }
    int vectors() const { return nvec_; }
    int constants() const { return ncon_; }
    int values() const { return nval_; }
    int groups() const { return static_cast<int>(groups_.size()); }

    /**
     * @brief A known field at the points of Q: the codes' values of vectors[vec][offset : offset + dim]
     *        (the field's block of a system vector).  Returns the index of the first value array;
     *        code i is value array first + i.
     */
    int add_field(const Cache2D &Q, int vec, int offset, int dim, const std::vector<int> &codes)
    {
        check_field(vec, offset, dim, codes, Q.ncodes(), Q.space().dim());
        Field f;
        f.Q = &Q; f.vec = vec; f.offset = offset; f.dim = dim; f.codes = codes; f.first = nval_;
        f.npts = static_cast<std::size_t>(Q.nent()) * Q.nq();
        fields_.push_back(f);
        nval_ += static_cast<int>(codes.size());
        return f.first;
    }
    /// @brief A known field on side s (0 '-', 1 '+') of the interior facets F.
    int add_field(const InteriorFacetCache2D &F, int vec, int offset, int dim, int side, const std::vector<int> &codes)
    {
        check_field(vec, offset, dim, codes, F.ncodes(), F.space().dim());
        if (side != 0 && side != 1) throw std::invalid_argument("FormAssembler2D::add_field: side is 0 ('-') or 1 ('+')");
        Field f;
        f.F = &F; f.vec = vec; f.offset = offset; f.dim = dim; f.side = side; f.codes = codes; f.first = nval_;
        f.npts = static_cast<std::size_t>(F.nf()) * F.nq();
        fields_.push_back(f);
        nval_ += static_cast<int>(codes.size());
        return f.first;
    }

    /**
     * @brief A group of terms on the cells or boundary facets of QT (the test block's cache; for a
     *        rank-0 form the cache of the point set), with QS the trial block's cache (rank 2, same
     *        entries and rule), rows from toff and columns from roff.  Returns the group index.
     */
    int add_group(const Cache2D &QT, const Cache2D *QS, int toff, int roff)
    {
        if (rank_ == 2 && QS == nullptr) throw std::invalid_argument("FormAssembler2D::add_group: a rank-2 form needs the trial cache");
        if (QS != nullptr && (QT.nent() != QS->nent() || QT.nq() != QS->nq() || QT.on_facets() != QS->on_facets()))
            throw std::invalid_argument("FormAssembler2D::add_group: test and trial caches differ (entries, points or kind)");
        if (toff < 0 || roff < 0) throw std::invalid_argument("FormAssembler2D::add_group: negative offset");
        Group g;
        g.QT = &QT; g.QS = QS; g.toff = toff; g.roff = roff;
        g.npts = static_cast<std::size_t>(QT.nent()) * QT.nq();
        groups_.push_back(g);
        return static_cast<int>(groups_.size()) - 1;
    }
    /// @brief A group of dS terms on the interior facets FT (test) and FS (trial, rank 2).
    int add_group(const InteriorFacetCache2D &FT, const InteriorFacetCache2D *FS, int toff, int roff)
    {
        if (rank_ == 2 && FS == nullptr) throw std::invalid_argument("FormAssembler2D::add_group: a rank-2 form needs the trial cache");
        if (FS != nullptr && (FT.nf() != FS->nf() || FT.nq() != FS->nq()))
            throw std::invalid_argument("FormAssembler2D::add_group: test and trial facet caches differ");
        if (toff < 0 || roff < 0) throw std::invalid_argument("FormAssembler2D::add_group: negative offset");
        Group g;
        g.FT = &FT; g.FS = FS; g.toff = toff; g.roff = roff;
        g.npts = static_cast<std::size_t>(FT.nf()) * FT.nq();
        groups_.push_back(g);
        return static_cast<int>(groups_.size()) - 1;
    }

    /**
     * @brief A term of group g: coefficient = prog(inputs = the value arrays named, constants =
     *        consts[k] for the indices named), test derivative code a on side sa, trial code b on
     *        side sb (sides matter for dS groups only).
     */
    void add_term(int g, const Integrand &prog, const std::vector<int> &inputs, const std::vector<int> &consts,
                  int a, int b, int sa, int sb)
    {
        if (g < 0 || g >= static_cast<int>(groups_.size())) throw std::invalid_argument("FormAssembler2D::add_term: no such group");
        Group &G = groups_[g];
        if (static_cast<int>(inputs.size()) != prog.inputs()) throw std::invalid_argument("FormAssembler2D::add_term: the program takes "
                                                                                           + std::to_string(prog.inputs()) + " inputs");
        if (static_cast<int>(consts.size()) != prog.constants()) throw std::invalid_argument("FormAssembler2D::add_term: the program takes "
                                                                                             + std::to_string(prog.constants()) + " constants");
        for (int i : inputs)
        {
            if (i < 0 || i >= nval_) throw std::invalid_argument("FormAssembler2D::add_term: input out of range");
            if (value_npts(i) != G.npts) throw std::invalid_argument("FormAssembler2D::add_term: an input lives on another point set");
        }
        for (int k : consts)
            if (k < 0 || k >= ncon_) throw std::invalid_argument("FormAssembler2D::add_term: constant out of range");
        if (prog.needs_normal() && G.QT != nullptr && !G.QT->on_facets())
            throw std::invalid_argument("FormAssembler2D::add_term: the coefficient reads the normal on a cell group");
        const int ncT = G.QT ? G.QT->ncodes() : G.FT->ncodes();
        if (rank_ >= 1 && (a < 0 || a >= ncT)) throw std::invalid_argument("FormAssembler2D::add_term: test code out of range");
        if (rank_ == 2)
        {
            const int ncS = G.QS ? G.QS->ncodes() : G.FS->ncodes();
            if (b < 0 || b >= ncS) throw std::invalid_argument("FormAssembler2D::add_term: trial code out of range");
        }
        if ((sa != 0 && sa != 1) || (sb != 0 && sb != 1)) throw std::invalid_argument("FormAssembler2D::add_term: a side is 0 or 1");
        Term t;
        t.prog = prog; t.inputs = inputs; t.consts = consts; t.a = a; t.b = b; t.sa = sa; t.sb = sb;
        G.terms.push_back(std::move(t));
    }

    /// @brief Rank 0: the sum of the integrals.
    double assemble_scalar(const std::vector<VectorRef> &vectors, const std::vector<double> &consts) const
    {
        if (rank_ != 0) throw std::invalid_argument("FormAssembler2D::assemble_scalar: not a rank-0 form");
        Values V = evaluate_fields(vectors, consts);
        double out = 0.0;
        for (const Group &G : groups_)
        {
            std::vector<std::vector<double>> cs = coefficients(G, V, consts);
            for (const auto &c : cs)
                out += G.QT ? assemble_scalar_2d(*G.QT, c) : assemble_facet_scalar_2d(*G.FT, c);
        }
        return out;
    }

    /// @brief Rank 1: f (n values, zeroed here) += the vector of the form.
    void assemble_vector(const std::vector<VectorRef> &vectors, const std::vector<double> &consts, double *f, std::size_t n) const
    {
        if (rank_ != 1) throw std::invalid_argument("FormAssembler2D::assemble_vector: not a rank-1 form");
        for (const Group &G : groups_)
        {
            const std::size_t dim = static_cast<std::size_t>(G.QT ? G.QT->space().dim() : G.FT->space().dim());
            if (static_cast<std::size_t>(G.toff) + dim > n) throw std::invalid_argument("FormAssembler2D::assemble_vector: f is too short");
        }
        Values V = evaluate_fields(vectors, consts);
        for (std::size_t i = 0; i < n; ++i) f[i] = 0.0;
        for (const Group &G : groups_)
        {
            std::vector<std::vector<double>> cs = coefficients(G, V, consts);
            std::vector<int> a(G.terms.size()), sa(G.terms.size());
            for (std::size_t t = 0; t < G.terms.size(); ++t) { a[t] = G.terms[t].a; sa[t] = G.terms[t].sa; }
            const std::size_t dim = static_cast<std::size_t>(G.QT ? G.QT->space().dim() : G.FT->space().dim());
            std::vector<double> block(dim, 0.0);          // the group's own vector, added to f as the Python driver did
            if (G.QT) assemble_vector_2d(*G.QT, a, cs, block);
            else assemble_facet_vector_2d(*G.FT, sa, a, cs, block);
            double *fo = f + G.toff;
            for (std::size_t i = 0; i < dim; ++i) fo[i] += block[i];
        }
    }

    /**
     * @brief Rank 0 with complex values (the complex-step derivative): vectors and constants as
     *        real and imaginary parts (an imaginary vector may be null: a real field).  Every map
     *        from coefficients to values and every kernel is real and linear, so they act on the two
     *        parts separately; only the coefficients are complex.
     */
    void assemble_scalar_complex(const std::vector<VectorRef> &vre, const std::vector<VectorRef> &vim,
                                 const std::vector<double> &cre, const std::vector<double> &cim, double &out_re, double &out_im) const
    {
        if (rank_ != 0) throw std::invalid_argument("FormAssembler2D::assemble_scalar_complex: not a rank-0 form");
        if (cim.size() != cre.size()) throw std::invalid_argument("FormAssembler2D: the constants' parts differ in length");
        Values Vr = evaluate_fields(vre, cre), Vi = evaluate_fields(vim, cre, true);
        out_re = 0.0; out_im = 0.0;
        std::vector<std::vector<double>> cr, ci;
        for (const Group &G : groups_)
        {
            coefficients_complex(G, Vr, Vi, cre, cim, cr, ci);
            for (std::size_t t = 0; t < cr.size(); ++t)
            {
                out_re += G.QT ? assemble_scalar_2d(*G.QT, cr[t]) : assemble_facet_scalar_2d(*G.FT, cr[t]);
                out_im += G.QT ? assemble_scalar_2d(*G.QT, ci[t]) : assemble_facet_scalar_2d(*G.FT, ci[t]);
            }
        }
    }

    /// @brief Rank 1 with complex values: f_re + i f_im (n values each, zeroed here).
    void assemble_vector_complex(const std::vector<VectorRef> &vre, const std::vector<VectorRef> &vim,
                                 const std::vector<double> &cre, const std::vector<double> &cim,
                                 double *f_re, double *f_im, std::size_t n) const
    {
        if (rank_ != 1) throw std::invalid_argument("FormAssembler2D::assemble_vector_complex: not a rank-1 form");
        if (cim.size() != cre.size()) throw std::invalid_argument("FormAssembler2D: the constants' parts differ in length");
        for (const Group &G : groups_)
        {
            const std::size_t dim = static_cast<std::size_t>(G.QT ? G.QT->space().dim() : G.FT->space().dim());
            if (static_cast<std::size_t>(G.toff) + dim > n) throw std::invalid_argument("FormAssembler2D::assemble_vector_complex: f is too short");
        }
        Values Vr = evaluate_fields(vre, cre), Vi = evaluate_fields(vim, cre, true);
        for (std::size_t i = 0; i < n; ++i) { f_re[i] = 0.0; f_im[i] = 0.0; }
        std::vector<std::vector<double>> cr, ci;
        for (const Group &G : groups_)
        {
            coefficients_complex(G, Vr, Vi, cre, cim, cr, ci);
            std::vector<int> a(G.terms.size()), sa(G.terms.size());
            for (std::size_t t = 0; t < G.terms.size(); ++t) { a[t] = G.terms[t].a; sa[t] = G.terms[t].sa; }
            const std::size_t dim = static_cast<std::size_t>(G.QT ? G.QT->space().dim() : G.FT->space().dim());
            for (int part = 0; part < 2; ++part)
            {
                std::vector<double> block(dim, 0.0);
                const std::vector<std::vector<double>> &cs = part ? ci : cr;
                if (G.QT) assemble_vector_2d(*G.QT, a, cs, block);
                else assemble_facet_vector_2d(*G.FT, sa, a, cs, block);
                double *fo = (part ? f_im : f_re) + G.toff;
                for (std::size_t i = 0; i < dim; ++i) fo[i] += block[i];
            }
        }
    }

    /// @brief Rank 2: K += the matrix of the form (K zeroed by the caller, on the form's pattern).
    void assemble_matrix(const std::vector<VectorRef> &vectors, const std::vector<double> &consts, CSRMatrix &K) const
    {
        if (rank_ != 2) throw std::invalid_argument("FormAssembler2D::assemble_matrix: not a rank-2 form");
        Values V = evaluate_fields(vectors, consts);
        for (const Group &G : groups_)
        {
            std::vector<std::vector<double>> cs = coefficients(G, V, consts);
            const std::size_t nt = G.terms.size();
            std::vector<int> a(nt), b(nt), sa(nt), sb(nt);
            for (std::size_t t = 0; t < nt; ++t) { a[t] = G.terms[t].a; b[t] = G.terms[t].b; sa[t] = G.terms[t].sa; sb[t] = G.terms[t].sb; }
            if (G.QT) assemble_matrix_2d(*G.QT, *G.QS, a, b, cs, K, G.toff, G.roff);
            else assemble_facet_matrix_2d(*G.FT, *G.FS, sa, sb, a, b, cs, K, G.toff, G.roff);
        }
    }

private:
    struct Field {
        const Cache2D *Q = nullptr;
        const InteriorFacetCache2D *F = nullptr;
        int vec = 0, offset = 0, dim = 0, side = 0, first = 0;
        std::size_t npts = 0;
        std::vector<int> codes;
    };
    struct Term {
        Integrand prog{std::vector<int>{0, 0, 0, 0}, 1, 1, 0};
        std::vector<int> inputs, consts;
        int a = 0, b = 0, sa = 0, sb = 0;
    };
    struct Group {
        const Cache2D *QT = nullptr, *QS = nullptr;
        const InteriorFacetCache2D *FT = nullptr, *FS = nullptr;
        int toff = 0, roff = 0;
        std::size_t npts = 0;
        std::vector<Term> terms;
    };
    struct Values {
        std::vector<std::vector<double>> buf;      // one buffer per field, its codes one after the other
        std::vector<const double *> at;            // value array i -> npts doubles
    };

    void check_field(int vec, int offset, int dim, const std::vector<int> &codes, int ncodes, int space_dim) const
    {
        if (vec < 0 || vec >= nvec_) throw std::invalid_argument("FormAssembler2D::add_field: vector index out of range");
        if (offset < 0 || dim < 0) throw std::invalid_argument("FormAssembler2D::add_field: negative offset or dim");
        if (dim != space_dim) throw std::invalid_argument("FormAssembler2D::add_field: dim != the dimension of the field's space");
        if (codes.empty()) throw std::invalid_argument("FormAssembler2D::add_field: no derivative codes");
        for (int m : codes)
            if (m < 0 || m >= ncodes) throw std::invalid_argument("FormAssembler2D::add_field: derivative code out of range");
    }

    std::size_t value_npts(int i) const
    {
        for (const Field &f : fields_)
            if (i >= f.first && i < f.first + static_cast<int>(f.codes.size())) return f.npts;
        return 0;
    }

    /// @brief The field values; with allow_null a vector given as null (the imaginary part of a
    ///        real field) gives no values (null value arrays, read as zeros).
    Values evaluate_fields(const std::vector<VectorRef> &vectors, const std::vector<double> &consts, bool allow_null = false) const
    {
        if (static_cast<int>(vectors.size()) != nvec_) throw std::invalid_argument("FormAssembler2D: " + std::to_string(nvec_) + " coefficient vectors expected");
        if (static_cast<int>(consts.size()) != ncon_) throw std::invalid_argument("FormAssembler2D: " + std::to_string(ncon_) + " constants expected");
        Values V;
        V.buf.resize(fields_.size());
        V.at.assign(static_cast<std::size_t>(nval_), nullptr);
        for (std::size_t k = 0; k < fields_.size(); ++k)
        {
            const Field &f = fields_[k];
            const VectorRef &v = vectors[static_cast<std::size_t>(f.vec)];
            if (allow_null && v.p == nullptr) continue;                 // a real field: no imaginary part
            if (v.p == nullptr || static_cast<std::size_t>(f.offset) + f.dim > v.n)
                throw std::invalid_argument("FormAssembler2D: coefficient vector " + std::to_string(f.vec) + " has " + std::to_string(v.n) +
                                            " entries, the form reads " + std::to_string(f.offset + f.dim));
            const double *c = v.p + f.offset;
            std::vector<double> &buf = V.buf[k];
            buf.assign(f.codes.size() * f.npts, 0.0);
            if (f.Q != nullptr)
            {
                if (!f.Q->is_vector())
                    f.Q->eval_multi(c, false, f.codes.data(), static_cast<int>(f.codes.size()), buf.data());
                else
                    for (std::size_t i = 0; i < f.codes.size(); ++i) f.Q->eval_into(c, f.codes[i], buf.data() + i * f.npts);
            }
            else
                for (std::size_t i = 0; i < f.codes.size(); ++i) f.F->eval_into(c, f.side, f.codes[i], buf.data() + i * f.npts);
            for (std::size_t i = 0; i < f.codes.size(); ++i) V.at[static_cast<std::size_t>(f.first) + i] = buf.data() + i * f.npts;
        }
        return V;
    }

    /// The complex coefficients of the terms of G: (re, im) per term.
    void coefficients_complex(const Group &G, const Values &Vr, const Values &Vi, const std::vector<double> &cr,
                              const std::vector<double> &ci, std::vector<std::vector<double>> &out_re,
                              std::vector<std::vector<double>> &out_im) const
    {
        const double *xs, *ys, *nx = nullptr, *ny = nullptr;
        if (G.QT) { xs = G.QT->x().data(); ys = G.QT->y().data(); if (G.QT->on_facets()) { nx = G.QT->normal_x().data(); ny = G.QT->normal_y().data(); } }
        else { xs = G.FT->x().data(); ys = G.FT->y().data(); nx = G.FT->normal_x().data(); ny = G.FT->normal_y().data(); }
        out_re.assign(G.terms.size(), {}); out_im.assign(G.terms.size(), {});
        for (std::size_t t = 0; t < G.terms.size(); ++t)
        {
            const Term &T = G.terms[t];
            std::vector<const double *> ir(T.inputs.size()), ii(T.inputs.size());
            for (std::size_t i = 0; i < T.inputs.size(); ++i)
            {
                ir[i] = Vr.at[static_cast<std::size_t>(T.inputs[i])];
                ii[i] = Vi.at[static_cast<std::size_t>(T.inputs[i])];
            }
            std::vector<double> kr(T.consts.size()), ki(T.consts.size());
            for (std::size_t k = 0; k < T.consts.size(); ++k) { kr[k] = cr[static_cast<std::size_t>(T.consts[k])]; ki[k] = ci[static_cast<std::size_t>(T.consts[k])]; }
            out_re[t].assign(G.npts, 0.0); out_im[t].assign(G.npts, 0.0);
            T.prog.evaluate_complex(G.npts, xs, ys, nx, ny, ir, ii, kr, ki, out_re[t].data(), out_im[t].data());
        }
    }

    std::vector<std::vector<double>> coefficients(const Group &G, const Values &V, const std::vector<double> &consts) const
    {
        const double *xs, *ys, *nx = nullptr, *ny = nullptr;
        if (G.QT) { xs = G.QT->x().data(); ys = G.QT->y().data(); if (G.QT->on_facets()) { nx = G.QT->normal_x().data(); ny = G.QT->normal_y().data(); } }
        else { xs = G.FT->x().data(); ys = G.FT->y().data(); nx = G.FT->normal_x().data(); ny = G.FT->normal_y().data(); }
        std::vector<std::vector<double>> cs(G.terms.size());
        for (std::size_t t = 0; t < G.terms.size(); ++t)
        {
            const Term &T = G.terms[t];
            std::vector<const double *> in(T.inputs.size());
            for (std::size_t i = 0; i < T.inputs.size(); ++i) in[i] = V.at[static_cast<std::size_t>(T.inputs[i])];
            std::vector<double> cv(T.consts.size());
            for (std::size_t k = 0; k < T.consts.size(); ++k) cv[k] = consts[static_cast<std::size_t>(T.consts[k])];
            cs[t].assign(G.npts, 0.0);
            T.prog.evaluate(G.npts, xs, ys, nx, ny, in, cv, cs[t].data());
        }
        return cs;
    }

    int rank_, nvec_, ncon_, nval_ = 0;
    std::vector<Field> fields_;
    std::vector<Group> groups_;
};

} // namespace femd

#endif // FEMD_FORMS_ASSEMBLER_2D_HPP
