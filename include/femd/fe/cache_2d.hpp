//
//  cache_2d.hpp  --  quadrature points of a 2D Lagrange space (triangles or quadrilaterals):
//  on cells (dx) or on boundary facets (ds), with the reference basis tabulated once.
//
//  An entry is a cell (dx) or a facet seen from its cell (ds).  For each entry e and point q
//  the cache holds the physical point, the weight (reference weight times |det J| for a cell,
//  times the edge length for a facet), J^{-1} at the point (so bilinear quadrilaterals, whose
//  Jacobian varies inside the cell, are handled exactly like affine triangles) and, on facets,
//  the outward unit normal.  The basis is tabulated on the reference cell only, once per VARIANT
//  (one for cells, one per local edge for facets).  Physical derivatives are J^{-T} times the
//  reference ones, formed on the fly by row().  Per-point arrays are laid out [e*nq + q].
//
//  Rules: triangles use TriangleQuadrature(degree); quadrilaterals the tensor Gauss-Legendre rule
//  with (degree+3)/2 points per direction (one more than the degree needs, for the Jacobian).
//  Facets use Gauss-Legendre along the edge.
//
//  Derivative codes: 0 value, 1 d/dx, 2 d/dy; for a vector family (Raviart-Thomas, Nedelec)
//  3 c + d for component c.  Such a cache also keeps J and det J per point, for the Piola map,
//  and row() applies the cell's edge signs (Space2D::physical_row).
//
#ifndef FEMD_FE_CACHE_2D_HPP
#define FEMD_FE_CACHE_2D_HPP

#include "femd/fe/space_2d.hpp"
#include "femd/quadrature/triangle.hpp"
#include "femd/quadrature/gauss_legendre.hpp"
#include "femd/quadrature/gauss_lobatto.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace femd {

class Cache2D {
public:
    struct Lobatto {};

    /// @brief Quadrilateral cells with the tensor npts x npts Gauss-Lobatto rule (npts >= 2), exact to
    ///        degree 2 npts - 3 per direction.  On a Lobatto Q_k space with npts = k + 1 the points are
    ///        the nodes bit for bit, so the mass matrix is exactly diagonal (the lumped mass).
    Cache2D(const Space2D &V, int npts, Lobatto) : V_(&V), facets_(false), degree_(2 * npts - 3), lobatto_(true)
    {
        if (V.nverts() != 4)
            throw std::invalid_argument("Cache2D: the Gauss-Lobatto rule is a tensor rule for quadrilaterals; "
                                        "triangles have none");
        GaussLobatto g(npts);
        std::vector<double> t = V.points_1d();
        if (V.equispaced() || static_cast<int>(t.size()) != npts)
        {
            t.assign(static_cast<std::size_t>(npts), 0.0);
            for (int i = 0; i < npts; ++i) t[i] = 0.5 * (g.node(i) + 1.0);
            t[0] = 0.0; t[npts - 1] = 1.0;
        }
        std::vector<double> rx, ry, rw;
        for (int j = 0; j < npts; ++j)
            for (int i = 0; i < npts; ++i)
            {
                rx.push_back(t[i]); ry.push_back(t[j]);
                rw.push_back(0.25 * g.weight(i) * g.weight(j));
            }
        build_cells(V, rx, ry, rw);
    }

    /// @brief Cells, with a rule exact to total degree `degree` (per direction on quadrilaterals).
    Cache2D(const Space2D &V, int degree) : V_(&V), facets_(false), degree_(degree)
    {
        std::vector<double> rx, ry, rw;
        if (V.nverts() == 3)
        {
            TriangleQuadrature rule(degree);
            for (int q = 0; q < rule.size(); ++q) { rx.push_back(rule.xi(q)); ry.push_back(rule.eta(q)); rw.push_back(rule.weight(q)); }
        }
        else
        {
            GaussLegendre g(std::max(1, (degree + 3) / 2));
            for (int j = 0; j < g.size(); ++j)
                for (int i = 0; i < g.size(); ++i)
                {
                    rx.push_back(0.5 * (g.node(i) + 1.0)); ry.push_back(0.5 * (g.node(j) + 1.0));
                    rw.push_back(0.25 * g.weight(i) * g.weight(j));
                }
        }
        build_cells(V, rx, ry, rw);
    }

private:
    void build_cells(const Space2D &V, const std::vector<double> &rx, const std::vector<double> &ry,
                     const std::vector<double> &rw)
    {
        nq_ = static_cast<int>(rw.size());
        nloc_ = V.nloc();
        stride_ = V.ref_stride();
        vector_ = V.ncomp() > 1;
        ref_.assign(static_cast<std::size_t>(nq_) * stride_, 0.0);
        for (int q = 0; q < nq_; ++q) V.ref_eval(rx[q], ry[q], 1, &ref_[static_cast<std::size_t>(q) * stride_]);
        nent_ = V.ncells();
        cell_.resize(static_cast<std::size_t>(nent_));
        var_.assign(static_cast<std::size_t>(nent_), 0);
        const std::size_t np = static_cast<std::size_t>(nent_) * nq_;
        x_.resize(np); y_.resize(np); w_.resize(np); Ji_.resize(4 * np);
        if (vector_) { J_.resize(4 * np); det_.resize(np); }
        FEMD_OMP_FOR_IF(detail::parallel_elements(nent_, nq_))
        for (int c = 0; c < nent_; ++c)
        {
            cell_[c] = c;
            for (int q = 0; q < nq_; ++q)
            {
                const std::size_t k = static_cast<std::size_t>(c) * nq_ + q;
                double J[4];
                V.map(c, rx[q], ry[q], x_[k], y_[k], J);
                const double d = J[0] * J[3] - J[1] * J[2];
                w_[k] = rw[q] * std::abs(d);
                double *Ji = &Ji_[4 * k];
                Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
                Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
                if (vector_) { std::copy(J, J + 4, &J_[4 * k]); det_[k] = d; }
            }
        }
    }

public:
    /// @brief Exterior facets whose marker is in `markers` (all exterior facets when `all`),
    ///        with Gauss-Legendre points exact to degree `degree` along each edge.
    Cache2D(const Space2D &V, int degree, std::vector<int> markers, bool all)
        : V_(&V), facets_(true), degree_(degree)
    {
        std::sort(markers.begin(), markers.end());
        const int nv = V.nverts();
        GaussLegendre g(std::max(1, (degree + (nv == 4 ? 3 : 2)) / 2));
        nq_ = g.size();
        nloc_ = V.nloc();
        stride_ = V.ref_stride();
        vector_ = V.ncomp() > 1;
        ref_.assign(static_cast<std::size_t>(nv) * nq_ * stride_, 0.0);
        std::vector<double> fx(static_cast<std::size_t>(nv) * nq_), fy(fx.size());
        for (int i = 0; i < nv; ++i)
        {
            const auto ab = V.edge_vertices(i);
            double ax, ay, bx, by;
            V.ref_vertex(ab[0], ax, ay);
            V.ref_vertex(ab[1], bx, by);
            for (int q = 0; q < nq_; ++q)
            {
                const double t = 0.5 * (g.node(q) + 1.0);
                fx[i * nq_ + q] = (1.0 - t) * ax + t * bx;
                fy[i * nq_ + q] = (1.0 - t) * ay + t * by;
                V.ref_eval(fx[i * nq_ + q], fy[i * nq_ + q], 1, &ref_[(static_cast<std::size_t>(i) * nq_ + q) * stride_]);
            }
        }
        for (const Facet2D &f : V.facets())
        {
            if (!f.exterior) continue;
            if (!all && !std::binary_search(markers.begin(), markers.end(), f.marker)) continue;
            cell_.push_back(f.cell);
            var_.push_back(f.local);
            marker_.push_back(f.marker);
        }
        nent_ = static_cast<int>(cell_.size());
        const std::size_t np = static_cast<std::size_t>(nent_) * nq_;
        x_.resize(np); y_.resize(np); w_.resize(np); nx_.resize(np); ny_.resize(np); Ji_.resize(4 * np);
        if (vector_) { J_.resize(4 * np); det_.resize(np); }
        for (int e = 0; e < nent_; ++e)
        {
            const int c = cell_[e], i = var_[e];
            const auto ab = V.edge_vertices(i);
            const Point A = V.vertex(V.cell_vertex(c, ab[0])), B = V.vertex(V.cell_vertex(c, ab[1]));
            const double dx = B.x - A.x, dy = B.y - A.y, len = std::hypot(dx, dy);
            for (int q = 0; q < nq_; ++q)
            {
                const std::size_t k = static_cast<std::size_t>(e) * nq_ + q;
                double J[4];
                V.map(c, fx[i * nq_ + q], fy[i * nq_ + q], x_[k], y_[k], J);
                w_[k] = 0.5 * g.weight(q) * len;
                nx_[k] = dy / len; ny_[k] = -dx / len;         // counter-clockwise cell: outward is to the right
                const double d = J[0] * J[3] - J[1] * J[2];
                double *Ji = &Ji_[4 * k];
                Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
                Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
                if (vector_) { std::copy(J, J + 4, &J_[4 * k]); det_[k] = d; }
            }
        }
    }

    const Space2D &space() const { return *V_; }
    bool on_facets() const { return facets_; }
    int degree() const { return degree_; }
    bool lobatto() const { return lobatto_; }
    int nent() const { return nent_; }
    int nq()   const { return nq_; }
    int nloc() const { return nloc_; }
    int cell(int e) const { return cell_[e]; }
    int variant(int e) const { return var_[e]; }
    const std::vector<int> &markers() const { return marker_; }
    const std::vector<double> &x() const { return x_; }
    const std::vector<double> &y() const { return y_; }
    const std::vector<double> &weights() const { return w_; }
    const std::vector<double> &normal_x() const { return nx_; }
    const std::vector<double> &normal_y() const { return ny_; }
    double weight(int e, int q) const { return w_[static_cast<std::size_t>(e) * nq_ + q]; }
    const int *dofs(int e) const { return &V_->cell_dofs()[static_cast<std::size_t>(cell_[e]) * nloc_]; }
    const int *raw_dofs(int e) const { return &V_->cell_raw_dofs()[static_cast<std::size_t>(cell_[e]) * nloc_]; }

    const double *ref(int v, int q) const { return &ref_[(static_cast<std::size_t>(v) * nq_ + q) * stride_]; }
    /// @brief Number of derivative codes of the space (3, or 6 for a vector family).
    int ncodes() const { return V_->ncodes(); }

    /// @brief Physical row of derivative code m (0 value, 1 d/dx, 2 d/dy; 3 c + d for a vector
    ///        family) at point q of entry e.
    void row(int e, int q, int m, double *out) const
    {
        const double *R = ref(var_[e], q);
        if (vector_)
        {
            const std::size_t k = static_cast<std::size_t>(e) * nq_ + q;
            V_->physical_row(cell_[e], R, &J_[4 * k], &Ji_[4 * k], det_[k], m, out);
            return;
        }
        const int n = nloc_;
        if (m == 0) { std::copy(R, R + n, out); return; }
        const double *Ji = &Ji_[4 * (static_cast<std::size_t>(e) * nq_ + q)];
        const double a = (m == 1) ? Ji[0] : Ji[1];
        const double b = (m == 1) ? Ji[2] : Ji[3];
        for (int l = 0; l < n; ++l) out[l] = a * R[n + l] + b * R[2 * n + l];
    }

    std::vector<double> at_points(const std::vector<double> &c, int m) const
    {
        if (static_cast<int>(c.size()) != V_->dim()) throw std::invalid_argument("at_points: coefficient length != dim");
        return eval(c, m, false);
    }
    std::vector<double> at_points_raw(const std::vector<double> &r, int m) const
    {
        if (static_cast<int>(r.size()) != V_->raw_dim()) throw std::invalid_argument("at_points_raw: coefficient length != raw_dim");
        return eval(r, m, true);
    }

private:
    std::vector<double> eval(const std::vector<double> &c, int m, bool raw) const
    {
        if (m < 0 || m >= V_->ncodes())
            throw std::invalid_argument(vector_ ? "derivative code is 3 c + d: component c, d = 0 (value), 1 (d/dx), 2 (d/dy)"
                                                : "derivative code is 0 (value), 1 (d/dx) or 2 (d/dy)");
        std::vector<double> out(static_cast<std::size_t>(nent_) * nq_, 0.0);
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(nent_, nq_))
        {
        std::vector<double> rowbuf(static_cast<std::size_t>(nloc_)), loc(static_cast<std::size_t>(nloc_));
        FEMD_OMP_FOR
        for (int e = 0; e < nent_; ++e)
        {
            const int *d = raw ? raw_dofs(e) : dofs(e);
            for (int l = 0; l < nloc_; ++l) loc[l] = d[l] >= 0 ? c[d[l]] : 0.0;
            for (int q = 0; q < nq_; ++q)
            {
                row(e, q, m, rowbuf.data());
                double s = 0.0;
                for (int l = 0; l < nloc_; ++l) s += rowbuf[l] * loc[l];
                out[static_cast<std::size_t>(e) * nq_ + q] = s;
            }
        }
        }
        return out;
    }

    const Space2D *V_;
    bool facets_;
    int degree_, nent_ = 0, nq_ = 0, nloc_ = 0, stride_ = 0;
    bool lobatto_ = false, vector_ = false;
    std::vector<double> ref_;
    std::vector<int> cell_, var_, marker_;
    std::vector<double> x_, y_, w_, nx_, ny_, Ji_, J_, det_;
};

} // namespace femd

#endif // FEMD_FE_CACHE_2D_HPP
