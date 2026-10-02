//
//  facet_cache_2d.hpp  --  quadrature on the interior facets of a 2D space (the dS measure):
//  the points of every interior facet, seen from both cells, with the basis of each side.
//
//  Facet f joins cell[0] (side 0, '-') and cell[1] (side 1, '+'), Space2D::interior_facets().
//  Its points are the Gauss-Legendre points of the '-' cell's local edge, run from that edge's
//  first vertex; on the '+' side the same physical points (moved by the shift of a periodic
//  facet) are reached along the '+' cell's local edge, in the same or the reverse direction.  The
//  reference basis is tabulated once per variant (local edge, direction), six in all on
//  triangles, so the cache stores per facet only its cells, variants and maps.
//
//  Per point: the physical point (on the '-' side), the weight (Gauss weight times the edge
//  length) and the unit normal, outward from the '-' cell.  Rows are physical (Space2D::
//  physical_row), so scalar and vector families (Piola maps, edge signs) work alike.  Per-point
//  arrays are laid out [f*nq + q].  Derivative codes as in Cache2D.
//
#ifndef FEMD_FE_FACET_CACHE_2D_HPP
#define FEMD_FE_FACET_CACHE_2D_HPP

#include "femd/fe/space_2d.hpp"
#include "femd/quadrature/gauss_legendre.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class InteriorFacetCache2D {
public:
    /// @brief The interior facets of V with a Gauss rule exact to `degree` along each edge.
    InteriorFacetCache2D(const Space2D &V, int degree) : V_(&V), degree_(degree)
    {
        if (!V.affine())
            throw std::invalid_argument("InteriorFacetCache2D: interior facets are implemented for affine cells (triangles)");
        const int nv = V.nverts();
        GaussLegendre g(std::max(1, (degree + 2) / 2));
        nq_ = g.size();
        nloc_ = V.nloc();
        stride_ = V.ref_stride();
        std::vector<double> t(static_cast<std::size_t>(nq_)), w(t.size());
        for (int q = 0; q < nq_; ++q) { t[q] = 0.5 * (g.node(q) + 1.0); w[q] = 0.5 * g.weight(q); }
        // variants: (local edge e, direction d), points at t (d = 0) or 1 - t (d = 1) from its first vertex
        nvar_ = 2 * nv;
        ref_.assign(static_cast<std::size_t>(nvar_) * nq_ * stride_, 0.0);
        for (int e = 0; e < nv; ++e)
        {
            const auto ab = V.edge_vertices(e);
            double ax, ay, bx, by;
            V.ref_vertex(ab[0], ax, ay);
            V.ref_vertex(ab[1], bx, by);
            for (int d = 0; d < 2; ++d)
                for (int q = 0; q < nq_; ++q)
                {
                    const double s = d == 0 ? t[q] : 1.0 - t[q];
                    V.ref_eval(ax + s * (bx - ax), ay + s * (by - ay), 1,
                               &ref_[(static_cast<std::size_t>(2 * e + d) * nq_ + q) * stride_]);
                }
        }
        const auto &F = V.interior_facets();
        nf_ = static_cast<int>(F.size());
        const std::size_t np = static_cast<std::size_t>(nf_) * nq_;
        x_.resize(np); y_.resize(np); w_.resize(np); nx_.resize(np); ny_.resize(np);
        cell_.resize(2 * static_cast<std::size_t>(nf_));
        var_.resize(cell_.size());
        J_.resize(8 * static_cast<std::size_t>(nf_)); Ji_.resize(J_.size()); det_.resize(cell_.size());
        std::atomic<bool> bad{false};
        FEMD_OMP_FOR_IF(nf_ > FEMD_OMP_THRESHOLD)
        for (int f = 0; f < nf_; ++f)
        {
            const InteriorFacet2D &fa = F[f];
            for (int s = 0; s < 2; ++s)
            {
                const int c = fa.cell[s];
                cell_[2 * f + s] = c;
                double xx, yy, *J = &J_[8 * static_cast<std::size_t>(f) + 4 * s], *Ji = &Ji_[8 * static_cast<std::size_t>(f) + 4 * s];
                V.map(c, 1.0 / 3.0, 1.0 / 3.0, xx, yy, J);
                const double d = J[0] * J[3] - J[1] * J[2];
                det_[2 * f + s] = d;
                Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
                Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
            }
            // the '-' side's edge, from its first vertex
            const auto ab = V.edge_vertices(fa.local[0]);
            const Point A = V.vertex(V.cell_vertex(fa.cell[0], ab[0])), B = V.vertex(V.cell_vertex(fa.cell[0], ab[1]));
            const double dx = B.x - A.x, dy = B.y - A.y, len = std::hypot(dx, dy);
            var_[2 * f] = 2 * fa.local[0];
            // the '+' side: the same points, moved by the shift; forward when A lands on its first vertex
            const auto cd = V.edge_vertices(fa.local[1]);
            const Point C = V.vertex(V.cell_vertex(fa.cell[1], cd[0])), D = V.vertex(V.cell_vertex(fa.cell[1], cd[1]));
            const double Ax = A.x + fa.shift[0], Ay = A.y + fa.shift[1], Bx = B.x + fa.shift[0], By = B.y + fa.shift[1];
            const double fwd = std::hypot(Ax - C.x, Ay - C.y) + std::hypot(Bx - D.x, By - D.y);
            const double rev = std::hypot(Ax - D.x, Ay - D.y) + std::hypot(Bx - C.x, By - C.y);
            if (std::min(fwd, rev) > 1e-8 * std::max(len, 1e-300)) bad.store(true, std::memory_order_relaxed);
            var_[2 * f + 1] = 2 * fa.local[1] + (rev < fwd ? 1 : 0);
            for (int q = 0; q < nq_; ++q)
            {
                const std::size_t k = static_cast<std::size_t>(f) * nq_ + q;
                x_[k] = A.x + t[q] * dx; y_[k] = A.y + t[q] * dy;
                w_[k] = w[q] * len;
                nx_[k] = dy / len; ny_[k] = -dx / len;            // outward from the counter-clockwise '-' cell
            }
        }
        if (bad.load()) throw std::invalid_argument("InteriorFacetCache2D: the two sides of a facet do not match (a periodic pair "
                                             "that is not a translate)");
    }

    const Space2D &space() const { return *V_; }
    int degree() const { return degree_; }
    int nf() const { return nf_; }
    int nq() const { return nq_; }
    int nloc() const { return nloc_; }
    int ncodes() const { return V_->ncodes(); }
    int cell(int f, int s) const { return cell_[2 * f + s]; }
    const std::vector<double> &x() const { return x_; }
    const std::vector<double> &y() const { return y_; }
    const std::vector<double> &weights() const { return w_; }
    const std::vector<double> &normal_x() const { return nx_; }
    const std::vector<double> &normal_y() const { return ny_; }
    const int *dofs(int f, int s) const { return &V_->cell_dofs()[static_cast<std::size_t>(cell_[2 * f + s]) * nloc_]; }
    const int *raw_dofs(int f, int s) const { return &V_->cell_raw_dofs()[static_cast<std::size_t>(cell_[2 * f + s]) * nloc_]; }

    /// @brief Physical row of derivative code m at point q of facet f, side s (0 '-', 1 '+').
    void row(int f, int q, int s, int m, double *out) const
    {
        const double *R = &ref_[(static_cast<std::size_t>(var_[2 * f + s]) * nq_ + q) * stride_];
        const std::size_t o = 8 * static_cast<std::size_t>(f) + 4 * s;
        V_->physical_row(cell_[2 * f + s], R, &J_[o], &Ji_[o], det_[2 * f + s], m, out);
    }

    /// @brief Values of code m on side s at every point, for adapted (raw = false) or raw coefficients.
    std::vector<double> at_points(const std::vector<double> &c, int s, int m, bool raw = false) const
    {
        if (static_cast<int>(c.size()) != (raw ? V_->raw_dim() : V_->dim()))
            throw std::invalid_argument("InteriorFacetCache2D::at_points: coefficient length does not match the space");
        if (s < 0 || s > 1) throw std::invalid_argument("InteriorFacetCache2D::at_points: side is 0 ('-') or 1 ('+')");
        if (m < 0 || m >= V_->ncodes()) throw std::invalid_argument("InteriorFacetCache2D::at_points: derivative code out of range");
        std::vector<double> out(static_cast<std::size_t>(nf_) * nq_, 0.0);
        FEMD_OMP_PARALLEL_IF(detail::parallel_elements(nf_, nq_))
        {
        std::vector<double> r(static_cast<std::size_t>(nloc_)), loc(r.size());
        FEMD_OMP_FOR
        for (int f = 0; f < nf_; ++f)
        {
            const int *d = raw ? raw_dofs(f, s) : dofs(f, s);
            for (int l = 0; l < nloc_; ++l) loc[l] = d[l] >= 0 ? c[d[l]] : 0.0;
            for (int q = 0; q < nq_; ++q)
            {
                row(f, q, s, m, r.data());
                double v = 0.0;
                for (int l = 0; l < nloc_; ++l) v += r[l] * loc[l];
                out[static_cast<std::size_t>(f) * nq_ + q] = v;
            }
        }
        }
        return out;
    }

private:
    const Space2D *V_;
    int degree_, nf_ = 0, nq_ = 0, nloc_ = 0, stride_ = 0, nvar_ = 0;
    std::vector<double> ref_;
    std::vector<int> cell_, var_;
    std::vector<double> x_, y_, w_, nx_, ny_, J_, Ji_, det_;
};

} // namespace femd

#endif // FEMD_FE_FACET_CACHE_2D_HPP
