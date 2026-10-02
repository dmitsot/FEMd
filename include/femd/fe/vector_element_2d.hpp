//
//  vector_element_2d.hpp  --  the H(div) and H(curl) families on triangles, Raviart-Thomas RT_k
//  and Nedelec of the first kind N1curl_k, and the broken (discontinuous) P_k space DG_k that
//  is their partner in mixed methods.
//
//  Degree convention (UFL's): RT_k and N1curl_k for k >= 1 contain P_{k-1}^2 and lie in P_k^2;
//  RT_1 and N1curl_1 are the lowest-order elements (one function per edge).  dim RT_k per
//  triangle = k (k + 2): k per edge, k (k - 1) inside.
//
//      RT_k     = P_{k-1}^2 + x P~_{k-1}                (normal component continuous)
//      N1curl_k = P_{k-1}^2 + (-y, x) P~_{k-1}          (tangential component continuous)
//
//  P~_{k-1} the homogeneous polynomials of degree k-1.  On the reference triangle the spanning
//  set is (psi_m, 0), (0, psi_m) for the Dubiner modes psi_m of degree <= k-1, plus x psi_t
//  (RT) or (-y, x) psi_t (N1curl) for the modes psi_t of degree exactly k-1, whose top-degree
//  parts span P~_{k-1}.
//
//  Degrees of freedom (the basis is their dual, built by inverting the generalized Vandermonde
//  matrix V_im = l_i(p_m) as simplex_basis.hpp does for Lagrange):
//    edge e, j = 0..k-1:  l(phi) = phi(p_ej) . w_e, at the k Gauss-Legendre points p_ej of the edge
//        (counted from its first vertex), with w_e = rot(t_e) = (t_y, -t_x) for RT and w_e = t_e
//        for N1curl, t_e the edge vector (not normalized).  Under the Piola maps these are
//        invariant: phi . rot(J t) = phihat . rot(t) (contravariant) and phi . (J t) = phihat . t
//        (covariant), so the physical degree of freedom is the normal (tangential) component
//        times the edge length, at the point.  rot(t) is the outward normal times |t| on a
//        counter-clockwise triangle.  On RT_k (N1curl_k) the normal (tangential) trace is in
//        P_{k-1}, so these k values are equivalent to the moments against P_{k-1}(e).
//    inside, k >= 2:  l(phi) = int_T phihat_c psi_i, c = 0, 1, psi_i the Dubiner modes of degree
//        <= k-2, on the reference triangle (physical functions are pulled back first).
//  Interpolation (interpolate_raw) is the canonical, commuting one: on each edge the trace
//  u . w is projected in L^2 onto P_{k-1}(e), with a Gauss rule of k + 3 points, and the
//  projection's values at the k points are the coefficients; the interior moments of the
//  pulled-back field use a rule exact to degree 2k + 4.  So div (curl) of the interpolant is the
//  L^2 projection of div u (curl u) onto DG_{k-1}, to quadrature accuracy.
//
//  A global edge runs from its lower to its higher vertex index; its j-th degree of freedom sits
//  at the j-th point from the lower vertex, with w the vector of that direction.  A cell whose
//  local edge runs the other way reads the points in reverse, and its local function is minus
//  the global one (Space2D::sign).
//
//  Local order: the k functions of edge 0, of edge 1, of edge 2 (edge i opposite vertex i, as for
//  Lagrange), then the interior ones [3k + c nI + i].
//
#ifndef FEMD_FE_VECTOR_ELEMENT_2D_HPP
#define FEMD_FE_VECTOR_ELEMENT_2D_HPP

#include "femd/fe/simplex_basis.hpp"
#include "femd/fe/space_2d.hpp"
#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/point_locator.hpp"
#include "femd/quadrature/gauss_legendre.hpp"
#include "femd/quadrature/triangle.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

enum class VectorFamily { RT, N1curl };

/// @brief The reference RT_k or N1curl_k element on T = {(0,0), (1,0), (0,1)}.
class ReferenceVectorTriangle {
public:
    ReferenceVectorTriangle(VectorFamily fam, int k) : fam_(fam), k_(k)
    {
        if (k < 1) throw std::invalid_argument("ReferenceVectorTriangle: degree must be >= 1 (the lowest order is 1)");
        if (k > 12) throw std::invalid_argument("ReferenceVectorTriangle: degree " + std::to_string(k) + " is above the supported 12");
        n_ = k * (k + 2);
        nm_ = simplex::nmodes(k - 1);
        nI_ = k >= 2 ? simplex::nmodes(k - 2) : 0;
        {
            int m = 0;
            for (int i = 0; i <= k - 1; ++i)
                for (int j = 0; j <= k - 1 - i; ++j, ++m)
                    if (i + j == k - 1) top_.push_back(m);
        }
        if (2 * nm_ + static_cast<int>(top_.size()) != n_) throw std::logic_error("ReferenceVectorTriangle: spanning set count");
        // symmetric Gauss-Legendre points on [0, 1]
        GaussLegendre g(k);
        std::vector<double> z(static_cast<std::size_t>(k)), zw(z.size());
        for (int j = 0; j < k; ++j) { z[j] = g.node(j); zw[j] = g.weight(j); }
        if (k > 1 && z[0] > z[k - 1]) { std::reverse(z.begin(), z.end()); std::reverse(zw.begin(), zw.end()); }
        s_.resize(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) s_[j] = 0.5 * (1.0 + 0.5 * (z[j] - z[k - 1 - j]));
        // the edge projection for interpolation: dof_j = sum_q P[j, q] f(t_q), the L^2 projection
        // of f onto P_{k-1}[0, 1] evaluated at s_j (its mass matrix in the Lagrange basis at the
        // Gauss points is diag(w_j / 2), the k-point rule being exact to degree 2k - 1)
        {
            GaussLegendre ge(k + 3);
            nqE_ = ge.size();
            for (int q = 0; q < nqE_; ++q) tE_.push_back(0.5 * (ge.node(q) + 1.0));
            PE_.assign(static_cast<std::size_t>(k) * nqE_, 0.0);
            for (int j = 0; j < k; ++j)
            {
                const double wj = 0.5 * 0.5 * (zw[j] + zw[k - 1 - j]);
                for (int q = 0; q < nqE_; ++q)
                {
                    double L = 1.0;
                    for (int i = 0; i < k; ++i)
                        if (i != j) L *= (tE_[q] - s_[i]) / (s_[j] - s_[i]);
                    PE_[static_cast<std::size_t>(j) * nqE_ + q] = 0.5 * ge.weight(q) * L / wj;
                }
            }
        }
        // interior moments: a rule exact for the products (degree k + k - 2), and with room for
        // the interpolation of fields that are not polynomials
        if (nI_ > 0)
        {
            TriangleQuadrature rule(2 * k + 4);
            nqI_ = rule.size();
            std::vector<double> psi(static_cast<std::size_t>(nI_));
            Wint_.assign(static_cast<std::size_t>(nI_) * nqI_, 0.0);
            for (int q = 0; q < nqI_; ++q)
            {
                qx_.push_back(rule.xi(q)); qy_.push_back(rule.eta(q));
                simplex::dubiner(k - 2, rule.xi(q), rule.eta(q), psi.data(), nullptr, nullptr);
                for (int i = 0; i < nI_; ++i) Wint_[static_cast<std::size_t>(i) * nqI_ + q] = rule.weight(q) * psi[i];
            }
        }
        // the generalized Vandermonde matrix and its inverse
        std::vector<double> V(static_cast<std::size_t>(n_) * n_, 0.0), P(static_cast<std::size_t>(6) * n_);
        static const double vx[3] = {0.0, 1.0, 0.0}, vy[3] = {0.0, 0.0, 1.0};
        for (int e = 0; e < 3; ++e)
        {
            const int a = (e + 1) % 3, b = (e + 2) % 3;
            const double tx = vx[b] - vx[a], ty = vy[b] - vy[a];
            double w0, w1;
            weight(tx, ty, w0, w1);
            for (int j = 0; j < k; ++j)
            {
                span(vx[a] + s_[j] * tx, vy[a] + s_[j] * ty, P.data());
                double *row = &V[static_cast<std::size_t>(e * k + j) * n_];
                for (int m = 0; m < n_; ++m) row[m] = w0 * P[m] + w1 * P[3 * n_ + m];
            }
        }
        for (int q = 0; q < nqI_; ++q)
        {
            span(qx_[q], qy_[q], P.data());
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < nI_; ++i)
                {
                    const double w = Wint_[static_cast<std::size_t>(i) * nqI_ + q];
                    double *row = &V[static_cast<std::size_t>(3 * k + c * nI_ + i) * n_];
                    for (int m = 0; m < n_; ++m) row[m] += w * P[static_cast<std::size_t>(c) * 3 * n_ + m];
                }
        }
        C_ = simplex::inverse(V, n_);
    }

    VectorFamily family() const { return fam_; }
    int degree() const { return k_; }
    int size() const { return n_; }
    int n_interior_modes() const { return nI_; }
    /// @brief Point j (0..k-1) on [0, 1] of an edge, counted from its first vertex.
    double edge_point(int j) const { return s_[j]; }
    /// @brief The weight vector w of an edge with vector t: rot(t) (RT) or t (N1curl).
    void weight(double tx, double ty, double &w0, double &w1) const
    {
        if (fam_ == VectorFamily::RT) { w0 = ty; w1 = -tx; }
        else { w0 = tx; w1 = ty; }
    }
    /// @brief The edge rule of interpolation: nq points t_q on [0, 1] and dof_j = sum_q P(j, q) f(t_q).
    int edge_rule_points() const { return nqE_; }
    double edge_rule_point(int q) const { return tE_[q]; }
    double edge_projection(int j, int q) const { return PE_[static_cast<std::size_t>(j) * nqE_ + q]; }
    /// @brief The interior rule: points and the moment weights W[i nq + q] = w_q psi_i(q).
    int interior_points() const { return nqI_; }
    double interior_xi(int q) const { return qx_[q]; }
    double interior_eta(int q) const { return qy_[q]; }
    double interior_weight(int i, int q) const { return Wint_[static_cast<std::size_t>(i) * nqI_ + q]; }

    /// @brief out[(c 3 + r) n + l]: component c of phi_l (r = 0), its d/dxi (1) and d/deta (2).
    void eval(double xi, double eta, double *out) const
    {
        std::vector<double> P(static_cast<std::size_t>(6) * n_);
        span(xi, eta, P.data());
        for (int r = 0; r < 6; ++r)
        {
            const double *pr = &P[static_cast<std::size_t>(r) * n_];
            double *o = out + static_cast<std::size_t>(r) * n_;
            for (int l = 0; l < n_; ++l) o[l] = 0.0;
            for (int m = 0; m < n_; ++m)
            {
                const double a = pr[m];
                if (a == 0.0) continue;
                const double *row = &C_[static_cast<std::size_t>(m) * n_];
                for (int l = 0; l < n_; ++l) o[l] += a * row[l];
            }
        }
    }

private:
    /// @brief The spanning functions p_m: P[(c 3 + r) n + m], component c, r = value, d/dxi, d/deta.
    void span(double xi, double eta, double *P) const
    {
        std::fill(P, P + 6 * n_, 0.0);
        std::vector<double> v(static_cast<std::size_t>(nm_)), dx(v.size()), dy(v.size());
        simplex::dubiner(k_ - 1, xi, eta, v.data(), dx.data(), dy.data());
        auto at = [&](int c, int r, int m) -> double & { return P[static_cast<std::size_t>(c * 3 + r) * n_ + m]; };
        for (int m = 0; m < nm_; ++m)
        {
            at(0, 0, m) = v[m]; at(0, 1, m) = dx[m]; at(0, 2, m) = dy[m];
            at(1, 0, nm_ + m) = v[m]; at(1, 1, nm_ + m) = dx[m]; at(1, 2, nm_ + m) = dy[m];
        }
        for (std::size_t t = 0; t < top_.size(); ++t)
        {
            const int m = 2 * nm_ + static_cast<int>(t), s = top_[t];
            const double p = v[s], px = dx[s], py = dy[s];
            if (fam_ == VectorFamily::RT)               // (xi p, eta p)
            {
                at(0, 0, m) = xi * p;  at(0, 1, m) = p + xi * px;  at(0, 2, m) = xi * py;
                at(1, 0, m) = eta * p; at(1, 1, m) = eta * px;     at(1, 2, m) = p + eta * py;
            }
            else                                        // (-eta p, xi p)
            {
                at(0, 0, m) = -eta * p; at(0, 1, m) = -eta * px;    at(0, 2, m) = -p - eta * py;
                at(1, 0, m) = xi * p;   at(1, 1, m) = p + xi * px;  at(1, 2, m) = xi * py;
            }
        }
    }

    VectorFamily fam_;
    int k_, n_ = 0, nm_ = 0, nI_ = 0, nqI_ = 0, nqE_ = 0;
    std::vector<int> top_;
    std::vector<double> s_, tE_, PE_, qx_, qy_, Wint_, C_;
};

/// @brief Affine triangle geometry and point location shared by the spaces of this header.
class TriangleSpace2D : public Space2D {
public:
    int nverts() const override { return 3; }
    std::array<int, 2> edge_vertices(int e) const override { return {(e + 1) % 3, (e + 2) % 3}; }
    void ref_vertex(int v, double &xi, double &eta) const override
    {
        static const double vx[3] = {0.0, 1.0, 0.0}, vy[3] = {0.0, 0.0, 1.0};
        xi = vx[v]; eta = vy[v];
    }
    void map(int c, double xi, double eta, double &x, double &y, double *J) const override
    {
        const Point P = origin(c);
        const double *Jc = jacobian(c);
        x = P.x + Jc[0] * xi + Jc[1] * eta;
        y = P.y + Jc[2] * xi + Jc[3] * eta;
        if (J) for (int i = 0; i < 4; ++i) J[i] = Jc[i];
    }
    bool to_reference(int c, double x, double y, double &xi, double &eta) const override
    {
        const Point P = origin(c);
        const double *Ji = jacobian_inverse(c);
        const double dx = x - P.x, dy = y - P.y;
        xi = Ji[0] * dx + Ji[1] * dy;
        eta = Ji[2] * dx + Ji[3] * dy;
        const double tol = 1e-10;
        return xi >= -tol && eta >= -tol && xi + eta <= 1.0 + tol;
    }
    int locate(const Point &p) const override
    {
        if (!locator_) locator_ = std::make_unique<PointLocator>(*mesh_);
        return locator_->locate(p);
    }
    void prepare_locate() const override
    {
        if (!locator_) locator_ = std::make_unique<PointLocator>(*mesh_);
        locator_->prepare();
    }
    int locate_prepared(const Point &p) const override { return locator_->find(p); }
    bool affine() const override { return true; }
    std::string cell_name() const override { return "triangle"; }

    const Mesh2D &mesh() const { return *mesh_; }
    const double *jacobian(int c) const { return &J_[static_cast<std::size_t>(c) * 4]; }
    const double *jacobian_inverse(int c) const { return &Jinv_[static_cast<std::size_t>(c) * 4]; }
    double det(int c) const { return det_[c]; }
    Point origin(int c) const { return mesh_->point(mesh_->triangle(c)[0]); }

protected:
    TriangleSpace2D(const Mesh2D &mesh, int k, std::vector<int> dirichlet, bool all_exterior, const char *who)
        : Space2D(k, std::move(dirichlet), all_exterior), mesh_(&mesh)
    {
        if (mesh.ntriangles() != mesh.tri_capacity())
            throw std::invalid_argument(std::string(who) + ": the mesh has free triangle slots; compact() it first");
        const int nc = mesh.ntriangles();
        J_.resize(static_cast<std::size_t>(nc) * 4);
        Jinv_.resize(J_.size());
        det_.resize(static_cast<std::size_t>(nc));
        for (int c = 0; c < nc; ++c)
        {
            const auto &t = mesh.triangle(c);
            const Point P0 = mesh.point(t[0]), P1 = mesh.point(t[1]), P2 = mesh.point(t[2]);
            double *J = &J_[static_cast<std::size_t>(c) * 4], *Ji = &Jinv_[static_cast<std::size_t>(c) * 4];
            J[0] = P1.x - P0.x; J[1] = P2.x - P0.x;
            J[2] = P1.y - P0.y; J[3] = P2.y - P0.y;
            const double d = J[0] * J[3] - J[1] * J[2];
            if (!(d > 0.0)) throw std::invalid_argument(std::string(who) + ": cell " + std::to_string(c) + " is degenerate or clockwise");
            det_[c] = d;
            Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
            Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
        }
    }

    Topology topology() const
    {
        const Mesh2D &mesh = *mesh_;
        const int nc = mesh.ntriangles();
        Topology T;
        T.pts = mesh.points();
        T.cellv.resize(static_cast<std::size_t>(nc) * 3);
        T.nbr.resize(T.cellv.size());
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < 3; ++i)
            {
                T.cellv[static_cast<std::size_t>(c) * 3 + i] = mesh.triangle(c)[i];
                T.nbr[static_cast<std::size_t>(c) * 3 + i] = mesh.neighbour(c, i);
            }
        const Mesh2D *M = mesh_;
        T.segment_marker = [M](int a, int b) { const int s = M->segment_index(a, b); return s < 0 ? -1 : M->segment_markers()[s]; };
        return T;
    }

    const Mesh2D *mesh_;
    std::vector<double> J_, Jinv_, det_;
    mutable std::unique_ptr<PointLocator> locator_;
};

/**
 * @brief RT_k (H(div)) or N1curl_k (H(curl)) on a triangular Mesh2D, k >= 1.
 *
 * Essential conditions (u . n for RT, u x n for N1curl) are eliminated on the sides named in
 * `dirichlet` (and every exterior side when all_exterior): their edge degrees of freedom are
 * removed, as Space2D does for Lagrange nodes.
 */
class VectorElementSpace2D : public TriangleSpace2D {
public:
    VectorElementSpace2D(const Mesh2D &mesh, VectorFamily fam, int k, std::vector<int> dirichlet = {}, bool all_exterior = false)
        : TriangleSpace2D(mesh, k, std::move(dirichlet), all_exterior, fam == VectorFamily::RT ? "RTSpace" : "N1curlSpace"),
          ref_(fam, k)
    {
        build(topology(), ref_.size());
    }

    // ---- the family --------------------------------------------------------------------------
    void ref_node(int l, double &xi, double &eta) const override
    {
        if (l < 3 * k_)
        {
            const int e = l / k_, j = l % k_;
            double ax, ay, bx, by;
            ref_vertex((e + 1) % 3, ax, ay);
            ref_vertex((e + 2) % 3, bx, by);
            const double s = ref_.edge_point(j);
            xi = ax + s * (bx - ax); eta = ay + s * (by - ay);
            return;
        }
        xi = eta = 1.0 / 3.0;                 // an interior moment: no point of its own
    }
    int edge_node(int e, int j) const override { return e * k_ + j; }
    int first_interior() const override { return 3 * k_; }
    int n_interior() const override { return k_ * (k_ - 1); }
    void ref_eval(double xi, double eta, int, double *out) const override { ref_.eval(xi, eta, out); }
    bool equispaced() const override { return true; }
    int dofs_per_vertex() const override { return 0; }
    int dofs_per_edge() const override { return k_; }
    int ncomp() const override { return 2; }
    int piola() const override { return ref_.family() == VectorFamily::RT ? 1 : 2; }
    bool edge_signs() const override { return true; }

    VectorFamily family() const { return ref_.family(); }
    const ReferenceVectorTriangle &reference() const { return ref_; }

    // ---- interpolation --------------------------------------------------------------------------
    /// @brief Points where a field is sampled for interpolate_raw: the edge rule on every global
    ///        edge (edge by edge, from its lower vertex), then the interior rule of every cell
    ///        (k >= 2).  (n, 2) row-major.
    std::vector<double> interpolation_points() const
    {
        const int ne = nedges(), nq = ref_.interior_points(), nqE = ref_.edge_rule_points();
        std::vector<double> out;
        out.reserve(2 * (static_cast<std::size_t>(ne) * nqE + static_cast<std::size_t>(ncells_) * nq));
        for (int g = 0; g < ne; ++g)
        {
            const Point A = pts_[edges_[g][0]], B = pts_[edges_[g][1]];
            for (int q = 0; q < nqE; ++q)
            {
                const double s = ref_.edge_rule_point(q);
                out.push_back(A.x + s * (B.x - A.x)); out.push_back(A.y + s * (B.y - A.y));
            }
        }
        for (int c = 0; c < ncells_; ++c)
            for (int q = 0; q < nq; ++q)
            {
                double x, y;
                map(c, ref_.interior_xi(q), ref_.interior_eta(q), x, y, nullptr);
                out.push_back(x); out.push_back(y);
            }
        return out;
    }

    /// @brief The raw coefficients of the canonical interpolant from the field's values F (n, 2)
    ///        at interpolation_points().
    std::vector<double> interpolate_raw(const std::vector<double> &F) const
    {
        const int ne = nedges(), nq = ref_.interior_points(), k = k_, nI = ref_.n_interior_modes();
        const int nqE = ref_.edge_rule_points();
        const std::size_t epts = static_cast<std::size_t>(ne) * nqE;
        const std::size_t npts = epts + static_cast<std::size_t>(ncells_) * nq;
        if (F.size() != 2 * npts) throw std::invalid_argument("interpolate_raw: need (n, 2) values at interpolation_points()");
        std::vector<double> raw(static_cast<std::size_t>(raw_dim_), 0.0);
        for (int g = 0; g < ne; ++g)
        {
            const Point A = pts_[edges_[g][0]], B = pts_[edges_[g][1]];
            double w0, w1;
            ref_.weight(B.x - A.x, B.y - A.y, w0, w1);
            for (int q = 0; q < nqE; ++q)
            {
                const std::size_t p = static_cast<std::size_t>(g) * nqE + q;
                const double f = F[2 * p] * w0 + F[2 * p + 1] * w1;
                for (int j = 0; j < k; ++j) raw[static_cast<std::size_t>(g) * k + j] += ref_.edge_projection(j, q) * f;
            }
        }
        const int nint = n_interior();
        for (int c = 0; c < ncells_ && nI > 0; ++c)
        {
            const double *J = jacobian(c), *Ji = jacobian_inverse(c);
            const double d = det(c);
            for (int q = 0; q < nq; ++q)
            {
                const std::size_t p = epts + static_cast<std::size_t>(c) * nq + q;
                const double fx = F[2 * p], fy = F[2 * p + 1];
                double h0, h1;                                  // the pull-back phihat
                if (ref_.family() == VectorFamily::RT) { h0 = d * (Ji[0] * fx + Ji[1] * fy); h1 = d * (Ji[2] * fx + Ji[3] * fy); }
                else { h0 = J[0] * fx + J[2] * fy; h1 = J[1] * fx + J[3] * fy; }
                for (int i = 0; i < nI; ++i)
                {
                    const double w = ref_.interior_weight(i, q);
                    raw[static_cast<std::size_t>(ne) * k + static_cast<std::size_t>(c) * nint + i] += w * h0;
                    raw[static_cast<std::size_t>(ne) * k + static_cast<std::size_t>(c) * nint + nI + i] += w * h1;
                }
            }
        }
        return raw;
    }

private:
    ReferenceVectorTriangle ref_;
};

/**
 * @brief Broken P_k (k >= 0) on a triangular Mesh2D: every cell carries its own nodal basis, so
 *        dim = ncells (k+1)(k+2)/2 and nothing is shared or eliminated.  The pressure of mixed
 *        methods (RT_k x DG_{k-1}).  The nodes are those of LagrangeSpace2D (equispaced by default),
 *        k = 0 the centroid.
 */
class DGSpace2D : public TriangleSpace2D {
public:
    DGSpace2D(const Mesh2D &mesh, int k, bool equispaced = true)
        : TriangleSpace2D(mesh, check(k), {}, false, "DGSpace2D"), ref_(k, equispaced)
    {
        build(topology(), ref_.size());
    }
    void ref_node(int l, double &xi, double &eta) const override { xi = ref_.node_xi(l); eta = ref_.node_eta(l); }
    int edge_node(int, int) const override { throw std::logic_error("DGSpace2D: no edge nodes"); }
    int first_interior() const override { return 0; }
    int n_interior() const override { return ref_.size(); }
    void ref_eval(double xi, double eta, int nder, double *out) const override { ref_.eval(xi, eta, nder, out); }
    bool equispaced() const override { return ref_.equispaced(); }
    int dofs_per_vertex() const override { return 0; }
    int dofs_per_edge() const override { return 0; }
    const ReferenceTriangle &reference() const { return ref_; }

private:
    static int check(int k)
    {
        if (k < 0) throw std::invalid_argument("DGSpace2D: degree must be >= 0");
        return k;
    }
    ReferenceTriangle ref_;
};

} // namespace femd

#endif // FEMD_FE_VECTOR_ELEMENT_2D_HPP
