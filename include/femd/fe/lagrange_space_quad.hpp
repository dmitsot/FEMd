//
//  lagrange_space_quad.hpp  --  continuous Lagrange Q_k (k >= 1) on a QuadMesh.
//
//  Reference square [0,1]^2 with vertices (0,0), (1,0), (1,1), (0,1), counter-clockwise.
//  Local edge e runs from vertex e to vertex (e+1)%4 (the QuadMesh convention).  The nodes are
//  the tensor product of the k+1 Gauss-Lobatto points t_0..t_k of [0,1] (symmetrized, so an edge
//  read backwards gives the same points), and the basis is the tensor product of the 1D Lagrange
//  polynomials on them, nodal by construction.  Local order: the 4 vertices, the k-1 nodes of
//  each edge along its direction, then the (k-1)^2 interior nodes, row by row.
//
//  Geometry is bilinear, x(xi, eta) = sum_v N_v(xi, eta) P_v, so J varies inside a cell (it is
//  constant only for parallelograms).  The inverse map is found by Newton's method.
//
#ifndef FEMD_FE_LAGRANGE_SPACE_QUAD_HPP
#define FEMD_FE_LAGRANGE_SPACE_QUAD_HPP

#include "femd/fe/dg_space.hpp"          // detail::gauss_lobatto_nodes
#include "femd/fe/space_2d.hpp"
#include "femd/mesh/quad_mesh.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class ReferenceQuad {
public:
    /// @param equispaced  true (the default): uniform tensor nodes j/k. false: Gauss-Lobatto (same space, other basis)
    explicit ReferenceQuad(int k, bool equispaced = true) : k_(k), equi_(equispaced)
    {
        if (k < 1) throw std::invalid_argument("ReferenceQuad: degree must be >= 1");
        if (k > 20) throw std::invalid_argument("ReferenceQuad: degree " + std::to_string(k) + " is above the supported 20");
        std::vector<double> z;
        if (equispaced) for (int j = 0; j <= k; ++j) z.push_back(-1.0 + 2.0 * j / k);
        else z = detail::gauss_lobatto_nodes(k);
        t_.resize(static_cast<std::size_t>(k) + 1);
        for (int j = 0; j <= k; ++j) t_[j] = 0.5 * (1.0 + 0.5 * (z[j] - z[k - j]));
        t_[0] = 0.0; t_[k] = 1.0;
        w_.assign(static_cast<std::size_t>(k) + 1, 1.0);          // barycentric weights of the 1D basis
        for (int j = 0; j <= k; ++j)
            for (int m = 0; m <= k; ++m) if (m != j) w_[j] /= (t_[j] - t_[m]);
        // local node l -> tensor index (i, j)
        const int n = (k + 1) * (k + 1);
        ij_.assign(static_cast<std::size_t>(n), {0, 0});
        ij_[0] = {0, 0}; ij_[1] = {k, 0}; ij_[2] = {k, k}; ij_[3] = {0, k};
        int l = 4;
        for (int j = 1; j < k; ++j) ij_[l++] = {j, 0};              // edge 0: bottom, left to right
        for (int j = 1; j < k; ++j) ij_[l++] = {k, j};              // edge 1: right, bottom to top
        for (int j = 1; j < k; ++j) ij_[l++] = {k - j, k};          // edge 2: top, right to left
        for (int j = 1; j < k; ++j) ij_[l++] = {0, k - j};          // edge 3: left, top to bottom
        for (int j = 1; j < k; ++j)
            for (int i = 1; i < k; ++i) ij_[l++] = {i, j};
    }

    int degree() const { return k_; }
    bool equispaced() const { return equi_; }
    int size() const { return (k_ + 1) * (k_ + 1); }
    double node_xi(int l) const { return t_[ij_[l][0]]; }
    double node_eta(int l) const { return t_[ij_[l][1]]; }
    const std::vector<double> &points_1d() const { return t_; }
    int edge_node(int e, int j) const { return 4 + e * (k_ - 1) + j; }
    int first_interior() const { return 4 + 4 * (k_ - 1); }

    /// @brief 1D Lagrange values L_j(t) and derivatives on the k+1 points.
    void basis_1d(double t, double *L, double *dL) const
    {
        const int k = k_;
        for (int j = 0; j <= k; ++j)
        {
            double p = 1.0, dp = 0.0;
            for (int m = 0; m <= k; ++m)
            {
                if (m == j) continue;
                // product rule: d(prod) = sum over one factor differentiated
                dp = dp * (t - t_[m]) + p;
                p *= (t - t_[m]);
            }
            L[j] = w_[j] * p;
            if (dL) dL[j] = w_[j] * dp;
        }
    }

    /// @brief Values [0, n), d/dxi [n, 2n), d/deta [2n, 3n) at (xi, eta).
    void eval(double xi, double eta, int nder, double *out) const
    {
        const int k = k_, n = size();
        std::vector<double> a(static_cast<std::size_t>(4) * (k + 1));
        double *Lx = a.data(), *dLx = Lx + (k + 1), *Ly = dLx + (k + 1), *dLy = Ly + (k + 1);
        basis_1d(xi, Lx, dLx);
        basis_1d(eta, Ly, dLy);
        for (int l = 0; l < n; ++l)
        {
            const int i = ij_[l][0], j = ij_[l][1];
            out[l] = Lx[i] * Ly[j];
            if (nder >= 1)
            {
                out[n + l] = dLx[i] * Ly[j];
                out[2 * n + l] = Lx[i] * dLy[j];
            }
        }
    }

private:
    int k_;
    bool equi_ = false;
    std::vector<double> t_, w_;
    std::vector<std::array<int, 2>> ij_;
};

class LagrangeSpaceQ : public Space2D {
public:
    LagrangeSpaceQ(const QuadMesh &mesh, int k, std::vector<int> dirichlet = {}, bool all_exterior = false, bool equispaced = true)
        : Space2D(k, std::move(dirichlet), all_exterior), mesh_(&mesh), ref_(check_degree(k), equispaced)
    {
        const int nc = mesh.nquads();
        Topology T;
        T.pts = mesh.points();
        T.cellv.resize(static_cast<std::size_t>(nc) * 4);
        T.nbr.resize(T.cellv.size());
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < 4; ++i)
            {
                T.cellv[static_cast<std::size_t>(c) * 4 + i] = mesh.quad(c)[i];
                T.nbr[static_cast<std::size_t>(c) * 4 + i] = mesh.neighbour(c, i);
            }
        const QuadMesh *M = &mesh;
        T.segment_marker = [M](int a, int b) { const int s = M->segment_index(a, b); return s < 0 ? -1 : M->segment_markers()[s]; };
        par_.resize(static_cast<std::size_t>(nc));
        for (int c = 0; c < nc; ++c)
        {
            const auto &q = mesh.quad(c);
            const Point &P0 = mesh.point(q[0]), &P1 = mesh.point(q[1]), &P2 = mesh.point(q[2]), &P3 = mesh.point(q[3]);
            const double s = std::abs(P0.x - P1.x + P2.x - P3.x) + std::abs(P0.y - P1.y + P2.y - P3.y);
            const double L = std::abs(P1.x - P0.x) + std::abs(P1.y - P0.y) + std::abs(P3.x - P0.x) + std::abs(P3.y - P0.y);
            par_[c] = s <= 1e-14 * L;                                 // a parallelogram: J is constant
            all_affine_ = all_affine_ && par_[c];
        }
        build(std::move(T), ref_.size());
    }

    // ---- the family --------------------------------------------------------------------------
    int nverts() const override { return 4; }
    std::array<int, 2> edge_vertices(int e) const override { return {e, (e + 1) % 4}; }
    void ref_vertex(int v, double &xi, double &eta) const override
    {
        static const double vx[4] = {0.0, 1.0, 1.0, 0.0}, vy[4] = {0.0, 0.0, 1.0, 1.0};
        xi = vx[v]; eta = vy[v];
    }
    void ref_node(int l, double &xi, double &eta) const override { xi = ref_.node_xi(l); eta = ref_.node_eta(l); }
    int edge_node(int e, int j) const override { return ref_.edge_node(e, j); }
    int first_interior() const override { return ref_.first_interior(); }
    int n_interior() const override { return (k_ - 1) * (k_ - 1); }
    void ref_eval(double xi, double eta, int nder, double *out) const override { ref_.eval(xi, eta, nder, out); }
    void map(int c, double xi, double eta, double &x, double &y, double *J) const override
    {
        const auto &q = mesh_->quad(c);
        const Point &P0 = mesh_->point(q[0]), &P1 = mesh_->point(q[1]), &P2 = mesh_->point(q[2]), &P3 = mesh_->point(q[3]);
        const double a0 = (1 - xi) * (1 - eta), a1 = xi * (1 - eta), a2 = xi * eta, a3 = (1 - xi) * eta;
        x = a0 * P0.x + a1 * P1.x + a2 * P2.x + a3 * P3.x;
        y = a0 * P0.y + a1 * P1.y + a2 * P2.y + a3 * P3.y;
        if (J)
        {
            J[0] = (P1.x - P0.x) * (1 - eta) + (P2.x - P3.x) * eta;       // dx/dxi
            J[1] = (P3.x - P0.x) * (1 - xi) + (P2.x - P1.x) * xi;         // dx/deta
            J[2] = (P1.y - P0.y) * (1 - eta) + (P2.y - P3.y) * eta;       // dy/dxi
            J[3] = (P3.y - P0.y) * (1 - xi) + (P2.y - P1.y) * xi;         // dy/deta
        }
    }
    bool to_reference(int c, double x, double y, double &xi, double &eta) const override
    {
        xi = 0.5; eta = 0.5;
        for (int it = 0; it < 30; ++it)
        {
            double fx, fy, J[4];
            map(c, xi, eta, fx, fy, J);
            fx -= x; fy -= y;
            const double d = J[0] * J[3] - J[1] * J[2];
            const double dxi = ( J[3] * fx - J[1] * fy) / d;
            const double deta = (-J[2] * fx + J[0] * fy) / d;
            xi -= dxi; eta -= deta;
            if (std::abs(dxi) + std::abs(deta) < 1e-15) break;
        }
        const double tol = 1e-10;
        return xi >= -tol && eta >= -tol && xi <= 1 + tol && eta <= 1 + tol;
    }
    int locate(const Point &p) const override { return mesh_->locate(p); }
    void prepare_locate() const override { mesh_->prepare_locate(); }
    int locate_prepared(const Point &p) const override { return mesh_->locate_prepared(p); }
    bool affine() const override { return all_affine_; }
    std::string cell_name() const override { return "quadrilateral"; }
    bool equispaced() const override { return ref_.equispaced(); }
    std::vector<double> points_1d() const override { return ref_.points_1d(); }

    const QuadMesh &mesh() const { return *mesh_; }
    const ReferenceQuad &reference() const { return ref_; }
    bool parallelogram(int c) const { return par_[c]; }

private:
    static int check_degree(int k)
    {
        if (k < 1) throw std::invalid_argument("LagrangeSpaceQ: degree must be >= 1");
        return k;
    }
    const QuadMesh *mesh_;
    ReferenceQuad ref_;
    std::vector<char> par_;
    bool all_affine_ = true;
};

} // namespace femd

#endif // FEMD_FE_LAGRANGE_SPACE_QUAD_HPP
