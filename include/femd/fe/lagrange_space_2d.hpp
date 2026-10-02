//
//  lagrange_space_2d.hpp  --  continuous Lagrange P_k (k >= 1) on a triangular Mesh2D.
//
//  The numbering, the Dirichlet elimination, the facets and point evaluation are shared
//  with the quadrilateral family (space_2d.hpp).  This class supplies the reference
//  triangle (simplex_basis.hpp: warp-and-blend nodes, edge i opposite vertex i, from vertex
//  (i+1)%3 to (i+2)%3, the convention of Mesh2D's neighbour array) and the affine geometry
//  x = P0 + J (xi, eta), J = [P1 - P0, P2 - P0].
//
#ifndef FEMD_FE_LAGRANGE_SPACE_2D_HPP
#define FEMD_FE_LAGRANGE_SPACE_2D_HPP

#include "femd/fe/simplex_basis.hpp"
#include "femd/fe/space_2d.hpp"
#include "femd/mesh/mesh2d.hpp"
#include "femd/mesh/point_locator.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class LagrangeSpace2D : public Space2D {
public:
    /**
     * @param mesh       a compacted Mesh2D (no free triangle slots); it must outlive the space
     * @param k          degree >= 1
     * @param dirichlet  markers whose facets carry a Dirichlet condition (eliminated)
     * @param all_exterior  also eliminate every exterior facet, whatever its marker
     */
    LagrangeSpace2D(const Mesh2D &mesh, int k, std::vector<int> dirichlet = {}, bool all_exterior = false, bool equispaced = true)
        : Space2D(k, std::move(dirichlet), all_exterior), mesh_(&mesh), ref_(check_degree(k), equispaced)
    {
        if (mesh.ntriangles() != mesh.tri_capacity())
            throw std::invalid_argument("LagrangeSpace2D: the mesh has free triangle slots; compact() it first");
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
            if (!(d > 0.0)) throw std::invalid_argument("LagrangeSpace2D: cell " + std::to_string(c) + " is degenerate or clockwise");
            det_[c] = d;
            Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
            Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
        }
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
        const Mesh2D *M = &mesh;
        T.segment_marker = [M](int a, int b) { const int s = M->segment_index(a, b); return s < 0 ? -1 : M->segment_markers()[s]; };
        build(std::move(T), ref_.size());
    }

    // ---- the family --------------------------------------------------------------------------
    int nverts() const override { return 3; }
    std::array<int, 2> edge_vertices(int e) const override { return {(e + 1) % 3, (e + 2) % 3}; }
    void ref_vertex(int v, double &xi, double &eta) const override
    {
        static const double vx[3] = {0.0, 1.0, 0.0}, vy[3] = {0.0, 0.0, 1.0};
        xi = vx[v]; eta = vy[v];
    }
    void ref_node(int l, double &xi, double &eta) const override { xi = ref_.node_xi(l); eta = ref_.node_eta(l); }
    int edge_node(int e, int j) const override { return ref_.edge_node(e, j); }
    int first_interior() const override { return ref_.first_interior(); }
    int n_interior() const override { return (k_ - 1) * (k_ - 2) / 2; }
    void ref_eval(double xi, double eta, int nder, double *out) const override { ref_.eval(xi, eta, nder, out); }
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
    bool equispaced() const override { return ref_.equispaced(); }

    // ---- triangle-specific access (kept for the tests and the 2D code built on it) -----------
    const Mesh2D &mesh() const { return *mesh_; }
    const ReferenceTriangle &reference() const { return ref_; }
    const double *jacobian(int c) const { return &J_[static_cast<std::size_t>(c) * 4]; }
    const double *jacobian_inverse(int c) const { return &Jinv_[static_cast<std::size_t>(c) * 4]; }
    double det(int c) const { return det_[c]; }
    Point origin(int c) const { return mesh_->point(mesh_->triangle(c)[0]); }
    Point to_physical(int c, double xi, double eta) const { double x, y; map(c, xi, eta, x, y, nullptr); return Point{x, y}; }

private:
    static int check_degree(int k)
    {
        if (k < 1) throw std::invalid_argument("LagrangeSpace2D: degree must be >= 1 (P0 is a DG space)");
        return k;
    }
    const Mesh2D *mesh_;
    ReferenceTriangle ref_;
    std::vector<double> J_, Jinv_, det_;
    mutable std::unique_ptr<PointLocator> locator_;
};

} // namespace femd

#endif // FEMD_FE_LAGRANGE_SPACE_2D_HPP
