//
//  dg_tables.hpp  --  the reference tables of the vertex limiter of a broken P_k space on
//  triangles (DGSpace2D, fd.VertexLimiter): the weights of the cell mean, the L2 projection onto
//  P1 written as the P1 values at the three vertices, and the P1 (barycentric) functions at the
//  nodes.  Computed once per space with the triangle rule of degree 2k + 2.
//
#ifndef FEMD_FE_DG_TABLES_HPP
#define FEMD_FE_DG_TABLES_HPP

#include "femd/fe/space_2d.hpp"
#include "femd/quadrature/triangle.hpp"
#include "femd/util/dense.hpp"
#include <vector>

namespace femd {

struct DGVertexTables {
    int nloc = 0;
    std::vector<double> mean;   ///< nloc: cell mean = sum_l mean[l] c_l
    std::vector<double> Pi;     ///< 3 x nloc (row-major): vertex values of the L2 projection onto P1
    std::vector<double> E;      ///< nloc x 3: the barycentric coordinates of the nodes
};

inline DGVertexTables dg_vertex_tables(const Space2D &V)
{
    if (V.nverts() != 3 || V.ncomp() != 1) throw std::invalid_argument("dg_vertex_tables: a scalar space on triangles");
    const int n = V.nloc();
    TriangleQuadrature Q(2 * V.degree() + 2);
    const int nq = Q.size();
    std::vector<double> tab(static_cast<std::size_t>(nq) * n), row(static_cast<std::size_t>(3) * n);
    for (int q = 0; q < nq; ++q)
    {
        V.ref_eval(Q.xi(q), Q.eta(q), 0, row.data());
        for (int l = 0; l < n; ++l) tab[static_cast<std::size_t>(q) * n + l] = row[l];
    }
    DGVertexTables T;
    T.nloc = n;
    double wsum = 0.0;
    for (int q = 0; q < nq; ++q) wsum += Q.weight(q);
    T.mean.assign(static_cast<std::size_t>(n), 0.0);
    for (int l = 0; l < n; ++l)
    {
        double s = 0.0;
        for (int q = 0; q < nq; ++q) s += Q.weight(q) * tab[static_cast<std::size_t>(q) * n + l];
        T.mean[l] = s / wsum;
    }
    // M1 = int lam lam^T, the P1 mass matrix of the reference triangle; R = int lam N^T
    std::vector<double> M1(9, 0.0), R(static_cast<std::size_t>(3) * n, 0.0);
    for (int q = 0; q < nq; ++q)
    {
        const double lam[3] = {1.0 - Q.xi(q) - Q.eta(q), Q.xi(q), Q.eta(q)}, w = Q.weight(q);
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b) M1[3 * a + b] += lam[a] * (w * lam[b]);
            for (int l = 0; l < n; ++l) R[static_cast<std::size_t>(a) * n + l] += lam[a] * (w * tab[static_cast<std::size_t>(q) * n + l]);
        }
    }
    T.Pi = DenseLU(M1, 3).solve_many(R, n);
    T.E.assign(static_cast<std::size_t>(n) * 3, 0.0);
    for (int l = 0; l < n; ++l)
    {
        double xi, eta;
        V.ref_node(l, xi, eta);
        T.E[3 * l] = 1.0 - xi - eta; T.E[3 * l + 1] = xi; T.E[3 * l + 2] = eta;
    }
    return T;
}

} // namespace femd

#endif // FEMD_FE_DG_TABLES_HPP
