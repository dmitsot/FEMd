//
//  slip.hpp  --  the strong slip condition u . n = 0 for a vector field on a 2D mesh: the slip
//  nodes with their unit normals, and the matrix Z with x_B = Z x (orthonormal columns) that
//  maps the coefficients x of the system with the condition to those of the system without it.
//
//  At every node on a slip side the two velocity coefficients (u_x, u_y) are replaced by one,
//  the tangential component: u_i = a_i t_i with t_i the unit tangent.  A form is assembled
//  without the condition and reduced to Z^T A Z, a residual to Z^T r.
//
//  The normal at a node:
//    - a node inside a boundary edge: the normal of that edge ("conservative"), or the blend
//      of the two vertex normals along the edge ("smooth", second order on a curved wall);
//    - a vertex between two slip edges: the length-weighted mean of their normals, the
//      mass-conserving normal n_i ~ int phi_i n ds (Engelman, Sani and Gresho 1982);
//    - a corner, where two edge normals differ by more than corner_angle degrees: both
//      components are fixed, u_i = 0;
//    - a vertex between a slip edge and a side without a Dirichlet condition: that edge's normal.
//  Nodes on a Dirichlet side keep their Dirichlet condition (Dirichlet wins).  With periodic
//  conditions the copies of one node get the normalized sum of their normals, and a node that
//  is a corner in one copy is a corner.
//
//  This is the arithmetic of the former femd/slip.py, step by step (stable orders, sums left
//  to right), so Z is the same.
//
#ifndef FEMD_FE_SLIP_HPP
#define FEMD_FE_SLIP_HPP

#include "femd/fe/space_2d.hpp"
#include "femd/sparse/csr.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

/// @brief Slip nodes of one scalar space: raw node numbers, unit normals (2 per node) and the
///        raw vertex nodes that are corners.
struct SlipNodes {
    std::vector<int> nodes;
    std::vector<double> normals;
    std::vector<int> corners;
};

/// @brief One slip condition of a system: the two velocity fields (indices into the fields),
///        the side markers, the corner angle in degrees and the normal ("conservative" or "smooth").
struct SlipSpec {
    int f0 = 0, f1 = 1;
    std::vector<int> markers;
    double corner_angle = 45.0;
    std::string normal = "conservative";
};

/// @brief Z (base dim x reduced dim) and, per spec, the nodes kept (raw numbers of the first
///        field), their normals and the corners.
struct SlipSystem {
    CSRMatrix Z;
    std::vector<SlipNodes> info;
};

namespace detail {
inline std::string marker_list(const std::vector<int> &m)
{
    std::string s = "[";
    for (std::size_t i = 0; i < m.size(); ++i) s += (i ? ", " : "") + std::to_string(m[i]);
    return s + "]";
}
} // namespace detail

/**
 * @brief The raw nodes of the scalar space f on the sides `markers` (sorted, unique), with
 *        their unit normals, and the raw vertex nodes that are corners.
 */
inline SlipNodes slip_nodes(const Space2D &f, const std::vector<int> &markers, double corner_angle = 45.0,
                            const std::string &normal = "conservative")
{
    // the exterior edges on the slip sides, in facet order, with their outward normals
    std::vector<int> ea, eb, eg;                       // the two vertices and the global edge
    std::vector<double> nux, nuy, len, tx, ty;
    for (const Facet2D &F : f.facets())
    {
        if (!F.exterior || !std::binary_search(markers.begin(), markers.end(), F.marker)) continue;
        const std::array<int, 2> E = f.edge(F.edge);
        const Point &a = f.vertex(E[0]), &b = f.vertex(E[1]);
        const double L = std::hypot(b.x - a.x, b.y - a.y);
        const double tauX = (b.x - a.x) / L, tauY = (b.y - a.y) / L;
        double nX = tauY, nY = -tauX;
        double cx = 0.0, cy = 0.0;
        const int nv = f.nverts();
        for (int v = 0; v < nv; ++v) { cx += f.vertex(f.cell_vertex(F.cell, v)).x; cy += f.vertex(f.cell_vertex(F.cell, v)).y; }
        cx /= nv; cy /= nv;
        const double mx = 0.5 * (a.x + b.x) - cx, my = 0.5 * (a.y + b.y) - cy;
        if (mx * nX + my * nY < 0) { nX = -nX; nY = -nY; }         // outward
        ea.push_back(E[0]); eb.push_back(E[1]); eg.push_back(F.edge);
        nux.push_back(nX); nuy.push_back(nY); len.push_back(L); tx.push_back(tauX); ty.push_back(tauY);
    }
    if (ea.empty()) throw std::invalid_argument("slip: no exterior side carries marker(s) " + detail::marker_list(markers));
    const std::size_t ne = ea.size();

    // the vertices: every slip edge at the vertex (as first ends, then as second ends), stable
    std::vector<int> ends(2 * ne);
    for (std::size_t s = 0; s < ne; ++s) { ends[s] = ea[s]; ends[ne + s] = eb[s]; }
    std::vector<std::size_t> order(2 * ne);
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&ends](std::size_t x, std::size_t y) { return ends[x] < ends[y]; });
    std::vector<int> verts, count;
    std::vector<double> accx, accy;
    std::vector<std::size_t> start;
    for (std::size_t q = 0; q < order.size(); ++q)
    {
        const std::size_t o = order[q], s = o % ne;
        const double px = nux[s] * len[s], py = nuy[s] * len[s];
        if (verts.empty() || verts.back() != ends[o])
        {
            verts.push_back(ends[o]); count.push_back(1); start.push_back(q); accx.push_back(px); accy.push_back(py);
        }
        else { ++count.back(); accx.back() += px; accy.back() += py; }
    }
    const double cos_min = std::cos(corner_angle * (M_PI / 180.0));
    const std::size_t nvx = verts.size();
    std::vector<char> corner(nvx, 0);
    std::vector<double> vnx(nvx), vny(nvx);
    for (std::size_t j = 0; j < nvx; ++j)
    {
        if (count[j] > 1)
        {
            double cmin = std::numeric_limits<double>::infinity();
            for (int p = 0; p < count[j]; ++p)
                for (int q = 0; q < count[j]; ++q)
                {
                    const std::size_t sp = order[start[j] + p] % ne, sq = order[start[j] + q] % ne;
                    cmin = std::min(cmin, nux[sp] * nux[sq] + nuy[sp] * nuy[sq]);
                }
            corner[j] = cmin < cos_min - 1e-12;
        }
        const double h = std::hypot(accx[j], accy[j]);
        vnx[j] = accx[j] / h; vny[j] = accy[j] / h;
    }
    SlipNodes out;
    for (std::size_t j = 0; j < nvx; ++j)
        if (!corner[j]) { out.nodes.push_back(verts[j]); out.normals.push_back(vnx[j]); out.normals.push_back(vny[j]); }
        else out.corners.push_back(verts[j]);

    const int k = f.degree(), nvp = f.npoints();
    if (k > 1)                                                      // the nodes inside the slip edges
    {
        const bool smooth = normal == "smooth";
        if (!smooth && normal != "conservative")
            throw std::invalid_argument("slip_normal: 'conservative' or 'smooth', got '" + normal + "'");
        std::vector<double> vgx, vgy;                               // the normals of the good vertices
        std::vector<char> good;
        if (smooth)
        {
            vgx.assign(static_cast<std::size_t>(nvp), 0.0); vgy.assign(static_cast<std::size_t>(nvp), 0.0);
            good.assign(static_cast<std::size_t>(nvp), 0);
            for (std::size_t j = 0; j < nvx; ++j)
                if (count[j] == 2 && !corner[j]) { vgx[verts[j]] = vnx[j]; vgy[verts[j]] = vny[j]; good[verts[j]] = 1; }
        }
        const std::vector<double> &X = f.raw_coordinates();
        for (std::size_t s = 0; s < ne; ++s)
            for (int j = 0; j < k - 1; ++j)
            {
                const int node = nvp + eg[s] * (k - 1) + j;
                out.nodes.push_back(node);
                double nX = nux[s], nY = nuy[s];
                if (smooth)
                {
                    const Point &A = f.vertex(ea[s]);
                    const double t = ((X[2 * node] - A.x) * tx[s] + (X[2 * node + 1] - A.y) * ty[s]) / len[s];
                    double nax = vgx[ea[s]], nay = vgy[ea[s]], nbx = vgx[eb[s]], nby = vgy[eb[s]];
                    const bool ga = nax != 0.0 || nay != 0.0, gb = nbx != 0.0 || nby != 0.0;
                    auto ref = [nX, nY](double mx, double my, double &rx, double &ry) {
                        const double d = 2 * (mx * nX + my * nY);
                        rx = d * nX - mx; ry = d * nY - my;
                    };
                    if (!ga) { if (gb) ref(nbx, nby, nax, nay); else { nax = nX; nay = nY; } }
                    if (!gb) { if (ga) ref(nax, nay, nbx, nby); else { nbx = nX; nby = nY; } }
                    const double bx = (1 - t) * nax + t * nbx, by = (1 - t) * nay + t * nby;
                    const double h = std::hypot(bx, by);
                    nX = bx / h; nY = by / h;
                }
                out.normals.push_back(nX); out.normals.push_back(nY);
            }
    }
    return out;
}

/**
 * @brief Z for the slip specs on the system of the scalar fields `fields` (numbered from
 *        offsets[f], the system without the condition), and per spec the nodes kept, their
 *        normals and the corners (raw numbers of the spec's first field).
 */
inline SlipSystem slip_matrix(const std::vector<const Space2D *> &fields, const std::vector<int> &offsets,
                              const std::vector<SlipSpec> &specs)
{
    if (offsets.size() != fields.size() + 1) throw std::invalid_argument("slip_matrix: one offset per field and the total");
    const int nB = offsets.back();
    std::vector<char> keep(static_cast<std::size_t>(nB), 1);
    std::vector<int> R0, R1;
    std::vector<double> TX, TY;
    SlipSystem out;
    for (const SlipSpec &s : specs)
    {
        if (s.f0 < 0 || s.f1 < 0 || s.f0 >= static_cast<int>(fields.size()) || s.f1 >= static_cast<int>(fields.size()))
            throw std::invalid_argument("slip_matrix: field out of range");
        const Space2D &F0 = *fields[s.f0], &F1 = *fields[s.f1];
        if (F0.raw_dim() != F1.raw_dim()) throw std::invalid_argument("slip: the two components must have the same Dirichlet conditions");
        for (int r = 0; r < F0.raw_dim(); ++r)
            if (F0.raw_to_adapted(r) != F1.raw_to_adapted(r))
                throw std::invalid_argument("slip: the two components must have the same Dirichlet conditions");
        std::vector<int> mk = s.markers;
        std::sort(mk.begin(), mk.end());
        mk.erase(std::unique(mk.begin(), mk.end()), mk.end());
        SlipNodes sn = slip_nodes(F0, mk, s.corner_angle, s.normal);
        auto r2a = [&F0](int r) { return F0.raw_to_adapted(r); };
        // Dirichlet wins
        std::vector<int> nodes, corners, ad;
        std::vector<double> nrm;
        for (std::size_t i = 0; i < sn.nodes.size(); ++i)
            if (r2a(sn.nodes[i]) >= 0) { nodes.push_back(sn.nodes[i]); nrm.push_back(sn.normals[2 * i]); nrm.push_back(sn.normals[2 * i + 1]); }
        for (int c : sn.corners) if (r2a(c) >= 0) corners.push_back(c);
        for (int n : nodes) ad.push_back(r2a(n));
        std::vector<int> sorted_ad(ad);
        std::sort(sorted_ad.begin(), sorted_ad.end());
        if (!ad.empty() && std::adjacent_find(sorted_ad.begin(), sorted_ad.end()) != sorted_ad.end())
        {
            // periodic: the copies of one node become one, with the normalized sum of their normals
            std::vector<std::size_t> idx(ad.size());
            std::iota(idx.begin(), idx.end(), std::size_t(0));
            std::stable_sort(idx.begin(), idx.end(), [&ad](std::size_t x, std::size_t y) { return ad[x] < ad[y]; });
            std::vector<int> nn, na;
            std::vector<double> sx, sy;
            for (std::size_t q = 0; q < idx.size(); ++q)
            {
                const std::size_t i = idx[q];
                if (na.empty() || na.back() != ad[i]) { na.push_back(ad[i]); nn.push_back(nodes[i]); sx.push_back(0.0); sy.push_back(0.0); }
            }
            for (std::size_t i = 0; i < ad.size(); ++i)                // added in the original order
            {
                const std::size_t u = static_cast<std::size_t>(std::lower_bound(na.begin(), na.end(), ad[i]) - na.begin());
                sx[u] += nrm[2 * i]; sy[u] += nrm[2 * i + 1];
            }
            std::vector<int> cad;                                      // corners: one per adapted node, by value
            {
                std::vector<std::size_t> ci(corners.size());
                std::iota(ci.begin(), ci.end(), std::size_t(0));
                std::stable_sort(ci.begin(), ci.end(), [&](std::size_t x, std::size_t y) { return r2a(corners[x]) < r2a(corners[y]); });
                std::vector<int> nc;
                for (std::size_t q = 0; q < ci.size(); ++q)
                    if (cad.empty() || cad.back() != r2a(corners[ci[q]])) { cad.push_back(r2a(corners[ci[q]])); nc.push_back(corners[ci[q]]); }
                corners = nc;
            }
            nodes.clear(); nrm.clear(); ad.clear();
            for (std::size_t u = 0; u < na.size(); ++u)
            {
                if (std::binary_search(cad.begin(), cad.end(), na[u])) continue;     // a corner in one copy wins
                const double h = std::hypot(sx[u], sy[u]);
                nodes.push_back(nn[u]); ad.push_back(na[u]); nrm.push_back(sx[u] / h); nrm.push_back(sy[u] / h);
            }
        }
        const int o0 = offsets[s.f0], o1 = offsets[s.f1];
        for (std::size_t i = 0; i < ad.size(); ++i)
        {
            R0.push_back(o0 + ad[i]); R1.push_back(o1 + ad[i]);
            keep[static_cast<std::size_t>(o1 + ad[i])] = 0;
            TX.push_back(-nrm[2 * i + 1]); TY.push_back(nrm[2 * i]);
        }
        for (int c : corners) { keep[static_cast<std::size_t>(o0 + r2a(c))] = 0; keep[static_cast<std::size_t>(o1 + r2a(c))] = 0; }
        SlipNodes info;
        info.nodes = nodes; info.normals = nrm; info.corners = corners;
        out.info.push_back(std::move(info));
    }
    std::vector<int> col(static_cast<std::size_t>(nB), -1), rows;
    int nk = 0;
    for (int r = 0; r < nB; ++r) if (keep[r]) { col[r] = nk++; rows.push_back(r); }
    std::vector<double> vals(rows.size(), 1.0);
    for (std::size_t i = 0; i < R0.size(); ++i)
        if (col[R0[i]] >= 0) vals[static_cast<std::size_t>(col[R0[i]])] = TX[i];
    std::vector<int> I, J;
    std::vector<double> V;
    for (std::size_t q = 0; q < rows.size(); ++q) { I.push_back(rows[q]); J.push_back(col[rows[q]]); V.push_back(vals[q]); }
    for (std::size_t i = 0; i < R1.size(); ++i)
    {
        if (col[R0[i]] < 0) throw std::invalid_argument("slip: a tangential node of one condition is a corner of another");
        I.push_back(R1[i]); J.push_back(col[R0[i]]); V.push_back(TY[i]);
    }
    out.Z = from_triplets(nB, nk, I, J, V);
    return out;
}

} // namespace femd

#endif // FEMD_FE_SLIP_HPP
