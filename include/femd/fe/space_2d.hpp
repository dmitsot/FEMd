//
//  space_2d.hpp  --  what every continuous Lagrange space on a 2D mesh shares, whatever
//  the cell: the numbering of the nodes, the Dirichlet elimination, the facets, and point
//  evaluation.  LagrangeSpace2D (P_k on triangles) and LagrangeSpaceQ (Q_k on quadrilaterals)
//  derive from it and supply the reference element and the geometry.
//
//  Raw numbering (every node, nothing eliminated), with dv = dofs_per_vertex() (1 or 0) and
//  de = dofs_per_edge() (k-1 for Lagrange):
//      vertex v                          -> v                                  (npoints, when dv = 1)
//      node j of global edge g           -> npoints dv + g de + j              (de per edge)
//      interior node j of cell c         -> npoints dv + nedges de + c nint + j
//  A global edge runs from its lower to its higher vertex index; a cell whose local edge runs
//  the other way reads its edge nodes in reverse.  Reference edge nodes are symmetric (the 1D
//  Gauss-Lobatto points), so the two cells see the same points and the basis is continuous.
//
//  Vector-valued families (Raviart-Thomas, Nedelec: vector_element_2d.hpp) have ncomp() = 2,
//  dv = 0, and a Piola map: piola() = 1 is contravariant, phi = J phihat / det J (H(div)), and
//  piola() = 2 covariant, phi = J^{-T} phihat (H(curl)).  Their edge degrees of freedom carry an
//  orientation: where a cell's local edge runs against the global edge, the cell's basis
//  function is minus the global one, sign(c, l) = -1 (edge_signs()).  Derivative codes are then
//  3 c + d: component c (0 or 1) and d = 0 value, 1 d/dx, 2 d/dy.  ref_eval writes, for such a
//  family, 3 ncomp nloc values [(c 3 + r) n + l] (r = value, d/dxi, d/deta), derivatives always.
//
//  Dirichlet conditions are eliminated as the 1D constraint operator does: the raw nodes on
//  every facet whose marker is in the Dirichlet set are removed, the others renumbered
//  consecutively (adapted numbering, 0..dim-1, in raw order).
//
//  Periodic conditions identify raw nodes: identify(rep) maps every raw node r to the raw node
//  rep[r] it is identified with (rep[rep[r]] == rep[r]).  The adapted numbering then counts one
//  degree of freedom per class, in raw order of the representatives, and every member of a class
//  has the class's adapted index; a class is eliminated when any member lies on a Dirichlet side.
//  Assembly needs nothing new: element contributions of identified nodes add up.
//
//  Facets: every exterior edge (no neighbour) and every segment edge, with the segment's
//  marker (0 for an exterior edge that is not a segment).
//
//  Geometry: map(c, xi, eta) -> (x, y) and its Jacobian J = d(x, y)/d(xi, eta), row-major.
//  Affine for triangles, bilinear for quadrilaterals.
//
#ifndef FEMD_FE_SPACE_2D_HPP
#define FEMD_FE_SPACE_2D_HPP

#include "femd/mesh/mesh2d.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace femd {

struct Facet2D {
    int cell = -1;       ///< the cell on the side the facet is integrated from
    int local = -1;      ///< local edge of that cell
    int marker = 0;      ///< segment marker, 0 when the edge is not a segment
    bool exterior = false;
    int edge = -1;       ///< global edge index
};

/// An interior facet (for dS): the edge between cell[0] (the '-' side, whose outward normal is the
/// facet's normal) and cell[1] (the '+' side).  A periodic facet joins an exterior edge of cell[0]
/// to one of cell[1]: the point x on the '-' side is x + shift on the '+' side.
struct InteriorFacet2D {
    int cell[2] = {-1, -1};
    int local[2] = {-1, -1};
    int edge = -1;               ///< global edge of the '-' side
    double shift[2] = {0.0, 0.0};
    bool periodic = false;
};

class Space2D {
public:
    virtual ~Space2D() = default;

    // ---- the family: reference element and geometry ------------------------------------
    /// @brief Vertices per cell: 3 (triangle) or 4 (quadrilateral).
    virtual int nverts() const = 0;
    /// @brief Local vertices (a, b) of local edge e, in the direction its nodes are counted.
    virtual std::array<int, 2> edge_vertices(int e) const = 0;
    /// @brief Reference coordinates of local vertex v.
    virtual void ref_vertex(int v, double &xi, double &eta) const = 0;
    /// @brief Reference coordinates of local node l.
    virtual void ref_node(int l, double &xi, double &eta) const = 0;
    /// @brief Local index of node j (0..k-2) of local edge e, counted from edge_vertices(e)[0].
    virtual int edge_node(int e, int j) const = 0;
    virtual int first_interior() const = 0;
    virtual int n_interior() const = 0;
    /// @brief Values [0, n), d/dxi [n, 2n), d/deta [2n, 3n) (the last two when nder >= 1).
    virtual void ref_eval(double xi, double eta, int nder, double *out) const = 0;
    /// @brief The 1D node coordinates on [0, 1] of a tensor-product element (quadrilaterals), empty otherwise.
    virtual std::vector<double> points_1d() const { return {}; }
    /// @brief (x, y) = F_c(xi, eta) and, when J is not null, J = [[dx/dxi, dx/deta], [dy/dxi, dy/deta]].
    virtual void map(int c, double xi, double eta, double &x, double &y, double *J) const = 0;
    /// @brief The reference point of (x, y) in cell c; false when it is not inside (to a tolerance).
    virtual bool to_reference(int c, double x, double y, double &xi, double &eta) const = 0;
    /// @brief The cell containing p, or -1.
    virtual int locate(const Point &p) const = 0;
    /// @brief Build the point-location grid now.  Then locate_prepared() is const and stateless,
    ///        so several threads may call it at once (locate() walks from the previous answer).
    virtual void prepare_locate() const = 0;
    virtual int locate_prepared(const Point &p) const = 0;
    /// @brief True when every cell map is affine (J constant per cell).
    virtual bool affine() const = 0;
    virtual std::string cell_name() const = 0;
    /// @brief True for uniform nodes, false for the Gauss-Lobatto based ones (warp-and-blend, GLL tensor).
    virtual bool equispaced() const = 0;
    /// @brief Degrees of freedom at each vertex (1 for continuous Lagrange, 0 for broken and vector families).
    virtual int dofs_per_vertex() const { return 1; }
    /// @brief Degrees of freedom on each edge, shared by the two cells (k-1 for Lagrange).
    virtual int dofs_per_edge() const { return k_ - 1; }
    /// @brief Components of a basis function: 1 (scalar) or 2 (Raviart-Thomas, Nedelec).
    virtual int ncomp() const { return 1; }
    /// @brief 0: none (scalar), 1: contravariant Piola (H(div)), 2: covariant Piola (H(curl)).
    virtual int piola() const { return 0; }
    /// @brief True when a reversed edge flips the sign of its basis functions (vector families).
    virtual bool edge_signs() const { return false; }
    /// @brief Number of derivative codes: 3 per component.
    int ncodes() const { return 3 * ncomp(); }
    /// @brief Values per point that ref_eval writes with derivatives.
    int ref_stride() const { return 3 * ncomp() * nloc_; }

    // ---- sizes ----------------------------------------------------------------------------
    int degree()   const { return k_; }
    int ncells()   const { return ncells_; }
    int nedges()   const { return static_cast<int>(edges_.size()); }
    int npoints()  const { return static_cast<int>(pts_.size()); }
    int nloc()     const { return nloc_; }
    int raw_dim()  const { return raw_dim_; }
    int dim()      const { return dim_; }
    int n_constraints() const { return raw_dim_ - dim_; }
    const std::vector<int> &dirichlet_markers() const { return dirichlet_; }
    bool all_exterior() const { return all_exterior_; }

    // ---- numbering --------------------------------------------------------------------------
    int raw_dof(int c, int l) const { return cell_raw_[static_cast<std::size_t>(c) * nloc_ + l]; }
    int dof(int c, int l) const { return cell_dof_[static_cast<std::size_t>(c) * nloc_ + l]; }
    const std::vector<int> &cell_dofs() const { return cell_dof_; }
    const std::vector<int> &cell_raw_dofs() const { return cell_raw_; }
    int raw_to_adapted(int r) const { return raw2ad_[r]; }
    int adapted_to_raw(int a) const { return ad2raw_[a]; }
    const std::vector<int> &constrained() const { return constrained_; }
    const std::vector<double> &raw_coordinates() const { return raw_xy_; }
    int cell_vertex(int c, int v) const { return cellv_[static_cast<std::size_t>(c) * nv_ + v]; }
    /// @brief +1, or -1 where local function l of cell c is minus the global one (a reversed edge).
    double sign(int c, int l) const { return sign_.empty() ? 1.0 : sign_[static_cast<std::size_t>(c) * nloc_ + l]; }
    const std::vector<double> &signs() const { return sign_; }
    const Point &vertex(int v) const { return pts_[v]; }

    std::vector<double> prolongate(const std::vector<double> &c) const
    {
        if (static_cast<int>(c.size()) != dim_) throw std::invalid_argument("prolongate: length != dim");
        std::vector<double> r(static_cast<std::size_t>(raw_dim_), 0.0);
        for (int i = 0; i < raw_dim_; ++i) if (raw2ad_[i] >= 0) r[i] = c[raw2ad_[i]];    // identified nodes share
        return r;
    }

    // ---- periodic identification -------------------------------------------------------------
    /// @brief rep[r]: the raw node that raw node r is identified with (r itself when none).
    const std::vector<int> &representatives() const { return rep_; }
    bool periodic() const { return periodic_; }
    /**
     * @brief Identify raw nodes (periodic conditions) and renumber.  rep[r] is the representative
     * of r's class, with rep[rep[r]] == rep[r].  A class is eliminated when any member is on a
     * Dirichlet side.  Replaces any earlier identification.
     */
    void identify(const std::vector<int> &rep)
    {
        if (static_cast<int>(rep.size()) != raw_dim_) throw std::invalid_argument("identify: length != raw_dim");
        for (int r = 0; r < raw_dim_; ++r)
        {
            const int q = rep[r];
            if (q < 0 || q >= raw_dim_) throw std::invalid_argument("identify: representative out of range");
            if (rep[q] != q) throw std::invalid_argument("identify: the representative of a class must represent itself");
        }
        rep_ = rep;
        periodic_ = false;
        for (int r = 0; r < raw_dim_ && !periodic_; ++r) periodic_ = rep_[r] != r;
        number();
    }
    std::vector<double> restrict_raw(const std::vector<double> &r) const
    {
        if (static_cast<int>(r.size()) != raw_dim_) throw std::invalid_argument("restrict: length != raw_dim");
        std::vector<double> c(static_cast<std::size_t>(dim_));
        for (int a = 0; a < dim_; ++a) c[a] = r[ad2raw_[a]];
        return c;
    }

    // ---- facets ------------------------------------------------------------------------------
    const std::vector<Facet2D> &facets() const { return facets_; }
    /// @brief The interior facets: every edge shared by two cells (the lower cell index is '-'),
    ///        then the periodic ones added by add_periodic_facets.
    const std::vector<InteriorFacet2D> &interior_facets() const { return interior_; }
    /**
     * @brief Join exterior edges in pairs into periodic interior facets: local edge la[i] of cell
     *        ca[i] (the '-' side) with local edge lb[i] of cell cb[i] ('+'), the second being the first
     *        moved by (sx, sy).  Replaces any earlier periodic facets.
     */
    void set_periodic_facets(const std::vector<int> &ca, const std::vector<int> &la, const std::vector<int> &cb,
                             const std::vector<int> &lb, double sx, double sy)
    {
        set_periodic_facets(ca, la, cb, lb, std::vector<double>(ca.size(), sx), std::vector<double>(ca.size(), sy));
    }
    void set_periodic_facets(const std::vector<int> &ca, const std::vector<int> &la, const std::vector<int> &cb,
                             const std::vector<int> &lb, const std::vector<double> &sx, const std::vector<double> &sy)
    {
        const std::size_t n = ca.size();
        if (la.size() != n || cb.size() != n || lb.size() != n || sx.size() != n || sy.size() != n)
            throw std::invalid_argument("set_periodic_facets: lengths differ");
        interior_.resize(n_interior_plain_);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (ca[i] < 0 || ca[i] >= ncells_ || cb[i] < 0 || cb[i] >= ncells_ || la[i] < 0 || la[i] >= nv_ || lb[i] < 0 || lb[i] >= nv_)
                throw std::invalid_argument("set_periodic_facets: cell or local edge out of range");
            InteriorFacet2D f;
            f.cell[0] = ca[i]; f.local[0] = la[i]; f.cell[1] = cb[i]; f.local[1] = lb[i];
            f.edge = cell_edge(ca[i], la[i]);
            f.shift[0] = sx[i]; f.shift[1] = sy[i];
            f.periodic = true;
            interior_.push_back(f);
        }
    }
    int n_periodic_facets() const { return static_cast<int>(interior_.size()) - n_interior_plain_; }
    std::array<int, 2> edge(int g) const { return edges_[g]; }
    int cell_edge(int c, int i) const { return cell_edge_[static_cast<std::size_t>(c) * nv_ + i]; }

    // ---- evaluation ----------------------------------------------------------------------------
    /**
     * @brief u, du/dx or du/dy (deriv 0, 1, 2) at points, for RAW coefficients.  NaN outside the mesh.
     * @param cells [out, optional] the cell of each point (-1 outside)
     */
    std::vector<double> evaluate_raw(const std::vector<double> &raw, const std::vector<Point> &pts, int deriv,
                                     std::vector<int> *cells = nullptr) const
    {
        if (static_cast<int>(raw.size()) != raw_dim_) throw std::invalid_argument("evaluate: coefficient length != raw_dim");
        if (deriv < 0 || deriv >= ncodes())
            throw std::invalid_argument(ncomp() == 1 ? "evaluate: deriv is 0 (value), 1 (d/dx) or 2 (d/dy)"
                                                     : "evaluate: the code is 3 c + d, component c and d = 0, 1 (d/dx), 2 (d/dy)");
        std::vector<double> out(pts.size(), std::numeric_limits<double>::quiet_NaN());
        if (cells) cells->assign(pts.size(), -1);
        const int n = nloc_;
        // a large batch is located through the grid alone, the same answers for any number of threads
        const long np = static_cast<long>(pts.size());
        const bool par = np > FEMD_OMP_THRESHOLD;
        if (par) prepare_locate();
        FEMD_OMP_PARALLEL_IF(par)
        {
        std::vector<double> tab(static_cast<std::size_t>(ref_stride())), row(static_cast<std::size_t>(n));
        FEMD_OMP_FOR
        for (long i = 0; i < np; ++i)
        {
            const int c = par ? locate_prepared(pts[i]) : locate(pts[i]);
            if (cells) (*cells)[i] = c;
            if (c < 0) continue;
            double xi, eta;
            to_reference(c, pts[i].x, pts[i].y, xi, eta);
            out[i] = value_at(raw, c, xi, eta, deriv, tab.data(), row.data());
        }
        }
        return out;
    }

    /**
     * @brief The same at reference points (xi[i], eta[i]) of given cells: no point location, and
     *        a field that is two-valued on an edge (broken, or a vector family's other component)
     *        is read from the cell named.
     */
    std::vector<double> evaluate_ref(const std::vector<double> &raw, const std::vector<int> &cells, const std::vector<double> &xi,
                                     const std::vector<double> &eta, int deriv) const
    {
        if (static_cast<int>(raw.size()) != raw_dim_) throw std::invalid_argument("evaluate_ref: coefficient length != raw_dim");
        if (cells.size() != xi.size() || xi.size() != eta.size()) throw std::invalid_argument("evaluate_ref: lengths differ");
        if (deriv < 0 || deriv >= ncodes()) throw std::invalid_argument("evaluate_ref: derivative code out of range");
        const long np = static_cast<long>(cells.size());
        std::vector<double> out(cells.size(), std::numeric_limits<double>::quiet_NaN());
        FEMD_OMP_PARALLEL_IF(np > FEMD_OMP_THRESHOLD)
        {
        std::vector<double> tab(static_cast<std::size_t>(ref_stride())), row(static_cast<std::size_t>(nloc_));
        FEMD_OMP_FOR
        for (long i = 0; i < np; ++i)
        {
            const int c = cells[i];
            if (c < 0 || c >= ncells_) continue;
            out[i] = value_at(raw, c, xi[i], eta[i], deriv, tab.data(), row.data());
        }
        }
        return out;
    }

    /**
     * @brief The physical row of derivative code m at one point of cell c: R is the reference
     *        tabulation there (ref_eval with derivatives), J the Jacobian, Ji its inverse, det = det J.
     *        Scalar families need only Ji; vector ones apply their Piola map and the edge signs.
     */
    void physical_row(int c, const double *R, const double *J, const double *Ji, double det, int m, double *out) const
    {
        const int n = nloc_;
        if (ncomp() == 1)
        {
            if (m == 0) { std::copy(R, R + n, out); return; }
            const double a = (m == 1) ? Ji[0] : Ji[1];
            const double b = (m == 1) ? Ji[2] : Ji[3];
            for (int l = 0; l < n; ++l) out[l] = a * R[n + l] + b * R[2 * n + l];
            return;
        }
        const int comp = m / 3, d = m % 3;
        double M[2];                                   // phi_comp = M[0] phihat_0 + M[1] phihat_1
        for (int b = 0; b < 2; ++b) M[b] = piola() == 1 ? J[comp * 2 + b] / det : Ji[b * 2 + comp];
        const double *R0 = R, *R1 = R + 3 * n;         // components of phihat: value, d/dxi, d/deta
        if (d == 0)
            for (int l = 0; l < n; ++l) out[l] = M[0] * R0[l] + M[1] * R1[l];
        else
        {
            const double a = Ji[d - 1], b = Ji[2 + d - 1];          // d xi / d x_d, d eta / d x_d
            for (int l = 0; l < n; ++l)
                out[l] = M[0] * (a * R0[n + l] + b * R0[2 * n + l]) + M[1] * (a * R1[n + l] + b * R1[2 * n + l]);
        }
        if (!sign_.empty())
        {
            const double *s = &sign_[static_cast<std::size_t>(c) * n];
            for (int l = 0; l < n; ++l) out[l] *= s[l];
        }
    }

    /// @brief The field with RAW coefficients, code m, at (xi, eta) of cell c (tab, row: work space).
    double value_at(const std::vector<double> &raw, int c, double xi, double eta, int m, double *tab, double *row) const
    {
        ref_eval(xi, eta, (m != 0 || ncomp() > 1) ? 1 : 0, tab);
        double x, y, J[4], Ji[4];
        map(c, xi, eta, x, y, J);
        const double d = J[0] * J[3] - J[1] * J[2];
        Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
        Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
        physical_row(c, tab, J, Ji, d, m, row);
        double s = 0.0;
        for (int l = 0; l < nloc_; ++l) s += row[l] * raw[raw_dof(c, l)];
        return s;
    }

    /// @brief J^{-1} at (xi, eta) of cell c, row-major; returns det J.
    double inverse_jacobian(int c, double xi, double eta, double *Ji) const
    {
        double x, y, J[4];
        map(c, xi, eta, x, y, J);
        const double d = J[0] * J[3] - J[1] * J[2];
        Ji[0] =  J[3] / d; Ji[1] = -J[1] / d;
        Ji[2] = -J[2] / d; Ji[3] =  J[0] / d;
        return d;
    }

protected:
    /// Everything the derived constructor knows about its mesh, in one place.
    struct Topology {
        std::vector<Point> pts;
        std::vector<int> cellv;                        ///< ncells * nverts, counter-clockwise
        std::vector<int> nbr;                          ///< ncells * nverts, neighbour across local edge i, -1 outside
        std::function<int(int, int)> segment_marker;   ///< marker of the segment (a, b), -1 when it is not one
    };

    Space2D(int k, std::vector<int> dirichlet, bool all_exterior) : k_(k), dirichlet_(std::move(dirichlet)), all_exterior_(all_exterior)
    {
        std::sort(dirichlet_.begin(), dirichlet_.end());
        dirichlet_.erase(std::unique(dirichlet_.begin(), dirichlet_.end()), dirichlet_.end());
    }

    /// @brief Numbering, facets, coordinates and elimination; call once the family is ready (nloc known).
    void build(Topology T, int nloc)
    {
        nv_ = nverts();
        nloc_ = nloc;
        pts_ = std::move(T.pts);
        cellv_ = std::move(T.cellv);
        ncells_ = static_cast<int>(cellv_.size()) / nv_;
        build_edges(T);
        build_dofs();
    }

private:
    static std::int64_t key(int a, int b)
    {
        if (a > b) std::swap(a, b);
        return (static_cast<std::int64_t>(a) << 32) | static_cast<std::uint32_t>(b);
    }

    void build_edges(const Topology &T)
    {
        cell_edge_.assign(static_cast<std::size_t>(ncells_) * nv_, -1);
        cell_flip_.assign(static_cast<std::size_t>(ncells_) * nv_, 0);
        std::unordered_map<std::int64_t, int> emap;
        emap.reserve(static_cast<std::size_t>(ncells_) * 2);
        for (int c = 0; c < ncells_; ++c)
            for (int i = 0; i < nv_; ++i)
            {
                const auto ab = edge_vertices(i);
                const int a = cell_vertex(c, ab[0]), b = cell_vertex(c, ab[1]);
                auto it = emap.find(key(a, b));
                int g;
                if (it == emap.end())
                {
                    g = static_cast<int>(edges_.size());
                    edges_.push_back({std::min(a, b), std::max(a, b)});
                    emap.emplace(key(a, b), g);
                }
                else g = it->second;
                cell_edge_[static_cast<std::size_t>(c) * nv_ + i] = g;
                cell_flip_[static_cast<std::size_t>(c) * nv_ + i] = (a > b) ? 1 : 0;
            }
        // interior facets: the two (cell, local edge) of every shared edge, '-' first met
        {
            std::vector<int> first(edges_.size(), -1);
            interior_.clear();
            for (int c = 0; c < ncells_; ++c)
                for (int i = 0; i < nv_; ++i)
                {
                    const int g = cell_edge(c, i);
                    if (first[g] < 0) { first[g] = c * nv_ + i; continue; }
                    InteriorFacet2D f;
                    f.cell[0] = first[g] / nv_; f.local[0] = first[g] % nv_;
                    f.cell[1] = c; f.local[1] = i;
                    f.edge = g;
                    interior_.push_back(f);
                }
            n_interior_plain_ = static_cast<int>(interior_.size());
        }
        std::vector<char> seen(edges_.size(), 0);
        for (int c = 0; c < ncells_; ++c)
            for (int i = 0; i < nv_; ++i)
            {
                const int g = cell_edge(c, i);
                const auto ab = edge_vertices(i);
                const int a = cell_vertex(c, ab[0]), b = cell_vertex(c, ab[1]);
                const bool ext = T.nbr[static_cast<std::size_t>(c) * nv_ + i] < 0;
                const int m = T.segment_marker(a, b);
                if (!ext && m < 0) continue;
                if (!ext && seen[g]) continue;
                seen[g] = 1;
                Facet2D f;
                f.cell = c; f.local = i; f.exterior = ext; f.edge = g;
                f.marker = m >= 0 ? m : 0;
                facets_.push_back(f);
            }
    }

    void build_dofs()
    {
        const int n = nloc_, nvp = npoints(), ne = nedges(), nint = n_interior();
        const int dv = dofs_per_vertex(), de = dofs_per_edge(), voff = nvp * dv;
        if (dv != 0 && dv != 1) throw std::logic_error("Space2D: dofs_per_vertex must be 0 or 1");
        raw_dim_ = voff + ne * de + ncells_ * nint;
        cell_raw_.assign(static_cast<std::size_t>(ncells_) * n, -1);
        sign_.clear();
        if (edge_signs()) sign_.assign(static_cast<std::size_t>(ncells_) * n, 1.0);
        for (int c = 0; c < ncells_; ++c)
        {
            int *d = &cell_raw_[static_cast<std::size_t>(c) * n];
            if (dv) for (int v = 0; v < nv_; ++v) d[v] = cell_vertex(c, v);
            for (int i = 0; i < nv_; ++i)
            {
                const int g = cell_edge(c, i);
                const bool flip = cell_flip_[static_cast<std::size_t>(c) * nv_ + i] != 0;
                for (int j = 0; j < de; ++j)
                {
                    const int l = edge_node(i, j);
                    d[l] = voff + g * de + (flip ? (de - 1 - j) : j);
                    if (flip && !sign_.empty()) sign_[static_cast<std::size_t>(c) * n + l] = -1.0;
                }
            }
            const int f0 = first_interior();
            for (int j = 0; j < nint; ++j) d[f0 + j] = voff + ne * de + c * nint + j;
        }
        raw_xy_.assign(static_cast<std::size_t>(raw_dim_) * 2, std::numeric_limits<double>::quiet_NaN());
        std::vector<char> done(static_cast<std::size_t>(raw_dim_), 0);
        if (dv) for (int v = 0; v < nvp; ++v) { raw_xy_[2 * v] = pts_[v].x; raw_xy_[2 * v + 1] = pts_[v].y; done[v] = 1; }
        for (int c = 0; c < ncells_; ++c)
            for (int l = dv ? nv_ : 0; l < n; ++l)
            {
                const int r = cell_raw_[static_cast<std::size_t>(c) * n + l];
                if (done[r]) continue;
                double xi, eta, x, y;
                ref_node(l, xi, eta);
                map(c, xi, eta, x, y, nullptr);
                raw_xy_[2 * r] = x; raw_xy_[2 * r + 1] = y;
                done[r] = 1;
            }
        fixed_.assign(static_cast<std::size_t>(raw_dim_), 0);
        for (const Facet2D &f : facets_)
        {
            const bool on = (all_exterior_ && f.exterior) || std::binary_search(dirichlet_.begin(), dirichlet_.end(), f.marker);
            if (!on) continue;
            const auto &e = edges_[f.edge];
            if (dv) fixed_[e[0]] = fixed_[e[1]] = 1;
            for (int j = 0; j < de; ++j) fixed_[voff + f.edge * de + j] = 1;
        }
        rep_.resize(static_cast<std::size_t>(raw_dim_));
        for (int r = 0; r < raw_dim_; ++r) rep_[r] = r;
        periodic_ = false;
        number();
    }

    /// @brief The adapted numbering from fixed_ (Dirichlet nodes) and rep_ (identified nodes).
    void number()
    {
        std::vector<char> gone(static_cast<std::size_t>(raw_dim_), 0);      // per class, at its representative
        for (int r = 0; r < raw_dim_; ++r) if (fixed_[r]) gone[rep_[r]] = 1;
        raw2ad_.assign(static_cast<std::size_t>(raw_dim_), -1);
        ad2raw_.clear(); constrained_.clear();
        for (int r = 0; r < raw_dim_; ++r)
            if (rep_[r] == r && !gone[r])
            {
                raw2ad_[r] = static_cast<int>(ad2raw_.size());
                ad2raw_.push_back(r);
            }
        for (int r = 0; r < raw_dim_; ++r)
        {
            if (gone[rep_[r]]) { constrained_.push_back(r); continue; }
            raw2ad_[r] = raw2ad_[rep_[r]];
        }
        dim_ = static_cast<int>(ad2raw_.size());
        cell_dof_.resize(cell_raw_.size());
        for (std::size_t i = 0; i < cell_raw_.size(); ++i) cell_dof_[i] = raw2ad_[cell_raw_[i]];
    }

protected:
    int k_;
    int nv_ = 3, nloc_ = 0, ncells_ = 0, raw_dim_ = 0, dim_ = 0;
    std::vector<int> dirichlet_;
    bool all_exterior_;
    std::vector<Point> pts_;
    std::vector<int> cellv_;
    std::vector<std::array<int, 2>> edges_;
    std::vector<int> cell_edge_;
    std::vector<char> cell_flip_;
    std::vector<double> sign_;             ///< ncells * nloc, -1 at reversed edge functions (vector families), empty otherwise
    std::vector<Facet2D> facets_;
    std::vector<InteriorFacet2D> interior_;
    int n_interior_plain_ = 0;
    std::vector<int> cell_raw_, cell_dof_, raw2ad_, ad2raw_, constrained_;
    std::vector<int> rep_;                 ///< periodic identification: the representative of each raw node
    std::vector<char> fixed_;              ///< raw nodes on Dirichlet sides
    bool periodic_ = false;
    std::vector<double> raw_xy_;
};

} // namespace femd

#endif // FEMD_FE_SPACE_2D_HPP
