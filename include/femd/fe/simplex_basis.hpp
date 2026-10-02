//
//  simplex_basis.hpp  --  the nodal Lagrange basis of degree k on the reference triangle.
//
//  Reference triangle T: vertices (0,0), (1,0), (0,1) in (xi, eta), barycentric
//  coordinates b0 = 1 - xi - eta, b1 = xi, b2 = eta.  Edge i is the edge OPPOSITE
//  vertex i, from vertex (i+1)%3 to vertex (i+2)%3, the convention of Mesh2D's
//  neighbour array.
//
//  Nodes: Warburton's warp-and-blend points (Hesthaven and Warburton, Nodal
//  Discontinuous Galerkin Methods, 2008, Section 6.1), whose edge nodes are the 1D
//  Gauss-Lobatto points.  For k <= 2 they coincide with the equispaced points; for
//  larger k their Lebesgue constant grows far more slowly.  The edge nodes are then
//  overwritten with symmetrized Gauss-Lobatto positions, so the node at position j
//  from one end of an edge is the node at position k-j from the other to rounding:
//  two triangles sharing an edge see the same points.
//
//  Local order: the 3 vertices, then the k-1 nodes of edge 0, edge 1, edge 2 (each
//  running from its first vertex to its second), then the (k-1)(k-2)/2 interior nodes.
//
//  Basis: phi = psi V^{-1}, psi the orthonormal Dubiner (Proriol-Koornwinder) basis,
//  V_ij = psi_j(node_i), so phi_k(node_i) = delta_ik to rounding whatever k is.
//
#ifndef FEMD_FE_SIMPLEX_BASIS_HPP
#define FEMD_FE_SIMPLEX_BASIS_HPP

#include "femd/fe/dg_space.hpp"          // detail::gauss_lobatto_nodes
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {
namespace simplex {

/// @brief Orthonormal Jacobi polynomials P_0..P_N^{(alpha,beta)} at x (Hesthaven's JacobiP).
inline void jacobi_all(double x, double alpha, double beta, int N, double *P)
{
    const double ab = alpha + beta;
    const double gamma0 = std::pow(2.0, ab + 1.0) / (ab + 1.0) * std::tgamma(alpha + 1.0) * std::tgamma(beta + 1.0) / std::tgamma(ab + 1.0);
    P[0] = 1.0 / std::sqrt(gamma0);
    if (N == 0) return;
    const double gamma1 = (alpha + 1.0) * (beta + 1.0) / (ab + 3.0) * gamma0;
    P[1] = ((ab + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / std::sqrt(gamma1);
    double aold = 2.0 / (2.0 + ab) * std::sqrt((alpha + 1.0) * (beta + 1.0) / (ab + 3.0));
    for (int i = 1; i < N; ++i)
    {
        const double h1 = 2.0 * i + ab;
        const double anew = 2.0 / (h1 + 2.0) * std::sqrt((i + 1.0) * (i + 1.0 + ab) * (i + 1.0 + alpha) * (i + 1.0 + beta) / (h1 + 1.0) / (h1 + 3.0));
        const double bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2.0);
        P[i + 1] = 1.0 / anew * (-aold * P[i - 1] + (x - bnew) * P[i]);
        aold = anew;
    }
}

inline double jacobi(double x, double alpha, double beta, int n)
{
    std::vector<double> P(static_cast<std::size_t>(n) + 1);
    jacobi_all(x, alpha, beta, n, P.data());
    return P[n];
}

inline double jacobi_derivative(double x, double alpha, double beta, int n)
{
    if (n == 0) return 0.0;
    return std::sqrt(n * (n + alpha + beta + 1.0)) * jacobi(x, alpha + 1.0, beta + 1.0, n - 1);
}

inline int nmodes(int k) { return (k + 1) * (k + 2) / 2; }

/**
 * @brief The Dubiner modes psi_m, m over (i, j) with i + j <= k in the order i = 0..k,
 *        j = 0..k-i, and their derivatives in xi and eta, at (xi, eta) of T.
 * @param v, dxi, deta  nmodes(k) values each; dxi and deta may be null.
 */
inline void dubiner(int k, double xi, double eta, double *v, double *dxi, double *deta)
{
    const double r = 2.0 * xi - 1.0, s = 2.0 * eta - 1.0;
    const double a = (std::abs(1.0 - s) > 1e-14) ? 2.0 * (1.0 + r) / (1.0 - s) - 1.0 : -1.0;
    const double b = s;
    std::vector<double> Pa(static_cast<std::size_t>(k) + 1), dPa(static_cast<std::size_t>(k) + 1);
    jacobi_all(a, 0.0, 0.0, k, Pa.data());
    for (int i = 0; i <= k; ++i) dPa[i] = jacobi_derivative(a, 0.0, 0.0, i);
    std::vector<double> Pb(static_cast<std::size_t>(k) + 1);
    int m = 0;
    const double hb = 0.5 * (1.0 - b);
    for (int i = 0; i <= k; ++i)
    {
        jacobi_all(b, 2.0 * i + 1.0, 0.0, k - i, Pb.data());
        const double hbi = std::pow(hb, i);                  // (0.5(1-b))^i
        const double hbim1 = i > 0 ? std::pow(hb, i - 1) : 0.0;
        for (int j = 0; j <= k - i; ++j, ++m)
        {
            const double fa = Pa[i], gb = Pb[j];
            // psi = sqrt(2) P_i(a) P_j^{(2i+1,0)}(b) (1-b)^i = 2^{i+1/2} fa gb hb^i
            const double c = std::pow(2.0, i + 0.5);
            v[m] = c * fa * gb * hbi;
            if (!dxi && !deta) continue;
            const double dfa = dPa[i];
            const double dgb = jacobi_derivative(b, 2.0 * i + 1.0, 0.0, j);
            double dr = dfa * gb;
            if (i > 0) dr *= hbim1;
            double ds = dfa * gb * 0.5 * (1.0 + a);
            if (i > 0) ds *= hbim1;
            double tmp = dgb * hbi;
            if (i > 0) tmp -= 0.5 * i * gb * hbim1;
            ds += fa * tmp;
            dr *= c; ds *= c;
            if (dxi)  dxi[m]  = 2.0 * dr;                        // d/dxi = 2 d/dr
            if (deta) deta[m] = 2.0 * ds;
        }
    }
}

/// @brief Gauss-Jordan inverse of a dense row-major n x n matrix, with partial pivoting.
inline std::vector<double> inverse(std::vector<double> A, int n)
{
    std::vector<double> X(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) X[static_cast<std::size_t>(i) * n + i] = 1.0;
    for (int c = 0; c < n; ++c)
    {
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::abs(A[static_cast<std::size_t>(r) * n + c]) > std::abs(A[static_cast<std::size_t>(piv) * n + c])) piv = r;
        if (A[static_cast<std::size_t>(piv) * n + c] == 0.0) throw std::runtime_error("simplex::inverse: singular matrix");
        if (piv != c)
            for (int j = 0; j < n; ++j)
            {
                std::swap(A[static_cast<std::size_t>(piv) * n + j], A[static_cast<std::size_t>(c) * n + j]);
                std::swap(X[static_cast<std::size_t>(piv) * n + j], X[static_cast<std::size_t>(c) * n + j]);
            }
        const double d = A[static_cast<std::size_t>(c) * n + c];
        for (int j = 0; j < n; ++j) { A[static_cast<std::size_t>(c) * n + j] /= d; X[static_cast<std::size_t>(c) * n + j] /= d; }
        for (int r = 0; r < n; ++r)
        {
            if (r == c) continue;
            const double f = A[static_cast<std::size_t>(r) * n + c];
            if (f == 0.0) continue;
            for (int j = 0; j < n; ++j)
            {
                A[static_cast<std::size_t>(r) * n + j] -= f * A[static_cast<std::size_t>(c) * n + j];
                X[static_cast<std::size_t>(r) * n + j] -= f * X[static_cast<std::size_t>(c) * n + j];
            }
        }
    }
    return X;
}

/// @brief Hesthaven's warp factor: the Gauss-Lobatto minus equispaced displacement,
///        interpolated at r and divided by the blend 1 - r^2.
inline double warp_factor(int N, double r, const std::vector<double> &lgl)
{
    double w = 0.0;
    for (int i = 0; i <= N; ++i)
    {
        const double ri = -1.0 + 2.0 * i / N;
        double L = 1.0;
        for (int j = 0; j <= N; ++j)
            if (j != i) L *= (r - (-1.0 + 2.0 * j / N)) / (ri - (-1.0 + 2.0 * j / N));
        w += L * (lgl[i] - ri);
    }
    if (std::abs(r) < 1.0 - 1e-10) return w / (1.0 - r * r);
    return 0.0;
}

/// @brief The warp-and-blend nodes of degree N >= 1 as (xi, eta) on T, lattice order.
inline std::vector<std::array<double, 2>> warp_blend(int N)
{
    static const double alpopt[] = {0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
                                    1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258};
    const double alpha = N < 16 ? alpopt[N - 1] : 5.0 / 3.0;
    const std::vector<double> lgl = detail::gauss_lobatto_nodes(N);
    const double pi = 3.14159265358979323846, s3 = std::sqrt(3.0);
    std::vector<std::array<double, 2>> out;
    for (int n = 0; n <= N; ++n)
        for (int m = 0; m <= N - n; ++m)
        {
            const double L1 = static_cast<double>(n) / N, L3 = static_cast<double>(m) / N, L2 = 1.0 - L1 - L3;
            double x = -L2 + L3, y = (-L2 - L3 + 2.0 * L1) / s3;
            const double blend1 = 4.0 * L2 * L3, blend2 = 4.0 * L1 * L3, blend3 = 4.0 * L1 * L2;
            const double warp1 = blend1 * warp_factor(N, L3 - L2, lgl) * (1.0 + (alpha * L1) * (alpha * L1));
            const double warp2 = blend2 * warp_factor(N, L1 - L3, lgl) * (1.0 + (alpha * L2) * (alpha * L2));
            const double warp3 = blend3 * warp_factor(N, L2 - L1, lgl) * (1.0 + (alpha * L3) * (alpha * L3));
            x += warp1 + std::cos(2.0 * pi / 3.0) * warp2 + std::cos(4.0 * pi / 3.0) * warp3;
            y += std::sin(2.0 * pi / 3.0) * warp2 + std::sin(4.0 * pi / 3.0) * warp3;
            // equilateral (x, y) -> Hesthaven (r, s) -> T
            const double l1 = (s3 * y + 1.0) / 3.0, l2 = (-3.0 * x - s3 * y + 2.0) / 6.0, l3 = (3.0 * x - s3 * y + 2.0) / 6.0;
            const double r = -l2 + l3 - l1, s = -l2 - l3 + l1;
            out.push_back({0.5 * (r + 1.0), 0.5 * (s + 1.0)});
        }
    return out;
}

} // namespace simplex

/**
 * @brief The reference Lagrange element of degree k >= 0 on T (k = 0: one node at the centroid).
 *        equispaced = true (the default) puts the nodes on the uniform lattice (i/k, j/k). false gives the
 *        warp-and-blend points; the space is the same, the basis and its conditioning are not.
 */
class ReferenceTriangle {
public:
    explicit ReferenceTriangle(int k, bool equispaced = true) : k_(k), n_(simplex::nmodes(k)), equi_(equispaced)
    {
        if (k < 0) throw std::invalid_argument("ReferenceTriangle: degree must be >= 0");
        if (k > 20) throw std::invalid_argument("ReferenceTriangle: degree " + std::to_string(k) + " is above the supported 20");
        build_nodes();
        std::vector<double> V(static_cast<std::size_t>(n_) * n_);
        std::vector<double> psi(static_cast<std::size_t>(n_));
        for (int i = 0; i < n_; ++i)
        {
            simplex::dubiner(k_, xi_[i], eta_[i], psi.data(), nullptr, nullptr);
            for (int m = 0; m < n_; ++m) V[static_cast<std::size_t>(i) * n_ + m] = psi[m];
        }
        Vinv_ = simplex::inverse(V, n_);
    }

    int degree() const { return k_; }
    int size() const { return n_; }
    bool equispaced() const { return equi_; }
    double node_xi(int i) const { return xi_[i]; }
    double node_eta(int i) const { return eta_[i]; }
    /// @brief Local index of node j (0-based, 0..k-2) of edge e, running from vertex (e+1)%3 to (e+2)%3.
    int edge_node(int e, int j) const { return 3 + e * (k_ - 1) + j; }
    int first_interior() const { return k_ == 0 ? 0 : 3 + 3 * (k_ - 1); }

    /**
     * @brief phi_l and, for nder >= 1, d phi_l / d xi and d phi_l / d eta at (xi, eta).
     * @param out  (1 + 2*[nder>=1]) * size() values: [0..n) values, [n..2n) d/dxi, [2n..3n) d/deta
     */
    void eval(double xi, double eta, int nder, double *out) const
    {
        std::vector<double> psi(static_cast<std::size_t>(3) * n_);
        double *p0 = psi.data(), *p1 = p0 + n_, *p2 = p1 + n_;
        simplex::dubiner(k_, xi, eta, p0, nder >= 1 ? p1 : nullptr, nder >= 1 ? p2 : nullptr);
        const int nrows = nder >= 1 ? 3 : 1;
        for (int r = 0; r < nrows; ++r)
        {
            const double *pr = psi.data() + static_cast<std::size_t>(r) * n_;
            double *o = out + static_cast<std::size_t>(r) * n_;
            for (int l = 0; l < n_; ++l) o[l] = 0.0;
            for (int m = 0; m < n_; ++m)
            {
                const double a = pr[m];
                if (a == 0.0) continue;
                const double *row = &Vinv_[static_cast<std::size_t>(m) * n_];
                for (int l = 0; l < n_; ++l) o[l] += a * row[l];
            }
        }
    }

private:
    void build_nodes()
    {
        xi_.clear(); eta_.clear();
        if (k_ == 0) { xi_.push_back(1.0 / 3.0); eta_.push_back(1.0 / 3.0); return; }
        const double vx[3] = {0.0, 1.0, 0.0}, vy[3] = {0.0, 0.0, 1.0};
        for (int v = 0; v < 3; ++v) { xi_.push_back(vx[v]); eta_.push_back(vy[v]); }
        // edge nodes: symmetrized Gauss-Lobatto positions, or the uniform ones
        std::vector<double> z = equi_ ? std::vector<double>() : detail::gauss_lobatto_nodes(k_);
        if (equi_) for (int j = 0; j <= k_; ++j) z.push_back(-1.0 + 2.0 * j / k_);
        for (int j = 0; j <= k_; ++j) { double s = 0.5 * (z[j] - z[k_ - j]); z[j] = s; }
        for (int e = 0; e < 3; ++e)
        {
            const int a = (e + 1) % 3, b = (e + 2) % 3;
            for (int j = 1; j < k_; ++j)
            {
                const double t = 0.5 * (1.0 + z[j]);
                xi_.push_back((1.0 - t) * vx[a] + t * vx[b]);
                eta_.push_back((1.0 - t) * vy[a] + t * vy[b]);
            }
        }
        // interior nodes: the uniform lattice, or the warp-and-blend points off the boundary
        if (k_ >= 3 && equi_)
        {
            for (int j = 1; j < k_; ++j)
                for (int i = 1; i + j < k_; ++i) { xi_.push_back(static_cast<double>(i) / k_); eta_.push_back(static_cast<double>(j) / k_); }
        }
        else if (k_ >= 3)
        {
            const double tol = 1e-10;
            for (const auto &p : simplex::warp_blend(k_))
            {
                const double b0 = 1.0 - p[0] - p[1];
                if (p[0] > tol && p[1] > tol && b0 > tol) { xi_.push_back(p[0]); eta_.push_back(p[1]); }
            }
        }
        if (static_cast<int>(xi_.size()) != n_)
            throw std::logic_error("ReferenceTriangle: node count " + std::to_string(xi_.size()) + " != " + std::to_string(n_));
    }

    int k_, n_;
    bool equi_ = false;
    std::vector<double> xi_, eta_, Vinv_;
};

} // namespace femd

#endif // FEMD_FE_SIMPLEX_BASIS_HPP
