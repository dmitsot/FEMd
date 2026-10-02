//
//  block_assembly.hpp  --  the rank-1 and rank-2 kernels for a ProductSpace:
//  block (I,J) of the global matrix, scattered through the product numbering.
//
//      K[g(I,i), g(J,j)] += integral c (D^a N^I_i)(D^b N^J_j)
//
//  The two caches must be built on the same mesh with the same rule (checked).
//  An off-diagonal block sets K.symmetric = false, since the assembler cannot
//  know whether the transposed block will also be assembled with the same
//  coefficient; the caller may reset the flag when the system is symmetric.
//
#ifndef FEMD_FORMS_BLOCK_ASSEMBLY_HPP
#define FEMD_FORMS_BLOCK_ASSEMBLY_HPP

#include "femd/fe/product_space.hpp"
#include "femd/forms/assembly.hpp"
#include <algorithm>
#include <cmath>

namespace femd {

namespace detail {
inline void check_caches(const QuadratureCache &QI, const QuadratureCache &QJ)
{
    if (QI.nelem() != QJ.nelem() || QI.nq() != QJ.nq())
        throw std::invalid_argument("block assembly: the two caches must share the mesh and the quadrature rule");
    // same element count is not the same mesh: compare the quadrature nodes themselves
    const std::vector<double> &a = QI.nodes(), &b = QJ.nodes();
    double scale = 0.0, worst = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k)
    {
        scale = std::max(scale, std::abs(a[k]));
        worst = std::max(worst, std::abs(a[k] - b[k]));
    }
    if (worst > 1e-13 * std::max(1.0, scale))
        throw std::invalid_argument("block assembly: the two caches have the same number of elements but different grids");
}
}

/**
 * @brief A rectangular block as triplets:  K_ij += integral c (D^a T_i)(D^b S_j),
 *        T_i the test basis (cache QT, rows) and S_j the trial basis (cache QS, columns).
 *
 * The two spaces share the mesh (checked) and may differ in family, degree,
 * continuity and boundary conditions, so the result is QT.space.dim x QS.space.dim.
 * Triplets are appended, duplicates included; the caller sums them (COO -> CSR).
 */
inline void assemble_rect(const QuadratureCache &QT, const QuadratureCache &QS, int a, int b, const Coefficient &c,
                          std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals)
{
    detail::check_caches(QT, QS);
    if (a > QT.nder() || b > QS.nder()) throw std::invalid_argument("rectangular assembly: cache holds fewer derivatives than requested");
    std::vector<double> cq = c.at_quad(QT);
    const std::size_t stride = static_cast<std::size_t>(QT.nloc_max()) * QS.nloc_max();
    detail::by_elements(QT.nelem(), QT.nq(), stride,
        [&](int e, double *Ke) { detail::element_matrix(QT, QS, cq, e, a, b, Ke); },
        [&](int e, const double *Ke) {
            const int ni = QT.nloc(e), nj = QS.nloc(e);
            for (int ki = 0; ki < ni; ++ki)
                for (int kj = 0; kj < nj; ++kj)
                {
                    rows.push_back(QT.dof(e, ki)); cols.push_back(QS.dof(e, kj));
                    vals.push_back(Ke[static_cast<std::size_t>(ki) * nj + kj]);
                }
        });
}

/// @brief Block (I,J):  K[g(I,i), g(J,j)] += integral c (D^a N^I_i)(D^b N^J_j).
inline void assemble_matrix(const ProductSpace &P, int I, int J,
                            const QuadratureCache &QI, const QuadratureCache &QJ,
                            int a, int b, const Coefficient &c, AssembledMatrix &K)
{
    detail::check_caches(QI, QJ);
    if (a > QI.nder() || b > QJ.nder()) throw std::invalid_argument("block assembly: cache holds fewer derivatives than requested");
    if (I != J || a != b) K.symmetric = false;
    std::vector<double> cq = c.at_quad(QI);
    const std::size_t stride = static_cast<std::size_t>(QI.nloc_max()) * QJ.nloc_max();
    detail::by_elements(QI.nelem(), QI.nq(), stride,
        [&](int e, double *Ke) { detail::element_matrix(QI, QJ, cq, e, a, b, Ke); },
        [&](int e, const double *Ke) {
            const int ni = QI.nloc(e), nj = QJ.nloc(e);
            for (int ki = 0; ki < ni; ++ki)
            {
                const int gi = P.global(I, QI.dof(e, ki));
                for (int kj = 0; kj < nj; ++kj)
                    K.add(gi, P.global(J, QJ.dof(e, kj)), Ke[static_cast<std::size_t>(ki) * nj + kj]);
            }
        });
}

/// @brief Block I of a right-hand side:  f[g(I,i)] += integral c (D^a N^I_i).
inline void assemble_vector(const ProductSpace &P, int I, const QuadratureCache &QI,
                            int a, const Coefficient &c, std::vector<double> &f)
{
    if (a > QI.nder()) throw std::invalid_argument("block assembly: cache holds fewer derivatives than requested");
    std::vector<double> cq = c.at_quad(QI);
    detail::by_elements(QI.nelem(), QI.nq(), static_cast<std::size_t>(QI.nloc_max()),
        [&](int e, double *fe) { detail::element_vector(QI, cq, e, a, fe); },
        [&](int e, const double *fe) { for (int k = 0; k < QI.nloc(e); ++k) f[P.global(I, QI.dof(e, k))] += fe[k]; });
}

} // namespace femd

#endif // FEMD_FORMS_BLOCK_ASSEMBLY_HPP
