//
//  block.hpp  --  a sparse block matrix from sparse blocks: the s n x s n stage Jacobian
//  delta_ij M - dt a_ij J_i of an implicit Runge-Kutta step, or any other matrix made of CSR
//  blocks with coefficients, several blocks on one position summed.
//
//      std::vector<BlockEntry> e = {{0, 0, &M, 1.0}, {0, 0, &J, -dt * a00}, {0, 1, &J, -dt * a01}, ...};
//      CSRMatrix K = block_csr({0, n, 2 * n}, {0, n, 2 * n}, e);
//
#ifndef FEMD_SPARSE_BLOCK_HPP
#define FEMD_SPARSE_BLOCK_HPP

#include "femd/sparse/csr.hpp"
#include "femd/util/omp.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace femd {

struct BlockEntry {
    int i = 0, j = 0;                ///< block row and column
    const CSRMatrix *K = nullptr;    ///< the block
    double alpha = 1.0;              ///< its coefficient
};

/**
 * @brief The block matrix with block rows [roff[i], roff[i+1]) and block columns [coff[j], coff[j+1]).
 *        Each row of the result is the merge of the rows of the blocks in its block row, sorted,
 *        duplicates summed in the order the entries are given.
 */
inline CSRMatrix block_csr(const std::vector<int> &roff, const std::vector<int> &coff, const std::vector<BlockEntry> &entries)
{
    const int nbr = static_cast<int>(roff.size()) - 1, nbc = static_cast<int>(coff.size()) - 1;
    if (nbr < 1 || nbc < 1) throw std::invalid_argument("block_csr: offsets need at least two entries");
    for (const BlockEntry &e : entries)
    {
        if (e.i < 0 || e.i >= nbr || e.j < 0 || e.j >= nbc || e.K == nullptr) throw std::invalid_argument("block_csr: block position out of range");
        if (e.K->nrows() != roff[e.i + 1] - roff[e.i] || e.K->ncols() != coff[e.j + 1] - coff[e.j])
            throw std::invalid_argument("block_csr: a block does not fit its position");
    }
    const int nrows = roff[nbr], ncols = coff[nbc];
    auto P = std::make_shared<SparsityPattern>();
    P->nrows = nrows; P->ncols = ncols;
    P->indptr.assign(static_cast<std::size_t>(nrows) + 1, 0);
    // the entries of each block row, then the rows: gather (col, val) pairs, sort, merge
    std::vector<std::vector<int>> by_row(static_cast<std::size_t>(nbr));
    for (std::size_t e = 0; e < entries.size(); ++e) by_row[entries[e].i].push_back(static_cast<int>(e));
    std::vector<std::vector<std::pair<int, double>>> rows(static_cast<std::size_t>(nrows));
    for (int bi = 0; bi < nbr; ++bi)
    {
        const int r0 = roff[bi], r1 = roff[bi + 1];
        FEMD_OMP_FOR_IF(r1 - r0 > FEMD_OMP_THRESHOLD)
        for (int r = r0; r < r1; ++r)
        {
            auto &row = rows[r];
            for (int e : by_row[bi])
            {
                const BlockEntry &B = entries[e];
                const int rr = r - r0, c0 = coff[B.j];
                for (int p = B.K->indptr()[rr]; p < B.K->indptr()[rr + 1]; ++p)
                    row.push_back({c0 + B.K->indices()[p], B.alpha * B.K->data[p]});
            }
            std::stable_sort(row.begin(), row.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            std::size_t w = 0;
            for (std::size_t k = 0; k < row.size(); ++k)
            {
                if (w > 0 && row[w - 1].first == row[k].first) row[w - 1].second += row[k].second;
                else row[w++] = row[k];
            }
            row.resize(w);
        }
    }
    for (int r = 0; r < nrows; ++r) P->indptr[r + 1] = P->indptr[r] + static_cast<int>(rows[r].size());
    P->indices.resize(static_cast<std::size_t>(P->indptr[nrows]));
    CSRMatrix K(P, false);
    for (int r = 0; r < nrows; ++r)
    {
        int p = P->indptr[r];
        for (const auto &cv : rows[r]) { P->indices[p] = cv.first; K.data[p] = cv.second; ++p; }
    }
    return K;
}

} // namespace femd

#endif // FEMD_SPARSE_BLOCK_HPP
