//
//  prune.hpp  --  the pattern of a square CSR matrix without its exact zeros, for the
//  ordering and the symbolic analysis of the direct solvers (cholesky.hpp, ldlt.hpp, lu.hpp).
//
//  FEMd stores whole blocks in its patterns: a form on a ProductSpace gets every field pair,
//  also the pairs that no term of the form couples (the eta-u blocks of a Boussinesq left
//  side, the pressure block of Stokes).  Those entries are exactly zero.  An ordering and a
//  symbolic phase that take them for couplings join fields that are independent, and the
//  factor fills with entries that come out zero: 1.4 times the fill of the factor of the
//  matrix without them, on a Peregrine system of 224k unknowns.
//
//      ZeroPruning<double> Z;
//      Z.build(n, Ap, Ai, Ax, symmetric);   // keep the nonzero entries and the diagonal
//      if (Z.active) ...                    // something was dropped: factor Z.Bp, Z.Bi, values Z.gather(Ax)
//      Z.fits(Ax2)                          // every dropped entry is still zero: same analysis
//
//  The diagonal is always kept.  With symmetric = true an entry is kept when it or its mirror
//  is nonzero, so the pattern stays symmetric (the Cholesky and the LDL^T read one triangle).
//  A refactor with values that are nonzero where the analysis dropped them needs a new
//  analysis: build(..., grow = true) then keeps what was kept before as well, so a pattern
//  that changes back and forth is analysed again at most until it covers both.
//
#ifndef FEMD_SPARSE_PRUNE_HPP
#define FEMD_SPARSE_PRUNE_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace femd {

template <class T>
struct ZeroPruning {
    bool active = false;            // some entry was dropped
    int n = 0;
    std::vector<int> Bp, Bi;        // the kept pattern (CSR), when active
    std::vector<int> src;           // kept entry q is entry src[q] of the input
    std::vector<int> dropped;       // the positions of the dropped entries in the input
    std::vector<int> Ap0, Ai0;      // the input pattern, kept when active (for a new analysis)
    std::vector<char> keep;         // by input position, when active

    /// Decide which entries to keep.  grow: also keep those kept by the previous build.
    void build(int n_, const int *Ap, const int *Ai, const T *Ax, bool symmetric, bool grow = false)
    {
        n = n_;
        const int nnz = Ap[n];
        const bool had = grow && active && static_cast<int>(keep.size()) == nnz;
        std::vector<char> k(static_cast<std::size_t>(nnz), 0);
        int nkeep = 0;
        for (int i = 0; i < n; ++i)
            for (int p = Ap[i]; p < Ap[i + 1]; ++p)
            {
                const int j = Ai[p];
                bool on = j == i || Ax[p] != T(0) || (had && keep[p]);
                if (!on && symmetric)
                {
                    const int *b = Ai + Ap[j], *e = Ai + Ap[j + 1];
                    const int *m = std::lower_bound(b, e, i);
                    if (m != e && *m == i) on = Ax[m - Ai] != T(0) || (had && keep[m - Ai]);
                }
                k[p] = on ? 1 : 0;
                nkeep += on;
            }
        active = nkeep < nnz;
        dropped.clear(); src.clear(); Bp.clear(); Bi.clear();
        if (!active)
        {
            std::vector<char>().swap(keep);
            std::vector<int>().swap(Ap0); std::vector<int>().swap(Ai0);
            return;
        }
        keep.swap(k);
        Ap0.assign(Ap, Ap + n + 1);
        Ai0.assign(Ai, Ai + nnz);
        Bp.assign(static_cast<std::size_t>(n) + 1, 0);
        Bi.reserve(static_cast<std::size_t>(nkeep));
        src.reserve(static_cast<std::size_t>(nkeep));
        dropped.reserve(static_cast<std::size_t>(nnz - nkeep));
        for (int i = 0; i < n; ++i)
        {
            for (int p = Ap[i]; p < Ap[i + 1]; ++p)
                if (keep[p]) { Bi.push_back(Ai[p]); src.push_back(p); }
                else dropped.push_back(p);
            Bp[i + 1] = static_cast<int>(Bi.size());
        }
    }

    /// Every dropped entry of Ax (on the input pattern) is still exactly zero.
    bool fits(const T *Ax) const
    {
        for (int p : dropped) if (Ax[p] != T(0)) return false;
        return true;
    }

    /// The kept values of Ax, in the order of Bp, Bi.
    void gather(const T *Ax, std::vector<T> &Bx) const
    {
        Bx.resize(src.size());
        for (std::size_t q = 0; q < src.size(); ++q) Bx[q] = Ax[src[q]];
    }

    /// Entries dropped from the input.
    long long ndropped() const { return static_cast<long long>(dropped.size()); }
};

} // namespace femd

#endif // FEMD_SPARSE_PRUNE_HPP
