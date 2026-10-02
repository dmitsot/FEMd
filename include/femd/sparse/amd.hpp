//
//  amd.hpp  --  fill-reducing orderings for sparse symmetric factorizations.
//
//  amd_order() is an approximate minimum degree ordering: the method of Amestoy,
//  Davis and Duff (SIAM J. Matrix Anal. Appl. 17, 1996), written here from the
//  published description.  It eliminates on the quotient graph (variables and
//  elements in one integer array), bounds the external degree of each variable
//  instead of computing it exactly, merges indistinguishable variables into
//  supervariables (found by hashing their adjacency), eliminates variables that
//  lose all outside neighbours together with the pivot (mass elimination), and
//  absorbs elements that are covered by the new one (aggressive absorption).
//  Rows much denser than the rest (more than 10 sqrt(n) entries) are held back
//  and ordered last.  The result is post-ordered on the assembly tree.
//
//  The input is the pattern of a square matrix in CSR (or CSC) form; it is
//  symmetrized and the diagonal is ignored.  The output p is the new order:
//  the k-th pivot is the original row p[k].
//
#ifndef FEMD_SPARSE_AMD_HPP
#define FEMD_SPARSE_AMD_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace femd {
namespace ordering {

namespace detail {

inline int flip(int i) { return -i - 2; }

/// @brief Reset the marker array when the mark would overflow.
inline int wclear(int mark, int lemax, std::vector<int> &w, int n)
{
    if (mark < 2 || mark + lemax < 0)
    {
        for (int k = 0; k < n; ++k)
            if (w[k] != 0) w[k] = 1;
        mark = 2;
    }
    return mark;
}

/// @brief Symmetric adjacency (A + A^T) without the diagonal, as CSR arrays.
inline void symmetric_graph(int n, const int *Ap, const int *Ai, std::vector<int> &Cp, std::vector<int> &Ci)
{
    std::vector<int> cnt(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        for (int p = Ap[i]; p < Ap[i + 1]; ++p)
        {
            const int j = Ai[p];
            if (j < 0 || j >= n) throw std::invalid_argument("amd: column index out of range");
            if (j != i) { ++cnt[i]; ++cnt[j]; }
        }
    Cp.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) Cp[i + 1] = Cp[i] + cnt[i];
    std::vector<int> fill(Cp.begin(), Cp.end() - 1);
    Ci.assign(static_cast<std::size_t>(Cp[n]), 0);
    for (int i = 0; i < n; ++i)
        for (int p = Ap[i]; p < Ap[i + 1]; ++p)
        {
            const int j = Ai[p];
            if (j != i) { Ci[fill[i]++] = j; Ci[fill[j]++] = i; }
        }
    // sort and remove duplicates row by row (a structurally symmetric input lists each edge twice)
    int q = 0;
    std::vector<int> start(Cp);
    for (int i = 0; i < n; ++i)
    {
        auto b = Ci.begin() + start[i], e = Ci.begin() + start[i + 1];
        std::sort(b, e);
        auto u = std::unique(b, e);
        const int keep = static_cast<int>(u - b);
        std::copy(b, u, Ci.begin() + q);
        Cp[i] = q;
        q += keep;
    }
    Cp[n] = q;
    Ci.resize(static_cast<std::size_t>(q));
}

} // namespace detail

/// @brief The identity order.
inline std::vector<int> natural_order(int n)
{
    std::vector<int> p(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) p[k] = k;
    return p;
}

/**
 * @brief Reverse Cuthill-McKee ordering of the pattern of a square matrix (symmetrized, diagonal
 *        ignored).  Each connected component starts from a pseudo-peripheral node (the George-Liu
 *        search from a node of least degree), is numbered breadth first with the neighbours of a
 *        node taken by increasing degree, and the whole order is reversed.  It reduces the profile
 *        and bandwidth, which keeps the fill that ILU(0) and SSOR drop small.
 * @return p with p[k] the original index of the k-th row
 */
inline std::vector<int> rcm_order(int n, const int *Ap, const int *Ai)
{
    std::vector<int> Cp, Ci;
    detail::symmetric_graph(n, Ap, Ai, Cp, Ci);
    std::vector<int> deg(static_cast<std::size_t>(n)), level(static_cast<std::size_t>(n), -1), order;
    order.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) deg[i] = Cp[i + 1] - Cp[i];
    std::vector<char> done(static_cast<std::size_t>(n), 0);
    std::vector<int> queue(static_cast<std::size_t>(n)), touched;
    // breadth-first level structure from s among the nodes not yet numbered; returns the eccentricity,
    // and the last level in [queue[first_last], queue[end])
    auto bfs = [&](int s, int &first_last, int &end) {
        touched.clear();
        int head = 0, tail = 0, ecc = 0;
        queue[tail++] = s; level[s] = 0; touched.push_back(s);
        first_last = 0;
        while (head < tail)
        {
            const int v = queue[head++];
            if (level[v] > ecc) { ecc = level[v]; first_last = head - 1; }
            for (int p = Cp[v]; p < Cp[v + 1]; ++p)
            {
                const int w = Ci[p];
                if (done[w] || level[w] >= 0) continue;
                level[w] = level[v] + 1;
                queue[tail++] = w;
                touched.push_back(w);
            }
        }
        end = tail;
        for (int v : touched) level[v] = -1;
        return ecc;
    };
    std::vector<int> byd(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) byd[i] = i;
    std::stable_sort(byd.begin(), byd.end(), [&](int a, int b) { return deg[a] < deg[b]; });
    std::vector<int> nbr;
    for (int seed : byd)
    {
        if (done[seed]) continue;
        // George-Liu pseudo-peripheral node
        int s = seed, fl, end;
        int ecc = bfs(s, fl, end);
        for (int it = 0; it < 32; ++it)
        {
            int best = queue[fl];
            for (int t = fl; t < end; ++t) if (deg[queue[t]] < deg[best]) best = queue[t];
            int fl2, end2;
            const int e2 = bfs(best, fl2, end2);
            if (e2 <= ecc) break;
            s = best; ecc = e2;
            bfs(s, fl, end);
        }
        // Cuthill-McKee from s
        int head = static_cast<int>(order.size());
        order.push_back(s); done[s] = 1;
        while (head < static_cast<int>(order.size()))
        {
            const int v = order[head++];
            nbr.clear();
            for (int p = Cp[v]; p < Cp[v + 1]; ++p)
                if (!done[Ci[p]]) { nbr.push_back(Ci[p]); done[Ci[p]] = 1; }
            std::stable_sort(nbr.begin(), nbr.end(), [&](int a, int b) { return deg[a] < deg[b]; });
            order.insert(order.end(), nbr.begin(), nbr.end());
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

/// @brief The half-bandwidth max |pinv[i] - pinv[j]| of the pattern under the order p.
inline int bandwidth(int n, const int *Ap, const int *Ai, const std::vector<int> &p)
{
    std::vector<int> pinv(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) pinv[p[k]] = k;
    int b = 0;
    for (int i = 0; i < n; ++i)
        for (int q = Ap[i]; q < Ap[i + 1]; ++q) b = std::max(b, std::abs(pinv[i] - pinv[Ai[q]]));
    return b;
}

/**
 * @brief Approximate minimum degree ordering of the pattern of a square matrix.
 * @param n   dimension
 * @param Ap  row pointers (n + 1)
 * @param Ai  column indices
 * @return    p with p[k] the original index of the k-th pivot
 */
inline std::vector<int> amd_order(int n, const int *Ap, const int *Ai)
{
    using detail::flip;
    if (n <= 0) return {};
    std::vector<int> Cp, Ci;
    detail::symmetric_graph(n, Ap, Ai, Cp, Ci);
    const int cnz0 = Cp[n];
    int dense = static_cast<int>(std::max(16.0, 10.0 * std::sqrt(static_cast<double>(n))));
    dense = std::min(n - 2, dense);

    const std::size_t N1 = static_cast<std::size_t>(n) + 1;
    const long long want = static_cast<long long>(cnz0) + cnz0 / 5 + 2LL * n + 1;
    const int nzmax = static_cast<int>(want);
    std::vector<int> iw(static_cast<std::size_t>(nzmax));
    std::copy(Ci.begin(), Ci.end(), iw.begin());
    std::vector<int> pe(N1), len(N1), nv(N1), next(N1), head(N1), elen(N1), degree(N1), w(N1), hhead(N1), last(N1);
    for (int i = 0; i < n; ++i) { pe[i] = Cp[i]; len[i] = Cp[i + 1] - Cp[i]; }
    pe[n] = -1; len[n] = 0;
    int cnz = cnz0;
    for (int i = 0; i <= n; ++i)
    {
        last[i] = -1; head[i] = -1; next[i] = -1; hhead[i] = -1;
        nv[i] = 1; w[i] = 1; elen[i] = 0; degree[i] = len[i];
    }
    int lemax = 0;
    int mark = detail::wclear(0, 0, w, n);
    elen[n] = -2; w[n] = 0; nv[n] = 0;
    int nel = 0;
    // initial degree lists; empty rows are eliminated at once, dense rows are set aside
    for (int i = 0; i < n; ++i)
    {
        const int d = degree[i];
        if (d == 0)
        {
            elen[i] = -2; ++nel; pe[i] = -1; w[i] = 0;
        }
        else if (d > dense)
        {
            nv[i] = 0; elen[i] = -1; ++nel; pe[i] = flip(n); ++nv[n];
        }
        else
        {
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d];
            head[d] = i;
        }
    }

    int mindeg = 0;
    while (nel < n)
    {
        // ---- the pivot: a variable of least (approximate) degree
        int k = -1;
        for (; mindeg < n && (k = head[mindeg]) == -1; ++mindeg) {}
        if (k < 0) throw std::logic_error("amd: no pivot found");
        if (next[k] != -1) last[next[k]] = -1;
        head[mindeg] = next[k];
        const int elenk = elen[k];
        int nvk = nv[k];
        nel += nvk;

        // ---- garbage collection when the new element might not fit
        if (elenk > 0 && cnz + mindeg >= nzmax)
        {
            for (int j = 0; j < n; ++j)
            {
                const int p = pe[j];
                if (p >= 0) { pe[j] = iw[p]; iw[p] = flip(j); }
            }
            int q = 0;
            for (int p = 0; p < cnz;)
            {
                const int j = flip(iw[p++]);
                if (j >= 0)
                {
                    iw[q] = pe[j];
                    pe[j] = q++;
                    for (int t = 0; t < len[j] - 1; ++t) iw[q++] = iw[p++];
                }
            }
            cnz = q;
        }

        // ---- the new element L_k: the union of the pivot's elements and variables
        int dk = 0;
        nv[k] = -nvk;
        int p = pe[k];
        const int pk1 = (elenk == 0) ? p : cnz;
        int pk2 = pk1;
        for (int k1 = 1; k1 <= elenk + 1; ++k1)
        {
            int e, pj, ln;
            if (k1 > elenk) { e = k; pj = p; ln = len[k] - elenk; }
            else { e = iw[p++]; pj = pe[e]; ln = len[e]; }
            for (int k2 = 1; k2 <= ln; ++k2)
            {
                const int i = iw[pj++];
                const int nvi = nv[i];
                if (nvi <= 0) continue;
                dk += nvi;
                nv[i] = -nvi;
                iw[pk2++] = i;
                if (next[i] != -1) last[next[i]] = last[i];
                if (last[i] != -1) next[last[i]] = next[i];
                else head[degree[i]] = next[i];
            }
            if (e != k) { pe[e] = flip(k); w[e] = 0; }
        }
        if (elenk != 0) cnz = pk2;
        degree[k] = dk;
        pe[k] = pk1;
        len[k] = pk2 - pk1;
        elen[k] = -2;

        // ---- |L_e \ L_k| for every element e next to a variable of L_k
        mark = detail::wclear(mark, lemax, w, n);
        for (int pk = pk1; pk < pk2; ++pk)
        {
            const int i = iw[pk];
            const int eln = elen[i];
            if (eln <= 0) continue;
            const int nvi = -nv[i];
            const int wnvi = mark - nvi;
            for (int q = pe[i]; q <= pe[i] + eln - 1; ++q)
            {
                const int e = iw[q];
                if (w[e] >= mark) w[e] -= nvi;
                else if (w[e] != 0) w[e] = degree[e] + wnvi;
            }
        }

        // ---- degree update and element absorption
        for (int pk = pk1; pk < pk2; ++pk)
        {
            const int i = iw[pk];
            const int p1 = pe[i];
            const int p2 = p1 + elen[i] - 1;
            int pn = p1;
            unsigned long h = 0;
            int d = 0;
            for (int q = p1; q <= p2; ++q)
            {
                const int e = iw[q];
                if (w[e] != 0)
                {
                    const int dext = w[e] - mark;
                    if (dext > 0) { d += dext; iw[pn++] = e; h += static_cast<unsigned long>(e); }
                    else { pe[e] = flip(k); w[e] = 0; }              // aggressive absorption
                }
            }
            elen[i] = pn - p1 + 1;
            const int p3 = pn;
            const int p4 = p1 + len[i];
            for (int q = p2 + 1; q < p4; ++q)
            {
                const int j = iw[q];
                const int nvj = nv[j];
                if (nvj <= 0) continue;
                d += nvj;
                iw[pn++] = j;
                h += static_cast<unsigned long>(j);
            }
            if (d == 0)                                               // mass elimination
            {
                pe[i] = flip(k);
                const int nvi = -nv[i];
                dk -= nvi; nvk += nvi; nel += nvi;
                nv[i] = 0; elen[i] = -1;
            }
            else
            {
                degree[i] = std::min(degree[i], d);
                iw[pn] = iw[p3];
                iw[p3] = iw[p1];
                iw[p1] = k;
                len[i] = pn - p1 + 1;
                const int hb = static_cast<int>(h % static_cast<unsigned long>(n));
                next[i] = hhead[hb];
                hhead[hb] = i;
                last[i] = hb;                                         // last holds the hash here
            }
        }
        degree[k] = dk;
        lemax = std::max(lemax, dk);
        mark = detail::wclear(mark + lemax, lemax, w, n);

        // ---- supervariables: variables of L_k with identical adjacency merge
        for (int pk = pk1; pk < pk2; ++pk)
        {
            int i = iw[pk];
            if (nv[i] >= 0) continue;
            const int hb = last[i];
            i = hhead[hb];
            hhead[hb] = -1;
            for (; i != -1 && next[i] != -1; i = next[i], ++mark)
            {
                const int ln = len[i], eln = elen[i];
                for (int q = pe[i] + 1; q <= pe[i] + ln - 1; ++q) w[iw[q]] = mark;
                int jlast = i;
                for (int j = next[i]; j != -1;)
                {
                    bool ok = (len[j] == ln) && (elen[j] == eln);
                    for (int q = pe[j] + 1; ok && q <= pe[j] + ln - 1; ++q)
                        if (w[iw[q]] != mark) ok = false;
                    if (ok)
                    {
                        pe[j] = flip(i);
                        nv[i] += nv[j];
                        nv[j] = 0;
                        elen[j] = -1;
                        j = next[j];
                        next[jlast] = j;
                    }
                    else
                    {
                        jlast = j;
                        j = next[j];
                    }
                }
            }
        }

        // ---- finalize L_k: principal variables back into the degree lists
        int pw = pk1;
        for (int pk = pk1; pk < pk2; ++pk)
        {
            const int i = iw[pk];
            const int nvi = -nv[i];
            if (nvi <= 0) continue;
            nv[i] = nvi;
            int d = degree[i] + dk - nvi;
            d = std::min(d, n - nel - nvi);
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d];
            last[i] = -1;
            head[d] = i;
            mindeg = std::min(mindeg, d);
            degree[i] = d;
            iw[pw++] = i;
        }
        nv[k] = nvk;
        len[k] = pw - pk1;
        if (len[k] == 0) { pe[k] = -1; w[k] = 0; }
        if (elenk != 0) cnz = pw;
    }

    // ---- post-order the assembly tree
    std::vector<int> parent(N1, -1), child(N1, -1), sibling(N1, -1);
    for (int i = 0; i < n; ++i) parent[i] = (pe[i] <= -2) ? flip(pe[i]) : -1;
    parent[n] = -1;
    // children lists: the variables merged into a node (nv = 0) first, then the elements it
    // absorbed are put in front of them, so that a post-order eliminates every absorbed
    // subtree before the variables that were merged with the node, and those just before it
    for (int pass = 0; pass < 2; ++pass)
        for (int j = n; j >= 0; --j)
        {
            const int q = parent[j];
            if (q < 0 || (pass == 0) != (nv[j] <= 0)) continue;
            sibling[j] = child[q];
            child[q] = j;
        }
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(n));
    std::vector<int> stack;
    for (int r = 0; r <= n; ++r)
    {
        if (parent[r] != -1) continue;
        stack.push_back(r);
        while (!stack.empty())
        {
            const int j = stack.back();
            const int c = child[j];
            if (c == -1)
            {
                stack.pop_back();
                if (j < n) order.push_back(j);
            }
            else
            {
                child[j] = sibling[c];                                // detach the first child
                stack.push_back(c);
            }
        }
    }
    if (static_cast<int>(order.size()) != n) throw std::logic_error("amd: the ordering lost nodes");
    return order;
}

} // namespace ordering
} // namespace femd

#endif // FEMD_SPARSE_AMD_HPP
