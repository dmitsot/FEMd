//
//  linear_solver.hpp  --  backend selection and factorization reuse over the
//  inherited direct solvers (design doc, Section 16).
//
//  A LinearSolver is an explicit object with a visible lifetime: construct it
//  once outside the time loop, call solve() as often as needed.  The inherited
//  backends factor on their first solve and keep the factor.
//
//  This is the ONE place where 0-based FEMd indices become the 1-based
//  matrix<T> indices of the inherited linalg (design doc, Section 11.3).
//
#ifndef FEMD_SOLVE_LINEAR_SOLVER_HPP
#define FEMD_SOLVE_LINEAR_SOLVER_HPP

#include "femd/forms/assembly.hpp"
#include "femd/linalg/linalg.hpp"
#ifdef FEMD_HAVE_FFTW
#include "femd/fft/circulant.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace femd {

// New members go at the END: the Python binding exposes the values.
enum class Solver { Auto, Dense, Band, SymBand, Cyclic, SymCyclic, Circulant, Diagonal };

inline bool has_fftw()
{
#ifdef FEMD_HAVE_FFTW
    return true;
#else
    return false;
#endif
}

inline const char *solver_name(Solver s)
{
    switch (s)
    {
    case Solver::Auto: return "Auto";
    case Solver::Dense: return "Dense";
    case Solver::Band: return "Band";
    case Solver::SymBand: return "SymBand";
    case Solver::Cyclic: return "Cyclic";
    case Solver::SymCyclic: return "SymCyclic";
    case Solver::Circulant: return "Circulant";
    case Solver::Diagonal: return "Diagonal";
    }
    return "?";
}

/// @brief Pick a backend from periodicity and symmetry alone (the flags; see detail::measure_band for Auto).
inline Solver autoselect(bool periodic, bool symmetric)
{
    if (periodic) return symmetric ? Solver::SymCyclic : Solver::Cyclic;
    return symmetric ? Solver::SymBand : Solver::Band;
}

namespace detail {

/// Copy the 0-based banded buffer into a 1-based solver matrix.  Overflow
/// entries are summed per (i,j) first, since assignment must be idempotent.
template <class M>
void fill(M &A, const AssembledMatrix &S)
{
    for (int i = 0; i < S.n; ++i)
    {
        const double *row = &S.band[static_cast<std::size_t>(i) * S.w];
        for (int off = -S.p; off <= S.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= S.n) continue;
            double v = row[off + S.p];
            if (v != 0.0) A(i + 1, j + 1) = v;
        }
    }
    if (!S.ov.empty())
    {
        std::map<std::pair<int, int>, double> corner;
        for (std::size_t k = 0; k < S.ov.size(); ++k) corner[std::make_pair(S.oi[k], S.oj[k])] += S.ov[k];
        for (const auto &kv : corner) A(kv.first.first + 1, kv.first.second + 1) = kv.second;
    }
}

/// Relative size below which an entry counts as zero when the structure is measured: a few
/// rounding errors, so the quadrature noise of an orthogonal basis (1e-17) is not structure.
constexpr double kZeroTol = 1e-14;

inline double max_abs(const AssembledMatrix &S)
{
    double m = 0.0;
    for (double v : S.band) m = std::max(m, std::abs(v));
    std::map<std::pair<int, int>, double> corner;
    for (std::size_t k = 0; k < S.ov.size(); ++k) corner[std::make_pair(S.oi[k], S.oj[k])] += S.ov[k];
    for (const auto &kv : corner) m = std::max(m, std::abs(kv.second));
    return m;
}

/**
 * @brief The band a factorization actually needs.
 * @param bw    largest |i - j| of an entry above zero_tol * max|entry|, ignoring the periodic wrap
 * @param cbw   the same measured cyclically, max min(|i-j|, n-|i-j|), when wrap entries are present
 * @param wrap  whether any entry above the threshold lies outside the plain band (a periodic corner)
 */
inline void measure_band(const AssembledMatrix &S, double zero_tol, int &bw, int &cbw, bool &wrap)
{
    const int n = S.n;
    const double thr = zero_tol * max_abs(S);
    bw = 0; cbw = 0; wrap = false;
    for (int i = 0; i < n; ++i)
        for (int off = -S.p; off <= S.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= n) continue;
            if (std::abs(S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)]) > thr)
            {
                int d = std::abs(off);
                bw = std::max(bw, d);
                cbw = std::max(cbw, std::min(d, n - d));
            }
        }
    std::map<std::pair<int, int>, double> corner;
    for (std::size_t k = 0; k < S.ov.size(); ++k) corner[std::make_pair(S.oi[k], S.oj[k])] += S.ov[k];
    for (const auto &kv : corner)
        if (std::abs(kv.second) > thr)
        {
            int d = std::abs(kv.first.first - kv.first.second);
            wrap = true;
            cbw = std::max(cbw, std::min(d, n - d));
        }
    if (!wrap) cbw = bw;
}

/**
 * @brief A copy of S stored with half-bandwidth p.  Every entry inside the new band is kept
 *        exactly; an entry outside it goes to the overflow list when it is above thr, and is
 *        dropped when it is not (so thr = 0 drops nothing but exact zeros).
 */
inline AssembledMatrix repack(const AssembledMatrix &S, int p, double thr)
{
    AssembledMatrix K(S.n, p);
    K.symmetric = S.symmetric;
    for (int i = 0; i < S.n; ++i)
        for (int off = -S.p; off <= S.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= S.n) continue;
            double v = S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)];
            if (v == 0.0) continue;
            if (std::abs(off) <= p || std::abs(v) > thr) K.add(i, j, v);
        }
    for (std::size_t k = 0; k < S.ov.size(); ++k)
        if (std::abs(S.oi[k] - S.oj[k]) <= p || std::abs(S.ov[k]) > thr) K.add(S.oi[k], S.oj[k], S.ov[k]);
    return K;
}

struct SolverBase {
    virtual ~SolverBase() = default;
    virtual void solve(matrix<double> &b, matrix<double> &x) = 0;
};

/**
 * @brief Banded LU with partial pivoting (the algorithm of LAPACK's dgbtf2), for the matrices the
 *        pivot-free banded solvers cannot take: symmetric indefinite ones (-u'' - k^2 u, a Robin
 *        term of the unstable sign) and nonsymmetric ones whose pivot-free LU breaks down.
 *
 * Row i is stored over the columns i - kl .. i + kl + ku: row interchanges widen U to kl + ku
 * super-diagonals.  Interchanges are applied to the columns right of the pivot only, and the
 * solve replays them step by step, as dgbtrs does.
 */
struct PivotedBandSolver : SolverBase {
    int n, kl, ku, w;
    std::vector<double> a;
    std::vector<int> piv;
    double &at(int i, int j) { return a[static_cast<std::size_t>(i) * w + (j - i + kl)]; }

    /// An n x n zero matrix of half-bandwidth kb, to be filled through at() and then factor()ed.
    PivotedBandSolver(int n_, int kb) : n(n_), kl(kb), ku(kb), w(3 * kb + 1),
        a(static_cast<std::size_t>(n_) * (3 * kb + 1), 0.0), piv(n_) {}

    explicit PivotedBandSolver(const AssembledMatrix &S) : PivotedBandSolver(S.n, S.p)
    {
        if (!S.ov.empty()) throw std::invalid_argument("PivotedBandSolver: the band cannot hold periodic corner entries");
        for (int i = 0; i < n; ++i)
            for (int off = -S.p; off <= S.p; ++off)
            {
                const int j = i + off;
                if (j >= 0 && j < n) at(i, j) = S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)];
            }
        factor();
    }

    void factor()
    {
        const int ue = kl + ku;                                   // super-diagonals of U after interchanges
        for (int k = 0; k < n; ++k)
        {
            const int last = std::min(n - 1, k + kl), cend = std::min(n - 1, k + ue);
            int p = k;
            double big = std::abs(at(k, k));
            for (int i = k + 1; i <= last; ++i)
                if (std::abs(at(i, k)) > big) { big = std::abs(at(i, k)); p = i; }
            if (!(big > 0.0)) throw std::domain_error("LinearSolver: the matrix is singular (pivoted banded LU, column " +
                                                      std::to_string(k + 1) + ")");
            piv[k] = p;
            if (p != k)
                for (int j = k; j <= cend; ++j) std::swap(at(k, j), at(p, j));
            const double d = at(k, k);
            for (int i = k + 1; i <= last; ++i)
            {
                const double l = at(i, k) / d;
                at(i, k) = l;
                if (l != 0.0)
                    for (int j = k + 1; j <= cend; ++j) at(i, j) -= l * at(k, j);
            }
        }
    }

    void solve(matrix<double> &b, matrix<double> &x) override
    {
        std::vector<double> y(n);
        for (int i = 0; i < n; ++i) y[i] = b(i + 1);
        for (int k = 0; k < n; ++k)                               // L, with the interchanges replayed
        {
            if (piv[k] != k) std::swap(y[k], y[piv[k]]);
            const double yk = y[k];
            for (int i = k + 1; i <= std::min(n - 1, k + kl); ++i) y[i] -= at(i, k) * yk;
        }
        for (int i = n - 1; i >= 0; --i)                          // U
        {
            double s = y[i];
            for (int j = i + 1; j <= std::min(n - 1, i + kl + ku); ++j) s -= at(i, j) * y[j];
            y[i] = s / at(i, i);
        }
        for (int i = 0; i < n; ++i) x(i + 1) = y[i];
    }
};

/**
 * @brief Periodic banded LU with partial pivoting.  The cyclic ordering is folded,
 *        0, n-1, 1, n-2, 2, ..., which turns a matrix of cyclic half-bandwidth p into an ordinary
 *        band of half-bandwidth at most 2p, and PivotedBandSolver factors that.  No corner
 *        correction is involved, so it is stable whenever partial pivoting is, including where the
 *        band without the corners is singular or nearly so (where Sherman-Morrison-Woodbury is not).
 */
struct PivotedCyclicSolver : SolverBase {
    int n;
    std::vector<int> pos;                          // pos[i]: the folded index of row / column i
    std::unique_ptr<PivotedBandSolver> core;

    template <class F> static void for_entries(const AssembledMatrix &S, F f)
    {
        for (int i = 0; i < S.n; ++i)
            for (int off = -S.p; off <= S.p; ++off)
            {
                const int j = i + off;
                if (j < 0 || j >= S.n) continue;
                const double v = S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)];
                if (v != 0.0) f(i, j, v);
            }
        for (std::size_t k = 0; k < S.ov.size(); ++k)
            if (S.ov[k] != 0.0) f(S.oi[k], S.oj[k], S.ov[k]);
    }

    explicit PivotedCyclicSolver(const AssembledMatrix &S) : n(S.n), pos(S.n)
    {
        for (int t = 0; t < n; ++t) pos[(t % 2 == 0) ? t / 2 : n - 1 - t / 2] = t;
        int kb = 0;
        for_entries(S, [&](int i, int j, double) { kb = std::max(kb, std::abs(pos[i] - pos[j])); });
        core.reset(new PivotedBandSolver(n, kb));
        for_entries(S, [&](int i, int j, double v) { core->at(pos[i], pos[j]) += v; });
        core->factor();
    }

    void solve(matrix<double> &b, matrix<double> &x) override
    {
        matrix<double> bf(n), xf(n);
        for (int i = 0; i < n; ++i) bf(pos[i] + 1) = b(i + 1);
        core->solve(bf, xf);
        for (int i = 0; i < n; ++i) x(i + 1) = xf(pos[i] + 1);
    }
};

/**
 * @brief Whether a factored solver is backward stable on S: it solves S x = S x0 for a fixed
 *        smooth x0 and requires the normwise backward error |S x - b|_inf / (|S|_inf |x|_inf + |b|_inf)
 *        to be at most tol and every value finite.  One solve and two products, O(n p).
 *        It catches what a pivot test cannot, such as a corner correction (Sherman-Morrison-Woodbury)
 *        built on a nearly singular band.
 */
inline bool solve_is_stable(const AssembledMatrix &S, SolverBase &impl, double tol = 1e-11)
{
    const int n = S.n;
    std::vector<double> x0(n), Sx(n, 0.0), rowsum(n, 0.0);
    for (int i = 0; i < n; ++i) x0[i] = 1.0 + 0.5 * std::sin(1.0 + 0.7 * i);
    PivotedCyclicSolver::for_entries(S, [&](int i, int j, double v) { Sx[i] += v * x0[j]; rowsum[i] += std::abs(v); });
    matrix<double> b(n), x(n);
    double bmax = 0.0, anorm = 0.0;
    for (int i = 0; i < n; ++i) { b(i + 1) = Sx[i]; bmax = std::max(bmax, std::abs(Sx[i])); anorm = std::max(anorm, rowsum[i]); }
    impl.solve(b, x);
    std::vector<double> xs(n), r(n);
    double xmax = 0.0;
    for (int i = 0; i < n; ++i)
    {
        xs[i] = x(i + 1);
        if (!std::isfinite(xs[i])) return false;
        xmax = std::max(xmax, std::abs(xs[i]));
        r[i] = -Sx[i];
    }
    PivotedCyclicSolver::for_entries(S, [&](int i, int j, double v) { r[i] += v * xs[j]; });
    double rmax = 0.0;
    for (int i = 0; i < n; ++i) rmax = std::max(rmax, std::abs(r[i]));
    const double denom = anorm * xmax + bmax;
    return std::isfinite(rmax) && rmax <= tol * denom;
}

/// First column of the assembled matrix, including the periodic corner entries.
inline std::vector<double> first_column(const AssembledMatrix &S)
{
    std::vector<double> c(S.n, 0.0);
    for (int i = 0; i < S.n; ++i) c[i] = S.at(i, 0);
    return c;
}

/// Max deviation of the matrix from the circulant generated by its first column, relative to max |entry|.
inline double circulant_defect(const AssembledMatrix &S)
{
    std::vector<double> c = first_column(S);
    double scale = 0.0, worst = 0.0;
    for (double v : c) scale = std::max(scale, std::abs(v));
    if (scale == 0.0) return 0.0;
    FEMD_OMP_FOR_REDUCE_IF(max, worst, S.n > FEMD_OMP_THRESHOLD)      // a maximum: exact in any order
    for (int i = 0; i < S.n; ++i)
    {
        for (int off = -S.p; off <= S.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= S.n) continue;
            int k = ((i - j) % S.n + S.n) % S.n;        // circulant: A_ij = c[(i-j) mod n]
            worst = std::max(worst, std::abs(S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)] - c[k]));
        }
    }
    for (std::size_t q = 0; q < S.ov.size(); ++q)
    {
        int k = ((S.oi[q] - S.oj[q]) % S.n + S.n) % S.n;
        // overflow entries may be split across several adds; compare the summed entry
        worst = std::max(worst, std::abs(S.at(S.oi[q], S.oj[q]) - c[k]));
    }
    return worst / scale;
}

/**
 * @brief Measured structural properties of an assembled matrix.
 *
 * Everything here is measured from the entries, never taken from a flag or from
 * the space.  `AssembledMatrix::symmetric` is a declaration made by the
 * assembler from the roles of its terms and is writable, so it can disagree
 * with `symmetry` below; `p` is the half-bandwidth the space reserves, which
 * can exceed the `bandwidth` a particular form actually reaches.
 */
struct Structure {
    int n = 0;                       ///< dimension
    int declared_uband = 0;          ///< AssembledMatrix::p, the space's reserved half-bandwidth
    int bandwidth = 0;               ///< measured max |i-j| over the nonzeros
    int cyclic_bandwidth = 0;        ///< measured max min(|i-j|, n-|i-j|)
    int nnz = 0;                     ///< stored nonzeros
    int corner_entries = 0;          ///< nonzeros outside |i-j| <= declared_uband (the periodic wrap)
    int block_period = 0;            ///< smallest q dividing n with A shift-invariant by q; 1 = circulant, 0 = none
    double scale = 0.0;              ///< max |A_ij|
    double symmetry = 0.0;           ///< max |A_ij - A_ji| / scale
    double skew = 0.0;               ///< max |A_ij + A_ji| / scale
    double circulant = 0.0;          ///< circulant_defect(), the relative defect against the first column
    double diagonal_dominance = 0.0; ///< min_i (|A_ii| - sum_{j != i} |A_ij|) / scale; > 0 means strictly dominant
};

/**
 * @brief Measure the structure of an assembled matrix.
 *
 * O(n(2b+1)) with b the measured cyclic bandwidth, plus O(d(n)) shift tests for
 * the block period, d(n) the number of divisors of n.
 *
 * Entries with |a| <= kZeroTol * max|a| count as zero throughout (nnz, bandwidths, corners).
 *
 * @param S   the assembled matrix.
 * @param tol relative tolerance for the shift-invariance test behind block_period.
 * @return the measurements.
 */
inline Structure structure(const AssembledMatrix &S, double tol = 1e-12)
{
    Structure r;
    r.n = S.n;
    r.declared_uband = S.p;
    if (S.n <= 0) return r;
    const int n = S.n;

    // sum any repeated overflow triplets first, so a split entry counts once
    std::map<std::pair<int, int>, double> corner;
    for (std::size_t q = 0; q < S.ov.size(); ++q) corner[std::make_pair(S.oi[q], S.oj[q])] += S.ov[q];

    // entries at rounding level relative to the largest (kZeroTol) are not structure: an
    // orthogonal basis leaves ~1e-17 quadrature noise where the exact matrix has zeros
    const double thr = kZeroTol * max_abs(S);
    auto keep = [&](double v) { return v != 0.0 && std::abs(v) > thr; };
    // counts and maxima only, so the parallel reduction is exact
    int nnz = 0, bw = 0, cbw = 0, corners = 0;
    double scale = 0.0;
    auto note = [&](int i, int j, double v, int &nz, int &b, int &cb, int &co, double &sc)
    {
        if (!keep(v)) return;
        ++nz;
        int d = std::abs(i - j);
        b = std::max(b, d);
        cb = std::max(cb, std::min(d, n - d));
        sc = std::max(sc, std::abs(v));
        if (d > S.p) ++co;
    };
    FEMD_OMP(parallel for schedule(static) reduction(+:nnz, corners) reduction(max:bw, cbw, scale) if(n > FEMD_OMP_THRESHOLD))
    for (int i = 0; i < n; ++i)
        for (int off = -S.p; off <= S.p; ++off)
        {
            int j = i + off;
            if (j >= 0 && j < n) note(i, j, S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)], nnz, bw, cbw, corners, scale);
        }
    for (const auto &kv : corner) note(kv.first.first, kv.first.second, kv.second, nnz, bw, cbw, corners, scale);
    r.nnz = nnz; r.bandwidth = bw; r.cyclic_bandwidth = cbw; r.corner_entries = corners; r.scale = scale;
    if (r.scale == 0.0) { r.block_period = 1; return r; }   // the zero matrix is everything at once

    // gather the entries by cyclic offset: E[i*W + k] = A(i, j_of(i,k))
    const int cb = r.cyclic_bandwidth;
    const bool wide = (2 * cb + 1 >= n);
    const int W = wide ? n : 2 * cb + 1;
    auto slot  = [&](int i, int j) { int d = ((j - i) % n + n) % n; return wide ? d : (d <= cb ? cb + d : cb - (n - d)); };
    auto col   = [&](int i, int k) { int sft = wide ? k : k - cb; return ((i + sft) % n + n) % n; };
    std::vector<double> E(static_cast<std::size_t>(n) * W, 0.0);
    FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)                  // row i writes only row i of E
    for (int i = 0; i < n; ++i)
        for (int off = -S.p; off <= S.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= n) continue;
            double v = S.band[static_cast<std::size_t>(i) * S.w + (off + S.p)];
            if (keep(v)) E[static_cast<std::size_t>(i) * W + slot(i, j)] += v;
        }
    for (const auto &kv : corner)
        if (keep(kv.second)) E[static_cast<std::size_t>(kv.first.first) * W + slot(kv.first.first, kv.first.second)] += kv.second;

    // symmetry, skew and diagonal dominance
    double sym = 0.0, skw = 0.0, dom = std::numeric_limits<double>::infinity();
    FEMD_OMP(parallel for schedule(static) reduction(max:sym, skw) reduction(min:dom) if(n > FEMD_OMP_THRESHOLD))
    for (int i = 0; i < n; ++i)
    {
        double diag = 0.0, offsum = 0.0;
        for (int k = 0; k < W; ++k)
        {
            int j = col(i, k);
            double a = E[static_cast<std::size_t>(i) * W + k];
            double b = E[static_cast<std::size_t>(j) * W + slot(j, i)];
            sym = std::max(sym, std::abs(a - b));
            skw = std::max(skw, std::abs(a + b));
            if (j == i) diag = a; else offsum += std::abs(a);
        }
        double d = std::abs(diag) - offsum;
        dom = std::min(dom, d);
    }
    r.symmetry = sym / r.scale;
    r.skew = skw / r.scale;
    r.diagonal_dominance = dom / r.scale;
    r.circulant = circulant_defect(S);

    // smallest shift q dividing n that leaves every cyclic diagonal invariant
    for (int q = 1; q < n; ++q)
    {
        if (n % q != 0) continue;
        double worst = 0.0;
        for (int i = 0; i < n && worst <= tol * r.scale; ++i)
            for (int k = 0; k < W; ++k)
                worst = std::max(worst, std::abs(E[static_cast<std::size_t>(i) * W + k] -
                                                 E[static_cast<std::size_t>((i + q) % n) * W + k]));
        if (worst <= tol * r.scale) { r.block_period = q; break; }
    }
    return r;
}

#ifdef FEMD_HAVE_FFTW
struct CirculantSolver : SolverBase {
    CirculantMatrix C;
    explicit CirculantSolver(const AssembledMatrix &S) : C(first_column(S)) {}
    void solve(matrix<double> &b, matrix<double> &x) override
    {
        std::vector<double> bb(b.rows());
        for (int i = 0; i < b.rows(); ++i) bb[i] = b(i + 1);
        std::vector<double> xx = C.solve(bb);
        for (int i = 0; i < b.rows(); ++i) x(i + 1) = xx[i];
    }
};
#endif

/// The diagonal backend: x = b / diag(K).
struct DiagonalSolver : SolverBase {
    std::vector<double> inv;
    explicit DiagonalSolver(const AssembledMatrix &S) : inv(S.n)
    {
        for (int i = 0; i < S.n; ++i)
        {
            double d = S.at(i, i);
            if (d == 0.0) throw std::runtime_error("LinearSolver: zero on the diagonal of a diagonal matrix");
            inv[i] = 1.0 / d;
        }
    }
    void solve(matrix<double> &b, matrix<double> &x) override
    {
        for (int i = 0; i < static_cast<int>(inv.size()); ++i) x(i + 1) = b(i + 1) * inv[i];
    }
};

template <class M>
struct SolverImpl : SolverBase {
    M A;
    template <class... Args>
    SolverImpl(const AssembledMatrix &S, Args... args) : A(args...) { A = 0.0; fill(A, S); }
    void solve(matrix<double> &b, matrix<double> &x) override { A.solve(b, x); }
};

} // namespace detail

class LinearSolver {
public:
    /**
     * @param K         assembled matrix (0-based banded buffer)
     * @param periodic  whether the space is periodic (corner entries may be present)
     * @param s         backend.  Auto measures the matrix: Diagonal when only the diagonal is
     *                  above kZeroTol * max|entry|; otherwise the banded solvers, cyclic only when
     *                  the matrix has wrap entries, Cholesky when K.symmetric is set.  Every banded
     *                  backend factors the band the entries reach, not the band the space reserves.
     */
    LinearSolver(const AssembledMatrix &Kin, bool periodic, Solver s = Solver::Auto) : n_(Kin.n)
    {
        int bw, cbw; bool wrap;
        const bool automatic = (s == Solver::Auto);
        // Auto ignores rounding-size entries; an explicit backend keeps every nonzero
        const double ztol = automatic ? detail::kZeroTol : 0.0;
        detail::measure_band(Kin, ztol, bw, cbw, wrap);
        const double scale = detail::max_abs(Kin);
        // the zero matrix has no structure to measure: keep the flag choice (it cannot be solved anyway)
        const bool measured = automatic && scale > 0.0;
        if (automatic)
        {
            if (!measured) s = autoselect(periodic, Kin.symmetric);
            else if (bw == 0 && !wrap) s = Solver::Diagonal;
            else s = autoselect(periodic && wrap, Kin.symmetric);
        }
        if (s == Solver::Diagonal && (bw != 0 || wrap))
            throw std::invalid_argument("LinearSolver: the diagonal backend needs a diagonal matrix (measured half-bandwidth "
                                        + std::to_string(wrap ? cbw : bw) + ")");
        // factor the band the entries reach; drop only what is below the zero threshold outside it
        const double thr = ztol * scale;
        const bool cyclic = (s == Solver::Cyclic || s == Solver::SymCyclic || s == Solver::Circulant || s == Solver::Dense);
        const int pnew = (s == Solver::Diagonal) ? 0 : (cyclic ? cbw : bw);
        AssembledMatrix Kc;
        const bool shrink = (s != Solver::Circulant && s != Solver::Dense && pnew < Kin.p);
        if (shrink) Kc = detail::repack(Kin, pnew, thr);
        const AssembledMatrix &K = shrink ? Kc : Kin;
        if (measured && periodic && !wrap) periodic = false;     // no wrap entries: the plain band solvers apply
        if (!periodic && (s == Solver::Cyclic || s == Solver::SymCyclic))
            throw std::invalid_argument("LinearSolver: cyclic backend requested for a non-periodic space");
        if (periodic && (s == Solver::Band || s == Solver::SymBand) && !K.ov.empty())
            throw std::invalid_argument("LinearSolver: banded backend cannot hold the periodic corner entries");
        if (!K.symmetric && (s == Solver::SymBand || s == Solver::SymCyclic))
            throw std::invalid_argument("LinearSolver: symmetric backend requested for a nonsymmetric matrix");
        if (s == Solver::Circulant)
        {
            if (!periodic) throw std::invalid_argument("LinearSolver: the circulant (FFT) backend needs a periodic space");
            // Measured, not inferred from the grid: the FFT inverts the circulant generated by the
            // first column, so it is exact only for a matrix that IS that circulant.  The Python
            // layer turns this into a message naming the cause (grid, basis or coefficient).
            double d = detail::circulant_defect(K);
            if (d > 1e-10)
            {
                std::ostringstream msg;
                msg.precision(2);
                msg << std::scientific << "LinearSolver: matrix is not circulant (relative defect " << d
                    << ", tolerance 1e-10); the FFT backend needs a uniform grid, a shift-invariant basis "
                       "and constant coefficients";
                throw std::invalid_argument(msg.str());
            }
#ifndef FEMD_HAVE_FFTW
            throw std::runtime_error("LinearSolver: the circulant backend needs FFTW; this build has none (see has_fftw())");
#endif
        }
        backend_ = s;
        // Auto factors now, so that a matrix the chosen solver cannot take is caught here and handed
        // to one that can: Cholesky refuses an indefinite matrix, and pivot-free banded LU is checked
        // for a zero pivot or large growth.  Both then fall back to the pivoted banded LU.  Periodic
        // corners are handled below.  Explicit backends keep their choice.
        if (measured && (s == Solver::SymBand || s == Solver::Band))      // (a zero matrix is left lazy)
        {
            bool ok = true;
            if (s == Solver::SymBand)
            {
                auto *impl = new detail::SolverImpl<symbandmatrix<double>>(K, n_, K.p);
                impl_.reset(impl);
                try { impl->A.LU(); } catch (const std::domain_error &) { ok = false; }
            }
            else
            {
                auto *impl = new detail::SolverImpl<bandmatrix<double>>(K, n_, n_, K.p, K.p);
                impl_.reset(impl);
                ok = impl->A.LU_checked(1e10);
            }
            if (!ok)
            {
                impl_.reset(new detail::PivotedBandSolver(K));
                backend_ = Solver::Band;
                pivoted_ = true;
            }
            return;
        }
        // Periodic corners: the Cholesky or the pivot-free LU of the band, corrected for the corners
        // by Sherman-Morrison-Woodbury, then a backward-error check of one solve.  A matrix that is
        // not positive definite, a band that is nearly singular, or growth in the pivot-free LU goes
        // to the folded periodic LU with partial pivoting.
        if (measured && (s == Solver::SymCyclic || s == Solver::Cyclic))
        {
            bool ok = true;
            try
            {
                if (s == Solver::SymCyclic)
                {
                    auto *impl = new detail::SolverImpl<symcyclicmatrix<double>>(K, n_, K.p);
                    impl_.reset(impl);
                    impl->A.LU();
                }
                else
                {
                    auto *impl = new detail::SolverImpl<cyclicmatrix<double>>(K, n_, K.p);
                    impl_.reset(impl);
                    impl->A.LU();
                }
                ok = detail::solve_is_stable(K, *impl_);
            }
            catch (const std::domain_error &) { ok = false; }
            if (!ok)
            {
                impl_.reset(new detail::PivotedCyclicSolver(K));
                backend_ = Solver::Cyclic;
                pivoted_ = true;
            }
            return;
        }
        switch (s)
        {
        case Solver::Dense:     impl_.reset(new detail::SolverImpl<matrix<double>>(K, n_, n_)); break;
        case Solver::Band:      impl_.reset(new detail::SolverImpl<bandmatrix<double>>(K, n_, n_, K.p, K.p)); break;
        case Solver::SymBand:   impl_.reset(new detail::SolverImpl<symbandmatrix<double>>(K, n_, K.p)); break;
        case Solver::Cyclic:    impl_.reset(new detail::SolverImpl<cyclicmatrix<double>>(K, n_, K.p)); break;
        case Solver::SymCyclic: impl_.reset(new detail::SolverImpl<symcyclicmatrix<double>>(K, n_, K.p)); break;
        case Solver::Circulant:
#ifdef FEMD_HAVE_FFTW
            impl_.reset(new detail::CirculantSolver(K));
#endif
            break;
        case Solver::Diagonal:  impl_.reset(new detail::DiagonalSolver(K)); break;
        case Solver::Auto: break;   // unreachable
        }
    }

    int size() const { return n_; }
    Solver backend() const { return backend_; }
    /// @brief Whether Auto fell back to a banded LU with partial pivoting (the matrix is indefinite,
    ///        or the pivot-free LU broke down); backend() then reports Band, or Cyclic for a matrix
    ///        with periodic corners (the folded periodic LU).
    bool pivoted() const { return pivoted_; }

    /// @brief Solve K x = b for a 0-based std::vector right-hand side.
    std::vector<double> solve(const std::vector<double> &b) const
    {
        if (static_cast<int>(b.size()) != n_) throw std::invalid_argument("LinearSolver::solve: wrong right-hand side length");
        matrix<double> B(n_), X(n_);
        for (int i = 0; i < n_; ++i) B(i + 1) = b[i];
        impl_->solve(B, X);
        std::vector<double> x(n_);
        for (int i = 0; i < n_; ++i) x[i] = X(i + 1);
        return x;
    }

    /// @brief Several right-hand sides, one factorization.
    std::vector<std::vector<double>> solve(const std::vector<std::vector<double>> &bs) const
    {
        std::vector<std::vector<double>> xs;
        xs.reserve(bs.size());
        for (const auto &b : bs) xs.push_back(solve(b));
        return xs;
    }

private:
    int n_;
    bool pivoted_ = false;
    Solver backend_ = Solver::Auto;
    mutable std::unique_ptr<detail::SolverBase> impl_;
};

} // namespace femd

#endif // FEMD_SOLVE_LINEAR_SOLVER_HPP
