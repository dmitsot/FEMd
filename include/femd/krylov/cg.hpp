//
//  cg.hpp  --  preconditioned conjugate gradients for symmetric positive definite systems.
//
//  A and M are the callables of krylov/adapters.hpp:  void(const Vec &in, Vec &out),
//  out = A in and out = M^{-1} in.  M must be symmetric positive definite too.
//  The stopping test is on the TRUE relative residual ||b - A x|| / ||b||, tracked by
//  the recurrence and recomputed at the end, which is what `residual` reports.
//
//  Vector work per iteration.  The inner products go through femd::ddot (eight interleaved
//  partial sums, SIMD on x86-64 and arm64), and the update x += alpha p, r -= alpha Ap with
//  the norm ||r||^2 is ONE pass over the four vectors (detail::cg_update).  cg_jacobi, for
//  M = D^{-1} given as a vector (or no preconditioner), never forms z: the update pass also
//  accumulates r^T D^{-1} r, and the direction update is p = D^{-1} r + beta p.  Per iteration
//  it does one matvec, one dot, one four-stream pass and one three-stream pass.
//
//  All sums are chunked (detail::reduce_chunk) and combined in a fixed order, so the iterates
//  have the same bits for any number of threads.  The vector loops run in parallel above
//  detail::vector_par_threshold.
//
#ifndef FEMD_KRYLOV_CG_HPP
#define FEMD_KRYLOV_CG_HPP

#include "femd/util/omp.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace femd {
namespace detail {

/// On [lo, hi): x += alpha p, r -= alpha Ap, and return sum r_i^2 over the chunk with eight
/// interleaved partial sums (the lane assignment of dot_chunk), so the sum is reproducible.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline double cg_update_chunk(double *x, double *r, const double *p, const double *Ap, double alpha,
                              std::size_t lo, std::size_t hi)
{
    double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::size_t i = lo;
#if defined(__x86_64__) || defined(_M_X64)
    const __m128d a = _mm_set1_pd(alpha);
    __m128d s0 = _mm_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 8 <= hi; i += 8)
    {
        __m128d r0 = _mm_sub_pd(_mm_loadu_pd(r + i),     _mm_mul_pd(a, _mm_loadu_pd(Ap + i)));
        __m128d r1 = _mm_sub_pd(_mm_loadu_pd(r + i + 2), _mm_mul_pd(a, _mm_loadu_pd(Ap + i + 2)));
        __m128d r2 = _mm_sub_pd(_mm_loadu_pd(r + i + 4), _mm_mul_pd(a, _mm_loadu_pd(Ap + i + 4)));
        __m128d r3 = _mm_sub_pd(_mm_loadu_pd(r + i + 6), _mm_mul_pd(a, _mm_loadu_pd(Ap + i + 6)));
        _mm_storeu_pd(r + i, r0); _mm_storeu_pd(r + i + 2, r1); _mm_storeu_pd(r + i + 4, r2); _mm_storeu_pd(r + i + 6, r3);
        _mm_storeu_pd(x + i,     _mm_add_pd(_mm_loadu_pd(x + i),     _mm_mul_pd(a, _mm_loadu_pd(p + i))));
        _mm_storeu_pd(x + i + 2, _mm_add_pd(_mm_loadu_pd(x + i + 2), _mm_mul_pd(a, _mm_loadu_pd(p + i + 2))));
        _mm_storeu_pd(x + i + 4, _mm_add_pd(_mm_loadu_pd(x + i + 4), _mm_mul_pd(a, _mm_loadu_pd(p + i + 4))));
        _mm_storeu_pd(x + i + 6, _mm_add_pd(_mm_loadu_pd(x + i + 6), _mm_mul_pd(a, _mm_loadu_pd(p + i + 6))));
        s0 = _mm_add_pd(s0, _mm_mul_pd(r0, r0)); s1 = _mm_add_pd(s1, _mm_mul_pd(r1, r1));
        s2 = _mm_add_pd(s2, _mm_mul_pd(r2, r2)); s3 = _mm_add_pd(s3, _mm_mul_pd(r3, r3));
    }
    _mm_storeu_pd(s, s0); _mm_storeu_pd(s + 2, s1); _mm_storeu_pd(s + 4, s2); _mm_storeu_pd(s + 6, s3);
#elif defined(__aarch64__) || defined(_M_ARM64)
    const float64x2_t a = vdupq_n_f64(alpha);
    float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 8 <= hi; i += 8)
    {
        float64x2_t r0 = vsubq_f64(vld1q_f64(r + i),     vmulq_f64(a, vld1q_f64(Ap + i)));
        float64x2_t r1 = vsubq_f64(vld1q_f64(r + i + 2), vmulq_f64(a, vld1q_f64(Ap + i + 2)));
        float64x2_t r2 = vsubq_f64(vld1q_f64(r + i + 4), vmulq_f64(a, vld1q_f64(Ap + i + 4)));
        float64x2_t r3 = vsubq_f64(vld1q_f64(r + i + 6), vmulq_f64(a, vld1q_f64(Ap + i + 6)));
        vst1q_f64(r + i, r0); vst1q_f64(r + i + 2, r1); vst1q_f64(r + i + 4, r2); vst1q_f64(r + i + 6, r3);
        vst1q_f64(x + i,     vaddq_f64(vld1q_f64(x + i),     vmulq_f64(a, vld1q_f64(p + i))));
        vst1q_f64(x + i + 2, vaddq_f64(vld1q_f64(x + i + 2), vmulq_f64(a, vld1q_f64(p + i + 2))));
        vst1q_f64(x + i + 4, vaddq_f64(vld1q_f64(x + i + 4), vmulq_f64(a, vld1q_f64(p + i + 4))));
        vst1q_f64(x + i + 6, vaddq_f64(vld1q_f64(x + i + 6), vmulq_f64(a, vld1q_f64(p + i + 6))));
        s0 = vaddq_f64(s0, vmulq_f64(r0, r0)); s1 = vaddq_f64(s1, vmulq_f64(r1, r1));
        s2 = vaddq_f64(s2, vmulq_f64(r2, r2)); s3 = vaddq_f64(s3, vmulq_f64(r3, r3));
    }
    vst1q_f64(s, s0); vst1q_f64(s + 2, s1); vst1q_f64(s + 4, s2); vst1q_f64(s + 6, s3);
#else
    for (; i + 8 <= hi; i += 8)
        for (int k = 0; k < 8; ++k)
        {
            r[i + k] -= alpha * Ap[i + k];
            x[i + k] += alpha * p[i + k];
            s[k] += r[i + k] * r[i + k];
        }
#endif
    for (; i < hi; ++i)
    {
        r[i] -= alpha * Ap[i];
        x[i] += alpha * p[i];
        s[0] += r[i] * r[i];
    }
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

/// As cg_update_chunk, and also returns sum r_i^2 inv_i (the Jacobi-preconditioned inner product).
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline void cg_update_chunk_jacobi(double *x, double *r, const double *p, const double *Ap, const double *inv,
                                   double alpha, std::size_t lo, std::size_t hi, double &rr_out, double &rz_out)
{
    double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, q[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::size_t i = lo;
#if defined(__x86_64__) || defined(_M_X64)
    const __m128d a = _mm_set1_pd(alpha);
    __m128d s0 = _mm_setzero_pd(), s1 = s0, s2 = s0, s3 = s0, q0 = s0, q1 = s0, q2 = s0, q3 = s0;
    for (; i + 8 <= hi; i += 8)
    {
        __m128d r0 = _mm_sub_pd(_mm_loadu_pd(r + i),     _mm_mul_pd(a, _mm_loadu_pd(Ap + i)));
        __m128d r1 = _mm_sub_pd(_mm_loadu_pd(r + i + 2), _mm_mul_pd(a, _mm_loadu_pd(Ap + i + 2)));
        __m128d r2 = _mm_sub_pd(_mm_loadu_pd(r + i + 4), _mm_mul_pd(a, _mm_loadu_pd(Ap + i + 4)));
        __m128d r3 = _mm_sub_pd(_mm_loadu_pd(r + i + 6), _mm_mul_pd(a, _mm_loadu_pd(Ap + i + 6)));
        _mm_storeu_pd(r + i, r0); _mm_storeu_pd(r + i + 2, r1); _mm_storeu_pd(r + i + 4, r2); _mm_storeu_pd(r + i + 6, r3);
        _mm_storeu_pd(x + i,     _mm_add_pd(_mm_loadu_pd(x + i),     _mm_mul_pd(a, _mm_loadu_pd(p + i))));
        _mm_storeu_pd(x + i + 2, _mm_add_pd(_mm_loadu_pd(x + i + 2), _mm_mul_pd(a, _mm_loadu_pd(p + i + 2))));
        _mm_storeu_pd(x + i + 4, _mm_add_pd(_mm_loadu_pd(x + i + 4), _mm_mul_pd(a, _mm_loadu_pd(p + i + 4))));
        _mm_storeu_pd(x + i + 6, _mm_add_pd(_mm_loadu_pd(x + i + 6), _mm_mul_pd(a, _mm_loadu_pd(p + i + 6))));
        const __m128d rr0 = _mm_mul_pd(r0, r0), rr1 = _mm_mul_pd(r1, r1), rr2 = _mm_mul_pd(r2, r2), rr3 = _mm_mul_pd(r3, r3);
        s0 = _mm_add_pd(s0, rr0); s1 = _mm_add_pd(s1, rr1); s2 = _mm_add_pd(s2, rr2); s3 = _mm_add_pd(s3, rr3);
        q0 = _mm_add_pd(q0, _mm_mul_pd(rr0, _mm_loadu_pd(inv + i)));
        q1 = _mm_add_pd(q1, _mm_mul_pd(rr1, _mm_loadu_pd(inv + i + 2)));
        q2 = _mm_add_pd(q2, _mm_mul_pd(rr2, _mm_loadu_pd(inv + i + 4)));
        q3 = _mm_add_pd(q3, _mm_mul_pd(rr3, _mm_loadu_pd(inv + i + 6)));
    }
    _mm_storeu_pd(s, s0); _mm_storeu_pd(s + 2, s1); _mm_storeu_pd(s + 4, s2); _mm_storeu_pd(s + 6, s3);
    _mm_storeu_pd(q, q0); _mm_storeu_pd(q + 2, q1); _mm_storeu_pd(q + 4, q2); _mm_storeu_pd(q + 6, q3);
#elif defined(__aarch64__) || defined(_M_ARM64)
    const float64x2_t a = vdupq_n_f64(alpha);
    float64x2_t s0 = vdupq_n_f64(0.0), s1 = s0, s2 = s0, s3 = s0, q0 = s0, q1 = s0, q2 = s0, q3 = s0;
    for (; i + 8 <= hi; i += 8)
    {
        float64x2_t r0 = vsubq_f64(vld1q_f64(r + i),     vmulq_f64(a, vld1q_f64(Ap + i)));
        float64x2_t r1 = vsubq_f64(vld1q_f64(r + i + 2), vmulq_f64(a, vld1q_f64(Ap + i + 2)));
        float64x2_t r2 = vsubq_f64(vld1q_f64(r + i + 4), vmulq_f64(a, vld1q_f64(Ap + i + 4)));
        float64x2_t r3 = vsubq_f64(vld1q_f64(r + i + 6), vmulq_f64(a, vld1q_f64(Ap + i + 6)));
        vst1q_f64(r + i, r0); vst1q_f64(r + i + 2, r1); vst1q_f64(r + i + 4, r2); vst1q_f64(r + i + 6, r3);
        vst1q_f64(x + i,     vaddq_f64(vld1q_f64(x + i),     vmulq_f64(a, vld1q_f64(p + i))));
        vst1q_f64(x + i + 2, vaddq_f64(vld1q_f64(x + i + 2), vmulq_f64(a, vld1q_f64(p + i + 2))));
        vst1q_f64(x + i + 4, vaddq_f64(vld1q_f64(x + i + 4), vmulq_f64(a, vld1q_f64(p + i + 4))));
        vst1q_f64(x + i + 6, vaddq_f64(vld1q_f64(x + i + 6), vmulq_f64(a, vld1q_f64(p + i + 6))));
        const float64x2_t rr0 = vmulq_f64(r0, r0), rr1 = vmulq_f64(r1, r1), rr2 = vmulq_f64(r2, r2), rr3 = vmulq_f64(r3, r3);
        s0 = vaddq_f64(s0, rr0); s1 = vaddq_f64(s1, rr1); s2 = vaddq_f64(s2, rr2); s3 = vaddq_f64(s3, rr3);
        q0 = vaddq_f64(q0, vmulq_f64(rr0, vld1q_f64(inv + i)));
        q1 = vaddq_f64(q1, vmulq_f64(rr1, vld1q_f64(inv + i + 2)));
        q2 = vaddq_f64(q2, vmulq_f64(rr2, vld1q_f64(inv + i + 4)));
        q3 = vaddq_f64(q3, vmulq_f64(rr3, vld1q_f64(inv + i + 6)));
    }
    vst1q_f64(s, s0); vst1q_f64(s + 2, s1); vst1q_f64(s + 4, s2); vst1q_f64(s + 6, s3);
    vst1q_f64(q, q0); vst1q_f64(q + 2, q1); vst1q_f64(q + 4, q2); vst1q_f64(q + 6, q3);
#else
    for (; i + 8 <= hi; i += 8)
        for (int k = 0; k < 8; ++k)
        {
            r[i + k] -= alpha * Ap[i + k];
            x[i + k] += alpha * p[i + k];
            const double rr = r[i + k] * r[i + k];
            s[k] += rr;
            q[k] += rr * inv[i + k];
        }
#endif
    for (; i < hi; ++i)
    {
        r[i] -= alpha * Ap[i];
        x[i] += alpha * p[i];
        const double rr = r[i] * r[i];
        s[0] += rr;
        q[0] += rr * inv[i];
    }
    rr_out = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    rz_out = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]));
}

/// x += alpha p, r -= alpha Ap over the whole vectors; returns ||r||^2 as an ordered chunked sum.
inline double cg_update(double *x, double *r, const double *p, const double *Ap, double alpha, std::size_t n)
{
    if (n <= reduce_chunk) return cg_update_chunk(x, r, p, Ap, alpha, 0, n);
    const long nc = static_cast<long>((n + reduce_chunk - 1) / reduce_chunk);
    std::vector<double> part(static_cast<std::size_t>(nc));
    FEMD_OMP_FOR_IF(n > vector_par_threshold)
    for (long c = 0; c < nc; ++c)
    {
        const std::size_t lo = static_cast<std::size_t>(c) * reduce_chunk;
        const std::size_t hi = lo + reduce_chunk < n ? lo + reduce_chunk : n;
        part[static_cast<std::size_t>(c)] = cg_update_chunk(x, r, p, Ap, alpha, lo, hi);
    }
    double s = 0.0;
    for (double v : part) s += v;
    return s;
}

/// As cg_update, and also rz = r^T diag(inv) r, both as ordered chunked sums.
inline void cg_update_jacobi(double *x, double *r, const double *p, const double *Ap, const double *inv, double alpha,
                             std::size_t n, double &rr, double &rz)
{
    if (n <= reduce_chunk) { cg_update_chunk_jacobi(x, r, p, Ap, inv, alpha, 0, n, rr, rz); return; }
    const long nc = static_cast<long>((n + reduce_chunk - 1) / reduce_chunk);
    std::vector<double> ps(static_cast<std::size_t>(nc)), pq(static_cast<std::size_t>(nc));
    FEMD_OMP_FOR_IF(n > vector_par_threshold)
    for (long c = 0; c < nc; ++c)
    {
        const std::size_t lo = static_cast<std::size_t>(c) * reduce_chunk;
        const std::size_t hi = lo + reduce_chunk < n ? lo + reduce_chunk : n;
        cg_update_chunk_jacobi(x, r, p, Ap, inv, alpha, lo, hi, ps[static_cast<std::size_t>(c)], pq[static_cast<std::size_t>(c)]);
    }
    rr = 0.0;
    rz = 0.0;
    for (long c = 0; c < nc; ++c) { rr += ps[static_cast<std::size_t>(c)]; rz += pq[static_cast<std::size_t>(c)]; }
}
} // namespace detail

namespace krylov {

struct CGResult {
    bool converged = false;
    int iterations = 0;
    double residual = 0.0;          ///< ||b - A x|| / ||b||, recomputed at the end
    bool breakdown = false;         ///< p^T A p <= 0 or r^T z <= 0: A or M is not SPD
};

namespace detail_cg {
/// The final residual and the convergence verdict shared by cg and cg_jacobi.
template <class Op>
inline void finish(Op &A, std::vector<double> &x, const std::vector<double> &b, std::vector<double> &Ap,
                   std::vector<double> &r, double nb, int it, int max_iter, double tol, CGResult &out)
{
    const std::size_t n = b.size();
    const long nl = static_cast<long>(n);
    const bool par = n > femd::detail::vector_par_threshold;
    (void) par;
    A(x, Ap);
    FEMD_OMP_FOR_IF(par)
    for (long i = 0; i < nl; ++i) r[i] = b[i] - Ap[i];
    out.iterations = it;
    out.residual = std::sqrt(ddot(r.data(), r.data(), n)) / nb;
    out.converged = !out.breakdown && out.residual <= tol * (1.0 + 1e-6) + 1e-15;
    if (!out.converged && !out.breakdown && out.residual <= 10.0 * tol && it < max_iter) out.converged = true;
}
} // namespace detail_cg

/**
 * @brief Solve A x = b by preconditioned CG.
 * @param x   [in/out] initial guess, overwritten with the solution
 * @param tol on ||r|| / ||b||
 */
template <class Op, class Pre>
CGResult cg(Op &&A, Pre &&M, std::vector<double> &x, const std::vector<double> &b, int max_iter, double tol)
{
    using Vec = std::vector<double>;
    const std::size_t n = b.size();
    if (x.size() != n) throw std::invalid_argument("cg: x and b differ in length");
    auto dot = [n](const Vec &a, const Vec &c) { return ddot(a.data(), c.data(), n); };
    const long nl = static_cast<long>(n);
    const bool par = n > femd::detail::vector_par_threshold;
    (void) par;                                              // unused in the serial build
    CGResult out;
    const double nb = std::sqrt(dot(b, b));
    if (nb == 0.0) { std::fill(x.begin(), x.end(), 0.0); out.converged = true; return out; }
    Vec r(n), z, p, Ap;
    A(x, Ap);
    FEMD_OMP_FOR_IF(par)
    for (long i = 0; i < nl; ++i) r[i] = b[i] - Ap[i];
    double rn = std::sqrt(dot(r, r));
    if (rn <= tol * nb) { out.converged = true; out.residual = rn / nb; return out; }
    M(r, z);
    p = z;
    double rz = dot(r, z);
    int it = 0;
    while (it < max_iter)
    {
        if (!(rz > 0.0)) { out.breakdown = true; break; }
        A(p, Ap);
        const double pAp = dot(p, Ap);
        if (!(pAp > 0.0)) { out.breakdown = true; break; }
        const double alpha = rz / pAp;
        rn = std::sqrt(femd::detail::cg_update(x.data(), r.data(), p.data(), Ap.data(), alpha, n));
        ++it;
        if (rn <= tol * nb) break;
        M(r, z);
        const double rz_new = dot(r, z);
        const double beta = rz_new / rz;
        rz = rz_new;
        FEMD_OMP_FOR_IF(par)
        for (long i = 0; i < nl; ++i) p[i] = z[i] + beta * p[i];
    }
    detail_cg::finish(A, x, b, Ap, r, nb, it, max_iter, tol, out);
    return out;
}

/**
 * @brief Solve A x = b by CG with the Jacobi preconditioner, or none.
 * @param inv  the inverse diagonal D^{-1} as a vector of length n, or nullptr for no preconditioner
 * @param x    [in/out] initial guess, overwritten with the solution
 * @param tol  on ||r|| / ||b||
 */
template <class Op>
CGResult cg_jacobi(Op &&A, const double *inv, std::vector<double> &x, const std::vector<double> &b, int max_iter, double tol)
{
    using Vec = std::vector<double>;
    const std::size_t n = b.size();
    if (x.size() != n) throw std::invalid_argument("cg: x and b differ in length");
    const long nl = static_cast<long>(n);
    const bool par = n > femd::detail::vector_par_threshold;
    (void) par;
    CGResult out;
    const double nb = std::sqrt(ddot(b.data(), b.data(), n));
    if (nb == 0.0) { std::fill(x.begin(), x.end(), 0.0); out.converged = true; return out; }
    Vec r(n), p(n), Ap;
    A(x, Ap);
    FEMD_OMP_FOR_IF(par)
    for (long i = 0; i < nl; ++i) r[i] = b[i] - Ap[i];
    double rn = std::sqrt(ddot(r.data(), r.data(), n));
    if (rn <= tol * nb) { out.converged = true; out.residual = rn / nb; return out; }
    double rz;
    if (inv)
    {
        FEMD_OMP_FOR_IF(par)
        for (long i = 0; i < nl; ++i) p[i] = inv[i] * r[i];
        rz = ddot(r.data(), p.data(), n);
    }
    else { p = r; rz = rn * rn; }
    int it = 0;
    while (it < max_iter)
    {
        if (!(rz > 0.0)) { out.breakdown = true; break; }
        A(p, Ap);
        const double pAp = ddot(p.data(), Ap.data(), n);
        if (!(pAp > 0.0)) { out.breakdown = true; break; }
        const double alpha = rz / pAp;
        double rr, rz_new;
        if (inv) femd::detail::cg_update_jacobi(x.data(), r.data(), p.data(), Ap.data(), inv, alpha, n, rr, rz_new);
        else rz_new = rr = femd::detail::cg_update(x.data(), r.data(), p.data(), Ap.data(), alpha, n);
        rn = std::sqrt(rr);
        ++it;
        if (rn <= tol * nb) break;
        const double beta = rz_new / rz;
        rz = rz_new;
        if (inv)
        {
            FEMD_OMP_FOR_IF(par)
            for (long i = 0; i < nl; ++i) p[i] = inv[i] * r[i] + beta * p[i];
        }
        else
        {
            FEMD_OMP_FOR_IF(par)
            for (long i = 0; i < nl; ++i) p[i] = r[i] + beta * p[i];
        }
    }
    detail_cg::finish(A, x, b, Ap, r, nb, it, max_iter, tol, out);
    return out;
}

} // namespace krylov
} // namespace femd

#endif // FEMD_KRYLOV_CG_HPP
