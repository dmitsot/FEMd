//
//  omp.hpp  --  optional OpenMP helpers.
//
//  These macros expand to an OpenMP pragma when compiled with OpenMP
//  (_OPENMP defined, e.g. `make OMP=1`) and to NOTHING otherwise -- so the
//  serial build has no pragmas at all (no -Wunknown-pragmas, no flags needed).
//
//  Only used on genuinely data-parallel, race-free, large-n loops, always with
//  an `if(cond)` size guard so small problems don't pay fork-join overhead.
//  Sequential kernels (LU/Cholesky/Gauss-Seidel, triangular sweeps) are
//  deliberately NOT parallelised.  Scatter-adds go parallel only by row
//  ownership (owns_row), and sums only through ordered_sum, so that results
//  are the same bits for any number of threads.
//
//  In matrix.hpp these cover the dense matmul, the element-wise BLAS-1 ops
//  (=, +=, -=, +, -, unary -, scalar *) and the ddot/maxentry reductions.
//  reduction(+) is undefined for std::complex, so ddot guards the reduction
//  behind `if constexpr (is_floating_point)` and keeps complex on the serial path.
//
#ifndef FEMD_UTIL_OMP_HPP
#define FEMD_UTIL_OMP_HPP

// Below this many elements, fork-join overhead outweighs the work.
#ifndef FEMD_OMP_THRESHOLD
#define FEMD_OMP_THRESHOLD 4096
#endif

#include <cstddef>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#  define FEMD_PRAGMA(x) _Pragma(#x)
#  define FEMD_OMP_FOR_IF(cond) \
       FEMD_PRAGMA(omp parallel for schedule(static) if(cond))
#  define FEMD_OMP_FOR_REDUCE_IF(op, var, cond) \
       FEMD_PRAGMA(omp parallel for schedule(static) reduction(op:var) if(cond))
   // a parallel region (thread-private buffers declared inside it) and the loop within it
#  define FEMD_OMP_PARALLEL_IF(cond) FEMD_PRAGMA(omp parallel if(cond))
#  define FEMD_OMP_FOR FEMD_PRAGMA(omp for schedule(static))
#  define FEMD_PRAGMA_CRITICAL FEMD_PRAGMA(omp critical)
   // any other OpenMP directive, e.g. FEMD_OMP(parallel for reduction(max:a, b) if(n > 100))
#  define FEMD_OMP(x) FEMD_PRAGMA(omp x)
#else
#  define FEMD_OMP_FOR_IF(cond)
#  define FEMD_OMP_FOR_REDUCE_IF(op, var, cond)
#  define FEMD_OMP_PARALLEL_IF(cond)
#  define FEMD_OMP_FOR
#  define FEMD_PRAGMA_CRITICAL
#  define FEMD_OMP(x)
#endif

namespace femd {
namespace detail {
/// Element loops go parallel when elements x quadrature points passes the threshold.
inline bool parallel_elements(long nelem, long nq) { return nelem * nq > FEMD_OMP_THRESHOLD; }

/// Inside a parallel region: this thread's number and the team size (0 and 1 without OpenMP).
#ifdef _OPENMP
inline int thread_id() { return omp_get_thread_num(); }
inline int thread_count() { return omp_get_num_threads(); }
#else
inline int thread_id() { return 0; }
inline int thread_count() { return 1; }
#endif

/// Row ownership for a parallel scatter: rows go to threads in blocks of 16, so every entry is
/// added by one thread, in the serial order, and the result has the same bits for any team size.
inline bool owns_row(int row, int me, int nth) { return nth == 1 || ((row >> 4) % nth) == me; }

/// Chunk length of the ordered reductions.
constexpr std::size_t reduce_chunk = 4096;

/**
 * @brief sum_{i<n} f(i) in fixed chunks of reduce_chunk, the chunk sums added in order.
 *        The same bits for any number of threads and in the serial build.  Below one
 *        chunk it is the plain left-to-right sum.
 */
template <class F>
inline double ordered_sum(std::size_t n, F &&f, std::size_t par_threshold = FEMD_OMP_THRESHOLD)
{
    if (n <= reduce_chunk)
    {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += f(i);
        return s;
    }
    const long nc = static_cast<long>((n + reduce_chunk - 1) / reduce_chunk);
    std::vector<double> part(static_cast<std::size_t>(nc));
    FEMD_OMP_FOR_IF(n > par_threshold)
    for (long c = 0; c < nc; ++c)
    {
        const std::size_t lo = static_cast<std::size_t>(c) * reduce_chunk;
        const std::size_t hi = lo + reduce_chunk < n ? lo + reduce_chunk : n;
        double s = 0.0;
        for (std::size_t i = lo; i < hi; ++i) s += f(i);
        part[static_cast<std::size_t>(c)] = s;
    }
    double s = 0.0;
    for (double v : part) s += v;
    return s;
}

/// Vectors shorter than this are reduced on one thread: a dot product of 10^5 entries takes a few
/// microseconds, less than the hand-off to a team costs, and it is memory bound anyway.
constexpr std::size_t vector_par_threshold = 1u << 20;
} // namespace detail

namespace detail {
/// x . y over [lo, hi) with eight interleaved partial sums s[0..7] (s[k] takes the entries
/// i = k mod 8), combined in a fixed order at the end: the adds of one partial sum are
/// independent of the others, so the loop runs at the machine's throughput instead of its add
/// latency, and the result is the same bits every time.  On x86-64 and arm64 the eight sums
/// are held in four vector registers through intrinsics, with the same lane assignment, since
/// GCC 13 compiled the plain loop three times slower than that.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline double dot_chunk(const double *x, const double *y, std::size_t lo, std::size_t hi)
{
    double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::size_t i = lo;
#if defined(__x86_64__) || defined(_M_X64)
    __m128d a0 = _mm_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 8 <= hi; i += 8)
    {
        a0 = _mm_add_pd(a0, _mm_mul_pd(_mm_loadu_pd(x + i),     _mm_loadu_pd(y + i)));
        a1 = _mm_add_pd(a1, _mm_mul_pd(_mm_loadu_pd(x + i + 2), _mm_loadu_pd(y + i + 2)));
        a2 = _mm_add_pd(a2, _mm_mul_pd(_mm_loadu_pd(x + i + 4), _mm_loadu_pd(y + i + 4)));
        a3 = _mm_add_pd(a3, _mm_mul_pd(_mm_loadu_pd(x + i + 6), _mm_loadu_pd(y + i + 6)));
    }
    _mm_storeu_pd(s, a0); _mm_storeu_pd(s + 2, a1); _mm_storeu_pd(s + 4, a2); _mm_storeu_pd(s + 6, a3);
#elif defined(__aarch64__) || defined(_M_ARM64)
    float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 8 <= hi; i += 8)
    {
        a0 = vaddq_f64(a0, vmulq_f64(vld1q_f64(x + i),     vld1q_f64(y + i)));
        a1 = vaddq_f64(a1, vmulq_f64(vld1q_f64(x + i + 2), vld1q_f64(y + i + 2)));
        a2 = vaddq_f64(a2, vmulq_f64(vld1q_f64(x + i + 4), vld1q_f64(y + i + 4)));
        a3 = vaddq_f64(a3, vmulq_f64(vld1q_f64(x + i + 6), vld1q_f64(y + i + 6)));
    }
    vst1q_f64(s, a0); vst1q_f64(s + 2, a1); vst1q_f64(s + 4, a2); vst1q_f64(s + 6, a3);
#else
    for (; i + 8 <= hi; i += 8)
        for (int k = 0; k < 8; ++k) s[k] += x[i + k] * y[i + k];
#endif
    for (; i < hi; ++i) s[0] += x[i] * y[i];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}
} // namespace detail

/**
 * @brief The dot product sum_i x[i] y[i] of two vectors of length n.
 *        Chunks of detail::reduce_chunk entries (detail::dot_chunk), the chunk sums added in
 *        order: the same bits for any number of threads, and threaded only from
 *        detail::vector_par_threshold entries on.  The BLAS of some NumPy builds threads a dot
 *        product of 10^4 entries across every core, which on a laptop can cost hundreds of
 *        microseconds; this one costs a few.
 */
inline double ddot(const double *x, const double *y, std::size_t n)
{
    using detail::reduce_chunk;
    if (n <= reduce_chunk) return detail::dot_chunk(x, y, 0, n);
    const long nc = static_cast<long>((n + reduce_chunk - 1) / reduce_chunk);
    std::vector<double> part(static_cast<std::size_t>(nc));
    FEMD_OMP_FOR_IF(n > detail::vector_par_threshold)
    for (long c = 0; c < nc; ++c)
    {
        const std::size_t lo = static_cast<std::size_t>(c) * reduce_chunk;
        const std::size_t hi = lo + reduce_chunk < n ? lo + reduce_chunk : n;
        part[static_cast<std::size_t>(c)] = detail::dot_chunk(x, y, lo, hi);
    }
    double s = 0.0;
    for (double v : part) s += v;
    return s;
}
} // namespace femd

#endif // FEMD_UTIL_OMP_HPP
