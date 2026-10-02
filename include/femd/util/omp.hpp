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
inline double ordered_sum(std::size_t n, F &&f)
{
    if (n <= reduce_chunk)
    {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += f(i);
        return s;
    }
    const long nc = static_cast<long>((n + reduce_chunk - 1) / reduce_chunk);
    std::vector<double> part(static_cast<std::size_t>(nc));
    FEMD_OMP_FOR_IF(n > FEMD_OMP_THRESHOLD)
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
} // namespace detail
} // namespace femd

#endif // FEMD_UTIL_OMP_HPP
