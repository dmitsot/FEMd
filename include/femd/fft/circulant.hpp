//
//  circulant.hpp  --  circulant matrix solved/applied via FFT (FFTW backend).
//
//  A circulant matrix M defined by its first column c diagonalises as
//      M = F^{-1} diag(F c) F,
//  so  M x = b   <=>   x = ifft( fft(b) ./ fft(c) ),  in O(n log n) with O(n)
//  storage.  The eigenvalues lambda = fft(c) are precomputed once; each solve
//  is two FFTs and a pointwise divide using reused FFTW plans.
//
//  This is the M^{-1} primitive for the SMW iteration on nearly-circulant
//  systems (see poseidon/iterative/smw.hpp).
//
//  DEPENDENCY: this header needs FFTW3.  It is NOT part of the header-only core
//  (poseidon.hpp does not include it).  Compile/link with, e.g.:
//      c++ -std=c++17 -I/opt/homebrew/include ... -L/opt/homebrew/lib -lfftw3
//  Define FEMD_FFTW_OMP and link -lfftw3_omp for OpenMP-threaded transforms.
//
#ifndef FEMD_FFT_CIRCULANT_HPP
#define FEMD_FFT_CIRCULANT_HPP

#include "femd/util/omp.hpp"
#include <fftw3.h>
#include <complex>
#include <vector>
#include <cstddef>
#include <cassert>

#ifdef FEMD_FFTW_OMP
#include <omp.h>
#endif

namespace femd {

using Cplx = std::complex<double>;

class CirculantMatrix {
public:
    /**
     * @brief Build a circulant matrix from its first column.
     * @param c  the first column (length n); for symmetric circulants this equals the first row.
     * @post The eigenvalues (FFT of c) and the reusable FFTW plans are precomputed.
     */
    explicit CirculantMatrix(const std::vector<double> &c)
        : n_(static_cast<int>(c.size())), in_(n_), out_(n_), lambda_(n_)
    {
        assert(n_ >= 1);
        init_threads_once();

        fwd_ = fftw_plan_dft_1d(n_, cast(in_), cast(out_), FFTW_FORWARD, FFTW_ESTIMATE);
        bwd_ = fftw_plan_dft_1d(n_, cast(in_), cast(out_), FFTW_BACKWARD, FFTW_ESTIMATE);

        // eigenvalues lambda = fft(c)
        for (int i = 0; i < n_; ++i) in_[i] = Cplx(c[i], 0.0);
        fftw_execute(fwd_);
        for (int i = 0; i < n_; ++i) lambda_[i] = out_[i];
    }

    ~CirculantMatrix()
    {
        if (fwd_) fftw_destroy_plan(fwd_);
        if (bwd_) fftw_destroy_plan(bwd_);
    }

    // FFTW plans are bound to the internal buffers -> non-copyable, non-movable.
    CirculantMatrix(const CirculantMatrix &) = delete;
    CirculantMatrix &operator=(const CirculantMatrix &) = delete;

    /// @brief Matrix dimension. @return n.
    int size() const { return n_; }

    /// @brief Smallest |eigenvalue| -- a quick invertibility / conditioning check. @return min |lambda_k|.
    double min_abs_eigenvalue() const
    {
        double m = std::abs(lambda_[0]);
        for (int i = 1; i < n_; ++i) m = std::min(m, std::abs(lambda_[i]));
        return m;
    }

    /**
     * @brief Solve M x = b by FFT (O(n log n)).
     * @param b  right-hand side (length n).
     * @return the solution x = ifft(fft(b)/lambda).
     */
    std::vector<double> solve(const std::vector<double> &b) const
    {
        assert((int) b.size() == n_);
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) in_[i] = Cplx(b[i], 0.0);
        fftw_execute(fwd_);                       // out_ = fft(b)
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) in_[i] = out_[i] / lambda_[i];
        fftw_execute(bwd_);                       // out_ = n * ifft(...)
        std::vector<double> x(n_);
        double inv = 1.0 / n_;
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) x[i] = out_[i].real() * inv;
        return x;
    }

    /**
     * @brief Apply y = M x by FFT.
     * @param x  input vector (length n).
     * @return the product y = M x.
     */
    std::vector<double> matvec(const std::vector<double> &x) const
    {
        assert((int) x.size() == n_);
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) in_[i] = Cplx(x[i], 0.0);
        fftw_execute(fwd_);
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) in_[i] = out_[i] * lambda_[i];
        fftw_execute(bwd_);
        std::vector<double> y(n_);
        double inv = 1.0 / n_;
        FEMD_OMP_FOR_IF(n_ > FEMD_OMP_THRESHOLD)
        for (int i = 0; i < n_; ++i) y[i] = out_[i].real() * inv;
        return y;
    }

private:
    int n_;
    mutable std::vector<Cplx> in_, out_;
    std::vector<Cplx> lambda_;
    fftw_plan fwd_ = nullptr, bwd_ = nullptr;

    static fftw_complex *cast(std::vector<Cplx> &v)
    {
        return reinterpret_cast<fftw_complex *>(v.data());
    }

    static void init_threads_once()
    {
#ifdef FEMD_FFTW_OMP
        static bool done = []() {
            fftw_init_threads();
            fftw_plan_with_nthreads(omp_get_max_threads());
            return true;
        }();
        (void) done;
#endif
    }
};

} // namespace femd

#endif // FEMD_FFT_CIRCULANT_HPP
