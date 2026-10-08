//
//  fft.hpp  --  the discrete Fourier transform without FFTW, for the symbol of a circulant
//  matrix (Matrix.symbol()):  X_k = sum_j x_j exp(-2 pi i j k / n), numpy.fft.fft's convention.
//  A power of two by the iterative radix-2 algorithm; any other n by Bluestein's chirp-z
//  transform on a power-of-two length >= 2n - 1.  O(n log n), to rounding of numpy's pocketfft.
//
#ifndef FEMD_UTIL_FFT_HPP
#define FEMD_UTIL_FFT_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace femd {

namespace detail {
inline void fft_pow2(std::vector<std::complex<double>> &a, bool inverse)
{
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i)                // bit reversal
    {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double pi = std::acos(-1.0);
    for (std::size_t len = 2; len <= n; len <<= 1)
    {
        const double ang = 2 * pi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        std::vector<std::complex<double>> w(len / 2);
        for (std::size_t k = 0; k < len / 2; ++k) w[k] = std::polar(1.0, ang * static_cast<double>(k));   // exact angles, no drift
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < len / 2; ++k)
            {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w[k];
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
    }
}
} // namespace detail

/// @brief X_k = sum_j x_j exp(-2 pi i j k / n).
inline std::vector<std::complex<double>> fft(const std::vector<std::complex<double>> &x)
{
    const std::size_t n = x.size();
    if (n <= 1) return x;
    if ((n & (n - 1)) == 0)
    {
        std::vector<std::complex<double>> a(x);
        detail::fft_pow2(a, false);
        return a;
    }
    // Bluestein: X_k = conj(c_k) sum_j (x_j conj(c_j)) c_{k-j},  c_j = exp(i pi j^2 / n)
    std::size_t m = 1;
    while (m < 2 * n - 1) m <<= 1;
    const double pi = std::acos(-1.0);
    std::vector<std::complex<double>> c(n);
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::size_t j2 = (j * j) % (2 * n);              // exp(i pi j^2 / n) has period 2n in j^2
        c[j] = std::polar(1.0, pi * static_cast<double>(j2) / static_cast<double>(n));
    }
    std::vector<std::complex<double>> a(m, 0.0), b(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) a[j] = x[j] * std::conj(c[j]);
    b[0] = c[0];
    for (std::size_t j = 1; j < n; ++j) b[j] = b[m - j] = c[j];
    detail::fft_pow2(a, false);
    detail::fft_pow2(b, false);
    for (std::size_t k = 0; k < m; ++k) a[k] *= b[k];
    detail::fft_pow2(a, true);
    std::vector<std::complex<double>> X(n);
    for (std::size_t k = 0; k < n; ++k) X[k] = std::conj(c[k]) * a[k] / static_cast<double>(m);
    return X;
}

} // namespace femd

#endif // FEMD_UTIL_FFT_HPP
