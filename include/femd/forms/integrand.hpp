//
//  integrand.hpp  --  the coefficient of a form evaluated at the quadrature points in C++.
//
//  The Python form language turns the coefficient of a term, say (u1 d_x u1 + u2 d_y u1) for
//  the convection term of Navier-Stokes, into an expression tree.  Instead of walking that
//  tree with NumPy (one whole-array operation per node, one thread, a temporary per node),
//  Python compiles it once into a short program for this evaluator: registers hold one
//  value per point, each instruction reads one or two of them (or a constant, a coordinate,
//  a normal component, or a field value supplied as an input array) and writes one.  The
//  program runs over the points in blocks, in parallel with OpenMP, each point by one
//  thread, so the result does not depend on the number of threads.
//
//  Inputs are the field values and derivatives at the points, which Cache2D::at_points_multi
//  already computes in C++; constants are passed at each evaluation (an fd.Constant changes
//  between time steps).  Sums and products are chained left to right, as NumPy's were, and
//  powers use std::pow, so results agree with the NumPy evaluation to rounding.
//
#ifndef FEMD_FORMS_INTEGRAND_HPP
#define FEMD_FORMS_INTEGRAND_HPP

#include "femd/util/omp.hpp"
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class Integrand {
public:
    enum Op : int {
        CONST = 0,   // out = consts[a]
        X = 1,       // out = x
        Y = 2,       // out = y
        NX = 3,      // out = n_x  (a = +1 or -1, the sign of the side)
        NY = 4,      // out = n_y
        INPUT = 5,   // out = inputs[a]
        ADD = 6,     // out = r[a] + r[b]
        MUL = 7,     // out = r[a] * r[b]
        POW = 8,     // out = r[a] ** consts[b]
        SIN = 9, COS = 10, EXP = 11, LOG = 12, TANH = 13, SQRT = 14, SINH = 15, COSH = 16,
        SECH = 17, ABS = 18, SIGN = 19
    };

    /// @param code  4 integers per instruction: op, out register, a, b
    Integrand(const std::vector<int> &code, int nreg, int nconst, int ninput)
        : code_(code), nreg_(nreg), nconst_(nconst), ninput_(ninput)
    {
        if (code.size() % 4 != 0) throw std::invalid_argument("Integrand: the code has 4 integers per instruction");
        if (nreg < 1) throw std::invalid_argument("Integrand: at least one register");
        result_ = code.empty() ? -1 : code[code.size() - 4 + 1];
        for (std::size_t k = 0; k < code.size(); k += 4)
        {
            const int op = code[k], out = code[k + 1], a = code[k + 2], b = code[k + 3];
            if (out < 0 || out >= nreg) throw std::invalid_argument("Integrand: register out of range");
            if (op == CONST && (a < 0 || a >= nconst)) throw std::invalid_argument("Integrand: constant out of range");
            if (op == INPUT && (a < 0 || a >= ninput)) throw std::invalid_argument("Integrand: input out of range");
            if ((op == ADD || op == MUL) && (a < 0 || a >= nreg || b < 0 || b >= nreg))
                throw std::invalid_argument("Integrand: register out of range");
            if (op == POW && (a < 0 || a >= nreg || b < 0 || b >= nconst)) throw std::invalid_argument("Integrand: pow operands out of range");
            if (op >= SIN && op <= SIGN && (a < 0 || a >= nreg)) throw std::invalid_argument("Integrand: register out of range");
            if (op == X || op == Y) need_xy_ = true;
            if (op == NX || op == NY) need_normal_ = true;
            if (op < 0 || op > SIGN) throw std::invalid_argument("Integrand: unknown instruction " + std::to_string(op));
        }
    }

    int instructions() const { return static_cast<int>(code_.size() / 4); }
    int registers() const { return nreg_; }
    int constants() const { return nconst_; }
    int inputs() const { return ninput_; }
    bool needs_coordinates() const { return need_xy_; }
    bool needs_normal() const { return need_normal_; }

    /**
     * @brief The coefficient at npts points.
     * @param xs, ys       coordinates (may be null when the program does not read them)
     * @param nx, ny       the unit normal (may be null when not read)
     * @param inputs       ninput arrays of npts values
     * @param consts       nconst values
     * @param out          npts values
     */
    void evaluate(std::size_t npts, const double *xs, const double *ys, const double *nx, const double *ny,
                  const std::vector<const double *> &inputs, const std::vector<double> &consts, double *out) const
    {
        if (static_cast<int>(inputs.size()) != ninput_) throw std::invalid_argument("Integrand: wrong number of inputs");
        if (static_cast<int>(consts.size()) != nconst_) throw std::invalid_argument("Integrand: wrong number of constants");
        if (need_xy_ && (xs == nullptr || ys == nullptr)) throw std::invalid_argument("Integrand: the program reads x or y");
        if (need_normal_ && (nx == nullptr || ny == nullptr)) throw std::invalid_argument("Integrand: the program reads the normal");
        if (result_ < 0) throw std::invalid_argument("Integrand: empty program");
        constexpr std::size_t B = 256;                                   // points per block: registers stay in cache
        const long nblocks = static_cast<long>((npts + B - 1) / B);
        const bool par = npts > FEMD_OMP_THRESHOLD;
        (void)par;
        FEMD_OMP_PARALLEL_IF(par)
        {
            std::vector<double> reg(static_cast<std::size_t>(nreg_) * B);
            FEMD_OMP_FOR
            for (long blk = 0; blk < nblocks; ++blk)
            {
                const std::size_t lo = static_cast<std::size_t>(blk) * B, hi = lo + B < npts ? lo + B : npts, len = hi - lo;
                run(lo, len, xs, ys, nx, ny, inputs, consts, reg.data());
                const double *res = reg.data() + static_cast<std::size_t>(result_) * B;
                for (std::size_t q = 0; q < len; ++q) out[lo + q] = res[q];
            }
        }
    }

    /**
     * @brief The coefficient at npts points for complex inputs and constants, the complex-step
     *        derivative's case: re and im parts in separate arrays (an input's im may be null, a
     *        real field).  The arithmetic is NumPy's for complex arrays, so a form gives what the
     *        NumPy evaluation of its tree gave: products (ar br - ai bi, ar bi + ai br); z**2 as
     *        ((a - b)(a + b), 2ab); z**-1 by Smith's reciprocal; other integer powers |n| < 100 by
     *        binary powering (z**3 = z (z z)), negative ones then 1 / . by Smith's division,
     *        but the real power for a real value (imaginary part 0, as for x or a constant);
     *        z**0.5 = csqrt; any other power cpow; sin ... cosh the C library's complex functions;
     *        abs(z) = sign(Re z) z and sign(z) = sign(Re z), whose imaginary parts carry the
     *        derivative; sech(z) = 1 / cosh(z).
     */
    void evaluate_complex(std::size_t npts, const double *xs, const double *ys, const double *nx, const double *ny,
                          const std::vector<const double *> &in_re, const std::vector<const double *> &in_im,
                          const std::vector<double> &c_re, const std::vector<double> &c_im,
                          double *out_re, double *out_im) const
    {
        if (static_cast<int>(in_re.size()) != ninput_ || static_cast<int>(in_im.size()) != ninput_)
            throw std::invalid_argument("Integrand: wrong number of inputs");
        if (static_cast<int>(c_re.size()) != nconst_ || static_cast<int>(c_im.size()) != nconst_)
            throw std::invalid_argument("Integrand: wrong number of constants");
        if (need_xy_ && (xs == nullptr || ys == nullptr)) throw std::invalid_argument("Integrand: the program reads x or y");
        if (need_normal_ && (nx == nullptr || ny == nullptr)) throw std::invalid_argument("Integrand: the program reads the normal");
        if (result_ < 0) throw std::invalid_argument("Integrand: empty program");
        for (const double *p : in_re) if (p == nullptr) throw std::invalid_argument("Integrand: a real part is missing");
        constexpr std::size_t B = 256;
        const long nblocks = static_cast<long>((npts + B - 1) / B);
        const bool par = npts > FEMD_OMP_THRESHOLD;
        (void)par;
        FEMD_OMP_PARALLEL_IF(par)
        {
            std::vector<double> re(static_cast<std::size_t>(nreg_) * B), im(static_cast<std::size_t>(nreg_) * B);
            FEMD_OMP_FOR
            for (long blk = 0; blk < nblocks; ++blk)
            {
                const std::size_t lo = static_cast<std::size_t>(blk) * B, hi = lo + B < npts ? lo + B : npts, len = hi - lo;
                run_complex(lo, len, xs, ys, nx, ny, in_re, in_im, c_re, c_im, re.data(), im.data());
                const std::size_t r = static_cast<std::size_t>(result_) * B;
                for (std::size_t q = 0; q < len; ++q) { out_re[lo + q] = re[r + q]; out_im[lo + q] = im[r + q]; }
            }
        }
    }

private:
    std::vector<int> code_;
    int nreg_, nconst_, ninput_, result_ = -1;
    bool need_xy_ = false, need_normal_ = false;

    static double sech(double z)
    {
        const double e = std::exp(-std::abs(z));
        return 2.0 * e / (1.0 + e * e);
    }

    // ---- complex arithmetic, as NumPy does it ------------------------------------------------
    static void cmul(double ar, double ai, double br, double bi, double &r, double &i)
    {
        FEMD_NO_FP_CONTRACT
        r = ar * br - ai * bi; i = ar * bi + ai * br;
    }
    /// a / b, NumPy's (Smith's) complex division
    static void cdiv(double ar, double ai, double br, double bi, double &r, double &i)
    {
        FEMD_NO_FP_CONTRACT
        const double abr = std::abs(br), abi = std::abs(bi);
        if (abr >= abi)
        {
            if (abr == 0.0 && abi == 0.0) { r = ar / abr; i = ai / abi; return; }
            const double rat = bi / br, scl = 1.0 / (br + bi * rat);
            r = (ar + ai * rat) * scl; i = (ai - ar * rat) * scl;
        }
        else
        {
            const double rat = br / bi, scl = 1.0 / (bi + br * rat);
            r = (ar * rat + ai) * scl; i = (ai * rat - ar) * scl;
        }
    }
    /// 1 / z, NumPy's reciprocal
    static void crecip(double a, double b, double &r, double &i)
    {
        FEMD_NO_FP_CONTRACT
        if (std::abs(b) <= std::abs(a)) { const double t = b / a, d = a + b * t; r = 1.0 / d; i = -t / d; }
        else { const double t = a / b, d = a * t + b; r = t / d; i = -1.0 / d; }
    }
    /// z**n for real n, NumPy's rules (fast paths, then npy_cpow)
    static void cpow(double a, double b, double n, double &r, double &i)
    {
        FEMD_NO_FP_CONTRACT
        if (n == 2.0) { r = (a - b) * (a + b); i = a * b + b * a; return; }
        if (n == -1.0) { crecip(a, b, r, i); return; }
        if (n == 0.5) { const std::complex<double> s = std::sqrt(std::complex<double>(a, b)); r = s.real(); i = s.imag(); return; }
        if (n == 1.0) { r = a; i = b; return; }
        if (n == 0.0) { r = 1.0; i = 0.0; return; }
        if (a == 0.0 && b == 0.0)
        {
            if (n > 0.0) { r = 0.0; i = 0.0; }
            else { r = std::numeric_limits<double>::quiet_NaN(); i = r; }
            return;
        }
        const long k = static_cast<long>(n);
        const bool integer = static_cast<double>(k) == n;
        if (b == 0.0 && (integer || a >= 0.0)) { r = std::pow(a, n); i = 0.0; return; }   // a real value: the real power
        if (integer && k > -100 && k < 100)
        {
            if (k == 3) { double sr, si; cmul(a, b, a, b, sr, si); cmul(a, b, sr, si, r, i); return; }
            long m = k < 0 ? -k : k, mask = 1;
            double ar = 1.0, ai = 0.0, pr = a, pi = b;
            while (true)
            {
                if (m & mask) { double tr, ti; cmul(ar, ai, pr, pi, tr, ti); ar = tr; ai = ti; }
                mask <<= 1;
                if (m < mask || mask <= 0) break;
                double tr, ti; cmul(pr, pi, pr, pi, tr, ti); pr = tr; pi = ti;
            }
            if (k < 0) cdiv(1.0, 0.0, ar, ai, r, i);
            else { r = ar; i = ai; }
            return;
        }
        const std::complex<double> p = std::pow(std::complex<double>(a, b), std::complex<double>(n, 0.0));
        r = p.real(); i = p.imag();
    }
    static double sgn(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : (x == 0.0 ? 0.0 : x)); }

    void run_complex(std::size_t lo, std::size_t len, const double *xs, const double *ys, const double *nx, const double *ny,
                     const std::vector<const double *> &in_re, const std::vector<const double *> &in_im,
                     const std::vector<double> &c_re, const std::vector<double> &c_im, double *re, double *im) const
    {
        constexpr std::size_t B = 256;
        for (std::size_t k = 0; k < code_.size(); k += 4)
        {
            const int op = code_[k], a = code_[k + 2], b = code_[k + 3];
            double *oR = re + static_cast<std::size_t>(code_[k + 1]) * B, *oI = im + static_cast<std::size_t>(code_[k + 1]) * B;
            const double *aR = (op >= ADD) ? re + static_cast<std::size_t>(a) * B : nullptr;
            const double *aI = (op >= ADD) ? im + static_cast<std::size_t>(a) * B : nullptr;
            auto unary = [&](auto f) {
                for (std::size_t q = 0; q < len; ++q)
                {
                    const std::complex<double> z = f(std::complex<double>(aR[q], aI[q]));
                    oR[q] = z.real(); oI[q] = z.imag();
                }
            };
            switch (op)
            {
            case CONST: for (std::size_t q = 0; q < len; ++q) { oR[q] = c_re[a]; oI[q] = c_im[a]; } break;
            case X:     for (std::size_t q = 0; q < len; ++q) { oR[q] = xs[lo + q]; oI[q] = 0.0; } break;
            case Y:     for (std::size_t q = 0; q < len; ++q) { oR[q] = ys[lo + q]; oI[q] = 0.0; } break;
            case NX:    { const double s = a < 0 ? -1.0 : 1.0; for (std::size_t q = 0; q < len; ++q) { oR[q] = s * nx[lo + q]; oI[q] = 0.0; } break; }
            case NY:    { const double s = a < 0 ? -1.0 : 1.0; for (std::size_t q = 0; q < len; ++q) { oR[q] = s * ny[lo + q]; oI[q] = 0.0; } break; }
            case INPUT:
            {
                const double *r = in_re[a] + lo, *i = in_im[a] ? in_im[a] + lo : nullptr;
                for (std::size_t q = 0; q < len; ++q) { oR[q] = r[q]; oI[q] = i ? i[q] : 0.0; }
                break;
            }
            case ADD:
            {
                const double *bR = re + static_cast<std::size_t>(b) * B, *bI = im + static_cast<std::size_t>(b) * B;
                for (std::size_t q = 0; q < len; ++q) { oR[q] = aR[q] + bR[q]; oI[q] = aI[q] + bI[q]; }
                break;
            }
            case MUL:
            {
                const double *bR = re + static_cast<std::size_t>(b) * B, *bI = im + static_cast<std::size_t>(b) * B;
                for (std::size_t q = 0; q < len; ++q) cmul(aR[q], aI[q], bR[q], bI[q], oR[q], oI[q]);
                break;
            }
            case POW: { const double n = c_re[b]; for (std::size_t q = 0; q < len; ++q) cpow(aR[q], aI[q], n, oR[q], oI[q]); break; }
            case SIN:  unary([](std::complex<double> z) { return std::sin(z); }); break;
            case COS:  unary([](std::complex<double> z) { return std::cos(z); }); break;
            case EXP:  unary([](std::complex<double> z) { return std::exp(z); }); break;
            case LOG:  unary([](std::complex<double> z) { return std::log(z); }); break;
            case TANH: unary([](std::complex<double> z) { return std::tanh(z); }); break;
            case SQRT: unary([](std::complex<double> z) { return std::sqrt(z); }); break;
            case SINH: unary([](std::complex<double> z) { return std::sinh(z); }); break;
            case COSH: unary([](std::complex<double> z) { return std::cosh(z); }); break;
            case SECH:
                for (std::size_t q = 0; q < len; ++q)
                {
                    const std::complex<double> c = std::cosh(std::complex<double>(aR[q], aI[q]));
                    cdiv(1.0, 0.0, c.real(), c.imag(), oR[q], oI[q]);
                }
                break;
            case ABS:  for (std::size_t q = 0; q < len; ++q) { const double s = sgn(aR[q]); oR[q] = s * aR[q]; oI[q] = s * aI[q]; } break;
            case SIGN: for (std::size_t q = 0; q < len; ++q) { oR[q] = sgn(aR[q]); oI[q] = 0.0; } break;
            default: break;
            }
        }
    }

    void run(std::size_t lo, std::size_t len, const double *xs, const double *ys, const double *nx, const double *ny,
             const std::vector<const double *> &inputs, const std::vector<double> &consts, double *reg) const
    {
        constexpr std::size_t B = 256;
        for (std::size_t k = 0; k < code_.size(); k += 4)
        {
            const int op = code_[k], a = code_[k + 2], b = code_[k + 3];
            double *o = reg + static_cast<std::size_t>(code_[k + 1]) * B;
            const double *ra = (op >= ADD) ? reg + static_cast<std::size_t>(a) * B : nullptr;
            switch (op)
            {
            case CONST: { const double c = consts[a]; for (std::size_t q = 0; q < len; ++q) o[q] = c; break; }
            case X:     for (std::size_t q = 0; q < len; ++q) o[q] = xs[lo + q]; break;
            case Y:     for (std::size_t q = 0; q < len; ++q) o[q] = ys[lo + q]; break;
            case NX:    { const double sgn = a < 0 ? -1.0 : 1.0; for (std::size_t q = 0; q < len; ++q) o[q] = sgn * nx[lo + q]; break; }
            case NY:    { const double sgn = a < 0 ? -1.0 : 1.0; for (std::size_t q = 0; q < len; ++q) o[q] = sgn * ny[lo + q]; break; }
            case INPUT: { const double *in = inputs[a] + lo; for (std::size_t q = 0; q < len; ++q) o[q] = in[q]; break; }
            case ADD:   { const double *rb = reg + static_cast<std::size_t>(b) * B; for (std::size_t q = 0; q < len; ++q) o[q] = ra[q] + rb[q]; break; }
            case MUL:   { const double *rb = reg + static_cast<std::size_t>(b) * B; for (std::size_t q = 0; q < len; ++q) o[q] = ra[q] * rb[q]; break; }
            case POW:
            {
                const double n = consts[b];
                if (n == 2.0)      for (std::size_t q = 0; q < len; ++q) o[q] = ra[q] * ra[q];
                else if (n == -1.0) for (std::size_t q = 0; q < len; ++q) o[q] = 1.0 / ra[q];
                else if (n == 0.5) for (std::size_t q = 0; q < len; ++q) o[q] = std::sqrt(ra[q]);
                else               for (std::size_t q = 0; q < len; ++q) o[q] = std::pow(ra[q], n);
                break;
            }
            case SIN:  for (std::size_t q = 0; q < len; ++q) o[q] = std::sin(ra[q]); break;
            case COS:  for (std::size_t q = 0; q < len; ++q) o[q] = std::cos(ra[q]); break;
            case EXP:  for (std::size_t q = 0; q < len; ++q) o[q] = std::exp(ra[q]); break;
            case LOG:  for (std::size_t q = 0; q < len; ++q) o[q] = std::log(ra[q]); break;
            case TANH: for (std::size_t q = 0; q < len; ++q) o[q] = std::tanh(ra[q]); break;
            case SQRT: for (std::size_t q = 0; q < len; ++q) o[q] = std::sqrt(ra[q]); break;
            case SINH: for (std::size_t q = 0; q < len; ++q) o[q] = std::sinh(ra[q]); break;
            case COSH: for (std::size_t q = 0; q < len; ++q) o[q] = std::cosh(ra[q]); break;
            case SECH: for (std::size_t q = 0; q < len; ++q) o[q] = sech(ra[q]); break;
            case ABS:  for (std::size_t q = 0; q < len; ++q) o[q] = std::abs(ra[q]); break;
            case SIGN: for (std::size_t q = 0; q < len; ++q) o[q] = ra[q] > 0.0 ? 1.0 : (ra[q] < 0.0 ? -1.0 : 0.0); break;
            default: break;
            }
        }
    }
};

} // namespace femd

#endif // FEMD_FORMS_INTEGRAND_HPP
