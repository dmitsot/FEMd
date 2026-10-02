//
//  mesh1d.hpp  --  1D mesh: an interval partitioned into elements.
//
//  Stores the vertex array a = x_0 < x_1 < ... < x_n = b.  Uniformity is a
//  DETECTED property (is_uniform()), never an assumption: nothing downstream may
//  rely on a single h.  h() throws unless the mesh is uniform, so that any
//  code that assumes uniformity says so at the call site.
//
#ifndef FEMD_MESH_MESH1D_HPP
#define FEMD_MESH_MESH1D_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace femd {

class Mesh1D {
public:
    /// @brief Mesh from a strictly increasing vertex array (size >= 2).
    explicit Mesh1D(std::vector<double> vertices) : x_(std::move(vertices))
    {
        if (x_.size() < 2) throw std::invalid_argument("Mesh1D: need at least 2 vertices");
        for (std::size_t i = 1; i < x_.size(); ++i)
            if (!(x_[i] > x_[i - 1]))
                throw std::invalid_argument("Mesh1D: vertices must be strictly increasing (index " + std::to_string(i) + ")");
        detect_uniform();
    }

    /// @brief Equispaced mesh on [a,b] with nelem elements; right endpoint exact.
    static Mesh1D uniform(double a, double b, int nelem)
    {
        if (nelem < 1 || !(b > a)) throw std::invalid_argument("Mesh1D::uniform: need nelem >= 1 and b > a");
        std::vector<double> x(nelem + 1);
        double h = (b - a) / nelem;
        for (int i = 0; i < nelem; ++i) x[i] = a + i * h;
        x[nelem] = b;
        return Mesh1D(std::move(x));
    }

    int    nelem()  const { return static_cast<int>(x_.size()) - 1; }
    double a()      const { return x_.front(); }
    double b()      const { return x_.back(); }
    double xleft (int e) const { return x_[e]; }
    double xright(int e) const { return x_[e + 1]; }
    double h(int e) const { return x_[e + 1] - x_[e]; }
    double hmin() const { return hmin_; }
    double hmax() const { return hmax_; }
    const std::vector<double> &vertices() const { return x_; }

    /// @brief True when all elements have the same width to relative tolerance 1e-10.
    bool is_uniform() const { return uniform_; }
    /// @brief The common element width.  Throws if the mesh is not uniform.
    double h() const
    {
        if (!uniform_) throw std::logic_error("Mesh1D::h(): mesh is not uniform, use h(e)");
        return hmean_;
    }

    /**
     * @brief Element containing x.  x == b maps to the LAST element, and a
     *        point outside [a,b] by more than 1e-12*(b-a) throws.
     */
    int element_of(double x) const
    {
        double tol = 1e-12 * (b() - a());
        if (x < a() - tol || x > b() + tol)
            throw std::out_of_range("Mesh1D::element_of: x outside [a,b]");
        // first vertex strictly greater than x, minus one
        auto it = std::upper_bound(x_.begin(), x_.end(), x);
        int e = static_cast<int>(it - x_.begin()) - 1;
        if (e < 0) e = 0;
        if (e > nelem() - 1) e = nelem() - 1;
        return e;
    }

    /// @brief Map x in element e to xi in [-1,1].
    double to_reference(int e, double x) const { return (2.0 * x - x_[e] - x_[e + 1]) / h(e); }
    /// @brief Map xi in [-1,1] to x in element e.
    double from_reference(int e, double xi) const { return 0.5 * (x_[e] + x_[e + 1]) + 0.5 * h(e) * xi; }

private:
    void detect_uniform()
    {
        int n = nelem();
        hmean_ = (b() - a()) / n;
        hmin_ = hmax_ = h(0);
        double dev = 0.0;
        for (int e = 0; e < n; ++e)
        {
            double he = h(e);
            hmin_ = std::min(hmin_, he);
            hmax_ = std::max(hmax_, he);
            dev = std::max(dev, std::abs(he - hmean_));
        }
        uniform_ = dev <= 1e-10 * hmean_;
    }

    std::vector<double> x_;
    bool   uniform_ = false;
    double hmean_ = 0.0, hmin_ = 0.0, hmax_ = 0.0;
};

} // namespace femd

#endif // FEMD_MESH_MESH1D_HPP
