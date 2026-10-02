//
//  product_space.hpp  --  several FunctionSpaces on one mesh, each with its own
//  boundary conditions, numbered together by POSITION (design doc, 7.1).
//
//  Every degree of freedom of every field carries a coordinate (node or
//  Greville abscissa).  Sorting all of them by coordinate gives a global
//  numbering in which coupling is local in x, so the half-bandwidth is bounded
//  by sum_f (p_f + 1) d_f regardless of how ragged the fields are, and the
//  inherited banded / cyclic solvers apply unchanged.  Field-major blocking
//  would give bandwidth ~N and is used only as the dense oracle in tests.
//
//  Fields must share the mesh and must be all periodic or all not: a periodic
//  field beside a non-periodic one has corner entries the cyclic solvers cannot
//  represent alongside a non-wrapping field.
//
#ifndef FEMD_FE_PRODUCT_SPACE_HPP
#define FEMD_FE_PRODUCT_SPACE_HPP

#include "femd/fe/function_space.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace femd {

class ProductSpace {
public:
    enum class Ordering { Position, FieldMajor };

    explicit ProductSpace(std::vector<const FunctionSpace *> fields, Ordering ord = Ordering::Position)
        : fields_(std::move(fields)), ordering_(ord)
    {
        if (fields_.empty()) throw std::invalid_argument("ProductSpace: no fields");
        const Mesh1D &m0 = fields_[0]->mesh();
        periodic_ = fields_[0]->bc().periodic;
        for (std::size_t f = 1; f < fields_.size(); ++f)
        {
            if (fields_[f]->mesh().vertices() != m0.vertices())
                throw std::invalid_argument("ProductSpace: field " + std::to_string(f) + " lives on a different mesh");
            if (fields_[f]->bc().periodic != periodic_)
                throw std::invalid_argument("ProductSpace: fields must be all periodic or all non-periodic");
        }
        // collect (coordinate, field, local) and sort
        struct Item { double x; int f, j; };
        std::vector<Item> items;
        offset_.assign(fields_.size() + 1, 0);
        for (std::size_t f = 0; f < fields_.size(); ++f)
        {
            std::vector<double> xs = fields_[f]->dof_coordinates();
            for (int j = 0; j < fields_[f]->dim(); ++j) items.push_back(Item{xs[j], static_cast<int>(f), j});
            offset_[f + 1] = offset_[f] + fields_[f]->dim();
        }
        dim_ = offset_.back();
        if (ordering_ == Ordering::Position)
            std::stable_sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.x < b.x; });
        // FieldMajor: items are already field by field
        global_.assign(fields_.size(), {});
        for (std::size_t f = 0; f < fields_.size(); ++f) global_[f].assign(fields_[f]->dim(), -1);
        local_.resize(dim_);
        for (int g = 0; g < dim_; ++g)
        {
            global_[items[g].f][items[g].j] = g;
            local_[g] = std::make_pair(items[g].f, items[g].j);
        }
        // half-bandwidth from element coupling, and from facet coupling when a field is
        // discontinuous (a dS term couples every field on the two elements at a vertex)
        uband_ = 0;
        int ne = m0.nelem();
        std::vector<int> gl;
        auto couple = [&](int e0, int e1)
        {
            gl.clear();
            for (std::size_t f = 0; f < fields_.size(); ++f)
            {
                for (int j : fields_[f]->element_dofs(e0)) gl.push_back(global_[f][j]);
                if (e1 != e0) for (int j : fields_[f]->element_dofs(e1)) gl.push_back(global_[f][j]);
            }
            for (std::size_t x = 0; x < gl.size(); ++x)
                for (std::size_t y = x + 1; y < gl.size(); ++y)
                {
                    int d = std::abs(gl[x] - gl[y]);
                    if (periodic_) d = std::min(d, dim_ - d);
                    uband_ = std::max(uband_, d);
                }
        };
        for (int e = 0; e < ne; ++e) couple(e, e);
        bool broken = false;
        for (const FunctionSpace *f : fields_) broken = broken || f->broken();
        if (broken)
        {
            for (int e = 1; e < ne; ++e) couple(e - 1, e);
            if (periodic_ && ne > 1) couple(ne - 1, 0);
        }
    }

    int nfields() const { return static_cast<int>(fields_.size()); }
    int dim() const { return dim_; }
    int uband() const { return uband_; }
    bool periodic() const { return periodic_; }
    Ordering ordering() const { return ordering_; }
    const FunctionSpace &field(int f) const { return *fields_[f]; }
    const Mesh1D &mesh() const { return fields_[0]->mesh(); }

    /// @brief Global index of local dof j of field f.
    int global(int f, int j) const { return global_[f][j]; }
    /// @brief (field, local dof) of global index g.
    std::pair<int, int> local(int g) const { return local_[g]; }

    /// @brief Split a global vector into per-field coefficient vectors.
    template <class Scalar>
    std::vector<std::vector<Scalar>> split(const std::vector<Scalar> &x) const
    {
        std::vector<std::vector<Scalar>> parts(fields_.size());
        for (std::size_t f = 0; f < fields_.size(); ++f) parts[f].assign(fields_[f]->dim(), Scalar(0));
        for (int g = 0; g < dim_; ++g) parts[local_[g].first][local_[g].second] = x[g];
        return parts;
    }
    /// @brief Gather per-field coefficient vectors into a global vector.
    template <class Scalar>
    std::vector<Scalar> gather(const std::vector<std::vector<Scalar>> &parts) const
    {
        std::vector<Scalar> x(dim_, Scalar(0));
        for (std::size_t f = 0; f < fields_.size(); ++f)
            for (int j = 0; j < fields_[f]->dim(); ++j) x[global_[f][j]] = parts[f][j];
        return x;
    }

private:
    std::vector<const FunctionSpace *> fields_;
    Ordering ordering_;
    bool periodic_ = false;
    int dim_ = 0, uband_ = 0;
    std::vector<int> offset_;
    std::vector<std::vector<int>> global_;
    std::vector<std::pair<int, int>> local_;
};

} // namespace femd

#endif // FEMD_FE_PRODUCT_SPACE_HPP
