//
//  constraint_operator.hpp  --  the sparse operator C (n_adapted x n_raw) that
//  defines an adapted basis  N_j = sum_i C_ji B_i  from a raw one.
//
//  This file is the DATA STRUCTURE and its operations only.  It knows nothing
//  about splines, Lagrange, or boundary conditions: it is built from explicit
//  rows.  Construction from boundary functionals (Dirichlet, Neumann, ...) is
//  boundary_condition.hpp (M2).  Row j lists (raw index i, coefficient).
//  Both row and column access are stored, since cache building walks columns
//  (raw -> adapted) and evaluation walks rows (adapted -> raw).
//
#ifndef FEMD_FE_CONSTRAINT_OPERATOR_HPP
#define FEMD_FE_CONSTRAINT_OPERATOR_HPP

#include <stdexcept>
#include <utility>
#include <vector>

namespace femd {

class ConstraintOperator {
public:
    using Entry = std::pair<int, double>;          // (index, coefficient)
    using Row   = std::vector<Entry>;

    ConstraintOperator() = default;

    /// @brief Build from explicit rows: rows[j] = { (i, C_ji), ... }.
    ConstraintOperator(int n_raw, std::vector<Row> rows)
        : n_raw_(n_raw), rows_(std::move(rows)), cols_(n_raw)
    {
        for (int j = 0; j < n_adapted(); ++j)
            for (const Entry &en : rows_[j])
            {
                if (en.first < 0 || en.first >= n_raw_)
                    throw std::out_of_range("ConstraintOperator: raw index out of range");
                cols_[en.first].push_back(Entry(j, en.second));
            }
    }

    /// @brief The identity, n_adapted == n_raw (the "free" space).
    static ConstraintOperator identity(int n)
    {
        std::vector<Row> rows(n);
        for (int i = 0; i < n; ++i) rows[i].push_back(Entry(i, 1.0));
        return ConstraintOperator(n, std::move(rows));
    }

    int n_raw()     const { return n_raw_; }
    int n_adapted() const { return static_cast<int>(rows_.size()); }
    bool is_identity() const
    {
        if (n_raw_ != n_adapted()) return false;
        for (int j = 0; j < n_adapted(); ++j)
            if (rows_[j].size() != 1 || rows_[j][0].first != j || rows_[j][0].second != 1.0) return false;
        return true;
    }

    /// @brief Adapted row j: the raw functions and weights that make up N_j.
    const Row &row(int j) const { return rows_[j]; }
    /// @brief Raw column i: the adapted functions that raw B_i contributes to.
    const Row &col(int i) const { return cols_[i]; }

    /// @brief y = C x   (raw-length x  ->  adapted-length y).  Restriction of a vector.
    template <class Scalar>
    std::vector<Scalar> restrict_vector(const std::vector<Scalar> &x) const
    {
        std::vector<Scalar> y(n_adapted(), Scalar(0));
        for (int j = 0; j < n_adapted(); ++j)
            for (const Entry &en : rows_[j]) y[j] += en.second * x[en.first];
        return y;
    }

    /// @brief x = C^T y   (adapted-length y  ->  raw-length x).  Prolongation.
    template <class Scalar>
    std::vector<Scalar> prolongate(const std::vector<Scalar> &y) const
    {
        std::vector<Scalar> x(n_raw_, Scalar(0));
        for (int j = 0; j < n_adapted(); ++j)
            for (const Entry &en : rows_[j]) x[en.first] += en.second * y[j];
        return x;
    }

    /**
     * @brief Ka = C K C^T for a DENSE row-major raw matrix K (n_raw x n_raw).
     *        This is the Restrict-mode oracle of the design doc, Section 7.
     *        Dense on purpose: tests only.
     */
    std::vector<double> restrict_dense(const std::vector<double> &K) const
    {
        int n = n_raw_, na = n_adapted();
        if (static_cast<int>(K.size()) != n * n) throw std::invalid_argument("restrict_dense: K must be n_raw x n_raw");
        // T = C K   (na x n)
        std::vector<double> T(static_cast<std::size_t>(na) * n, 0.0);
        for (int j = 0; j < na; ++j)
            for (const Entry &en : rows_[j])
                for (int c = 0; c < n; ++c) T[j * n + c] += en.second * K[en.first * n + c];
        // Ka = T C^T  (na x na)
        std::vector<double> Ka(static_cast<std::size_t>(na) * na, 0.0);
        for (int j = 0; j < na; ++j)
            for (int k = 0; k < na; ++k)
                for (const Entry &en : rows_[k]) Ka[j * na + k] += T[j * n + en.first] * en.second;
        return Ka;
    }

private:
    int n_raw_ = 0;
    std::vector<Row> rows_;   // adapted j -> raw entries
    std::vector<Row> cols_;   // raw i     -> adapted entries
};

} // namespace femd

#endif // FEMD_FE_CONSTRAINT_OPERATOR_HPP
