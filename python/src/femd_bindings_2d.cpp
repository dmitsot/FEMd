//
//  femd_bindings_2d.cpp  --  nanobind layer for the 2D finite element spaces and the
//  CSR store: LagrangeSpace2D, Cache2D, the 2D kernels, SparsityPattern, CSRMatrix and
//  the sparse preconditioners.  Called from the module initializer in
//  femd_bindings.cpp (bind_2d), after Mesh2D is registered.
//
//  The CSR arrays are views that borrow the owning CSRMatrix (no copy): the Python
//  SparseMatrix builds its scipy.sparse.csr_matrix straight on them.
//
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include "femd/femd.hpp"
#include <memory>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace femd;

namespace {

using DArr  = nb::ndarray<const double, nb::shape<-1>, nb::c_contig>;
using IArr  = nb::ndarray<const int, nb::shape<-1>, nb::c_contig>;
using PArr  = nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig>;
using Vec   = nb::ndarray<double, nb::numpy, nb::shape<-1>, nb::c_contig>;
using IVec  = nb::ndarray<int, nb::numpy, nb::shape<-1>, nb::c_contig>;
using Mat   = nb::ndarray<double, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;
using IMat  = nb::ndarray<int, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;
using GilRelease = nb::gil_scoped_release;

std::vector<double> vec(const DArr &a) { return std::vector<double>(a.data(), a.data() + a.shape(0)); }
std::vector<int> ivec(const IArr &a) { return std::vector<int>(a.data(), a.data() + a.shape(0)); }

template <class T>
nb::ndarray<T, nb::numpy, nb::shape<-1>, nb::c_contig> np1(const std::vector<T> &v)
{
    T *p = new T[v.size()];
    std::copy(v.begin(), v.end(), p);
    nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<T *>(q); });
    return nb::ndarray<T, nb::numpy, nb::shape<-1>, nb::c_contig>(p, {v.size()}, owner);
}
template <class T>
nb::ndarray<T, nb::numpy, nb::shape<-1, -1>, nb::c_contig> np2(const std::vector<T> &v, std::size_t rows, std::size_t cols)
{
    T *p = new T[v.size()];
    std::copy(v.begin(), v.end(), p);
    nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<T *>(q); });
    return nb::ndarray<T, nb::numpy, nb::shape<-1, -1>, nb::c_contig>(p, {rows, cols}, owner);
}

std::vector<std::vector<double>> coeff_list(const std::vector<DArr> &c)
{
    std::vector<std::vector<double>> out;
    out.reserve(c.size());
    for (const auto &a : c) out.push_back(vec(a));
    return out;
}

std::vector<Point> points(const PArr &a)
{
    std::vector<Point> p(a.shape(0));
    for (std::size_t i = 0; i < a.shape(0); ++i) p[i] = Point{a(i, 0), a(i, 1)};
    return p;
}

// at_points of a scalar space without copying the coefficients or the result.
Vec at_points_scalar(const Cache2D &Q, const DArr &c, int d, bool raw)
{
    const std::size_t want = static_cast<std::size_t>(raw ? Q.space().raw_dim() : Q.space().dim());
    if (c.shape(0) != want)
        throw std::invalid_argument(raw ? "at_points_raw: coefficient length != raw_dim" : "at_points: coefficient length != dim");
    if (d < 0 || d > 2) throw std::invalid_argument("derivative code is 0 (value), 1 (d/dx) or 2 (d/dy)");
    const std::size_t np = static_cast<std::size_t>(Q.nent()) * Q.nq();
    double *p = new double[np];
    nb::capsule owner(p, [](void *v) noexcept { delete[] static_cast<double *>(v); });
    { GilRelease g; Q.eval_multi(c.data(), raw, &d, 1, p); }
    return Vec(p, {np}, owner);
}

} // namespace

void bind_2d(nb::module_ &m)
{
    // ---- sparse store --------------------------------------------------------
    nb::class_<SparsityPattern>(m, "SparsityPattern", "Row pointers and sorted column indices, shared by matrices.")
        .def_ro("nrows", &SparsityPattern::nrows)
        .def_ro("ncols", &SparsityPattern::ncols)
        .def_prop_ro("nnz", &SparsityPattern::nnz)
        .def_static("from_pairs", [](int nr, int nc, IArr I, IArr J) {
                return SparsityPattern::from_pairs(nr, nc, ivec(I), ivec(J)); },
            "nrows"_a, "ncols"_a, "I"_a, "J"_a);

    nb::class_<CSRMatrix>(m, "CSRMatrix", "Compressed sparse row matrix owned by FEMd.")
        .def(nb::init<std::shared_ptr<const SparsityPattern>, bool>(), "pattern"_a, "symmetric"_a = false)
        .def_static("from_arrays", [](int nr, int nc, IArr indptr, IArr indices, DArr data, bool sym) {
                if (indptr.shape(0) != static_cast<std::size_t>(nr) + 1) throw std::invalid_argument("CSRMatrix: indptr length != nrows + 1");
                if (indices.shape(0) != data.shape(0)) throw std::invalid_argument("CSRMatrix: indices and data differ in length");
                auto P = std::make_shared<SparsityPattern>();
                P->nrows = nr; P->ncols = nc; P->indptr = ivec(indptr); P->indices = ivec(indices);
                for (int i = 0; i < nr; ++i)          // SciPy may hand over unsorted rows: sort them with their values
                {
                    const int a = P->indptr[i], b = P->indptr[i + 1];
                    if (!std::is_sorted(P->indices.begin() + a, P->indices.begin() + b))
                        throw std::invalid_argument("CSRMatrix.from_arrays: column indices must be sorted (call sort_indices())");
                }
                CSRMatrix K(P, sym);
                K.data = vec(data);
                return K;
            }, "nrows"_a, "ncols"_a, "indptr"_a, "indices"_a, "data"_a, "symmetric"_a = false)
        .def_prop_ro("nrows", &CSRMatrix::nrows)
        .def_prop_ro("ncols", &CSRMatrix::ncols)
        .def_prop_ro("nnz", &CSRMatrix::nnz)
        .def_rw("symmetric", &CSRMatrix::symmetric)
        .def_ro("pattern", &CSRMatrix::pattern)
        .def_prop_ro("indptr", [](nb::object self) {
                const CSRMatrix &K = nb::cast<const CSRMatrix &>(self);
                return nb::ndarray<nb::numpy, const int, nb::shape<-1>, nb::c_contig>(K.indptr(), {K.pattern->indptr.size()}, self); })
        .def_prop_ro("indices", [](nb::object self) {
                const CSRMatrix &K = nb::cast<const CSRMatrix &>(self);
                return nb::ndarray<nb::numpy, const int, nb::shape<-1>, nb::c_contig>(K.indices(), {K.pattern->indices.size()}, self); })
        .def_prop_ro("data", [](nb::object self) {
                CSRMatrix &K = nb::cast<CSRMatrix &>(self);
                return nb::ndarray<nb::numpy, double, nb::shape<-1>, nb::c_contig>(K.data.data(), {K.data.size()}, self); },
            "Writable view of the values.")
        .def("matvec", [](const CSRMatrix &K, DArr x) {
                if (x.shape(0) != static_cast<std::size_t>(K.ncols())) throw std::invalid_argument("matvec: length mismatch");
                double *y = new double[K.nrows()];
                nb::capsule owner(y, [](void *q) noexcept { delete[] static_cast<double *>(q); });
                { GilRelease r; if (K.nrows() > 0) K.apply(x.data(), y); }
                return Vec(y, {static_cast<std::size_t>(K.nrows())}, owner); }, "x"_a)
        .def("matvec_into", [](const CSRMatrix &K, DArr x, Vec y) {
                if (x.shape(0) != static_cast<std::size_t>(K.ncols()) || y.shape(0) != static_cast<std::size_t>(K.nrows()))
                    throw std::invalid_argument("matvec_into: length mismatch");
                GilRelease r; if (K.nrows() > 0) K.apply(x.data(), y.data()); }, "x"_a, "y"_a)
        .def("inner", [](const CSRMatrix &K, DArr x, DArr y) {
                if (x.shape(0) != static_cast<std::size_t>(K.ncols()) || y.shape(0) != static_cast<std::size_t>(K.nrows()))
                    throw std::invalid_argument("inner(x, y): x needs ncols entries and y nrows");
                if (K.nrows() == 0) return 0.0;
                GilRelease r; return K.inner(x.data(), y.data()); }, "x"_a, "y"_a,
             "y^T K x without forming K x; an ordered sum, the same bits for any thread count.")
        .def("rmatvec", [](const CSRMatrix &K, DArr x) {
                if (x.shape(0) != static_cast<std::size_t>(K.nrows())) throw std::invalid_argument("rmatvec: length mismatch");
                std::vector<double> y(static_cast<std::size_t>(K.ncols()));
                K.apply_transpose(x.data(), y.data());
                return np1(y); }, "x"_a)
        .def("diagonal", [](const CSRMatrix &K) { return np1(K.diagonal()); })
        .def("dense", [](const CSRMatrix &K) { return np2(K.dense(), K.nrows(), K.ncols()); })
        .def("combine", [](const CSRMatrix &A, const CSRMatrix &B, double a, double b) { return combine(A, B, a, b); },
             "B"_a, "alpha"_a, "beta"_a, "alpha A + beta B (A's pattern when the patterns are shared)")
        .def("scaled", [](const CSRMatrix &A, double a) { return scaled(A, a); }, "a"_a)
        .def("transposed", [](const CSRMatrix &A) { return transposed(A); })
        .def("multiply", [](const CSRMatrix &A, const CSRMatrix &B) { return multiply(A, B); }, "B"_a)
        .def("asymmetry", [](const CSRMatrix &A) { return asymmetry(A); })
        .def("pruned", &CSRMatrix::pruned, "tol"_a = 0.0)
        .def("copy", [](const CSRMatrix &A) { return CSRMatrix(A); })
        .def("clear", &CSRMatrix::clear);

    nb::class_<SparsePreconditioner>(m, "SparsePreconditioner", "z = M^{-1} r on a CSRMatrix.")
        .def_prop_ro("size", &SparsePreconditioner::size)
        .def_prop_ro("name", &SparsePreconditioner::name)
        .def_prop_ro("ordering", &SparsePreconditioner::ordering)
        .def("apply", [](const SparsePreconditioner &P, DArr r) {
                if (r.shape(0) != static_cast<std::size_t>(P.size())) throw std::invalid_argument("preconditioner: length mismatch");
                std::vector<double> z(static_cast<std::size_t>(P.size()));
                { GilRelease g; P.apply(r.data(), z.data()); }
                return np1(z); }, "r"_a);
    nb::class_<JacobiPreconditioner, SparsePreconditioner>(m, "JacobiPreconditioner")
        .def(nb::init<const CSRMatrix &>(), "K"_a);
    nb::class_<SSORPreconditioner, SparsePreconditioner>(m, "SSORPreconditioner")
        .def(nb::init<const CSRMatrix &, double, const std::string &>(), "K"_a, "omega"_a = 1.0, "ordering"_a = "natural");
    nb::class_<ILU0Preconditioner, SparsePreconditioner>(m, "ILU0Preconditioner")
        .def(nb::init<const CSRMatrix &, const std::string &>(), "K"_a, "ordering"_a = "natural", nb::call_guard<GilRelease>())
        .def_prop_ro("min_pivot", &ILU0Preconditioner::min_pivot);

    // ---- sparse LDL^T (sparse/cholesky.hpp) and the AMD ordering (sparse/amd.hpp) ------
    nb::class_<SparseCholesky, SparsePreconditioner>(m, "SparseCholesky",
            "P K P^T = L D L^T of a symmetric matrix (both triangles stored), AMD or natural order.")
        .def("__init__", [](SparseCholesky *self, const CSRMatrix &K, const std::string &ordering, bool positive_definite) {
                SparseCholesky::Ordering o;
                if (ordering == "amd") o = SparseCholesky::Ordering::AMD;
                else if (ordering == "natural") o = SparseCholesky::Ordering::Natural;
                else throw std::invalid_argument("SparseCholesky: ordering is 'amd' or 'natural', got '" + ordering + "'");
                GilRelease g;
                new (self) SparseCholesky(K, o, positive_definite); },
             "K"_a, "ordering"_a = "amd", "positive_definite"_a = true)
        .def("refactor", &SparseCholesky::refactor, "K"_a, nb::call_guard<GilRelease>(),
             "New values on the same pattern: the numeric factorization only.")
        .def("solve", [](const SparseCholesky &C, DArr b) {
                if (b.shape(0) != static_cast<std::size_t>(C.size())) throw std::invalid_argument("SparseCholesky.solve: length mismatch");
                std::vector<double> x(static_cast<std::size_t>(C.size()));
                { GilRelease g; if (C.size() > 0) C.apply(b.data(), x.data()); }
                return np1(x); }, "b"_a, "x = K^{-1} b.")
        .def_prop_ro("nnz_L", &SparseCholesky::nnz_L)
        .def_prop_ro("flops", &SparseCholesky::flops)
        .def_prop_ro("min_pivot", &SparseCholesky::min_pivot)
        .def_prop_ro("max_pivot", &SparseCholesky::max_pivot)
        .def_prop_ro("ordering", &SparseCholesky::ordering_name)
        .def_prop_ro("positive_definite", &SparseCholesky::positive_definite)
        .def("permutation", [](const SparseCholesky &C) { return np1(C.permutation()); })
        .def("etree", [](const SparseCholesky &C) { return np1(C.etree()); })
        .def("factors", [](const SparseCholesky &C) {
                return nb::make_tuple(np1(C.Lp()), np1(C.Li()), np1(C.Lx()), np1(C.D())); },
             "(Lp, Li, Lx, D): the unit lower triangular L by columns (diagonal not stored) and D, permuted.");
    nb::class_<SparseLDLT, SparsePreconditioner>(m, "SparseLDLT",
            "P K P^T = L D L^T of a symmetric (possibly indefinite) matrix, multifrontal with 1x1 and 2x2 threshold pivoting.")
        .def("__init__", [](SparseLDLT *self, const CSRMatrix &K, const std::string &ordering, double u) {
                SparseLDLT::Ordering o;
                if (ordering == "amd") o = SparseLDLT::Ordering::AMD;
                else if (ordering == "natural") o = SparseLDLT::Ordering::Natural;
                else throw std::invalid_argument("SparseLDLT: ordering is 'amd' or 'natural', got '" + ordering + "'");
                GilRelease g;
                new (self) SparseLDLT(K, o, u); },
             "K"_a, "ordering"_a = "amd", "threshold"_a = 0.01)
        .def("refactor", &SparseLDLT::refactor, "K"_a, nb::call_guard<GilRelease>(),
             "New values on the same pattern: the numeric phase only.")
        .def("solve", [](const SparseLDLT &F, DArr b) {
                if (b.shape(0) != static_cast<std::size_t>(F.size())) throw std::invalid_argument("SparseLDLT.solve: length mismatch");
                std::vector<double> x(static_cast<std::size_t>(F.size()));
                { GilRelease g; if (F.size() > 0) F.apply(b.data(), x.data()); }
                return np1(x); }, "b"_a)
        .def_prop_ro("nnz_L", &SparseLDLT::nnz_L)
        .def_prop_ro("inertia", [](const SparseLDLT &F) { auto a = F.inertia(); return nb::make_tuple(a[0], a[1], a[2]); })
        .def_prop_ro("delayed", &SparseLDLT::delayed)
        .def_prop_ro("two_by_two", &SparseLDLT::two_by_two)
        .def_prop_ro("supernodes", &SparseLDLT::supernodes)
        .def_prop_ro("max_front", &SparseLDLT::max_front)
        .def_prop_ro("threshold", &SparseLDLT::threshold)
        .def_prop_ro("min_pivot", &SparseLDLT::min_pivot)
        .def_prop_ro("ordering", &SparseLDLT::ordering_name)
        .def("permutation", [](const SparseLDLT &F) { return np1(F.permutation()); });
    m.def("amd_order", [](const CSRMatrix &K) {
            if (K.nrows() != K.ncols()) throw std::invalid_argument("amd_order: the matrix must be square");
            std::vector<int> p;
            { GilRelease g; p = ordering::amd_order(K.nrows(), K.indptr(), K.indices()); }
            return np1(p); }, "K"_a, "The approximate minimum degree order of the pattern of K (symmetrized): p[k] is the k-th pivot.");
    m.def("rcm_order", [](const CSRMatrix &K) {
            if (K.nrows() != K.ncols()) throw std::invalid_argument("rcm_order: the matrix must be square");
            std::vector<int> p;
            { GilRelease g; p = ordering::rcm_order(K.nrows(), K.indptr(), K.indices()); }
            return np1(p); }, "K"_a, "The reverse Cuthill-McKee order of the pattern of K (symmetrized): p[k] is the k-th row.");
    m.def("bandwidth", [](const CSRMatrix &K, std::vector<int> p) {
            if (p.empty()) p = ordering::natural_order(K.nrows());
            return ordering::bandwidth(K.nrows(), K.indptr(), K.indices(), p); }, "K"_a, "p"_a = std::vector<int>(),
          "The half-bandwidth of K under the order p (natural when empty).");

    // ---- quadrature and the reference element ---------------------------------
    m.def("triangle_rule", [](int d) {
            TriangleQuadrature Q(d);
            std::vector<double> xi(Q.size()), eta(Q.size()), w(Q.size());
            for (int q = 0; q < Q.size(); ++q) { xi[q] = Q.xi(q); eta[q] = Q.eta(q); w[q] = Q.weight(q); }
            return nb::make_tuple(np1(xi), np1(eta), np1(w)); }, "degree"_a,
          "(xi, eta, w) of the rule on the reference triangle exact to total degree `degree`.");
    nb::class_<ReferenceTriangle>(m, "ReferenceTriangle")
        .def(nb::init<int, bool>(), "degree"_a, "equispaced"_a = true)
        .def_prop_ro("degree", &ReferenceTriangle::degree)
        .def_prop_ro("size", &ReferenceTriangle::size)
        .def("nodes", [](const ReferenceTriangle &R) {
                std::vector<double> v(2 * R.size());
                for (int i = 0; i < R.size(); ++i) { v[2 * i] = R.node_xi(i); v[2 * i + 1] = R.node_eta(i); }
                return np2(v, R.size(), 2); })
        .def("eval", [](const ReferenceTriangle &R, double xi, double eta, int nder) {
                const int rows = nder >= 1 ? 3 : 1;
                std::vector<double> out(static_cast<std::size_t>(rows) * R.size());
                R.eval(xi, eta, nder, out.data());
                return np2(out, rows, R.size()); }, "xi"_a, "eta"_a, "nder"_a = 0,
             "Rows: values, then d/dxi and d/deta when nder >= 1.");

    // ---- the space ---------------------------------------------------------------
    nb::class_<Space2D>(m, "Space2D", "Base of the 2D Lagrange spaces: numbering, elimination, facets, evaluation.")
        .def_prop_ro("degree", &Space2D::degree)
        .def_prop_ro("ncells", &Space2D::ncells)
        .def_prop_ro("nedges", &Space2D::nedges)
        .def_prop_ro("nloc", &Space2D::nloc)
        .def_prop_ro("raw_dim", &Space2D::raw_dim)
        .def_prop_ro("dim", &Space2D::dim)
        .def_prop_ro("n_constraints", &Space2D::n_constraints)
        .def_prop_ro("dirichlet_markers", &Space2D::dirichlet_markers)
        .def_prop_ro("all_exterior", &Space2D::all_exterior)
        .def_prop_ro("nverts", &Space2D::nverts)
        .def_prop_ro("affine", &Space2D::affine)
        .def_prop_ro("cell_name", &Space2D::cell_name)
        .def_prop_ro("equispaced", &Space2D::equispaced)
        .def_prop_ro("ncomp", &Space2D::ncomp)
        .def_prop_ro("piola", &Space2D::piola, "0 none, 1 contravariant (H(div)), 2 covariant (H(curl))")
        .def_prop_ro("ncodes", &Space2D::ncodes)
        .def_prop_ro("dofs_per_vertex", &Space2D::dofs_per_vertex)
        .def_prop_ro("dofs_per_edge", &Space2D::dofs_per_edge)
        .def_prop_ro("n_interior", &Space2D::n_interior)
        .def("cell_signs", [](const Space2D &V) {
                std::vector<double> s(static_cast<std::size_t>(V.ncells()) * V.nloc());
                for (int c = 0; c < V.ncells(); ++c) for (int l = 0; l < V.nloc(); ++l) s[static_cast<std::size_t>(c) * V.nloc() + l] = V.sign(c, l);
                return np2(s, V.ncells(), V.nloc()); }, "+1 or -1 per local function (reversed edges of a vector family)")
        .def("evaluate_ref", [](const Space2D &V, DArr raw, IArr cells, DArr xi, DArr eta, int code) {
                std::vector<double> r = vec(raw), a = vec(xi), b = vec(eta);
                std::vector<int> c = ivec(cells);
                std::vector<double> out;
                { GilRelease g; out = V.evaluate_ref(r, c, a, b, code); }
                return np1(out); }, "raw"_a, "cells"_a, "xi"_a, "eta"_a, "code"_a = 0,
             "Values (derivative code `code`) at reference points (xi, eta) of the given cells, raw coefficients.")
        .def("ref_eval", [](const Space2D &V, double xi, double eta, int nder) {
                const int rows = V.ncomp() > 1 ? 3 * V.ncomp() : (nder >= 1 ? 3 : 1);
                std::vector<double> out(static_cast<std::size_t>(rows) * V.nloc());
                V.ref_eval(xi, eta, nder, out.data());
                return np2(out, rows, V.nloc()); }, "xi"_a, "eta"_a, "nder"_a = 0)
        .def("ref_nodes", [](const Space2D &V) {
                std::vector<double> v(2 * V.nloc());
                for (int l = 0; l < V.nloc(); ++l) V.ref_node(l, v[2 * l], v[2 * l + 1]);
                return np2(v, V.nloc(), 2); })
        .def("cell_vertices", [](const Space2D &V) {
                std::vector<int> v(static_cast<std::size_t>(V.ncells()) * V.nverts());
                for (int c = 0; c < V.ncells(); ++c) for (int i = 0; i < V.nverts(); ++i) v[static_cast<std::size_t>(c) * V.nverts() + i] = V.cell_vertex(c, i);
                return np2(v, V.ncells(), V.nverts()); })
        .def("map_points", [](const Space2D &V, IArr cells, DArr xi, DArr eta) {
                const std::size_t n = cells.shape(0);
                if (xi.shape(0) != n || eta.shape(0) != n) throw std::invalid_argument("map_points: lengths differ");
                std::vector<double> out(2 * n), J(4 * n);
                for (std::size_t i = 0; i < n; ++i) V.map(cells(i), xi(i), eta(i), out[2 * i], out[2 * i + 1], &J[4 * i]);
                return nb::make_tuple(np2(out, n, 2), np2(J, n, 4)); }, "cells"_a, "xi"_a, "eta"_a,
             "Physical points and Jacobians [dx/dxi, dx/deta, dy/dxi, dy/deta] of reference points in the given cells.")
        .def("raw_coordinates", [](const Space2D &V) { return np2(V.raw_coordinates(), V.raw_dim(), 2); })
        .def("cell_dofs", [](const Space2D &V) { return np2(V.cell_dofs(), V.ncells(), V.nloc()); })
        .def("cell_raw_dofs", [](const Space2D &V) { return np2(V.cell_raw_dofs(), V.ncells(), V.nloc()); })
        .def("constrained", [](const Space2D &V) { return np1(V.constrained()); })
        .def("adapted_to_raw", [](const Space2D &V) {
                std::vector<int> v(V.dim()); for (int a = 0; a < V.dim(); ++a) v[a] = V.adapted_to_raw(a); return np1(v); })
        .def("raw_to_adapted", [](const Space2D &V) {
                std::vector<int> v(V.raw_dim()); for (int r = 0; r < V.raw_dim(); ++r) v[r] = V.raw_to_adapted(r); return np1(v); })
        .def("prolongate", [](const Space2D &V, DArr c) { return np1(V.prolongate(vec(c))); }, "c"_a)
        .def("restrict", [](const Space2D &V, DArr r) { return np1(V.restrict_raw(vec(r))); }, "raw"_a)
        .def("_identify", [](Space2D &V, IArr rep) {
                std::vector<int> r(rep.data(), rep.data() + rep.shape(0));
                V.identify(r); }, "rep"_a, "Identify raw nodes (periodic conditions): rep[r] is the representative of r.")
        .def("representatives", [](const Space2D &V) { return np1(V.representatives()); })
        .def_prop_ro("is_periodic", &Space2D::periodic)
        .def("edges", [](const Space2D &V) {
                std::vector<int> v(2 * V.nedges());
                for (int g = 0; g < V.nedges(); ++g) { auto e = V.edge(g); v[2 * g] = e[0]; v[2 * g + 1] = e[1]; }
                return np2(v, V.nedges(), 2); })
        .def("facets", [](const Space2D &V) {
                const auto &F = V.facets();
                std::vector<int> c(F.size()), l(F.size()), mk(F.size()), ex(F.size()), ed(F.size());
                for (std::size_t i = 0; i < F.size(); ++i) { c[i] = F[i].cell; l[i] = F[i].local; mk[i] = F[i].marker; ex[i] = F[i].exterior; ed[i] = F[i].edge; }
                return nb::make_tuple(np1(c), np1(l), np1(mk), np1(ex), np1(ed)); },
             "(cell, local edge, marker, exterior, global edge) of every facet")
        .def("interior_facets", [](const Space2D &V) {
                const auto &F = V.interior_facets();
                std::vector<int> c(2 * F.size()), l(2 * F.size()), e(F.size()), per(F.size());
                std::vector<double> sh(2 * F.size());
                for (std::size_t i = 0; i < F.size(); ++i)
                {
                    c[2 * i] = F[i].cell[0]; c[2 * i + 1] = F[i].cell[1]; l[2 * i] = F[i].local[0]; l[2 * i + 1] = F[i].local[1];
                    e[i] = F[i].edge; per[i] = F[i].periodic; sh[2 * i] = F[i].shift[0]; sh[2 * i + 1] = F[i].shift[1];
                }
                return nb::make_tuple(np2(c, F.size(), 2), np2(l, F.size(), 2), np1(e), np1(per), np2(sh, F.size(), 2)); },
             "(cells (nf, 2), local edges (nf, 2), edge, periodic, shift (nf, 2)) of the interior facets, '-' side first")
        .def("_set_periodic_facets", [](Space2D &V, IArr ca, IArr la, IArr cb, IArr lb, DArr sx, DArr sy) {
                V.set_periodic_facets(ivec(ca), ivec(la), ivec(cb), ivec(lb), vec(sx), vec(sy)); },
             "ca"_a, "la"_a, "cb"_a, "lb"_a, "sx"_a, "sy"_a)
        .def_prop_ro("n_periodic_facets", &Space2D::n_periodic_facets)
        .def("evaluate_raw", [](const Space2D &V, DArr raw, PArr pts, int deriv) {
                std::vector<double> r = vec(raw);
                std::vector<Point> p = points(pts);
                std::vector<double> out;
                { GilRelease g; out = V.evaluate_raw(r, p, deriv); }
                return np1(out); }, "raw"_a, "points"_a, "deriv"_a = 0)
        .def("locate", [](const Space2D &V, PArr pts) {
                std::vector<Point> p = points(pts);
                std::vector<int> cells;
                std::vector<double> zero(static_cast<std::size_t>(V.raw_dim()), 0.0);
                V.evaluate_raw(zero, p, 0, &cells);
                return np1(cells); }, "points"_a);

    nb::class_<LagrangeSpace2D, Space2D>(m, "LagrangeSpace2D")
        .def(nb::init<const Mesh2D &, int, std::vector<int>, bool, bool>(), "mesh"_a, "degree"_a, "dirichlet"_a = std::vector<int>{},
             "all_exterior"_a = false, "equispaced"_a = true, nb::keep_alive<1, 2>());
    nb::enum_<VectorFamily>(m, "VectorFamily")
        .value("RT", VectorFamily::RT)
        .value("N1curl", VectorFamily::N1curl);
    nb::class_<VectorElementSpace2D, Space2D>(m, "VectorElementSpace2D",
                                              "Raviart-Thomas RT_k or Nedelec N1curl_k on triangles (vector_element_2d.hpp).")
        .def(nb::init<const Mesh2D &, VectorFamily, int, std::vector<int>, bool>(), "mesh"_a, "family"_a, "degree"_a,
             "dirichlet"_a = std::vector<int>{}, "all_exterior"_a = false, nb::keep_alive<1, 2>())
        .def_prop_ro("vector_family", &VectorElementSpace2D::family)
        .def("interpolation_points", [](const VectorElementSpace2D &V) {
                std::vector<double> p = V.interpolation_points();
                return np2(p, p.size() / 2, 2); })
        .def("edge_projection", [](const VectorElementSpace2D &V) {
                const auto &R = V.reference();
                const int k = V.degree(), nq = R.edge_rule_points();
                std::vector<double> P(static_cast<std::size_t>(k) * nq), t(static_cast<std::size_t>(nq));
                for (int q = 0; q < nq; ++q) t[q] = R.edge_rule_point(q);
                for (int j = 0; j < k; ++j) for (int q = 0; q < nq; ++q) P[static_cast<std::size_t>(j) * nq + q] = R.edge_projection(j, q);
                return nb::make_tuple(np1(t), np2(P, k, nq)); },
             "(t, P): the edge rule on [0, 1] and dof_j = sum_q P[j, q] f(t_q), the L2 projection onto P_{k-1}")
        .def("interpolate_raw", [](const VectorElementSpace2D &V, PArr F) {
                std::vector<double> f(F.data(), F.data() + 2 * F.shape(0));
                return np1(V.interpolate_raw(f)); }, "values"_a);
    nb::class_<DGSpace2D, Space2D>(m, "DGSpace2D", "Broken P_k on triangles (vector_element_2d.hpp).")
        .def(nb::init<const Mesh2D &, int, bool>(), "mesh"_a, "degree"_a, "equispaced"_a = true, nb::keep_alive<1, 2>());
    nb::class_<LagrangeSpaceQ, Space2D>(m, "LagrangeSpaceQ")
        .def(nb::init<const QuadMesh &, int, std::vector<int>, bool, bool>(), "mesh"_a, "degree"_a, "dirichlet"_a = std::vector<int>{},
             "all_exterior"_a = false, "equispaced"_a = true, nb::keep_alive<1, 2>());
    nb::class_<ReferenceQuad>(m, "ReferenceQuad")
        .def(nb::init<int, bool>(), "degree"_a, "equispaced"_a = true)
        .def_prop_ro("degree", &ReferenceQuad::degree)
        .def_prop_ro("size", &ReferenceQuad::size)
        .def("nodes", [](const ReferenceQuad &R) {
                std::vector<double> v(2 * R.size());
                for (int i = 0; i < R.size(); ++i) { v[2 * i] = R.node_xi(i); v[2 * i + 1] = R.node_eta(i); }
                return np2(v, R.size(), 2); })
        .def("eval", [](const ReferenceQuad &R, double xi, double eta, int nder) {
                const int rows = nder >= 1 ? 3 : 1;
                std::vector<double> out(static_cast<std::size_t>(rows) * R.size());
                R.eval(xi, eta, nder, out.data());
                return np2(out, rows, R.size()); }, "xi"_a, "eta"_a, "nder"_a = 0);

    // ---- quadrilateral meshes -------------------------------------------------------------------
    nb::class_<QuadMesh>(m, "QuadMesh", "Quadrilateral mesh (mesh/quad_mesh.hpp).")
        .def("__init__", [](QuadMesh *self, PArr pts, nb::ndarray<const int, nb::shape<-1, 4>, nb::c_contig> quads,
                            nb::ndarray<const int, nb::shape<-1, 2>, nb::c_contig> segs, IArr smarks, IArr pmarks, int marker) {
                 std::vector<std::array<int, 4>> q(quads.shape(0));
                 for (std::size_t i = 0; i < q.size(); ++i) for (int j = 0; j < 4; ++j) q[i][j] = quads(i, j);
                 std::vector<std::array<int, 2>> s(segs.shape(0));
                 for (std::size_t i = 0; i < s.size(); ++i) { s[i][0] = segs(i, 0); s[i][1] = segs(i, 1); }
                 new (self) QuadMesh(points(pts), std::move(q), std::move(s), ivec(smarks), ivec(pmarks), marker);
             }, "points"_a, "quads"_a, "segments"_a, "segment_markers"_a, "point_markers"_a, "marker"_a = 1)
        .def_static("from_triangles", &QuadMesh::from_triangles, "mesh"_a)
        .def_prop_ro("npoints", &QuadMesh::npoints)
        .def_prop_ro("nquads", &QuadMesh::nquads)
        .def_prop_ro("nsegments", &QuadMesh::nsegments)
        .def_prop_ro("points", [](nb::object self) {
                 const QuadMesh &M = nb::cast<const QuadMesh &>(self);
                 return nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>, nb::c_contig>(
                     reinterpret_cast<const double *>(M.points().data()), {M.points().size(), 2}, self); })
        .def_prop_ro("quads", [](nb::object self) {
                 const QuadMesh &M = nb::cast<const QuadMesh &>(self);
                 return nb::ndarray<nb::numpy, const int, nb::shape<-1, 4>, nb::c_contig>(
                     reinterpret_cast<const int *>(M.quads().data()), {M.quads().size(), 4}, self); })
        .def_prop_ro("neighbours", [](nb::object self) {
                 const QuadMesh &M = nb::cast<const QuadMesh &>(self);
                 return nb::ndarray<nb::numpy, const int, nb::shape<-1, 4>, nb::c_contig>(
                     reinterpret_cast<const int *>(M.neighbours().data()), {M.neighbours().size(), 4}, self); })
        .def_prop_ro("segments", [](nb::object self) {
                 const QuadMesh &M = nb::cast<const QuadMesh &>(self);
                 return nb::ndarray<nb::numpy, const int, nb::shape<-1, 2>, nb::c_contig>(
                     reinterpret_cast<const int *>(M.segments().data()), {M.segments().size(), 2}, self); })
        .def_prop_ro("segment_markers", [](nb::object self) {
                 const QuadMesh &M = nb::cast<const QuadMesh &>(self);
                 return nb::ndarray<nb::numpy, const int, nb::shape<-1>, nb::c_contig>(M.segment_markers().data(), {M.segment_markers().size()}, self); })
        .def_prop_ro("point_markers", [](nb::object self) {
                 const QuadMesh &M = nb::cast<const QuadMesh &>(self);
                 return nb::ndarray<nb::numpy, const int, nb::shape<-1>, nb::c_contig>(M.point_markers().data(), {M.point_markers().size()}, self); })
        .def("areas", [](const QuadMesh &M) { std::vector<double> a(M.nquads()); for (int q = 0; q < M.nquads(); ++q) a[q] = M.area(q); return np1(a); })
        .def("min_angles", [](const QuadMesh &M) { std::vector<double> a(M.nquads()); for (int q = 0; q < M.nquads(); ++q) a[q] = M.min_angle(q); return np1(a); })
        .def("max_angles", [](const QuadMesh &M) { std::vector<double> a(M.nquads()); for (int q = 0; q < M.nquads(); ++q) a[q] = M.max_angle(q); return np1(a); })
        .def("aspects", [](const QuadMesh &M) { std::vector<double> a(M.nquads()); for (int q = 0; q < M.nquads(); ++q) a[q] = M.aspect(q); return np1(a); })
        .def("locate", [](const QuadMesh &M, PArr pts) {
                 std::vector<Point> p = points(pts);
                 std::vector<int> out(p.size());
                 const long n = static_cast<long>(p.size());
                 if (n > FEMD_OMP_THRESHOLD)
                 {
                     GilRelease g;
                     M.prepare_locate();                    // grid alone: the same answers for any team size
                     FEMD_OMP_FOR_IF(true)
                     for (long i = 0; i < n; ++i) out[i] = M.locate_prepared(p[i]);
                 }
                 else
                     for (long i = 0; i < n; ++i) out[i] = M.locate(p[i]);
                 return np1(out); }, "points"_a)
        .def("smooth", &QuadMesh::smooth, "passes"_a = 1)
        .def("validate", &QuadMesh::validate)
        .def("copy", [](const QuadMesh &M) { return QuadMesh(M); });

    nb::class_<Cache2D>(m, "Cache2D", "Quadrature points of a LagrangeSpace2D on its cells (dx) or boundary facets (ds).")
        .def(nb::init<const Space2D &, int>(), "space"_a, "degree"_a, nb::keep_alive<1, 2>())
        .def("__init__", [](Cache2D *self, const Space2D &V, int npts, bool lobatto) {
                 if (lobatto) new (self) Cache2D(V, npts, Cache2D::Lobatto{});
                 else new (self) Cache2D(V, npts);
             }, "space"_a, "npts"_a, "lobatto"_a, nb::keep_alive<1, 2>(),
             "With lobatto=True the tensor npts x npts Gauss-Lobatto rule on quadrilaterals.")
        .def_prop_ro("lobatto", &Cache2D::lobatto)
        .def(nb::init<const Space2D &, int, std::vector<int>, bool>(), "space"_a, "degree"_a, "markers"_a, "all"_a,
             nb::keep_alive<1, 2>())
        .def_prop_ro("nent", &Cache2D::nent)
        .def_prop_ro("nq", &Cache2D::nq)
        .def_prop_ro("nloc", &Cache2D::nloc)
        .def_prop_ro("degree", &Cache2D::degree)
        .def_prop_ro("on_facets", &Cache2D::on_facets)
        .def_prop_ro("ncodes", &Cache2D::ncodes)
        .def("x", [](const Cache2D &Q) { return np1(Q.x()); })
        .def("y", [](const Cache2D &Q) { return np1(Q.y()); })
        .def("points", [](const Cache2D &Q) {
                std::vector<double> v(2 * Q.x().size());
                for (std::size_t i = 0; i < Q.x().size(); ++i) { v[2 * i] = Q.x()[i]; v[2 * i + 1] = Q.y()[i]; }
                return np2(v, Q.x().size(), 2); })
        .def("weights", [](const Cache2D &Q) { return np1(Q.weights()); })
        .def("normal_x", [](const Cache2D &Q) { return np1(Q.normal_x()); })
        .def("normal_y", [](const Cache2D &Q) { return np1(Q.normal_y()); })
        .def("cells", [](const Cache2D &Q) { std::vector<int> c(Q.nent()); for (int e = 0; e < Q.nent(); ++e) c[e] = Q.cell(e); return np1(c); })
        .def("markers", [](const Cache2D &Q) { return np1(Q.markers()); })
        .def("at_points", [](const Cache2D &Q, DArr c, int d) {
                if (!Q.is_vector()) return at_points_scalar(Q, c, d, false);
                std::vector<double> cv = vec(c), out; { GilRelease g; out = Q.at_points(cv, d); } return np1(out); }, "c"_a, "deriv"_a = 0)
        .def("at_points_raw", [](const Cache2D &Q, DArr r, int d) {
                if (!Q.is_vector()) return at_points_scalar(Q, r, d, true);
                std::vector<double> rv = vec(r), out; { GilRelease g; out = Q.at_points_raw(rv, d); } return np1(out); }, "raw"_a, "deriv"_a = 0)
        .def("at_points_multi", [](const Cache2D &Q, DArr c, std::vector<int> codes, bool raw) {
                const std::size_t want = static_cast<std::size_t>(raw ? Q.space().raw_dim() : Q.space().dim());
                if (c.shape(0) != want) throw std::invalid_argument("at_points_multi: coefficient length != dim (raw_dim with raw=True)");
                const std::size_t np = static_cast<std::size_t>(Q.nent()) * Q.nq();
                double *p = new double[codes.size() * np];
                nb::capsule owner(p, [](void *v) noexcept { delete[] static_cast<double *>(v); });
                { GilRelease g; Q.eval_multi(c.data(), raw, codes.data(), static_cast<int>(codes.size()), p); }
                return Mat(p, {codes.size(), np}, owner); }, "c"_a, "codes"_a, "raw"_a = false,
             "Values (code 0), d/dx (1) and d/dy (2) of a scalar field at every point, one row per code, in one pass.");

    // ---- kernels -------------------------------------------------------------------
    nb::class_<InteriorFacetCache2D>(m, "InteriorFacetCache2D", "Gauss points on the interior facets of a 2D space (dS), both sides.")
        .def(nb::init<const Space2D &, int>(), "space"_a, "degree"_a, nb::keep_alive<1, 2>())
        .def_prop_ro("nf", &InteriorFacetCache2D::nf)
        .def_prop_ro("nq", &InteriorFacetCache2D::nq)
        .def_prop_ro("nloc", &InteriorFacetCache2D::nloc)
        .def_prop_ro("degree", &InteriorFacetCache2D::degree)
        .def_prop_ro("ncodes", &InteriorFacetCache2D::ncodes)
        .def("x", [](const InteriorFacetCache2D &F) { return np1(F.x()); })
        .def("y", [](const InteriorFacetCache2D &F) { return np1(F.y()); })
        .def("weights", [](const InteriorFacetCache2D &F) { return np1(F.weights()); })
        .def("normal_x", [](const InteriorFacetCache2D &F) { return np1(F.normal_x()); })
        .def("normal_y", [](const InteriorFacetCache2D &F) { return np1(F.normal_y()); })
        .def("cells", [](const InteriorFacetCache2D &F) {
                std::vector<int> c(2 * static_cast<std::size_t>(F.nf()));
                for (int f = 0; f < F.nf(); ++f) { c[2 * f] = F.cell(f, 0); c[2 * f + 1] = F.cell(f, 1); }
                return np2(c, F.nf(), 2); })
        .def("at_points", [](const InteriorFacetCache2D &F, DArr c, int side, int code) {
                std::vector<double> v = vec(c), out;
                { GilRelease g; out = F.at_points(v, side, code); }
                return np1(out); }, "c"_a, "side"_a, "code"_a)
        .def("at_points_raw", [](const InteriorFacetCache2D &F, DArr c, int side, int code) {
                std::vector<double> v = vec(c), out;
                { GilRelease g; out = F.at_points(v, side, code, true); }
                return np1(out); }, "raw"_a, "side"_a, "code"_a);
    m.def("facet_pattern", [](nb::list T, std::vector<int> toff, int nrows, nb::list S, std::vector<int> soff, int ncols) {
              std::vector<const Space2D *> t, s;
              for (nb::handle h : T) t.push_back(nb::cast<const Space2D *>(h));
              for (nb::handle h : S) s.push_back(nb::cast<const Space2D *>(h));
              GilRelease g;
              return facet_pattern(t, toff, nrows, s, soff, ncols); },
          "T"_a, "toff"_a, "nrows"_a, "S"_a, "soff"_a, "ncols"_a);
    m.def("assemble_facet_scalar_2d", [](const InteriorFacetCache2D &F, DArr c) { return assemble_facet_scalar_2d(F, vec(c)); },
          "F"_a, "coeff"_a);
    m.def("assemble_facet_vector_2d", [](const InteriorFacetCache2D &F, std::vector<int> side, std::vector<int> a, std::vector<DArr> c) {
              auto cc = coeff_list(c);
              std::vector<double> f(static_cast<std::size_t>(F.space().dim()), 0.0);
              { GilRelease g; assemble_facet_vector_2d(F, side, a, cc, f); }
              return np1(f); }, "F"_a, "side"_a, "a"_a, "coeffs"_a);
    m.def("assemble_facet_matrix_2d", [](const InteriorFacetCache2D &FT, const InteriorFacetCache2D &FS, std::vector<int> sa,
                                         std::vector<int> sb, std::vector<int> a, std::vector<int> b, std::vector<DArr> c,
                                         CSRMatrix &K, int roff, int coff) {
              auto cc = coeff_list(c);
              GilRelease g;
              assemble_facet_matrix_2d(FT, FS, sa, sb, a, b, cc, K, roff, coff); },
          "FT"_a, "FS"_a, "sa"_a, "sb"_a, "a"_a, "b"_a, "coeffs"_a, "K"_a, "row_offset"_a = 0, "col_offset"_a = 0);
    m.def("cell_pattern", &cell_pattern, "test"_a, "trial"_a, nb::call_guard<GilRelease>(),
          "CSR pattern of all test-trial couplings through a common cell.");
    m.def("product_pattern", [](nb::list T, std::vector<int> toff, int nrows, nb::list S, std::vector<int> soff, int ncols) {
              std::vector<const Space2D *> t, s;
              for (nb::handle h : T) t.push_back(nb::cast<const Space2D *>(h));
              for (nb::handle h : S) s.push_back(nb::cast<const Space2D *>(h));
              GilRelease g;
              return product_pattern(t, toff, nrows, s, soff, ncols);
          }, "test_fields"_a, "test_offsets"_a, "nrows"_a, "trial_fields"_a, "trial_offsets"_a, "ncols"_a,
          "CSR pattern of a system of fields numbered block by block.");
    m.def("assemble_matrix_2d", [](const Cache2D &QT, const Cache2D &QS, std::vector<int> a, std::vector<int> b,
                                   std::vector<DArr> c, CSRMatrix &K, int row_offset, int col_offset) {
              auto cc = coeff_list(c);
              GilRelease g;
              assemble_matrix_2d(QT, QS, a, b, cc, K, row_offset, col_offset);
          }, "QT"_a, "QS"_a, "a"_a, "b"_a, "coeffs"_a, "K"_a, "row_offset"_a = 0, "col_offset"_a = 0,
          "K += sum_t integral c_t (D^a_t T_i)(D^b_t S_j); codes 0 value, 1 d/dx, 2 d/dy.");
    m.def("assemble_vector_2d", [](const Cache2D &Q, std::vector<int> a, std::vector<DArr> c) {
              auto cc = coeff_list(c);
              std::vector<double> f(static_cast<std::size_t>(Q.space().dim()), 0.0);
              { GilRelease g; assemble_vector_2d(Q, a, cc, f); }
              return np1(f);
          }, "Q"_a, "a"_a, "coeffs"_a);
    m.def("assemble_scalar_2d", [](const Cache2D &Q, DArr c) { return assemble_scalar_2d(Q, vec(c)); }, "Q"_a, "coeff"_a);
}
