//
//  femd_bindings.cpp  --  nanobind layer over the C++ core (design doc M6).
//
//  The C++ objects are exposed almost one to one; the friendly grid-and-degree
//  API, bc parsing, scipy conversion and projections live in python/femd/.
//  All arrays cross as contiguous float64 / int32 NumPy views.  Indices are
//  0-based here, exactly as in the C++ core.
//
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>

#include "femd/femd.hpp"
#include "femd/krylov/adapters.hpp"
#include <complex>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;
using namespace femd;

using Vec  = nb::ndarray<double, nb::numpy, nb::shape<-1>, nb::c_contig>;
using IVec = nb::ndarray<int,    nb::numpy, nb::shape<-1>, nb::c_contig>;

// ---- helpers ---------------------------------------------------------------
static std::vector<double> to_vec(const nb::ndarray<const double, nb::shape<-1>, nb::c_contig> &a)
{
    return std::vector<double>(a.data(), a.data() + a.shape(0));
}
static Vec to_np(std::vector<double> v)
{
    double *p = new double[v.size()];
    std::copy(v.begin(), v.end(), p);
    nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<double *>(q); });
    return Vec(p, {v.size()}, owner);
}
static IVec to_np(std::vector<int> v)
{
    int *p = new int[v.size()];
    std::copy(v.begin(), v.end(), p);
    nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<int *>(q); });
    return IVec(p, {v.size()}, owner);
}
using DArr = nb::ndarray<const double, nb::shape<-1>, nb::c_contig>;
using IArr = nb::ndarray<const int, nb::shape<-1>, nb::c_contig>;

// Coefficient from a Python object: float, or array of values at the cache nodes.
static Coefficient make_coeff(nb::handle obj)
{
    if (nb::isinstance<nb::float_>(obj) || nb::isinstance<nb::int_>(obj)) return Coefficient(nb::cast<double>(obj));
    DArr a = nb::cast<DArr>(obj);
    return Coefficient(to_vec(a));
}

// Field evaluation without exposing Field<Scalar>: coefficients + points -> values
// The long loops below run without the GIL (other Python threads proceed) and, with
// OpenMP, over the points in parallel, each thread with its own evaluation buffers.
using GilRelease = nb::gil_scoped_release;

static Vec evaluate(const FunctionSpace &V, DArr coeffs, DArr x, int k)
{
    const double *c = coeffs.data(), *xs = x.data();
    const long np_ = static_cast<long>(x.shape(0));
    std::vector<double> out(static_cast<std::size_t>(np_));
    {
        GilRelease release;
        FEMD_OMP_PARALLEL_IF(np_ > FEMD_OMP_THRESHOLD)
        {
            std::vector<double> vals, raw;
            FEMD_OMP_FOR
            for (long i = 0; i < np_; ++i)
            {
                const int e = V.mesh().element_of(xs[i]);
                V.eval_on_element(e, xs[i], k, vals, raw);
                const std::vector<int> &d = V.element_dofs(e);
                const int na = static_cast<int>(d.size());
                double sum = 0.0;
                for (int kk = 0; kk < na; ++kk) sum += c[d[kk]] * vals[k * na + kk];
                out[static_cast<std::size_t>(i)] = sum;
            }
        }
    }
    return to_np(std::move(out));
}
static Vec at_quad(const FunctionSpace &V, const QuadratureCache &Q, DArr coeffs, int k)
{
    std::vector<double> out;
    {
        Field<double> u(V, to_vec(coeffs));
        GilRelease release;
        out = u.at_quad(Q, k);
    }
    return to_np(std::move(out));
}

// Basis at points: COO triplets (row = point, col = adapted dof) of d^k N_j(x_i)
static nb::tuple basis_at(const FunctionSpace &V, DArr x, int k)
{
    const double *xs = x.data();
    const long np_ = static_cast<long>(x.shape(0));
    std::vector<int> rows, cols; std::vector<double> vals;
    {
        GilRelease release;
        // pass 1: the element of each point and where its entries start
        std::vector<int> elem(static_cast<std::size_t>(np_));
        std::vector<std::size_t> start(static_cast<std::size_t>(np_) + 1, 0);
        for (long i = 0; i < np_; ++i)
        {
            elem[i] = V.mesh().element_of(xs[i]);
            start[i + 1] = start[i] + V.element_dofs(elem[i]).size();
        }
        rows.resize(start.back()); cols.resize(start.back()); vals.resize(start.back());
        // pass 2: the values, each point writing its own slots
        FEMD_OMP_PARALLEL_IF(np_ > FEMD_OMP_THRESHOLD)
        {
            std::vector<double> buf, raw;
            FEMD_OMP_FOR
            for (long i = 0; i < np_; ++i)
            {
                V.eval_on_element(elem[i], xs[i], k, buf, raw);
                const std::vector<int> &d = V.element_dofs(elem[i]);
                const int na = static_cast<int>(d.size());
                for (int kk = 0; kk < na; ++kk)
                {
                    rows[start[i] + kk] = static_cast<int>(i); cols[start[i] + kk] = d[kk]; vals[start[i] + kk] = buf[k * na + kk];
                }
            }
        }
    }
    return nb::make_tuple(to_np(std::move(rows)), to_np(std::move(cols)), to_np(std::move(vals)));
}

// AssembledMatrix -> COO triplets
static nb::tuple to_coo(const AssembledMatrix &K)
{
    std::vector<int> rows, cols; std::vector<double> vals;
    for (int i = 0; i < K.n; ++i)
        for (int off = -K.p; off <= K.p; ++off)
        {
            int j = i + off;
            if (j < 0 || j >= K.n) continue;
            double v = K.band[static_cast<std::size_t>(i) * K.w + (off + K.p)];
            if (v != 0.0) { rows.push_back(i); cols.push_back(j); vals.push_back(v); }
        }
    for (std::size_t k = 0; k < K.ov.size(); ++k) { rows.push_back(K.oi[k]); cols.push_back(K.oj[k]); vals.push_back(K.ov[k]); }
    return nb::make_tuple(to_np(std::move(rows)), to_np(std::move(cols)), to_np(std::move(vals)));
}

// ---- 2D mesh helpers -------------------------------------------------------
// Point and array<int,3> are standard-layout and tightly packed, so the C++
// vectors ARE the NumPy buffers: the arrays below are views that borrow the
// owning Mesh2D, with no copy and no conversion pass.
static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be two packed doubles");
static_assert(sizeof(std::array<int, 3>) == 3 * sizeof(int), "array<int,3> must be three packed ints");
static_assert(sizeof(std::array<int, 2>) == 2 * sizeof(int), "array<int,2> must be two packed ints");

using Arr2D = nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>, nb::c_contig>;
using ArrI3 = nb::ndarray<nb::numpy, const int,    nb::shape<-1, 3>, nb::c_contig>;
using ArrI2 = nb::ndarray<nb::numpy, const int,    nb::shape<-1, 2>, nb::c_contig>;
using ArrI1 = nb::ndarray<nb::numpy, const int,    nb::shape<-1>,    nb::c_contig>;

/// Points of a Domain ring, or any (n,2) array, as a vector<Point>.
static std::vector<Point> to_points(const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &a)
{
    std::vector<Point> p(a.shape(0));
    for (std::size_t i = 0; i < a.shape(0); ++i) p[i] = Point{a(i, 0), a(i, 1)};
    return p;
}

// ---- Krylov glue --------------------------------------------------------------
using CVec = nb::ndarray<std::complex<double>, nb::numpy, nb::shape<-1>, nb::c_contig>;
using CArr = nb::ndarray<const std::complex<double>, nb::shape<-1>, nb::c_contig>;
using KOp  = std::function<void(const std::vector<double> &, std::vector<double> &)>;

static CVec to_np(const std::vector<std::complex<double>> &v)
{
    auto *p = new std::complex<double>[v.size()];
    std::copy(v.begin(), v.end(), p);
    nb::capsule owner(p, [](void *q) noexcept { delete[] static_cast<std::complex<double> *>(q); });
    return CVec(p, {v.size()}, owner);
}

// A Python callable x -> y becomes a matvec / preconditioner; the Python side hands
// over callables that already return contiguous float64 arrays of the right length.
static KOp callable_op(nb::callable f, std::size_t n, const char *what)
{
    return [f, n, what](const std::vector<double> &x, std::vector<double> &y) {
        nb::object r = f(to_np(x));
        DArr a = nb::cast<DArr>(r);
        if (a.shape(0) != n) throw std::invalid_argument(std::string(what) + ": callable returned the wrong length");
        y.assign(a.data(), a.data() + a.shape(0));
    };
}

// An operator: an AssembledMatrix (kernel in C++) or a callable.
static KOp as_operator(nb::handle A, std::size_t n)
{
    if (nb::isinstance<AssembledMatrix>(A))
    {
        const AssembledMatrix *K = nb::cast<const AssembledMatrix *>(A);
        if (static_cast<std::size_t>(K->n) != n) throw std::invalid_argument("krylov: matrix and right-hand side differ in size");
        return krylov::matvec(*K);
    }
    if (nb::isinstance<CSRMatrix>(A))
    {
        const CSRMatrix *K = nb::cast<const CSRMatrix *>(A);
        if (static_cast<std::size_t>(K->nrows()) != n || K->ncols() != K->nrows())
            throw std::invalid_argument("krylov: matrix and right-hand side differ in size");
        return [K](const std::vector<double> &x, std::vector<double> &y) {
            y.resize(static_cast<std::size_t>(K->nrows()));
            if (K->nrows() > 0) K->apply(x.data(), y.data());
        };
    }
    return callable_op(nb::cast<nb::callable>(A), n, "krylov operator");
}

// A preconditioner: None (identity), a LinearSolver (factored in C++), or a callable.
static KOp as_precon(nb::handle M, std::size_t n)
{
    if (M.is_none()) return krylov::identity();
    if (nb::isinstance<LinearSolver>(M))
    {
        const LinearSolver *S = nb::cast<const LinearSolver *>(M);
        if (static_cast<std::size_t>(S->size()) != n) throw std::invalid_argument("krylov: preconditioner and right-hand side differ in size");
        return krylov::precon(*S);
    }
    if (nb::isinstance<SparsePreconditioner>(M))
    {
        const SparsePreconditioner *P = nb::cast<const SparsePreconditioner *>(M);
        if (static_cast<std::size_t>(P->size()) != n) throw std::invalid_argument("krylov: preconditioner and right-hand side differ in size");
        return [P](const std::vector<double> &r, std::vector<double> &z) { (*P)(r, z); };
    }
    return callable_op(nb::cast<nb::callable>(M), n, "krylov preconditioner");
}

void bind_2d(nb::module_ &m);     // femd_bindings_2d.cpp

NB_MODULE(_femd, m)
{
    m.doc() = "FEMd core bindings";
    m.attr("__version__") = FEMD_VERSION;
    // Spaces keep Python objects in their __dict__ (boundary data given as callables, caches), so
    // a script's globals and its spaces can form reference cycles that the interpreter breaks only
    // after nanobind has checked for leaks at exit.  Those objects are collected normally while the
    // program runs (tests/python/test_space_boundary_data.py checks it), so the exit-time report
    // would be a false alarm: switch it off.
    nb::set_leak_warnings(false);

    // ---- mesh -----------------------------------------------------------
    nb::class_<Mesh1D>(m, "Mesh1D")
        .def("__init__", [](Mesh1D *self, DArr v) { new (self) Mesh1D(to_vec(v)); }, "vertices"_a)
        .def_static("uniform", [](double a, double b, int n) { return Mesh1D::uniform(a, b, n); }, "a"_a, "b"_a, "nelem"_a)
        .def_prop_ro("nelem", &Mesh1D::nelem)
        .def_prop_ro("a", &Mesh1D::a).def_prop_ro("b", &Mesh1D::b)
        .def_prop_ro("is_uniform", [](const Mesh1D &m_) { return m_.is_uniform(); })
        .def("vertices", [](const Mesh1D &m_) { return to_np(m_.vertices()); })
        .def("h", [](const Mesh1D &m_, int e) { return m_.h(e); }, "e"_a)
        .def("element_of", &Mesh1D::element_of, "x"_a);


    // ---- 2D: predicates, domain, mesh, mesher --------------------------------
    m.def("orient2d", &pred::orient2d, "ax"_a, "ay"_a, "bx"_a, "by"_a, "cx"_a, "cy"_a,
          "Exact sign of the triangle area: +1 counter-clockwise, -1 clockwise, 0 collinear.");
    m.def("incircle", &pred::incircle, "ax"_a, "ay"_a, "bx"_a, "by"_a, "cx"_a, "cy"_a, "dx"_a, "dy"_a,
          "Exact sign of the in-circle test for a counter-clockwise triangle abc.");

    nb::class_<Domain>(m, "Domain")
        .def(nb::init<>())
        .def("boundary", [](Domain &d, const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &p,
                            std::vector<int> vm, int sm) -> Domain & {
                 auto pts = to_points(p);
                 if (vm.empty()) vm.assign(pts.size(), sm);
                 return d.boundary(std::move(pts), std::move(vm), sm);
             }, "points"_a, "vertex_markers"_a = std::vector<int>{}, "marker"_a = 1, nb::rv_policy::reference_internal)
        .def("hole", [](Domain &d, const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &p,
                        std::vector<int> vm, int sm) -> Domain & {
                 auto pts = to_points(p);
                 if (vm.empty()) vm.assign(pts.size(), sm);
                 return d.hole(std::move(pts), std::move(vm), sm);
             }, "points"_a, "vertex_markers"_a = std::vector<int>{}, "marker"_a = 2, nb::rv_policy::reference_internal)
        .def_prop_ro("nrings", &Domain::nrings)
        .def_prop_ro("nholes", &Domain::nholes)
        .def_prop_ro("nvertices", &Domain::nvertices)
        .def_prop_ro("smallest_input_angle", &Domain::smallest_input_angle)
        .def("contains", [](const Domain &d, double x, double y) { return d.contains(Point{x, y}); }, "x"_a, "y"_a)
        .def("bbox", [](const Domain &d) { Point lo, hi; d.bbox(lo, hi); return nb::make_tuple(lo.x, lo.y, hi.x, hi.y); });

    nb::class_<Mesh2D>(m, "Mesh2D")
        .def(nb::init<>())
        .def_prop_ro("npoints", &Mesh2D::npoints)
        .def_prop_ro("ntriangles", &Mesh2D::ntriangles)
        .def_prop_ro("nsegments", &Mesh2D::nsegments)
        .def_prop_ro("points", [](nb::object self) {
                 const Mesh2D &M = nb::cast<const Mesh2D &>(self);
                 return Arr2D(reinterpret_cast<const double *>(M.points().data()),
                              {M.points().size(), 2}, self);
             }, "(npoints, 2) float64 view of the vertex coordinates")
        .def_prop_ro("triangles", [](nb::object self) {
                 const Mesh2D &M = nb::cast<const Mesh2D &>(self);
                 return ArrI3(reinterpret_cast<const int *>(M.triangles().data()),
                              {M.triangles().size(), 3}, self);
             }, "(ntriangles, 3) int32 view; every triangle is counter-clockwise")
        .def_prop_ro("neighbours", [](nb::object self) {
                 const Mesh2D &M = nb::cast<const Mesh2D &>(self);
                 return ArrI3(reinterpret_cast<const int *>(M.neighbours().data()),
                              {M.neighbours().size(), 3}, self);
             }, "(ntriangles, 3) int32 view; [t,i] is the triangle across the edge opposite vertex i, or -1")
        .def_prop_ro("segments", [](nb::object self) {
                 const Mesh2D &M = nb::cast<const Mesh2D &>(self);
                 return ArrI2(reinterpret_cast<const int *>(M.segments().data()),
                              {M.segments().size(), 2}, self);
             }, "(nsegments, 2) int32 view of the constrained boundary edges")
        .def_prop_ro("point_markers", [](nb::object self) {
                 const Mesh2D &M = nb::cast<const Mesh2D &>(self);
                 return ArrI1(M.point_markers().data(), {M.point_markers().size()}, self);
             }, "(npoints,) int32 view; 0 for interior vertices")
        .def_prop_ro("segment_markers", [](nb::object self) {
                 const Mesh2D &M = nb::cast<const Mesh2D &>(self);
                 return ArrI1(M.segment_markers().data(), {M.segment_markers().size()}, self);
             }, "(nsegments,) int32 view")
        .def("areas", [](const Mesh2D &M) {
                 std::vector<double> a(M.tri_capacity());
                 for (int t = 0; t < M.tri_capacity(); ++t) a[t] = M.area(t);
                 return to_np(std::move(a));
             })
        .def("angles", [](const Mesh2D &M) {
                 std::vector<double> a(M.tri_capacity());
                 for (int t = 0; t < M.tri_capacity(); ++t) a[t] = M.min_angle(t);
                 return to_np(std::move(a));
             }, "smallest interior angle of each triangle, in degrees")
        .def("qualities", [](const Mesh2D &M) {
                 std::vector<double> a(M.tri_capacity());
                 for (int t = 0; t < M.tri_capacity(); ++t) a[t] = M.quality(t);
                 return to_np(std::move(a));
             }, "shape quality in [0,1]; 1 is equilateral")
        .def("boundary_edge_counts", [](const Mesh2D &M) {
                 std::vector<int> a(M.tri_capacity());
                 for (int t = 0; t < M.tri_capacity(); ++t) a[t] = M.boundary_edge_count(t);
                 return to_np(std::move(a));
             }, "how many of each triangle's three edges are boundary segments (0..3)")
        .def("has_interior_vertex", [](const Mesh2D &M) {
                 std::vector<int> a(M.tri_capacity());
                 for (int t = 0; t < M.tri_capacity(); ++t) a[t] = M.has_interior_vertex(t) ? 1 : 0;
                 return to_np(std::move(a));
             }, "1 where the triangle has at least one vertex off the boundary")
        .def("area", &Mesh2D::area, "t"_a)
        .def("boundary_edge_count", &Mesh2D::boundary_edge_count, "t"_a)
        .def("vertex_on_boundary", &Mesh2D::vertex_on_boundary, "v"_a)
        .def("quality", &Mesh2D::quality, "t"_a)
        .def("min_angle", &Mesh2D::min_angle, "t"_a)
        .def("validate", &Mesh2D::validate, "empty string when the mesh satisfies its invariants");

    // Point location.  A locator borrows its mesh (kept alive by nanobind) and caches the
    // bucket grid, so the Python Mesh2D keeps one and reuses it: Python meshes are never
    // changed in place, every operation returns a new one.
    nb::class_<PointLocator>(m, "PointLocator")
        .def(nb::init<const Mesh2D &>(), "mesh"_a, nb::keep_alive<1, 2>())
        .def("locate", [](PointLocator &L, const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &p,
                          int ntriangles) {
                 const std::size_t n = p.shape(0);
                 const double *d = p.data();
                 std::vector<int> out(n, -1);
                 {
                     GilRelease release;
                     // a batch comparable in size to the mesh pays for the grid up front
                     if (n >= 1000 && 8 * n >= static_cast<std::size_t>(ntriangles))
                     {
                         L.prepare();                 // then answered from the grid alone, in parallel
                         const long nl = static_cast<long>(n);
                         FEMD_OMP_FOR_IF(nl > FEMD_OMP_THRESHOLD)
                         for (long i = 0; i < nl; ++i) out[i] = L.find(Point{d[2 * i], d[2 * i + 1]});
                     }
                     else
                         for (std::size_t i = 0; i < n; ++i) out[i] = L.locate(Point{d[2 * i], d[2 * i + 1]});
                 }
                 return to_np(std::move(out));
             }, "points"_a, "ntriangles"_a, "Triangle containing each point, -1 outside the mesh.")
        .def_prop_ro("grid_built", &PointLocator::grid_built);

    // Smoothing and Delaunay restoration, standalone.  Same copy-on-return rule
    // as the refinement functions below, for the same reason.
    m.def("smooth", [](const Mesh2D &src, int passes, bool restore) {
              Mesh2D out = src;
              int n = laplacian_smooth(out, passes, restore);
              return nb::make_tuple(std::move(out), n);
          }, "mesh"_a, "passes"_a = 1, "restore_delaunay"_a = true);

    m.def("restore_delaunay", [](const Mesh2D &src, int max_sweeps) {
              Mesh2D out = src;
              int n = restore_delaunay(out, max_sweeps);
              return nb::make_tuple(std::move(out), n);
          }, "mesh"_a, "max_sweeps"_a = 64);

    // Meshes from arrays, and structured meshes.  All return (mesh, report).
    m.def("mesh_from_arrays", [](const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &p,
                                 const nb::ndarray<const int, nb::shape<-1, 3>, nb::c_contig> &t,
                                 const nb::ndarray<const int, nb::shape<-1, 2>, nb::c_contig> &s,
                                 std::vector<int> seg_markers, std::vector<int> point_markers, int marker) {
              auto pts = to_points(p);
              std::vector<std::array<int, 3>> tris(t.shape(0));
              for (std::size_t k = 0; k < t.shape(0); ++k) tris[k] = {t(k, 0), t(k, 1), t(k, 2)};
              std::vector<std::array<int, 2>> segs(s.shape(0));
              for (std::size_t k = 0; k < s.shape(0); ++k) segs[k] = {s(k, 0), s(k, 1)};
              FromArraysReport r;
              Mesh2D mesh = mesh_from_arrays(pts, std::move(tris), segs, seg_markers, point_markers, marker, &r);
              nb::dict rep;
              rep["flipped"] = r.flipped;
              rep["boundary_edges"] = r.boundary_edges;
              rep["boundary_loops"] = r.boundary_loops;
              return nb::make_tuple(std::move(mesh), rep);
          }, "points"_a, "triangles"_a, "segments"_a, "segment_markers"_a, "point_markers"_a, "marker"_a = 1);

    m.def("rectangle_mesh", [](double x0, double y0, double x1, double y1, int nx, int ny,
                               const std::string &diag, std::vector<int> mk) {
              if (mk.size() != 4) throw std::invalid_argument("rectangle_mesh: need four markers");
              Diagonal d = diagonal_from_name(diag);
              Mesh2D mesh = rectangle_mesh(x0, y0, x1, y1, nx, ny, d, {mk[0], mk[1], mk[2], mk[3]});
              nb::dict rep;
              rep["structured"] = nb::make_tuple(nx, ny, diagonal_name(d));
              return nb::make_tuple(std::move(mesh), rep);
          }, "x0"_a, "y0"_a, "x1"_a, "y1"_a, "nx"_a, "ny"_a, "diagonal"_a = "right",
             "markers"_a = std::vector<int>{1, 2, 3, 4});

    m.def("mapped_mesh", [](const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &b,
                            const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &r,
                            const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &t,
                            const nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig> &l,
                            const std::string &diag, std::vector<int> mk) {
              if (mk.size() != 4) throw std::invalid_argument("mapped_mesh: need four markers");
              Diagonal d = diagonal_from_name(diag);
              Mesh2D mesh = mapped_mesh(to_points(b), to_points(r), to_points(t), to_points(l), d,
                                        {mk[0], mk[1], mk[2], mk[3]});
              nb::dict rep;
              rep["structured"] = nb::make_tuple(static_cast<int>(b.shape(0)) - 1, static_cast<int>(l.shape(0)) - 1,
                                                 diagonal_name(d));
              return nb::make_tuple(std::move(mesh), rep);
          }, "bottom"_a, "right"_a, "top"_a, "left"_a, "diagonal"_a = "right",
             "markers"_a = std::vector<int>{1, 2, 3, 4});

    // Vertex removal.  Same copy-on-return rule.  The removed points are compacted
    // away, so the old-to-new vertex map is returned with the mesh.
    m.def("remove_vertices", [](const Mesh2D &src, std::vector<int> vertices, double min_angle,
                                double max_edge, double max_area, bool boundary) {
              Mesh2D out = src;
              RemoveOptions o;
              o.min_angle = min_angle; o.max_edge = max_edge; o.max_area = max_area; o.boundary = boundary;
              RemoveReport r;
              remove_vertices(out, vertices, o, &r);
              std::vector<int> pmap = out.compact(/*points_too=*/true);
              return nb::make_tuple(std::move(out), r.removed, pmap, r.kept, r.reasons, r.passes);
          }, "mesh"_a, "vertices"_a, "min_angle"_a = 0.0, "max_edge"_a = 0.0, "max_area"_a = 0.0,
             "boundary"_a = true);

    // Local refinement.  The C++ side refines in place; these return a refined
    // COPY, because the Python mesh arrays are views borrowed from the mesh
    // that owns them and refining in place would leave any array the caller is
    // still holding pointing at freed memory.
    m.def("split_triangles", [](const Mesh2D &src, std::vector<int> marked) {
              Mesh2D out = src;
              int n = split_triangles(out, marked);
              out.compact(/*points_too=*/false);
              return nb::make_tuple(std::move(out), n);
          }, "mesh"_a, "triangles"_a);

    m.def("bisect_triangles", [](const Mesh2D &src, std::vector<int> marked, int levels) {
              Mesh2D out = src;
              int n = bisect_triangles(out, marked, levels);
              out.compact(false);
              return nb::make_tuple(std::move(out), n);
          }, "mesh"_a, "triangles"_a, "levels"_a = 1);

    m.def("refine_triangles", [](const Mesh2D &src, std::vector<int> marked, double min_angle,
                                 double max_edge, double max_area, int max_points, double input_angle) {
              Mesh2D out = src;
              RefineOptions ro;
              ro.min_angle = min_angle; ro.max_edge = max_edge; ro.max_area = max_area;
              ro.max_points = max_points; ro.input_angle = input_angle;
              Point lo{1e300, 1e300}, hi{-1e300, -1e300};
              for (const Point &p : out.points()) {
                  lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y);
                  hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y);
              }
              ro.min_feature = 1e-9 * std::sqrt((hi.x-lo.x)*(hi.x-lo.x) + (hi.y-lo.y)*(hi.y-lo.y));
              int n = refine_triangles(out, marked, ro);
              out.compact(false);
              return nb::make_tuple(std::move(out), n);
          }, "mesh"_a, "triangles"_a, "min_angle"_a = 20.0, "max_edge"_a = 0.0, "max_area"_a = 0.0,
             "max_points"_a = 500000, "input_angle"_a = 180.0);

    m.def("triangulate", [](const Domain &d, double min_angle, double max_edge, double max_area,
                            int smooth_passes, int max_points, bool check, bool interior_vertex) {
              MeshOptions o;
              o.min_angle = min_angle; o.max_edge = max_edge; o.max_area = max_area;
              o.smooth_passes = smooth_passes; o.max_points = max_points; o.check = check;
              o.interior_vertex = interior_vertex;
              MeshReport r;
              Mesh2D mesh = triangulate(d, o, &r);
              nb::dict rep;
              rep["input_vertices"]  = r.input_vertices;
              rep["boundary_splits"] = r.boundary_splits;
              rep["refine_inserts"]  = r.refine_inserts;
              rep["smoothed"]        = r.smoothed;
              rep["min_angle"]       = r.min_angle;
              rep["min_quality"]     = r.min_quality;
              rep["max_area"]        = r.max_area;
              rep["total_area"]      = r.total_area;
              rep["corner_flips"]    = r.corner_flips;
              rep["corner_splits"]   = r.corner_splits;
              rep["corner_centres"]  = r.corner_centres;
              rep["boundary_only"]   = r.boundary_only;
              rep["two_boundary_edges"] = r.two_boundary_edges;
              return nb::make_tuple(std::move(mesh), rep);
          }, "domain"_a, "min_angle"_a = 20.0, "max_edge"_a = 0.0, "max_area"_a = 0.0,
             "smooth_passes"_a = 0, "max_points"_a = 500000, "check"_a = false,
             "interior_vertex"_a = false);

    // ---- boundary conditions ------------------------------------------------
    nb::class_<BoundaryCondition>(m, "BoundaryCondition")
        .def_static("free", &BoundaryCondition::free)
        .def_static("dirichlet", &BoundaryCondition::dirichlet)
        .def_static("neumann", &BoundaryCondition::neumann)
        .def_static("clamped", &BoundaryCondition::clamped)
        .def_static("robin", &BoundaryCondition::robin, "alpha"_a, "beta"_a)
        .def_static("derivative", &BoundaryCondition::derivative, "m"_a)
        .def_static("custom", [](std::vector<std::vector<std::pair<int, double>>> f) { return BoundaryCondition::custom(std::move(f)); }, "functionals"_a)
        .def_static("from_name", &BoundaryCondition::from_name, "name"_a)
        .def_prop_ro("name", [](const BoundaryCondition &b) { return b.name; })
        .def_prop_ro("count", &BoundaryCondition::count)
        .def_prop_ro("max_order", &BoundaryCondition::max_order)
        .def_prop_ro("functionals", [](const BoundaryCondition &b) { return b.functionals; });

    nb::class_<BCSpec>(m, "BCSpec")
        .def(nb::init<>())
        .def(nb::init<BoundaryCondition, BoundaryCondition>(), "left"_a, "right"_a)
        .def_static("periodic", &BCSpec::make_periodic)
        .def_static("from_name", &BCSpec::from_name, "name"_a)
        .def_prop_ro("is_periodic", [](const BCSpec &s) { return s.periodic; })
        .def_prop_ro("left", [](const BCSpec &s) { return s.left; })
        .def_prop_ro("right", [](const BCSpec &s) { return s.right; })
        .def("describe", &BCSpec::describe);

    // ---- spaces ---------------------------------------------------------------
    nb::class_<FunctionSpace>(m, "FunctionSpace")
        .def_prop_ro("dim", &FunctionSpace::dim)
        .def_prop_ro("degree", &FunctionSpace::degree)
        .def_prop_ro("uband", &FunctionSpace::uband)
        .def_prop_ro("nelem", &FunctionSpace::nelem)
        .def_prop_ro("raw_dim", &FunctionSpace::raw_dim)
        .def_prop_ro("nloc_max", &FunctionSpace::nloc_max)
        .def_prop_ro("mesh", &FunctionSpace::mesh, nb::rv_policy::reference_internal)
        .def_prop_ro("bc", &FunctionSpace::bc, nb::rv_policy::reference_internal)
        .def_prop_ro("n_constraints", &FunctionSpace::n_constraints)
        .def("dof_coordinates", [](const FunctionSpace &V) { return to_np(V.dof_coordinates()); })
        .def("element_dofs", [](const FunctionSpace &V, int e) { return to_np(V.element_dofs(e)); }, "e"_a)
        .def("lift", [](const FunctionSpace &V, DArr g) { return to_np(V.lift(to_vec(g))); }, "g"_a)
        .def("prolongate", [](const FunctionSpace &V, DArr c) { return to_np(V.constraints().prolongate(to_vec(c))); }, "coeffs"_a)
        .def("restrict", [](const FunctionSpace &V, DArr r) { return to_np(V.constraints().restrict_vector(to_vec(r))); }, "raw"_a)
        .def("evaluate", &evaluate, "coeffs"_a, "x"_a, "k"_a = 0)
        .def("at_quad", &at_quad, "cache"_a, "coeffs"_a, "k"_a = 0)
        .def("basis_at", &basis_at, "x"_a, "k"_a = 0);

    nb::class_<SplineSpace, FunctionSpace>(m, "SplineSpace")
        .def(nb::init<const Mesh1D &, int, const BCSpec &, int, const std::map<int, int> &>(),
             "mesh"_a, "degree"_a, "bc"_a = BCSpec::free(), "continuity"_a = -1, "continuity_at"_a = std::map<int, int>{})
        .def("greville", [](const SplineSpace &V) { return to_np(V.greville()); })
        .def("knots", [](const SplineSpace &V) { return to_np(V.knots().knots()); });

    nb::class_<LagrangeSpace, FunctionSpace>(m, "LagrangeSpace")
        .def(nb::init<const Mesh1D &, int, const BCSpec &, bool>(), "mesh"_a, "degree"_a, "bc"_a = BCSpec::free(), "equispaced"_a = true)
        .def_prop_ro("equispaced", &LagrangeSpace::equispaced)
        .def("nodes", [](const LagrangeSpace &V) { return to_np(V.nodes()); });

    // ---- quadrature cache -------------------------------------------------------
    nb::enum_<DGSpace::Basis>(m, "DGBasis")
        .value("Legendre", DGSpace::Basis::Legendre).value("Lobatto", DGSpace::Basis::Lobatto);

    nb::class_<DGSpace, FunctionSpace>(m, "DGSpace")
        .def(nb::init<const Mesh1D &, int, bool, DGSpace::Basis>(), "mesh"_a, "degree"_a, "periodic"_a = false,
             "basis"_a = DGSpace::Basis::Legendre)
        .def_prop_ro("basis", &DGSpace::basis)
        .def("reference_points", [](const DGSpace &V) { return to_np(V.reference_points()); })
        .def("interpolate_values", [](const DGSpace &V, DArr v) { return to_np(V.interpolate_values(to_vec(v))); }, "values"_a);

    nb::class_<FacetCache>(m, "FacetCache")
        .def(nb::init<const FunctionSpace &, int>(), "space"_a, "nder"_a, nb::call_guard<GilRelease>())
        .def_prop_ro("nf", &FacetCache::nf)
        .def_prop_ro("nder", &FacetCache::nder)
        .def("points", [](const FacetCache &F, int s) {
                 std::vector<double> x(F.nf());
                 for (int f = 0; f < F.nf(); ++f) x[f] = F.x(f, s);
                 return to_np(std::move(x));
             }, "side"_a = 0)
        .def("elements", [](const FacetCache &F, int s) {
                 std::vector<int> e(F.nf());
                 for (int f = 0; f < F.nf(); ++f) e[f] = F.element(f, s);
                 return to_np(std::move(e));
             }, "side"_a);

    // the cache reads its space, so it keeps it alive (keep_alive); dynamic_attr lets the Python side
    // record which space made it (Q._space) without a module-level table that would pin both forever
    nb::class_<QuadratureCache>(m, "QuadratureCache", nb::dynamic_attr())
        .def(nb::init<const FunctionSpace &, int, int, bool>(), "space"_a, "npts"_a, "nder"_a, "lobatto"_a = false,
             nb::call_guard<GilRelease>(), nb::keep_alive<1, 2>())
        .def_prop_ro("lobatto", &QuadratureCache::lobatto)
        .def_prop_ro("nelem", &QuadratureCache::nelem)
        .def_prop_ro("nq", &QuadratureCache::nq)
        .def_prop_ro("nder", &QuadratureCache::nder)
        .def("nodes", [](const QuadratureCache &Q) { return to_np(Q.nodes()); })
        .def("weights", [](const QuadratureCache &Q) { return to_np(Q.weights()); });

    // ---- assembled matrix + kernels ------------------------------------------------
    nb::class_<AssembledMatrix>(m, "AssembledMatrix")
        .def(nb::init<int, int>(), "n"_a, "uband"_a)
        .def_prop_ro("n", [](const AssembledMatrix &K) { return K.n; })
        .def_prop_ro("uband", [](const AssembledMatrix &K) { return K.p; })
        .def_prop_rw("symmetric", [](const AssembledMatrix &K) { return K.symmetric; }, [](AssembledMatrix &K, bool s) { K.symmetric = s; })
        .def("clear", &AssembledMatrix::clear)
        .def("add_entry", [](AssembledMatrix &K, int i, int j, double v) { K.add(i, j, v); }, "i"_a, "j"_a, "v"_a)
        .def("to_coo", &to_coo)
        .def("matvec",
             [](const AssembledMatrix &K, DArr x)
             {
                 if (static_cast<int>(x.shape(0)) != K.n)
                     throw std::invalid_argument("matvec: expected a vector of length " + std::to_string(K.n) +
                                                 ", got " + std::to_string(x.shape(0)));
                 // write straight into the NumPy buffer: no vector, no second copy
                 double *out = new double[static_cast<std::size_t>(K.n) > 0 ? K.n : 1];
                 nb::capsule owner(out, [](void *q) noexcept { delete[] static_cast<double *>(q); });
                 if (K.n > 0) { GilRelease release; K.apply(x.data(), out); }
                 return Vec(out, {static_cast<std::size_t>(K.n)}, owner);
             },
             "x"_a, "y = K x straight off the banded store, O(n(2p+1)); no sparse conversion.")
        .def("matvec_into",
             [](const AssembledMatrix &K, DArr x, nb::ndarray<double, nb::shape<-1>, nb::c_contig> y)
             {
                 if (static_cast<int>(x.shape(0)) != K.n || static_cast<int>(y.shape(0)) != K.n)
                     throw std::invalid_argument("matvec(x, out): x and out must both have length " + std::to_string(K.n));
                 const double *xb = x.data(), *xe = xb + K.n;
                 const double *yb = y.data(), *ye = yb + K.n;
                 if (K.n > 0 && xb < ye && yb < xe)      // the kernel reads x[i +- p] after writing y[i]
                     throw std::invalid_argument("matvec(x, out): out must not overlap x");
                 if (K.n > 0) { GilRelease release; K.apply(x.data(), y.data()); }
             },
             "x"_a, "out"_a, "out[:] = K x, no allocation.")
        .def("combine", [](const AssembledMatrix &A, const AssembledMatrix &B, double alpha, double beta)
             { return combine(A, B, alpha, beta); }, "B"_a, "alpha"_a, "beta"_a, "alpha*self + beta*B.",
             nb::call_guard<GilRelease>())
        .def("scaled", [](const AssembledMatrix &A, double a) { return scaled(A, a); }, "a"_a, "a*self.")
        .def("transposed", [](const AssembledMatrix &A) { return transposed(A); }, "self^T.")
        .def("multiply", [](const AssembledMatrix &A, const AssembledMatrix &B) { return multiply(A, B); },
             "B"_a, "self @ B, half-bandwidth p_self + p_B.", nb::call_guard<GilRelease>())
        .def("axpy", [](AssembledMatrix &A, const AssembledMatrix &B, double beta) { axpy(A, B, beta); },
             "B"_a, "beta"_a, "self += beta*B in place; requires self.uband >= B.uband.")
        .def("copy", [](const AssembledMatrix &A) { return scaled(A, 1.0); }, "An independent copy.")
        .def("add_matrix", [](AssembledMatrix &K, const QuadratureCache &Q, int a, int b, nb::handle c)
             { Coefficient cc = make_coeff(c); GilRelease release; assemble_matrix(Q, a, b, cc, K); },
             "cache"_a, "a"_a, "b"_a, "coeff"_a)
        .def("add_block", [](AssembledMatrix &K, const ProductSpace &P, int I, int J, const QuadratureCache &QI, const QuadratureCache &QJ, int a, int b, nb::handle c)
             { Coefficient cc = make_coeff(c); GilRelease release; assemble_matrix(P, I, J, QI, QJ, a, b, cc, K); },
             "product"_a, "I"_a, "J"_a, "cache_I"_a, "cache_J"_a, "a"_a, "b"_a, "coeff"_a)
        .def("add_coo", [](AssembledMatrix &K, const ProductSpace &P, int I, int J, IArr r, IArr c, DArr v)
             {
                 std::size_t m = r.shape(0);
                 if (c.shape(0) != m || v.shape(0) != m) throw std::invalid_argument("add_coo: rows, cols and vals differ in length");
                 const int *rp = r.data(), *cp = c.data(); const double *vp = v.data();
                 for (std::size_t k = 0; k < m; ++k) K.add(P.global(I, rp[k]), P.global(J, cp[k]), vp[k]);
             },
             "product"_a, "I"_a, "J"_a, "rows"_a, "cols"_a, "vals"_a,
             "Scatter block (I, J), given as field-local triplets, through the product numbering.")
        .def("add_scaled_block", [](AssembledMatrix &K, const ProductSpace &P, int I, int J, const AssembledMatrix &B, double alpha)
             {
                 GilRelease release;
                 for (int i = 0; i < B.n; ++i)
                 {
                     const int gi = P.global(I, i);
                     const double *row = &B.band[static_cast<std::size_t>(i) * B.w];
                     for (int off = -B.p; off <= B.p; ++off)
                     {
                         const int j = i + off;
                         if (j < 0 || j >= B.n || row[off + B.p] == 0.0) continue;
                         K.add(gi, P.global(J, j), alpha * row[off + B.p]);
                     }
                 }
                 for (std::size_t k = 0; k < B.ov.size(); ++k) K.add(P.global(I, B.oi[k]), P.global(J, B.oj[k]), alpha * B.ov[k]);
             },
             "product"_a, "I"_a, "J"_a, "block"_a, "alpha"_a = 1.0,
             "Scatter a whole field matrix, times alpha, into block (I, J) through the product numbering.");
    m.def("krylov_solve",
          [](nb::handle A, nb::handle M, DArr b, DArr x0, bool lgmres, int restart, int max_iter, double tol, int k_aug)
          {
              std::size_t n = b.shape(0);
              if (x0.shape(0) != n) throw std::invalid_argument("krylov: x0 and b differ in length");
              // held by the lambdas below; A and M stay alive through the Python call frame
              KOp Aop = as_operator(A, n), Mop = as_precon(M, n);
              std::vector<double> x = to_vec(x0), bv = to_vec(b);
              krylov::Result r = krylov::solve(Aop, Mop, x, bv, lgmres ? krylov::Method::LGMRES : krylov::Method::GMRES,
                                               restart, max_iter, tol, k_aug);
              return nb::make_tuple(to_np(std::move(x)), r.converged, r.iterations, r.residual, r.precond_residual);
          },
          "A"_a, "M"_a.none(), "b"_a, "x0"_a, "lgmres"_a, "restart"_a, "max_iter"_a, "tol"_a, "k_aug"_a,
          "Preconditioned GMRES / LGMRES from the vendored csnewton, A an AssembledMatrix or callable, "
          "M None, a LinearSolver or a callable.  Returns (x, converged, iterations, residual, precond_residual).");
    m.def("cg_solve",
          [](nb::handle A, nb::handle M, DArr b, DArr x0, int max_iter, double tol)
          {
              std::size_t n = b.shape(0);
              if (x0.shape(0) != n) throw std::invalid_argument("cg: x0 and b differ in length");
              KOp Aop = as_operator(A, n), Mop = as_precon(M, n);
              std::vector<double> x = to_vec(x0), bv = to_vec(b);
              krylov::CGResult r;
              {
                  // callables re-enter Python, so the GIL is kept unless both are native
                  bool native = !nb::isinstance<nb::callable>(A) && (M.is_none() || !nb::isinstance<nb::callable>(M));
                  if (native) { GilRelease release; r = krylov::cg(Aop, Mop, x, bv, max_iter, tol); }
                  else r = krylov::cg(Aop, Mop, x, bv, max_iter, tol);
              }
              return nb::make_tuple(to_np(std::move(x)), r.converged, r.iterations, r.residual, r.breakdown);
          },
          "A"_a, "M"_a.none(), "b"_a, "x0"_a, "max_iter"_a, "tol"_a,
          "Preconditioned conjugate gradients.  A an AssembledMatrix, CSRMatrix or callable; M None, a LinearSolver, "
          "a SparsePreconditioner or a callable.  Returns (x, converged, iterations, residual, breakdown).");
    // ---- explicit Runge-Kutta ------------------------------------------------------
    m.def("explicit_tableau", [](const std::string &name) {
              ExplicitTableau T = ExplicitTableau::named(name);
              return nb::make_tuple(to_np(T.A), to_np(T.b), to_np(T.c), T.order, T.name);
          }, "name"_a, "(A flattened row-major, b, c, order, canonical name) of a named explicit method.");
    nb::class_<ExplicitRK>(m, "ExplicitRK", "Explicit Runge-Kutta stepper for M u' = f(t, u) (timestep/explicit_rk.hpp).")
        .def("__init__", [](ExplicitRK *self, const std::string &name) { new (self) ExplicitRK(name); }, "method"_a)
        .def("__init__", [](ExplicitRK *self, DArr A, DArr b, DArr c, int order) {
                 ExplicitTableau T;
                 T.s = static_cast<int>(b.shape(0)); T.order = order; T.name = "tableau";
                 T.A = to_vec(A); T.b = to_vec(b); T.c = to_vec(c);
                 new (self) ExplicitRK(T);
             }, "A"_a, "b"_a, "c"_a, "order"_a = 0)
        .def_prop_ro("stages", [](const ExplicitRK &rk) { return rk.tableau().s; })
        .def_prop_ro("order", [](const ExplicitRK &rk) { return rk.tableau().order; })
        .def_prop_ro("name", [](const ExplicitRK &rk) { return rk.tableau().name; })
        .def_prop_ro("c", [](const ExplicitRK &rk) { return to_np(rk.tableau().c); })
        .def("step", [](ExplicitRK &rk, Vec u, double t, double dt, nb::callable f, nb::handle Minv) {
                 const std::size_t n = u.shape(0);
                 KOp Mop = as_precon(Minv, n);            // None: M = I; a LinearSolver in C++; or a callable
                 std::vector<double> uv(u.data(), u.data() + n), fb;
                 auto rate = [&](double tt, const std::vector<double> &U, std::vector<double> &k) {
                     nb::object r = f(tt, to_np(U));
                     DArr a = nb::cast<DArr>(r);
                     if (a.shape(0) != n) throw std::invalid_argument("ExplicitRK: f(t, U) returned the wrong length");
                     fb.assign(a.data(), a.data() + n);
                     Mop(fb, k);
                 };
                 rk.step(uv, t, dt, rate);
                 std::copy(uv.begin(), uv.end(), u.data());
             }, "u"_a, "t"_a, "dt"_a, "f"_a, "Minv"_a.none(),
             "One step in place: u <- u + dt sum b_i k_i, k_i = M^{-1} f(t + c_i dt, U_i).  Minv: None (M = I), a "
             "LinearSolver (the factored mass matrix, applied in C++) or a callable r -> M^{-1} r.");
    m.def("csnewton",
          [](nb::callable F, nb::handle M, DArr x0, int max_newton, double tol, double h, int restart,
             int gmres_max_iter, double gmres_tol, bool lgmres, int k_aug)
          {
              std::size_t n = x0.shape(0);
              auto Fgen = [F, n](const auto &z) {
                  using T = typename std::decay_t<decltype(z)>::value_type;
                  nb::object r = F(to_np(z));
                  if constexpr (std::is_same_v<T, double>)
                  {
                      DArr a = nb::cast<DArr>(r);
                      if (a.shape(0) != n) throw std::invalid_argument("csnewton: F returned the wrong length");
                      return std::vector<double>(a.data(), a.data() + n);
                  }
                  else
                  {
                      CArr a = nb::cast<CArr>(r);
                      if (a.shape(0) != n) throw std::invalid_argument("csnewton: F returned the wrong length");
                      return std::vector<std::complex<double>>(a.data(), a.data() + n);
                  }
              };
              KOp Mop = as_precon(M, n);
              std::vector<double> x = to_vec(x0);
              krylov::SerialBelowThreshold serial(n);
              csnewton::NewtonResult r = csnewton::csnewton(Fgen, Mop, x, max_newton, tol, h, restart,
                                                            gmres_max_iter, gmres_tol, lgmres, k_aug);
              return nb::make_tuple(to_np(std::move(x)), r.converged, r.newton_iters, r.gmres_iters, r.residual);
          },
          "F"_a, "M"_a.none(), "x0"_a, "max_newton"_a, "tol"_a, "h"_a, "restart"_a, "gmres_max_iter"_a,
          "gmres_tol"_a, "lgmres"_a, "k_aug"_a,
          "The vendored complex-step Newton-Krylov solver on a Python F that accepts real and complex arrays.");
    m.def("assemble_rect", [](const QuadratureCache &QT, const QuadratureCache &QS, int a, int b, nb::handle c)
          {
              std::vector<int> rows, cols; std::vector<double> vals;
              Coefficient cc = make_coeff(c);
              { GilRelease release; assemble_rect(QT, QS, a, b, cc, rows, cols, vals); }
              return nb::make_tuple(to_np(std::move(rows)), to_np(std::move(cols)), to_np(std::move(vals)));
          },
          "test_cache"_a, "trial_cache"_a, "a"_a, "b"_a, "coeff"_a,
          "Triplets of integral c (D^a T_i)(D^b S_j), test cache rows, trial cache columns, same mesh.");

    m.def("facet_matrix", [](AssembledMatrix &K, const FacetCache &FT, const FacetCache &FS, int st, int ss, int a, int b, DArr c) {
              std::vector<double> cv = to_vec(c);
              if (static_cast<int>(cv.size()) != FT.nf()) throw std::invalid_argument("facet_matrix: one coefficient per facet");
              GilRelease release;
              assemble_facet_matrix(K, FT, FS, st, ss, a, b, cv.data());
          }, "K"_a, "FT"_a, "FS"_a, "st"_a, "ss"_a, "a"_a, "b"_a, "c"_a);
    m.def("facet_triplets", [](const FacetCache &FT, const FacetCache &FS, int st, int ss, int a, int b, DArr c) {
              std::vector<double> cv = to_vec(c);
              if (static_cast<int>(cv.size()) != FT.nf()) throw std::invalid_argument("facet_triplets: one coefficient per facet");
              std::vector<int> r, cc; std::vector<double> v;
              facet_triplets(FT, FS, st, ss, a, b, cv.data(), r, cc, v);
              return nb::make_tuple(to_np(std::move(r)), to_np(std::move(cc)), to_np(std::move(v)));
          }, "FT"_a, "FS"_a, "st"_a, "ss"_a, "a"_a, "b"_a, "c"_a);
    m.def("facet_vector", [](const FacetCache &F, int s, int a, DArr c, int n) {
              std::vector<double> cv = to_vec(c);
              if (static_cast<int>(cv.size()) != F.nf()) throw std::invalid_argument("facet_vector: one coefficient per facet");
              return to_np(assemble_facet_vector(F, s, a, cv.data(), n));
          }, "F"_a, "side"_a, "a"_a, "c"_a, "n"_a);
    m.def("facet_values", [](const FacetCache &F, DArr coeffs, int s, int mder) {
              return to_np(facet_values(F, to_vec(coeffs), s, mder));
          }, "F"_a, "coeffs"_a, "side"_a, "m"_a);
    m.def("assemble_vector", [](const QuadratureCache &Q, int a, nb::handle c, int n)
          { std::vector<double> f(n, 0.0); Coefficient cc = make_coeff(c);
            { GilRelease release; assemble_vector(Q, a, cc, f); } return to_np(std::move(f)); },
          "cache"_a, "a"_a, "coeff"_a, "n"_a);
    m.def("assemble_block_vector", [](const ProductSpace &P, int I, const QuadratureCache &QI, int a, nb::handle c)
          { std::vector<double> f(P.dim(), 0.0); Coefficient cc = make_coeff(c);
            { GilRelease release; assemble_vector(P, I, QI, a, cc, f); } return to_np(std::move(f)); },
          "product"_a, "I"_a, "cache_I"_a, "a"_a, "coeff"_a);
    m.def("assemble_scalar", [](const QuadratureCache &Q, nb::handle c)
          { Coefficient cc = make_coeff(c); GilRelease release; return assemble_scalar(Q, cc); }, "cache"_a, "coeff"_a);

    // ---- product space ----------------------------------------------------------------
    nb::class_<ProductSpace>(m, "ProductSpace")
        .def("__init__", [](ProductSpace *self, std::vector<const FunctionSpace *> fields, bool position)
             { new (self) ProductSpace(std::move(fields), position ? ProductSpace::Ordering::Position : ProductSpace::Ordering::FieldMajor); },
             "fields"_a, "position"_a = true, nb::keep_alive<1, 2>())
        .def_prop_ro("dim", &ProductSpace::dim)
        .def_prop_ro("uband", &ProductSpace::uband)
        .def_prop_ro("nfields", &ProductSpace::nfields)
        .def_prop_ro("periodic", &ProductSpace::periodic)
        .def("global_index", &ProductSpace::global, "field"_a, "j"_a)
        .def("local_index", &ProductSpace::local, "g"_a)
        .def("split", [](const ProductSpace &P, DArr x) { auto parts = P.split(to_vec(x)); std::vector<Vec> out; for (auto &p : parts) out.push_back(to_np(std::move(p))); return out; }, "x"_a)
        .def("gather", [](const ProductSpace &P, std::vector<DArr> parts) { std::vector<std::vector<double>> v; for (auto &p : parts) v.push_back(to_vec(p)); return to_np(P.gather(v)); }, "parts"_a);

    // ---- solver ---------------------------------------------------------------------------
    nb::enum_<Solver>(m, "Solver")
        .value("Auto", Solver::Auto).value("Dense", Solver::Dense).value("Band", Solver::Band)
        .value("SymBand", Solver::SymBand).value("Cyclic", Solver::Cyclic).value("SymCyclic", Solver::SymCyclic)
        .value("Circulant", Solver::Circulant).value("Diagonal", Solver::Diagonal);
    m.def("has_fftw", &has_fftw, "Whether this build links FFTW (needed for Solver.Circulant).");
    m.def("has_openmp", []() {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }, "Whether this build was compiled with OpenMP.");
    m.def("set_num_threads", [](int k) {
        if (k < 1) throw std::invalid_argument("set_num_threads: need at least one thread");
#ifdef _OPENMP
        omp_set_num_threads(k);
#else
        (void) k;
#endif
    }, "k"_a, "Threads for the OpenMP loops (a no-op without OpenMP).");
    m.def("get_num_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Threads the OpenMP loops will use (1 without OpenMP).");
    m.def("omp_threshold", []() { return static_cast<long>(FEMD_OMP_THRESHOLD); },
          "Loops go parallel above this size (elements x quadrature points, points, or rows).");
    m.def("circulant_defect", [](const AssembledMatrix &K) { return detail::circulant_defect(K); }, "K"_a,
          "Relative deviation of K from the circulant generated by its first column (0 means exactly circulant).");

    nb::class_<detail::Structure>(m, "Structure", "Measured structure of an assembled matrix; see Matrix.classify().")
        .def_ro("n", &detail::Structure::n)
        .def_ro("declared_uband", &detail::Structure::declared_uband)
        .def_ro("bandwidth", &detail::Structure::bandwidth)
        .def_ro("cyclic_bandwidth", &detail::Structure::cyclic_bandwidth)
        .def_ro("nnz", &detail::Structure::nnz)
        .def_ro("corner_entries", &detail::Structure::corner_entries)
        .def_ro("block_period", &detail::Structure::block_period)
        .def_ro("scale", &detail::Structure::scale)
        .def_ro("symmetry", &detail::Structure::symmetry)
        .def_ro("skew", &detail::Structure::skew)
        .def_ro("circulant", &detail::Structure::circulant)
        .def_ro("diagonal_dominance", &detail::Structure::diagonal_dominance);

    m.def("structure", [](const AssembledMatrix &K, double tol) { return detail::structure(K, tol); },
          "K"_a, "tol"_a = 1e-12, nb::call_guard<GilRelease>(),
          "Measure bandwidth, symmetry, skew, circulant defect, block period and diagonal dominance.");
    m.def("first_column", [](const AssembledMatrix &K) { return to_np(detail::first_column(K)); }, "K"_a);

    nb::class_<LinearSolver>(m, "LinearSolver")
        .def(nb::init<const AssembledMatrix &, bool, Solver>(), "K"_a, "periodic"_a, "solver"_a = Solver::Auto,
             nb::call_guard<GilRelease>())
        .def_prop_ro("size", &LinearSolver::size)
        .def_prop_ro("backend", &LinearSolver::backend)
        .def_prop_ro("pivoted", &LinearSolver::pivoted,
                     "True when Auto fell back to a banded LU with partial pivoting (an indefinite matrix, or a pivot-free LU that broke down or, with periodic corners, failed its accuracy check).")
        .def("solve", [](const LinearSolver &s, DArr b)
             { std::vector<double> bv = to_vec(b), x; { GilRelease release; x = s.solve(bv); } return to_np(std::move(x)); }, "b"_a);

    bind_2d(m);
}
