"""
femd -- a Finite Element Library.
1D finite elements on B-splines of any degree, with classical Lagrange
elements as a second family, and continuous P_k Lagrange elements on
triangular meshes in 2D.  Grid and degree in, sparse matrices and vectors
out.  fd.IRK and fd.SSPRK step in time.

    import numpy as np, femd as fd
    grid = np.linspace(0, 1, 65)
    V = fd.SplineSpace(grid, degree=3, bc="periodic")
    Q = V.cache(npts=5, nder=1)
    A = fd.assemble(Q, [(0, 0, 1.0), (1, 1, 1.0)])      # (u,v) + (u',v')
    solve = A.solver()
    u = fd.Function(V).project(np.sin)                   # a Function carrying its coefficients
    u.vector, u(x), u + 0.5*dt*k                          # array, evaluation, linear algebra
"""
from __future__ import annotations

from . import _femd as _C
from ._femd import Mesh1D, BoundaryCondition, BCSpec, QuadratureCache, has_fftw, Solver
from ._femd import has_openmp, set_num_threads, get_num_threads, omp_threshold
from ._femd import ddot as _ddot
from ._femd import Domain, orient2d, incircle

__all__ = [
    "Mesh1D", "BoundaryCondition", "BCSpec", "QuadratureCache", "Solver", "LinearSolver", "RectMatrix", "block",
    "Mesh2D", "Domain", "triangulate", "circle", "orient2d", "incircle",
    "split_triangles", "bisect_triangles", "refine_triangles",
    "smooth", "restore_delaunay", "remove_vertices", "mesh_from_arrays", "rectangle_mesh", "mapped_mesh",
    "SplineSpace", "LagrangeSpace", "DGSpace", "FunctionSpace", "ProductSpace",
    "has_fftw", "has_openmp", "set_num_threads", "get_num_threads", "omp_threshold", "Robin", "Derivative", "Matrix", "MatrixInfo", "assemble", "assemble_vector", "assemble_block_vector", "assemble_scalar",
    "TestFunction", "TrialFunction", "Function", "ProductFunction", "TestFunctions", "TrialFunctions", "Functions", "Constant",
    "D", "dx", "ds", "dS", "jump", "avg", "x", "form", "Form", "set_quadrature_degree", "quadrature_degree",
    "Transfer", "mixed_mass", "solve",
    "gmres", "lgmres", "cg", "csnewton", "KrylovInfo", "CSNewtonInfo", "newton", "NewtonInfo", "residual_function",
    "newton_system", "IRK", "SSPRK", "ERK", "butcher", "TVBLimiter", "SlopeLimiter", "limited_slope", "minmod",
    "sin", "cos", "exp", "log", "tanh", "sqrt", "sinh", "cosh", "sech",
    "LagrangeSpace2D", "SparseMatrix", "SparseSolver", "Preconditioner",
    "y", "Dx", "grad", "div", "dot", "inner", "as_vector", "FacetNormal",
    "ProductSpace2D", "VectorFunctionSpace", "VectorFunction", "as_matrix", "outer", "transpose", "sym", "skew",
    "tr", "Identity", "nabla_grad",
    "QuadMesh", "quad_mesh_from_arrays", "quad_mesh_from_triangles", "rectangle_quad_mesh", "mapped_quad_mesh", "quadrangulate", "LagrangeSpaceQ",
    "write_vtk", "VTKSeries", "read_gmsh",
    "RTSpace", "N1curlSpace", "DGSpace2D", "VectorElementFunction", "curl", "rot", "eigs", "EigenResult",
    "sign", "max_value", "min_value", "VertexLimiter", "ddot",
]

__version__ = _C.__version__     # from include/femd/version.hpp, as is the package metadata


def ddot(x, y):
    """The dot product sum_i x_i y_i of two coefficient vectors (arrays or Functions of one space).

    Computed in C++ as an ordered chunked sum, with the same bits for any number of threads,
    and without BLAS.  NumPy's  x @ y  goes to the BLAS, and the OpenBLAS of some builds
    threads even a dot product of 10^4 entries across every core, which on a laptop can
    take hundreds of microseconds instead of a few.  Energies, dissipation rates and
    relaxation parameters are sums of such products, so a time loop can spend most of its
    time there.  For x^T A y see Matrix.inner and SparseMatrix.inner."""
    import numpy as _np
    xv = _np.asarray(getattr(x, "vector", x))
    yv = _np.asarray(getattr(y, "vector", y))
    if xv.ndim != 1 or yv.ndim != 1:
        raise ValueError("ddot: two vectors")
    if _np.iscomplexobj(xv) or _np.iscomplexobj(yv):
        raise TypeError("ddot: real vectors only; use numpy for complex ones")
    return _ddot(_np.ascontiguousarray(xv, dtype=_np.float64), _np.ascontiguousarray(yv, dtype=_np.float64))

# The package is split into modules; everything public is re-exported here, so
# `import femd as fd` sees one flat namespace, as it always has.
from ._util import _vec, _coeff  # noqa: E402,F401
from .forms import (TestFunction, TrialFunction, Function, ProductFunction, TestFunctions, TrialFunctions, Functions,  # noqa: E402
                    Constant, D, dx, ds, dS, jump, avg, x, form, Form, sin, cos, exp, log, tanh, sqrt, sinh, cosh, sech,
                    y, Dx, grad, div, dot, inner, as_vector, FacetNormal, VectorFunction, as_matrix, outer,
                    transpose, sym, skew, tr, Identity, nabla_grad)
from .spaces import (Robin, Derivative, SplineSpace, LagrangeSpace, DGSpace, FunctionSpace, ProductSpace,  # noqa: E402,F401
                     _SpaceMixin, _bcspec, _one_bc, _integer, _mesh, _space_of)
from .linalg import (LinearSolver, MatrixInfo, Matrix, RectMatrix, block, solve,  # noqa: E402
                     assemble, assemble_vector, assemble_block_vector, assemble_scalar)
from .transfer import Transfer, mixed_mass  # noqa: E402
from .krylov import gmres, lgmres, cg, csnewton, KrylovInfo, CSNewtonInfo  # noqa: E402
from .newton import newton, NewtonInfo, residual_function  # noqa: E402
from .timestep import newton_system, IRK, SSPRK, ERK, butcher  # noqa: E402
from .limiters import TVBLimiter, SlopeLimiter, limited_slope, VertexLimiter, minmod  # noqa: E402
from .mesh2d import (Mesh2D, triangulate, circle, split_triangles, bisect_triangles, refine_triangles,  # noqa: E402
                     remove_vertices, smooth, restore_delaunay, mesh_from_arrays, rectangle_mesh, mapped_mesh)
from .sparse import SparseMatrix, SparseSolver, Preconditioner  # noqa: E402
from .spaces2d import LagrangeSpace2D, LagrangeSpaceQ, ProductSpace2D, VectorFunctionSpace  # noqa: E402
from .quadmesh import QuadMesh, quad_mesh_from_arrays, quad_mesh_from_triangles, rectangle_quad_mesh, mapped_quad_mesh, quadrangulate  # noqa: E402
from .io import write_vtk, VTKSeries, read_gmsh  # noqa: E402
from .forms import curl, rot, VectorElementFunction, sign, max_value, min_value  # noqa: E402
from .forms import set_quadrature_degree, quadrature_degree  # noqa: E402
from .forms import abs_ as abs  # noqa: E402,A001  (fd.abs; not in __all__, so a star import keeps the builtin)
from .elements2d import RTSpace, N1curlSpace, DGSpace2D  # noqa: E402
from .eigen import eigs, EigenResult  # noqa: E402
