# FEMd 0.5 manual

FEMd is a Finite Element Library for problems in one and two space dimensions especially designed for running efficiently simulations on modern personal computers. Its core is header-only C++17 and is used from Python through bindings. The user provides a domain, a polynomial degree and boundary conditions, writes the equations in weak form using inner products, and FEMd does the rest. The design follows the FEniCS project, so users familiar with FEniCS will find the form language and the workflow familiar.

In one dimension the basis functions can be B-splines of any degree and continuity, classical Lagrange elements or discontinuous (DG) elements. Boundary conditions such as Dirichlet, Neumann, clamped, periodic and Robin are built into the basis, and banded, cyclic and FFT solvers exploit the structure of the resulting matrices.

In two dimensions FEMd generates its own meshes. The triangular mesh generator produces Delaunay meshes with a guaranteed minimum angle, holes and boundary markers, and a mesh can be refined locally, smoothed or coarsened. Structured and mapped meshes of triangles or quadrilaterals are also available, and meshes can be imported from Gmsh. On these meshes FEMd provides continuous $P_k$ elements on triangles and $Q_k$ elements on quadrilaterals of any degree, vector-valued and mixed spaces for systems such as the Stokes and Navier-Stokes equations, Raviart-Thomas, Nédélec and discontinuous Galerkin elements for mixed methods, electromagnetics and conservation laws, and Dirichlet, natural, periodic and slip boundary conditions. Solutions can be written to VTK files for visualization in ParaView.

Both dimensions share the same form language, with symbolic Jacobians for nonlinear problems. The solvers include sparse direct factorizations, preconditioned Krylov methods (CG, GMRES and LGMRES), Newton and Newton-Krylov iterations, implicit and explicit Runge-Kutta time steppers, and sparse eigenvalue solvers. Any other scheme can be built from the same pieces. OpenMP parallelism is optional.

FEMd was designed and developed by D. Mitsotakis. The Delaunay mesh generator was developed in collaboration with T. Katsaounis. The code has been tested and used by the developer only on macOS.

This is the user manual. Every function and class is documented in the [API reference](reference/index.md).

---


## 1. Installation

Requirements: a C++17 compiler, Python 3.10 or later, NumPy and SciPy. FFTW is optional and enables the FFT solver backend. Matplotlib is optional and is used by the plot methods and the examples.

Download the repository from GitHub, with `git clone` or with Code > Download ZIP, and install it from its folder:

```bash
cd FEMd
pip install .                      # builds the C++ extension, installs `femd`
```

Run `pip install .` again after changing anything under `include/`, `python/src/` or `python/femd/`. Editing your own scripts or the examples needs nothing. `pip` installs into the Python it belongs to, so run it from the environment your notebook kernel uses, then restart the kernel.

`fd.__version__` gives the installed version, for example `python -c "import femd; print(femd.__version__)"`. The C++ header `femd/version.hpp` has the same number as `FEMD_VERSION`.

### 1.1 OpenMP

```bash
brew install libomp                                  # macOS only; Linux compilers ship OpenMP
pip install . -C cmake.define.FEMD_OPENMP=ON       # the extension with OpenMP (and threaded FFTW if present)
```

```python
fd.has_openmp()          # True for such a build
fd.set_num_threads(4)    # threads for the parallel loops; fd.get_num_threads()
```

OpenMP speeds up both 1D and 2D computations.

**When a loop goes parallel.** Each parallel loop stays on one thread until its size passes a threshold, 4096 by default, so small problems do not pay the cost of starting threads. `fd.omp_threshold()` returns the value. The size is measured per loop.

| loop | parallel when | for example |
|---|---|---|
| assembly in 1D and 2D, quadrature tables | elements (or cells) × quadrature points > 4096 | cubic splines with 5 points, above about 820 elements. $P_2$ triangles, above about 600 cells |
| evaluation and location at points | points > 4096 | `u.at` at 5000 points |
| matrix-vector products, sums and products of matrices, sparsity patterns | rows > 4096 | a matrix of dimension 5000 |
| Krylov solvers (CG, GMRES, LGMRES) | unknowns > 4096 | |
| the solves with the sparse Cholesky factor | unknowns > 8192 | the mass solve of each explicit Runge-Kutta stage |

`fd.set_num_threads(k)` sets how many threads a parallel loop uses. It does not change the threshold, which is fixed when the library is built. To change it, rebuild with the new value, for example 2048:

```bash
pip install . -C cmake.define.FEMD_OPENMP=ON -C cmake.define.CMAKE_CXX_FLAGS="-DFEMD_OMP_THRESHOLD=2048"
```

Restart the Python kernel afterwards. `fd.omp_threshold()` then reports the new value. The value stays in the build folder, so a later `pip install .` keeps it. To return to 4096, rebuild with an empty flag:

```bash
pip install . -C cmake.define.FEMD_OPENMP=ON -C cmake.define.CMAKE_CXX_FLAGS=""
```

The solves with a sparse Cholesky factor, $L y = b$ and $L^T x = y$, are sequential by nature, each unknown needing the ones before it. They run in parallel through the elimination tree of the factor: the tree is cut into independent subtrees, solved concurrently, and the nodes above them, which hold the long columns, are swept as dense blocks (supernodes) with the rows of each block shared between the threads. On a $P_2$ matrix with 50 000 unknowns the solve took 3.0 ms on one thread and 1.4 ms on four. The solves with the sparse LU factors run in parallel the same way, through the elimination tree of the pattern of $L + U$, which holds the dependencies of both sweeps whatever pivots were taken; the top is swept by rows in blocks, so that the rows of a block gather their entries in parallel. On the stage matrix of the vortex-shedding problem (17 051 unknowns) a solve took 1.6 ms on one thread and 0.7 ms on four. The multifrontal LU factorization also runs in parallel, independent subtrees of its supernodal tree concurrently and the dense updates of the top fronts shared between the threads, with the same factors for any number of threads (44 ms on one thread, 19 on four for that matrix). The Cholesky and $LDL^T$ factorizations are still sequential.

Two things to know.

- **Set NumPy's BLAS to one thread** when OpenMP threads are used, `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`, or `threadpoolctl`), in the environment before Python starts. Otherwise the two thread pools compete for the cores: with 4 threads each, assembly at $n = 10^4$ ran five times slower than serially.
- **One OpenMP runtime per process.** On macOS the build links Homebrew's `libomp`. A Python environment that loads another runtime (conda's MKL brings Intel's) can fail with "OMP: Error #15". Build and run in an environment with one runtime.

The C++ kernels also release the GIL, whether or not OpenMP is on, so Python threads assembling different forms run at the same time.

## 2. Quick start

Solve $-u'' + u = f$ on $(0, 1)$ with $u(0) = 0$ and $u(1) = 1$, where $f = (\pi^2 + 1)\sin \pi x + x$ so that the exact solution is $u = x + \sin \pi x$, with cubic splines on 20 elements:

```python
import numpy as np
import matplotlib.pyplot as plt
import femd as fd
from femd import D, dx

grid = np.linspace(0, 1, 21)
V = fd.SplineSpace(grid, 3, bc="dirichlet", left=0.0, right=1.0)
u, v = fd.TrialFunction(V), fd.TestFunction(V)
f = (np.pi**2 + 1) * fd.sin(np.pi * fd.x) + fd.x

uh = fd.solve(D(u) * D(v) * dx + u * v * dx, f * v * dx)

xs = np.linspace(0, 1, 401)
print(np.max(np.abs(uh(xs) - (xs + np.sin(np.pi * xs)))))
plt.plot(xs, uh(xs))
```

The space is a grid, a degree and the boundary conditions, here with their values ([Section 3](#3-spaces-in-one-dimension), [Section 4](#4-boundary-conditions)). The weak form $\int_0^1 (u'v' + uv)\,dx = \int_0^1 f v\,dx$ is written as it reads ([Section 6](#6-forms)), and `fd.solve` assembles it, factors the matrix and solves ([Section 7](#7-solving)). The result `uh` is a `Function`: `uh(x)` evaluates it and `uh(x, 1)` gives its derivative ([Section 5](#5-functions)).

## 3. Spaces in one dimension

The spaces of this section live on an interval, given by its grid. Spaces on triangle and quadrilateral meshes are in [Section 11](#11-spaces-in-two-dimensions).

### 3.1 `SplineSpace(grid, degree, bc="free", continuity=None, continuity_at=None, left=None, right=None)`

B-splines of degree $p$ on the grid, $C^{p-1}$ across every interior vertex by default (`continuity=None`), the maximal smoothness for the degree: $C^1$ for quadratic and $C^2$ for cubic splines. `continuity=k` lowers the global continuity to $C^k$, `continuity_at= {node: k}` lowers it at chosen interior nodes only. `continuity=0` spans the same space as `LagrangeSpace` on a different (Bernstein) basis.

`left=` and `right=` give the values of the boundary conditions built in at the two ends, zero by default ([Section 4](#4-boundary-conditions)).

Every `k` must be an integer in $[0, p-1]$, and every node an interior one, `1..nelem-1`. Anything else raises. In particular `continuity=-1` is refused rather than read as the default. A discontinuous space is `DGSpace` ([Section 3.6](#36-dgspacegrid-degree-bcfree-basislegendre)), whose forms carry the interior facet terms `dS`, `jump` and `avg`. Leave `continuity` out for the default $C^{p-1}$.

The coefficients of a spline Function are **not** point values. `V. dof_coordinates()` gives the Greville abscissa each coefficient is associated with, and `u(x)` evaluates the function.

### 3.2 `LagrangeSpace(grid, degree, bc="free", nodes="equispaced", left=None, right=None)`

Classical $C^0$ nodal elements. Coefficients **are** the values at the nodes `V.nodes()`, so `u.interpolate(f)` is just `f` at the nodes. `left=` and `right=` give the values of the boundary conditions built in at the two ends, zero by default ([Section 4](#4-boundary-conditions)).

The nodes of each element are $p+1$ equally spaced points by default. `nodes="lobatto"` puts them at the $p+1$ Gauss-Lobatto points instead. Use `nodes="lobatto"` from about $p = 5$. For $p \le 2$ the two node sets coincide. Above that the space is the same and only the basis differs. The uniform one interpolates badly at high degree and leads to worse conditioned matrices. The same option exists on 2D meshes ([Section 11.1](#111-p_k-elements-on-triangles)). 

Derivatives of any order utilize the exact derivatives of the basis functions.

### 3.3 Dimension

`V.dim` is the number of degrees of freedom, and it is **not** the number of grid points. For $n_e$ elements:

| | free | one end constrained by $k$ functionals | periodic |
|---|---|---|---|
| splines, $C^{p-1}$ | $n_e + p$ | $-k$ per end | $n_e$ |
| Lagrange | $n_e p + 1$ | $-k$ per end | $n_e p$ |

### 3.4 Non-uniform grids

Any strictly increasing grid works, for both families and all boundary conditions. Nothing in the library assumes equal elements. `V.mesh` is the 1D mesh the space was built on, and two of its members describe the element widths.

| member | returns | meaning |
|---|---|---|
| `V.mesh.is_uniform` | bool | `True` when all elements have the same width. It is measured from the grid points (to a relative $10^{-10}$), not assumed. |
| `V.mesh.h(e)` | float | The width $x_{e+1} - x_e$ of element `e`, for `e` from 0 to `V.nelem - 1`. It works on any grid. |

```python
V = fd.SplineSpace(np.linspace(0, 1, 21), 3)
V.mesh.is_uniform                    # True
V.mesh.h(0)                          # 0.05

W = fd.SplineSpace([0, 0.1, 0.3, 0.6, 1.0], 3)
W.mesh.is_uniform                    # False
[W.mesh.h(e) for e in range(4)]      # [0.1, 0.2, 0.3, 0.4]
```

Uniformity matters only for the FFT solver backend, which needs equal elements on a periodic grid.

### 3.5 `ProductSpace(*fields)`

Several spaces on one mesh, each with its own boundary conditions and degree, numbered together by position so that the coupled matrix stays banded. Each field keeps its own boundary conditions, so `ProductSpace(fd.SplineSpace(grid, 3, bc="dirichlet"), fd.LagrangeSpace(grid, 2, bc=("free", "neumann")))` is fine. The one restriction is that the fields are either all periodic or all non-periodic. A periodic field cannot sit beside a non-periodic one, and the constructor refuses that mix. `P.split(x)` and `P.gather(parts)` move between the global vector and per-field vectors. [Section 6.6](#66-coupled-systems) shows the forms.

### 3.6 `DGSpace(grid, degree, bc="free", basis="legendre")`

Discontinuous piecewise polynomials of degree $p \ge 0$. Every element carries its own $p + 1$ functions, so `V.dim` is $n_e (p+1)$ and a function has two values at every interior vertex. The elements are coupled only through the facet terms of a DG formulation, written with `dS`, `jump` and `avg` ([Section 6.10](#610-interior-facets-ds-jump-avg)).

```python
V = fd.DGSpace(grid, 2)                         # Legendre basis: diagonal mass matrix
V = fd.DGSpace(grid, 3, basis="lobatto")        # nodal at the Gauss-Lobatto points
V = fd.DGSpace(grid, 0, bc="periodic")          # piecewise constants, finite volumes
```

- **`basis="legendre"`** (the default) is $P_0, \dots, P_p$ mapped to each element, with $P_l(1) = 1$. It is orthogonal, so the mass matrix is diagonal with entries $h_e/(2l+1)$.
- **`basis="lobatto"`** is the Lagrange basis on the $p+1$ Gauss-Lobatto points of each element, $p \ge 1$. The coefficients are the values at the nodes, and the two end nodes of an element are its traces.
- **`bc`** is `"free"` or `"periodic"`. Nothing is built into a DG space, so boundary data enter weakly through `ds`. Periodicity leaves the basis alone and adds the seam between the last and the first element as an interior facet. The matrices are then cyclic, and the cyclic solvers apply.

Derivatives `D(u, k)` are taken element by element, at any order (zero above $p$). The half-bandwidth is $2p+1$, since a facet couples two neighboring elements. `V.interpolate(f)` works element by element, at the Gauss points (Legendre) or at the Gauss-Lobatto nodes (Lobatto). The Lobatto interpolant samples the element ends, so at a vertex where $f$ jumps both elements get the one value $f$ has there, while the Legendre interpolant reproduces the jump.

### 3.7 The space protocol and `FunctionSpace`

Every space states what it is through three attributes, which the form compiler reads instead of testing types.

| attribute | `SplineSpace` | `LagrangeSpace` | `DGSpace` |
|---|---|---|---|
| `family` | `"spline"` | `"lagrange"` | `"dg"` |
| `tdim` | 1 | 1 | 1 |
| `broken` | `False` | `False` | `True` |

`fd.FunctionSpace(mesh, family, degree, **options)` builds a space by family name, so code can take the family as a parameter.

```python
V = fd.FunctionSpace(grid, "spline", 3, bc="dirichlet")     # = fd.SplineSpace(grid, 3, bc="dirichlet")
V = fd.FunctionSpace(grid, "spline", 3, bc="dirichlet", left=0.0, right=1.0)   # with the boundary values
V = fd.FunctionSpace(grid, "dg-lobatto", 2)                 # = fd.DGSpace(grid, 2, basis="lobatto")
```

Given a `Mesh2D` or a `QuadMesh` instead of a grid, `fd.FunctionSpace(mesh, "lagrange", k)` builds the $P_k$ or $Q_k$ space of [Section 11](#11-spaces-in-two-dimensions). On triangles the families `"DG"`, `"RT"` and `"N1curl"` give `fd.DGSpace2D`, `fd.RTSpace` and `fd.N1curlSpace` ([Section 11.4](#114-raviart-thomas-nédélec-and-broken-p_k)). These spaces have `tdim` 2, and their `family` is `"lagrange"`, `"dg"`, `"RT"` or `"N1curl"`.

## 4. Boundary conditions

```python
bc="free"                              # nothing built in (default)
bc="dirichlet"                         # u = 0 at both ends
bc=("dirichlet", "neumann")            # per end
bc=("free", fd.Robin(1.0, 0.5))        # left natural, right: u + 0.5 u' = 0 built in
bc="clamped"                           # u = u' = 0
bc=fd.Derivative(2)                    # u'' = 0 built in
bc="periodic"
```

Each end takes its own condition, so Dirichlet at one end and Neumann or Robin at the other is a pair. What is built in is always homogeneous. Data are added in one of two ways.

- **Natural data** (an end left `"free"`) enter the right-hand side through `ds`, [Section 6.4](#64-boundary-terms).
- **Data on a built-in condition** are given with the space, or to `fd.solve`, which lifts them.

```python
V = fd.SplineSpace(grid, 3, bc=("dirichlet", "free"), left=1.0)   # u(0) = 1, built in, with its value
u, v = fd.TrialFunction(V), fd.TestFunction(V)
a = D(u)*D(v)*dx
L = f*v*dx + g*v*ds("right")           # u'(1) = g, natural
uh = fd.solve(a, L)                    # the value comes from the space, and is lifted
```

**`left` and `right`.** They carry the values of the conditions built into the space: `left` at the left end of the grid, $x = a$ (the first grid point), and `right` at the right end, $x = b$ (the last one). Both default to `None`, which means zero data, so `bc="dirichlet"` alone gives $u(a) = u(b) = 0$, and an end without a value keeps homogeneous data. What a value means depends on the condition built in at that end.

| condition at that end | `left=` or `right=` | condition imposed |
|---|---|---|
| `"dirichlet"` | a number $g$ | $u = g$ |
| `"neumann"` | a number $g$ | $u' = g$ |
| `"clamped"` | a pair $(g_0, g_1)$ | $u = g_0$ and $u' = g_1$ |
| `fd.Robin(alpha, beta)` | a number $g$ | $\alpha u + \beta u' = g$ |
| `fd.Derivative(m)` | a number $g$ | $u^{(m)} = g$ |
| `"free"` | not allowed | natural data enter through `ds` |
| `"periodic"` | not allowed | the two ends are identified |

A value at an end with nothing built in raises an error when the space is made. With a derivative condition (Neumann, Robin, `Derivative`) and nonzero data, the data must also appear in the boundary term of the weak form ([Section 4.1](#41-lifting-step-by-step)).

Given to the space, as here, the values are used by every `fd.solve`, `fd.newton` and `fd.IRK` on it and by `V.lift()`, so they are written once. Given to a call instead, `fd.solve(a, L, left=1.0)`, they apply to that call only and replace the data of the space. `V.boundary_data` shows what the space holds. A callable of $t$ is data in time for `fd.IRK` ([Section 7.6](#76-implicit-runge-kutta-time-stepping-and-algebraic-systems)). With data, the unknown of `fd.newton` and `fd.IRK` is a Function of `V.unconstrained`, which can hold them.

### 4.1 Lifting, step by step

`fd.solve` writes $u = u_0 + u_g$. The lift $u_g$ satisfies the conditions with the data, and $u_0 \in V$ solves $a(u_0, v) = L(v) - a(u_g, v)$. The lift lives outside $V$, in `V.unconstrained`, the space with the same grid, degree and continuity and nothing built in. Its coefficients are the raw coefficients of $V$, so `V.prolongate(c)` is a vector of `V.unconstrained`. The same steps by hand:

```python
ug = V.lift(left=1.0)                  # a Function of V.unconstrained
A  = fd.form(a).assemble()
b  = fd.form(L).assemble() - fd.form(a).action(ug).assemble()   # L(v) - a(ug, v)
uh = A.solver().solve(b, lift=ug)      # V.prolongate(c) + ug, a Function of V.unconstrained
```

`a.action(w)` is the linear form $a(w, v)$, the trial function replaced by a Function. A Function of any space on the interval may appear in a form on $V$. One on the same grid is read at the quadrature nodes directly, and one on another grid is evaluated there.

**A derivative condition built in with nonzero data still needs its boundary term.** Integrating $-u''$ by parts leaves $u'(b)v(b) - u'(a)v(a)$, and a test function vanishes only at an end where $u$ itself is prescribed. So with `bc=("dirichlet", "neumann")` and $u'(1) = g$, the right-hand side is still `f*v*dx + g*v*ds("right")`, and the built-in version buys nothing over `"free"`. Built-in Robin $\alpha u + \beta u' = g$ likewise needs $u'(a) = (g - \alpha u(a))/\beta$ substituted into the boundary term. This is why the strong-versus-natural warning ([Section 6.8](#68-the-strong-versus-natural-warning)) recommends `"free"` for these conditions. Built-in derivative conditions pay off for higher-order operators, for instance clamped ends of a beam, where the conditions are essential.

## 5. Functions

```python
u = fd.Function(V)                     # zero; the name "u" is read from the variable
u.project(f)                           # L2 projection of a callable, returns u
u.interpolate(f)                       # interpolation at the dof coordinates (nodes / Greville points)
u.vector                               # NumPy array, writable in place
u(x), u(x, k)                          # value, k-th derivative at points
u.at_quad(Q, k)                        # values at the quadrature nodes of a cache
u.copy()
u.project_to(W), u.interpolate_to(W)   # a new Function in another space W (Section 5.4)
```

`u.project(f)` and `u.interpolate(f)` take a callable `f(x)`, a number, or an expression in `fd.x`, numbers, Constants and Functions. In `u.interpolate(fd.sin(np.pi*fd.x) + v**2 + fd.D(v, 1))` the Function `v` may live on another grid, and it is evaluated at the dof coordinates of `u`. Test and trial functions are not allowed. `u.interpolate` also takes the values at `V.dof_coordinates()`, and `u.project` the values at the nodes of its quadrature cache.

### 5.1 What `u.vector` is

`u.vector` holds the coefficients $c_j$ of the expansion

$$u_h(x) = \sum_{j=0}^{\dim V - 1} c_j\, N_j(x),$$

one entry per degree of freedom, indexed exactly as the rows and columns of every matrix on `V`. It is what a solve returns and what a form reads.

Whether a coefficient is a *value* depends on the family.

- **Lagrange.** The basis is interpolatory, $N_j(x_i) = \delta_{ij}$, so $c_j$ is the value of $u_h$ at node $j$: `u.vector[j] == u(V.nodes()[j])`, and `u.interpolate(f)` is just `f(V.nodes())`.
- **Splines.** The basis is not interpolatory. B-splines are overlapping positive bumps, several are nonzero at any point and none equals one at a node, so $c_j$ is a **weight**, not a value. `u(x)` blends the $p+1$ weights whose bumps cover $x$. Each weight is loosely located at its Greville abscissa `V.dof_coordinates()[j]`, and the polygon through $(\gamma_j, c_j)$ shadows the curve without touching it.

Consequences: plot a spline solution by evaluating it, `u(xs)`, never by plotting `u.vector`; set it from data with `u.project(f)` (L2 projection) or `u.interpolate(f)` (collocation at the Greville abscissae, a banded solve), not by writing samples into the vector; compare two solutions through `u1(xs)` and `u2(xs)`, or through their vectors only when both live in the same space. Everything *linear* in the discretization acts on `u.vector` directly, which is why a time stepper can live there: `u.assign(u + dt*k)` is exactly `u.vector + dt*k.vector`.

### 5.2 Arithmetic and `assign`

**Arithmetic is symbolic.** `u + 0.5*dt*k` is an expression, not a Function. It can be used in a form, passed to `assign`, or converted with `np.asarray`.

```python
w.assign(u + 0.5*dt*k)                 # evaluate a linear combination into w
u += dt * k.vector                     # in-place on the array
np.abs(u - v)                          # a linear combination converts to an array on demand
z = fd.Function(V, u + v)              # a new Function from a linear combination
w.assign(u * v)                        # ValueError: not linear, use project
```

Do not write `u = u + dt*k` in a loop: each iteration would wrap the previous expression, and the tree grows without bound. Write `u.assign(u + dt*k)`.

### 5.3 Names

The name is read from the source line at construction, so `eta = fd.Function(V)` is named `"eta"`, including in tuple assignments. Several `fd.Function` (or `fd.Constant`) calls on one line, as in `a, b = fd.Function(V), fd.Function(V)`, need Python 3.11 or later. Older versions raise an error there, so pass the names. Pass `fd.Function(V, "eta")` where that cannot work (a bare interpreter prompt, or a call nested inside a larger expression). The name is what `repr` shows and what the keyword override in `assemble` uses.

### 5.4 Moving a Function to another space

A coupled system with fields of different degree, a solution carried to a finer grid, a spline field written out on a Lagrange space: each needs a Function of `V` expressed in another space `W`.

```python
uW = u.project_to(W)                   # L2 projection, a Function in W named like u
uW = u.interpolate_to(W)               # collocation at W.dof_coordinates()
T  = W.transfer_matrix(V)              # the operator, assembled and factored once
T  = W.transfer_matrix(V, kind="interpolate")
uW = T(u)                              # Function in, Function out
cW = T @ u.vector                      # array in, array out
fd.Function(W).project(u)              # the same as u.project_to(W)
```

**The projection is exact.** It solves $M_W c_W = B c_V$ with the mixed mass matrix $B_{ij} = \int N^W_i N^V_j$ over the interval of $W$. When the grids differ, the integrand is a polynomial only between consecutive points of the union of the two grids, so $B$ is integrated there, with $\lfloor (p_V+p_W)/2 \rfloor + 1$ Gauss points per piece. A rule on the elements of $W$ alone cannot do this. The result is the true $L^2$ projection, $\left(u - u_W, v\right) = 0$ for every $v$ in $W$, and it conserves $\int u$ whenever constants lie in $W$. `fd.mixed_mass(W, V)` returns $B$ as a `scipy.sparse` matrix.

**Interpolation** solves $C c_W = E c_V$ with $C = N^W(\xi)$ and $E = N^V(\xi)$ at the dof coordinates $\xi$ of $W$. For a Lagrange target $C$ is the identity and nothing is solved. It is cheaper to build, but it is neither optimal in $L^2$ nor conservative.

Both reproduce $u$ to round-off whenever $V \subset W$. Examples are a refined grid, a higher degree with the same continuity (`continuity=p-1` for the target of a $C^{p-1}$ space of degree $p$), a spline into a Lagrange space of the same degree on the same grid, or a periodic space into a free one. Raising the degree of a spline space with default continuity is *not* nested, since the target is smoother.

**Boundary conditions.** The target's built-in conditions hold by construction. If the source violates one of them, for instance a free Function moved into a Dirichlet space, the result is the projection onto the constrained space and departs from $u$ near that end. A warning names the condition and the defect. Pass `check=False` to silence it. A non-periodic source moved into a periodic space is checked for $u(a) = u(b)$ the same way.

**Intervals.** The interval of `W` must lie inside that of `V`, so restricting a Function to a subinterval works and extending one does not.

**Cost.** `project_to` and `interpolate_to` build and factor a new operator on every call. In a loop build `T = W.transfer_matrix(V)` once. At $\dim V = 4099$ and $\dim W = 6001$ building took 9 ms and applying it 0.4 ms. `T.toarray()` gives the dense operator, since $M_W^{-1}B$ is not sparse even though $B$ is. `T.rhs` and `T.lhs` hold the two sparse factors.

## 6. Forms

### 6.1 Symbols and measures

```python
u = fd.TrialFunction(V)                # the unknown, becomes the column index
v = fd.TestFunction(V)                 # the test function, becomes the row index
w = fd.Function(V)                     # a known field, read live at assembly
D(u), D(u, k)                          # derivatives, k up to the continuity + 1 (any k on a DGSpace)
fd.x                                   # the coordinate
fd.sin, fd.cos, fd.exp, fd.log, fd.tanh, fd.sqrt   # pointwise functions of known fields and x
fd.sinh, fd.cosh, fd.sech, fd.abs, fd.sign         # more of them; the derivative of fd.abs is fd.sign
fd.max_value(a, b), fd.min_value(a, b)             # pointwise max and min, written with fd.abs
dx                                     # the interval measure; dx(quad_degree=n) overrides the rule
dx(scheme="lobatto")                   # the Gauss-Lobatto rule, a diagonal mass on Lobatto nodes
ds, ds("left"), ds("right")            # endpoint evaluation, for natural conditions with data
dS, jump(u), avg(u), u('-'), u('+')    # interior facets, Section 6.10
```

### 6.2 Compiling and assembling

```python
a = fd.form((D(u)*D(v) + u*v) * dx)    # rank 2 -> Matrix
L = fd.form(fd.sin(fd.x) * v * dx)     # rank 1 -> vector
E = fd.form((w*w + D(w)*D(w)) * dx)    # rank 0 -> float

A = a.assemble()                       # a constant form is assembled once and cached
b = L.assemble()
e = E.assemble()                       # uses w.vector as it is now
e = E.assemble(w=other)                # keyword override by name, for this call
```

Rank is decided by which symbols appear. A form must be linear in the test and trial functions, so `u*u*v` and `fd.sin(v)` are rejected. Nonlinearity in known fields is unrestricted: `w*D(w)*v`, `fd.sin(w)*v`, `fd.exp(D(w))*v`.

**What assembly does.** `fd.form` splits the expression into terms, each a test function (and a trial function for rank 2), a derivative code, a measure and a *coefficient*, the part made of known fields, constants and coordinates, for instance $u_1\,\partial_x u_1 + u_2\,\partial_y u_1$ in the convection term of Navier-Stokes. Assembling evaluates each coefficient at the quadrature points and hands it to the C++ kernels, which multiply it by the basis functions and the weights and add the products into the vector or matrix. The coefficient is evaluated in C++ too: the expression tree is compiled once into a short program (`Integrand`, one instruction per node, the values of the fields at the points as inputs), which runs over the points in parallel with the same bits for any number of threads. `fd.set_compiled_integrands(False)` goes back to the NumPy evaluation of the tree; `fd.compiled_integrands()` reads the switch. Complex field values (the complex-step derivative, Section 7.2) go through the same programs in complex arithmetic, with NumPy's rules for products, powers and the elementary functions, $|z| = \mathrm{sign}(\mathrm{Re} z)\, z$ and $\mathrm{sech} z = 1/\cosh z$, while the fields and the kernels, which are real and linear, act on the real and the imaginary parts separately. The two agree to rounding (the same left-to-right sums and products, `std::pow` and libm functions against NumPy's). Since 0.5 the whole assembly of a form is one C++ call (`FormAssembler2D`, `forms/assembler_2d.hpp`, and in 1D `FormAssembler1D`, `forms/assembler_1d.hpp`, whose coefficients go through the same `Integrand` programs): the form is compiled once into its point sets, the fields read at each of them and the terms with their programs, and each `assemble()` passes the current coefficient vectors and `Constant` values to C++, which evaluates the fields and the coefficients and runs the kernels with no Python in the loop and without the GIL, so several threads can assemble the same form with different data at the same time. The derivative terms of `jacobian()` are built once per field and reuse their compiled form. Complex field values (rank 0 and 1) are assembled there too. `set_compiled_integrands(False)` keeps the term-by-term Python driver, which gives the same numbers (to rounding where a 2D field's derivatives are read by some groups of terms and not others, where libm and NumPy round an elementary function differently, and in a rank-0 `dS` form in 1D, whose facets NumPy summed pairwise).

### 6.3 Jacobians

For a rank-1 form, `R.jacobian(w)` is the matrix $\partial R/\partial w$, obtained by symbolic differentiation of the coefficient tree and exact to round-off. `R.jacobian("w")` by name also works.

### 6.4 Boundary terms

A natural condition with data is a boundary term. For $-u''+u=f$ with $u'(1) + u(1) = g$:

```python
f = (1 + np.pi**2) * fd.cos(np.pi * fd.x)         # data as an expression in fd.x (or a Function)
a = fd.form((D(u)*D(v) + u*v) * dx + u*v * ds("right"))
L = fd.form(f * v * dx + (-1.0) * v * ds("right"))
```

Data enters a form as an expression in `fd.x`, as a `Function`, as a number or as a `fd.Constant` ([Section 6.9](#69-constants)). A plain Python callable is not accepted; project it into a `Function` first if it has no closed form.

`ds` has no meaning on a periodic space.

**Robin data in general.** For $\alpha u + \beta\,\partial u/\partial n = c$ at an end, where $\partial u/\partial n$ is the outward derivative ($u'$ at the right end, $-u'$ at the left end), add `(alpha/beta)*u*v*ds(end)` to the bilinear form and `(c/beta)*v*ds(end)` to the right-hand side, the same at both ends. With $\alpha/\beta > 0$ the problem is well posed. Written with $+u'$ at the left end instead, the boundary term changes sign and the matrix can become indefinite, which `Auto` then solves with a pivoted LU ([Section 7](#7-solving)).

### 6.5 Higher derivatives

`D(u, k)` is allowed when the space is $C^{k-1}$, so cubic splines take `D(u, 2)` and `D(u, 3)`, Lagrange only `D(u, 1)`. A `DGSpace` takes any order, element by element ([Section 3.6](#36-dgspacegrid-degree-bcfree-basislegendre)). Fourth-order operators are therefore primal on splines:

```python
V = fd.SplineSpace(grid, 3, bc="clamped")          # u = u' = 0 are essential here
a = fd.form(D(u, 2) * D(v, 2) * dx)                 # u''''
```

### 6.6 Coupled systems

```python
P = fd.ProductSpace(U, W)
(u1, u2), (v1, v2) = fd.TrialFunctions(P), fd.TestFunctions(P)
a = fd.form((D(u1)*D(v1) + u1*v1 + u2*v1 + D(u2)*D(v2) + u2*v2 + u1*v2) * dx)
A = a.assemble(); A.symmetric = True                # the assembler cannot see block symmetry
sol = A.solver().solve(L.assemble())                # a ProductFunction
c1, c2 = sol.split()                                # per-field Functions
W_ = fd.Function(P)                                  # a ProductFunction on P, one global vector (or fd.Functions(P))
W_.project([f1, f2])                                 # set it field by field, L2 projection (or interpolate)
(w1, w2) = W_                                        # its components for forms
```

The same system can be built one block at a time. A form whose test and trial functions come from two different spaces assembles to a `RectMatrix`, (stored in FEMd's compressed sparse row store, its products, transpose and algebra in C++), and `fd.block` puts the blocks together into one banded `Matrix` on the `ProductSpace`:

```python
u, phi = fd.TrialFunction(U), fd.TestFunction(U)
w, psi = fd.TrialFunction(W), fd.TestFunction(W)
A = fd.form((D(u)*D(phi) + u*phi) * dx).assemble()     # U x U, a Matrix
B = fd.form(w*phi * dx).assemble()                      # U <- W, a RectMatrix, U.dim x W.dim
C = fd.form(D(u)*psi * dx).assemble()                   # W <- U
M = fd.form((D(w)*D(psi) + w*psi) * dx).assemble()     # W x W
K = fd.block([[A, B], [C, M]])                          # on ProductSpace(U, W), banded
P = K.space
sol = K.solver().solve(P.gather([f1, f2]))              # a ProductFunction
uh, wh = sol.split()
```

Blocks are assembled by the same kernels as the product form and give the same matrix to round-off. They are the natural route when the blocks are reused on their own, for instance a coupling `B` applied in a splitting scheme, or `B.T` in a Schur complement. `fd.block` measures the symmetry of the result, so `[[A, B], [B.T, M]]` with symmetric `A` and `M` is solved by Cholesky without setting the flag by hand. `None` or `0` marks a zero block. Field $I$ must be one space for row $I$ and column $I$.

The two spaces of a rectangular form must share the grid. Across grids, use `fd.mixed_mass(W, V)`, which integrates the mass coupling exactly on the union of the two grids ([Section 5.4](#54-moving-a-function-to-another-space)).

### 6.7 Quadrature

The rule is inferred per form from the polynomial degree of its terms, so $u\,u_x\,v$ at $p=3$ gets 5 points and is exact. A non-polynomial function, of the data ($\sin x$) or of a field ($\sin u$), counts as a polynomial of degree $p+1$: its interpolant of degree $p$ plus the leading term of the remainder. So $f\,v$ is integrated exactly for degree $2p+1$ and an error norm $(u_h - u)^2$ for degree $2p+2$, which is what the square of the leading error term needs. This is the lowest rule that leaves the convergence order and the measured errors unchanged; raise it with `dx(quad_degree=n)` when the data vary on a scale below the mesh. To use one rule in every form of a code, call `fd.set_quadrature_degree(n)` once: every form made afterwards takes $n$ on all its measures, as if each carried `quad_degree=n` ($n$ Gauss points per element in 1D, the degree of exactness in 2D), a measure with its own `quad_degree=` still wins, and `fd.set_quadrature_degree(None)` returns to the inferred rules. `fd.quadrature_degree()` reads the setting. `dx(quad_degree=n)`, or `dx(n)`, overrides it with $n$ Gauss points per element. On a 2D mesh $n$ is the degree of exactness instead ([Section 12.2](#122-the-form-language-in-2d)). Each form builds its own quadrature caches at its first assembly and keeps them.

**Lumped mass.** `dx(scheme="lobatto")` integrates with the Gauss-Lobatto rule instead, and `dx(n, scheme="lobatto")` with $n \ge 2$ points per element. Without $n$ the rule has $p+1$ points, where $p$ is the highest degree of the term's test and trial spaces. These points are the nodes of `LagrangeSpace(grid, p, nodes="lobatto")`, so on that space `u*v*dx(scheme="lobatto")` is exactly diagonal. This is the lumped mass. `V.mass_matrix(lumped=True)` returns the same matrix, and on the other spaces the row sums of the mass matrix, $d_i = \sum_j M_{ij}$. A diagonal matrix is solved by the `Diagonal` backend, so an explicit time stepper needs no factorization. The rule is exact to degree $2p-1$, so the lumped mass is not the exact $\int u v$, but the eigenvalues of $-u''$ keep their order $2p$. Row sums are refused when one of them is not positive, which happens for equispaced Lagrange nodes at $p = 8$ and from $p = 10$. On a 2D mesh it is the tensor Gauss-Lobatto rule on quadrilaterals, so `u*v*dx(scheme="lobatto")` is exactly diagonal on a $Q_k$ space with `nodes="lobatto"`, and `mass_matrix(lumped=True)` works the same way ([Section 12.3](#123-matrices-and-solvers)). Triangles have no such rule.

```python
V = fd.LagrangeSpace(grid, 4, bc="dirichlet", nodes="lobatto")
u, v = fd.TrialFunction(V), fd.TestFunction(V)
M = fd.form(u*v*dx(scheme="lobatto")).assemble()     # diagonal
M = V.mass_matrix(lumped=True)                       # the same matrix
```

### 6.8 The strong-versus-natural warning

Compiling a symmetric form of order $2k$ on a space that has a derivative condition of order $\ge k$ built in emits a warning: that condition is natural for this operator and has been imposed strongly. Use `"free"` for the classical treatment. No warning for `"clamped"` under `D(u,2)*D(v,2)`, where $u'$ is essential.

### 6.9 Constants

A plain number in a form is fixed when the form is compiled. `fd.Constant` is a number whose value is read when the form is assembled, as a Function's coefficients are. It is the time in a source term, or a parameter swept without recompiling.

```python
t = fd.Constant(0.0)                               # named "t" after the variable
L = fd.form(fd.exp(-t) * fd.sin(np.pi * fd.x) * v * dx)
t.assign(0.5); b = L.assemble()                    # at t = 0.5
b = L.assemble(t=0.7)                              # this call only, t stays 0.5
```

A Constant can appear anywhere a number can: in rank 0, 1 and 2 forms, inside `fd.sin` and the other pointwise functions, in Jacobians (its derivative is zero), and in `assign` of a linear combination, `w.assign(u + dt*k)` with `dt` a Constant. `L.constants` lists the names. A form that contains a Constant is never cached, since its value can change between assemblies. Name inference works as for Functions ([Section 5.3](#53-names)), and `fd.Constant(0.0, "t")` names it explicitly.

### 6.10 Interior facets: `dS`, `jump`, `avg`

This section is about 1D meshes. On triangles the same measure and restrictions work with the facet normal, see [Section 12.9](#129-discontinuous-galerkin-on-triangles).

`dS` is the measure of the interior facets, which in 1D are the vertices between elements, plus the seam of a periodic space. On a facet every broken quantity has two values. `u('-')` is the value from the left element and `u('+')` from the right one, and

$$\mathrm{jump}(u) = u^- - u^+, \qquad \mathrm{avg}(u) = \tfrac12\left(u^- + u^+\right).$$

The outward normal of the left element is $+1$, so $\mathrm{jump}(u)$ is also the jump times that normal. An upwind flux for $a > 0$ is `a*u('-')`, and a Lax-Friedrichs flux is `avg(F) + 0.5*alpha*jump(u)`.

Upwind DG for $u_t + u_x = 0$, periodic:

```python
V = fd.DGSpace(grid, 2, bc="periodic")
u, v, w = fd.TrialFunction(V), fd.TestFunction(V), fd.Function(V)
M = fd.form(u*v*dx)
R = fd.form(w*D(v)*dx - w('-')*jump(v)*dS)          # (w, v_x) - sum over facets of w^- [v]
irk = fd.IRK(M, R, dt, "gauss", 3)
```

Symmetric interior penalty for $-u'' = f$, with $u(0) = u(1) = 0$ imposed weakly (Nitsche), $\sigma$ of order $p^2/h$:

```python
V = fd.DGSpace(grid, p)
u, v = fd.TrialFunction(V), fd.TestFunction(V)
a = fd.form(D(u)*D(v)*dx
            + (-avg(D(u))*jump(v) - avg(D(v))*jump(u) + sigma*jump(u)*jump(v)) * dS
            + (-D(u)*v - D(v)*u + sigma*u*v) * ds("right")
            + ( D(u)*v + D(v)*u + sigma*u*v) * ds("left"))
```

Both converge at order $p + 1$ in $L^2$, and the upwind scheme conserves $\int u$ to round-off across the seam. Measure the mass as `fd.form(w*dx).assemble()`. Summing `M @ c` gives $\int u \sum_i N_i$, which is not the mass for the Legendre basis, since it is not a partition of unity.

**Systems.** `dS` works on a `ProductSpace` too. Each field is restricted, jumped and averaged on its own, and a term may couple any two fields. The shallow water equations $h_t + m_x = 0$, $m_t + \left(m^2/h + g h^2/2\right)_x = 0$ with a Lax-Friedrichs flux:

```python
V = fd.DGSpace(grid, 2, bc="periodic")
W = fd.ProductSpace(V, V)
(h, m), (a, b) = fd.TrialFunctions(W), fd.TestFunctions(W)
q = fd.Functions(W, name="q")
qh, qm = q
F1, F2 = qm, qm**2/qh + 0.5*g*qh**2
M = fd.form(h*a*dx + m*b*dx)
R = fd.form(F1*D(a)*dx + F2*D(b)*dx
            - (avg(F1) + 0.5*alpha*jump(qh))*jump(a)*dS
            - (avg(F2) + 0.5*alpha*jump(qm))*jump(b)*dS)
rk = fd.SSPRK(M, R, dt, limiter=fd.TVBLimiter(W))    # or fd.IRK(M, R, dt), with the exact Jacobian
```

When a field of a `ProductSpace` is discontinuous, its half-bandwidth covers the coupling of neighboring elements through the facets, so the banded and cyclic solvers still apply. A product of continuous fields keeps the band of one element.

- **Rules.** Restrictions and `jump`/`avg` belong only in `dS` terms. In a `dS` term the test and trial functions must be restricted, and so must a Function of a broken space or a derivative. A Function of a continuous space may appear unrestricted, since its value is single-valued.
- **Nonlinear fluxes.** Restricted Functions differentiate like any other, so `R.jacobian(w)`, `fd.newton` and `fd.IRK` get the exact Jacobian of a flux such as `avg(0.5*w**2) + 0.5*alpha*jump(w)`.
- **Continuous spaces.** `dS` works on them too. `jump(u)` is then zero and `jump(D(u))` is the derivative jump, the continuous interior penalty term.
- **Symmetry.** A facet term that mixes sides clears the symmetric flag, even when the whole form is symmetric, as for interior penalty. `A.classify()` measures it, and `A.symmetric = True` then lets `Auto` pick Cholesky.

## 7. Solving

```python
solve = A.solver()                     # Auto: chosen from the measured structure (below)
solve = A.solver("fft")                # Circulant: periodic, uniform, constant coefficients
solve = A.solver("dense")              # pivoted LU, the oracle
k = solve.solve(b)                     # a Function in A's space
solve.solve(b, into=k)                 # write into an existing Function
solve.solve(b, lift=ug)                # V.prolongate(c) + ug, Section 4.1
y = A @ x                              # banded matvec, a Function in A's space
A.matvec(x, out=y_buf)                 # the same into a buffer you keep, for inner loops
uh = fd.solve(a, L)                    # assemble, lift the space's boundary values, solve
A.tocsr()                              # scipy.sparse, for anything else
```

| backend | requires | cost per solve |
|---|---|---|
| `Diagonal` | a diagonal matrix | $O(N)$, a division |
| `SymBand` | symmetric positive definite (`Auto` falls back to a pivoted `Band` LU otherwise) | $O(Np)$ |
| `Band` | none (no pivoting; fine for diffusion and mass terms) | $O(Np)$ |
| `SymCyclic`, `Cyclic` | periodic space | $O(Np)$ |
| `Circulant` | periodic, uniform grid, constant coefficients, FFTW | $O(N\log N)$ |
| `Dense` | none | $O(N^3)$ factor |

`Auto` measures the matrix instead of trusting the space. Entries below $10^{-14}$ times the largest entry count as zero, so the quadrature round-off of an orthogonal basis is not mistaken for structure. Then

- only the diagonal remains: `Diagonal` (the Legendre DG mass matrix),
- entries across the periodic seam: `SymCyclic` or `Cyclic`,
- otherwise `SymBand` or `Band`, even on a periodic space (the Lobatto DG mass, any form without facet or wrap coupling).

The banded backends factor the band the entries reach, not the band the space reserves. A DG space reserves $2p+1$ for facet terms, while a mass matrix only reaches $p$ (Lobatto) or 0 (Legendre). `Auto` picks a Cholesky backend when the `symmetric` flag is set, so an indefinite system fails loudly instead of silently. It never picks `Circulant`. An explicit backend counts every nonzero, so `A.solver("diagonal")` refuses a matrix with any off-diagonal entry. Measuring costs one pass over the band per `solver()` call.

**Matrices that are not positive definite.** `Auto` factors at once. A symmetric matrix that is not positive definite, such as the Helmholtz operator $-u'' - k^2 u$ or a boundary term of the unstable sign, is refused by the Cholesky and solved by a banded LU with partial pivoting instead. A pivot-free `Band` LU that meets a zero pivot is redone with pivoting too. `solve.pivoted` tells whether that happened. With periodic corners the fallback is a pivoted LU too, and `solve.backend` is then `Cyclic`. An explicit `SymBand` or `SymCyclic` raises an error on such a matrix instead of returning NaN.

**The FFT backend is guarded by measurement.** `A.solver("fft")` computes how far the matrix is from the circulant generated by its first column and refuses it above a relative defect of $10^{-10}$, since the FFT would otherwise invert a different matrix. The refusal names the cause: a non-uniform grid (with the range of element widths), periodic Lagrange of degree 2 or more, splines with reduced continuity or `continuity_at`, a coefficient that varies in $x$, or a coupled `ProductSpace`. Round-off in a uniform grid passes. Only the check for a periodic space comes before it, and the check for FFTW after it. `A.is_circulant()` and `fd.has_fftw()` tell you whether `Circulant` will work. `A.symbol()` gives the eigenvalues of a circulant operator, the discrete dispersion relation, without needing FFTW (it uses FEMd's own FFT, `util/fft.hpp`).

A singular system, for instance a coupled periodic problem whose operator annihilates $(c, -c)$, gives garbage from every backend. Well-posedness is the user's to check.

### 7.1 Krylov solvers

```python
x, info = fd.gmres(A, b, M=A0.solver())        # left-preconditioned GMRES(30)
x, info = fd.lgmres(A, b, M=A0)                # LGMRES(20, 2); a Matrix M is factored once
x, info = fd.gmres(lambda y: ..., b, M=lambda r: ...)   # any operator, any preconditioner
```

GMRES and LGMRES come from the vendored `csnewton`. `fd.cg(A, b, M=None)` is preconditioned conjugate gradients for a symmetric positive definite `A`, in 1D and 2D, and its `tol` bounds the true relative residual $\left\| b - Ax \right\| / \left\| b \right\|$. `A` is a `Matrix`, a 2D `SparseMatrix`, an `fd.block` system, anything with `.matvec`, or a callable. `M` is `None`, a `LinearSolver`, a `Matrix`, anything with `.solve`, or a callable $r \mapsto M^{-1} r$. With a `Matrix` or a `SparseMatrix` and a `LinearSolver`, a sparse Cholesky factor or a sparse `Preconditioner` (`A.preconditioner("ilu0")`, `"ssor"`, `"jacobi"`), the whole iteration runs in C++. Otherwise each application calls back into Python.

`tol` bounds $\|M^{-1} r\| / \|M^{-1} b\|$, the left-preconditioned relative residual, so it does not depend on how $M$ is scaled. `info.residual` is the true $\|b - Ax\| / \|b\|$, recomputed at the end, and a `RuntimeWarning` is issued when `tol` is not met. The solution is a Function or `ProductFunction` when `A` knows its space.

In 1D a direct banded solve is already $O(np)$, so a Krylov solver pays when an exact factorization is unavailable or expensive and a cheap one is close: a variable coefficient preconditioned by the constant-coefficient operator (19 iterations where plain GMRES stalls), a coupled system preconditioned by its factored diagonal blocks, or an FFT solve preconditioning a nearly circulant matrix.

### 7.2 Newton's method

A nonlinear problem is written as a **residual**, a form linear in the test function: find $u$ with $R(u; v) = 0$ for every $v \in V$. Newton's method solves $J(u)\,\delta = -R(u)$ and updates $u \leftarrow u + s\,\delta$, where $J = \partial R / \partial u$ is the Jacobian and $s \le 1$ the step length.

```python
V = fd.SplineSpace(grid, 3, bc=("dirichlet", "free"), left=1.0)   # u(0) = 1, with the space
v = fd.TestFunction(V)
u = fd.Function(V.unconstrained)                        # the unknown, holding the initial guess
R = fd.form((D(u)*D(v) + u**3*v - f*v) * dx - g*v*ds("right"))   # u'(1) = g, natural
info = fd.newton(R, u)                                  # u(0) = 1 from the space; u is updated in place
```

Each iteration of `fd.newton` does four things.

1. It assembles the residual vector $r = R(u)$.
2. It assembles the exact Jacobian `R.derivative(u).assemble()`, a banded `Matrix` in 1D and a `SparseMatrix` in 2D ([Section 7.3](#73-the-exact-jacobian)).
3. It solves $J\,d = -r$ with the matrix's `solver(backend)`, a banded or cyclic direct solve at $O(np)$ in 1D and a sparse factorization in 2D.
4. It halves $s$, starting from 1, until $\|R(u + s\,d)\| \le (1 - 10^{-4} s)\,\|R(u)\|$ (Armijo backtracking), then takes the step.

It stops when $\|R\| \le \max(\text{tol}, \text{rtol}\,\|R(u_0)\|)$, with $\|\cdot\|$ the Euclidean norm of the residual vector.

**Complex-step Newton, without forming the Jacobian.** With `linear="gmres"` (or `"lgmres"`) and `jv="complex-step"`, step 2 is skipped. Each Newton system is solved by GMRES, which needs only products $J v$, and each product is one evaluation of the residual at a complex field, $J v = \mathrm{Im}\, R(u + i h v)/h$ with $h = 10^{-20}$, exact to rounding since nothing is subtracted. The method and its convergence are analysed in D. Mitsotakis, *The complex-step Newton method and its convergence*, Numerische Mathematik 157 (2025) 993–1021, [doi:10.1007/s00211-025-01471-w](https://doi.org/10.1007/s00211-025-01471-w). With `precond=None` no Jacobian matrix is formed at all. The default `precond="frozen"` would assemble and factor $J$ at the first iterate to precondition GMRES, and `precond="linear"` factors only its constant-coefficient part ([Section 7.5](#75-newton-krylov)). The line search and the boundary data work as in the direct mode.

```python
info = fd.newton(R, u, linear="gmres",     # GMRES for each Newton system
                 jv="complex-step",        # J v = Im R(u + i h v) / h, no Jacobian formed
                 precond=None)             # no preconditioner: free of any Jacobian matrix
```

With `line_search=False` as well, this is the method of `fd.csnewton`, driven by `fd.newton` on the form. `examples/example2.ipynb` compares it with the direct solve.

| argument | default | meaning |
|---|---|---|
| `tol`, `rtol` | `1e-10`, `0` | absolute and relative stopping tolerances on $\|R\|$ |
| `maxiter` | `50` | Newton iterations |
| `line_search` | `True` | Armijo backtracking; `False` gives the classical full step |
| `backend` | `"Auto"` | the direct solver, as in `Matrix.solver` |
| `left`, `right`, `dirichlet` | `None` | data on the conditions built into $V$, `left=` and `right=` in 1D, `dirichlet=` in 2D ([Section 7.4](#74-boundary-data-and-coupled-systems)) |
| `linear`, `jv`, `precond`, `krylov_tol`, `restart`, `krylov_maxiter`, `cs_h` | `"direct"`, `"assembled"`, `"frozen"`, `1e-8`, `30`, `1000`, `1e-20` | Newton-Krylov ([Section 7.5](#75-newton-krylov)) |
| `verbose`, `warn` | `False`, `True` | print each iteration; warn when not converged |

The result is a `NewtonInfo` with `converged`, `iterations`, `residuals` (the norm at every iterate, the first one initial), `steps` (the line-search step lengths), `linear_iterations` and `message` (why it stopped early, if it did). The residual history shows the quadratic convergence of an exact Jacobian. For $-u'' + u^3 = f$ above, with cubic splines on 32 elements:

    ||R||: 6.9e+01  3.5e+00  8.8e-01  1.4e-01  6.2e-03  1.4e-05  7.2e-11

The line search matters away from the solution. For $10^{-3} u'' = \tanh(u) - f$ started at $u = 3$, the full Newton step overshoots and diverges, while the damped iteration converges. Near the solution the full step is accepted, so the line search costs nothing there.

`fd.newton` refuses instead of guessing. A singular Jacobian, a residual that is not finite along the Newton direction, or a line search that cannot reduce $\left\| R \right\|$ in 30 halvings ends the iteration with the reason in `info.message` and a `RuntimeWarning`, and `u` is left at the last good iterate.

### 7.3 The exact Jacobian

```python
J = R.derivative(u)              # a rank-2 Form: d R / d u, not yet assembled
A = J.assemble()                 # the same as R.jacobian(u), a RectMatrix for u in V.unconstrained
Jd = J.action(d).assemble()      # J applied to a Function d, matrix-free
A = R.jacobian(w, w=U)           # at the coefficient vector U, leaving w.vector untouched
Jc = R.derivative(u, part="constant").assemble()   # only the terms whose coefficient has no Function
```

The Jacobian is symbolic. The form compiler differentiates each term with respect to the field and its derivatives, with the product rule and the chain rule through `sin`, `cos`, `exp`, `log`, `tanh`, `sinh`, `cosh`, `sech`, `sqrt`, `abs`, `max_value`, `min_value` and powers, so it is exact and assembled with the same quadrature as the residual.

- **Where it is evaluated.** At the current `u.vector`. A keyword override, `R.jacobian(w, w=U)` or `J.assemble(w=U)`, evaluates it at another vector without changing the Function. This is how the Jacobian of a right-hand side $f(U)$ is taken at several stage values ([Section 7.6](#76-implicit-runge-kutta-time-stepping-and-algebraic-systems)).
- **Its trial space.** The space of `u` by default. For `u` in `V.unconstrained`, `space=V` gives the square Jacobian on $V$, the one Newton needs when $u$ carries boundary data. For a field of another space on the same grid, the result is a `RectMatrix`.
- **Several fields.** With respect to a `ProductFunction` it differentiates with respect to every field, and the result is a banded `Matrix` on the `ProductSpace`.
- **Its constant part.** `part="constant"` keeps the terms whose coefficient does not depend on any Function, the linear constant-coefficient part of the operator. It is assembled once and makes a good preconditioner.

### 7.4 Boundary data and coupled systems

Data on a built-in condition work as for `fd.solve` ([Section 4.1](#41-lifting-step-by-step)). The unknown is a Function of `V.unconstrained`, written $u = u_g + P c$ with the lift $u_g$ = `V.lift()` of the values given with the space (or of `left=`, `right=` given to `fd.newton`, which replace them) and $c \in V$. Newton moves $c$ only, so the data hold exactly at every iterate. The initial guess is whatever `u` holds, reduced to $V$ by subtracting $u_g$ and interpolating. Without data, `u` may be a Function of $V$ itself. On a 2D mesh `dirichlet=` takes the place of `left=` and `right=` ([Section 12.1](#121-boundary-conditions)). The unknown may then be a Function of any 2D space, `RTSpace` and `N1curlSpace` included, or a `ProductFunction` of a 2D `ProductSpace`, of $W$ itself when no field carries data and of `W.unconstrained` otherwise.

A built-in derivative condition with nonzero data still needs its `ds` term in `R`, since the test functions do not vanish there ([Section 4.1](#41-lifting-step-by-step)).

A coupled system uses a `ProductFunction`:

```python
P = fd.ProductSpace(V, W)
U = fd.Functions(P, "U")
a, b = U                                      # its fields, for the form
p, q = fd.TestFunctions(P)
R = fd.form((D(a)*D(p) + a*b*p - p + D(b)*D(q) + b**3*q - a*q) * dx)
fd.newton(R, U)                               # U.vector is updated; U.split() gives the fields
```

Boundary data on the fields work in the same way. The unknown is then a `ProductFunction` of `P.unconstrained`, and the data are those given with the fields, or `left=` and `right=` given to `fd.newton`, each a list with one entry per field and `None` for a field without data, as in `P.lift`. Newton moves the free coefficients of $P$ only, so the data of every field hold exactly at every iterate. All the `jv=` and `precond=` options of [Section 7.5](#75-newton-krylov) work on systems, and so does `fd.residual_function`.

```python
V = fd.SplineSpace(grid, 3, bc="dirichlet", left=1.0, right=2.0)
W = fd.LagrangeSpace(grid, 3, bc=("dirichlet", "free"), left=1.0)
P = fd.ProductSpace(V, W)
U = fd.Functions(P.unconstrained, "U")        # can hold the data of both fields
a, b = U
p, q = fd.TestFunctions(P)
R = fd.form((D(a)*D(p) + a*b*p - p + D(b)*D(q) + b**3*q - a*q) * dx)
fd.newton(R, U)                               # a(0) = 1, a(1) = 2 and b(0) = 1 at every iterate
fd.newton(R, U, left=[1.0, 1.0], right=[2.0, None])   # the same data given in the call
```

### 7.5 Newton-Krylov

`linear="gmres"` or `"lgmres"` solves the same Newton systems by Krylov iteration, with the Jacobian-vector product chosen by `jv=`:

| `jv=` | $J v$ from | cost per product |
|---|---|---|
| `"assembled"` (default) | $J$ assembled each step, banded kernel | one assembly per Newton step |
| `"exact"` | the action of `R.derivative(u)` on $v$, matrix-free | one rank-1 assembly |
| `"complex-step"` | $\mathrm{Im}\, R(u + ihv)/h$, $h = 10^{-20}$, exact to round-off | one complex residual |
| `"fd"` | a one-sided difference | one residual, $O(\sqrt{\varepsilon})$ accurate |

and the left preconditioner by `precond=`: `"frozen"` (the Jacobian at the first iterate, refactored when the iteration count grows, the default), `"linear"` (`R.derivative(u, part="constant")`, factored once), `"jacobian"` (every step, for testing), `None`, or a `LinearSolver`, `Matrix` or callable. `krylov_tol` bounds the preconditioned relative residual of each linear solve ([Section 7.1](#71-krylov-solvers)).

The complex step needs a residual that is analytic in $u$. Every operation of the form language is, and a residual form evaluates at complex field values directly: `R.assemble(u=z)` with a complex `z` gives a complex vector. For a system that is not a form, `fd.csnewton(F, x0, M=None)` runs the vendored complex-step Newton-Krylov solver on any NumPy function `F` that accepts complex arrays.

`fd.residual_function(R, u)` turns a residual form into such a function, for `fd.csnewton` or any other solver that works on plain arrays (`scipy.optimize.root`, a Newton loop of your own). It returns `F`, the residual as a function of the free coefficients $c$ of $V$, with the field $u = u_g + P c$ and the boundary data in the lift $u_g$, as `fd.newton` builds it internally, together with the initial guess `c0` taken from `u` and `set_u`, which writes a solution back into `u`. `F` accepts complex $c$ and leaves `u` alone. The boundary data come from the space, or from `left=`, `right=` (1D) and `dirichlet=` (2D) given to it.

```python
F, c0, set_u = fd.residual_function(R, u)
c, info = fd.csnewton(F, c0)            # matrix free: J v = Im F(c + i h v) / h
set_u(c)                                # u = u_g + P c, the boundary data included
```

The complex-step mode of `fd.newton` ([Section 7.2](#72-newtons-method)) does the same without leaving the form: `fd.newton(R, u, linear="gmres", jv="complex-step", precond=None, line_search=False)` takes the iterates of `fd.csnewton`.

In 1D the direct solve is already $O(np)$, so `linear="direct"` is usually the fastest. A Krylov solve pays when the Jacobian is expensive to assemble or to factor and a cheap operator is close to it.

### 7.6 Implicit Runge-Kutta time stepping and algebraic systems

`fd.newton` handles one unknown Function in one residual form. Two more entry points cover what does not fit that shape.

**`fd.IRK`** advances a semi-discrete system $M u' = f(t, u)$ by an implicit Runge-Kutta method. It builds the stage equations, their exact Jacobian and the Newton iteration, so a time loop is one call per step:

```python
M = fd.form(u*v*dx).assemble()
f = fd.form((D(w, 2)*D(v) - w*D(w)*v)*dx)        # KdV: f(w) = (w_xx, v_x) - (w w_x, v), w a Function of V
irk = fd.IRK(M, f, dt, method="gauss", stages=2)   # M u' = f(u)
for n in range(nsteps):
    info = irk.step(w)                             # w updated in place; info is a NewtonInfo
```

- **The right-hand side `rhs`** is usually one rank-1 `Form` in a Function of $V$, the whole of $f$, as `f` above in `w`. `fd.IRK` evaluates it with that Function replaced by the stage value $U$, `f.assemble(w=U)`, and takes the exact Jacobian from the form, `f.jacobian(w, w=U)`. Linear terms belong in the form too: the residual then never applies an assembled matrix, which is what keeps conserved quantities at round-off ([Section 7.7](#77-tolerances-round-off-and-conservation)).

  A list of terms, $f(U) = \sum_k c_k\,T_k(U)$, is for what one form cannot hold: an assembled `Matrix` $A$ (a linear term $A U$ with Jacobian $A$), a rank-2 `Form` (used as a `Matrix`, assembled once, or at every evaluation when it contains a Function or a Constant), or separately defined forms to combine with weights. Each entry is a term on its own ($c_k = 1$) or a pair `(c_k, T_k)`, so `[A, (-1.0, Nf)]` is $f(U) = A U - \mathcal N(U)$. A rank-1 form that contains no Function, such as `fd.form(fd.cos(t)*v*dx)`, is a source term. It is evaluated as it is and left out of the Jacobian, in `fd.IRK`, `fd.ERK` and `fd.SSPRK` alike. The Jacobian $f'(U)$ is then exact, the matrices plus `Form.jacobian` of each form. When the forms contain more than one Function, `unknown=` names the one that is the state. A callable `rhs` works too, with `jacobian=` a callable returning a `Matrix`.
- **The method.** `"gauss"` (Gauss-Legendre, order $2s$, symplectic, conserves every quadratic invariant) or `"radau"` (Radau IIA, order $2s - 1$, L-stable, stiffly accurate), with $s$ = 1, 2 or 3 stages, or `method=(A, b)` for any other tableau. `fd.butcher(method, s)` returns $(A, b, c)$.
- **The Newton iteration.** The $s$ stages are solved together. Their Jacobian $\delta_{ij} M - \Delta t\,a_{ij} f'(U_i)$ is assembled by `fd.block` into one banded matrix on `ProductSpace(V, ..., V)`, so each iteration is one direct solve. `newton="exact"` (the default in 1D, see [Section 12.6](#126-implicit-time-stepping) for 2D) re-evaluates it at every iteration, the classical Newton method. `newton="simplified"` factors it once per step at $u^n$ and reuses it, more iterations at a lower cost each. `newton="frozen"` keeps that factorization from step to step, renews it at once when Newton fails, and at the next step when Newton slows down (`refresh=`, the same rule as in 2D, [Section 12.6](#126-implicit-time-stepping)). A linear problem with a fixed step is then factored once. For a BBM solitary wave on 801 cubic splines, Gauss 2, 50 steps, `"exact"` takes 0.21 s (149 factorizations), `"simplified"` 0.10 s (50) and `"frozen"` 0.06 s (6). `irk.factorizations` counts them. The stages of one step start the next. `tol` (default $10^{-12}$), `rtol` and `maxiter` (default 20) apply to the stage residual, and the line search of `fd.newton` is off unless `line_search=True` is given, and a step whose Newton iteration fails raises with the reason. `tol` is absolute, and the round-off floor of $\|F\|$ grows with the problem (about $2\cdot 10^{-13}$ for KdV at $n = 400$, $3\cdot 10^{-10}$ at $n = 8000$). `xtol` (default $10^{-12}$) handles that. Newton also stops when the step says the stages are converged, that is when $\frac{\theta}{1-\theta}\|d_k\| \le \texttt{xtol}\,\|K\|$ with the contraction rate $\theta = \|d_k\| / \|d_{k-1}\|$ (as in Hairer and Wanner's IRK codes), or when a full step leaves $\|F\|$ where it was while $\|d_k\| \le \sqrt{\texttt{xtol}}\,\|K\|$. So large problems need no retuning of `tol`, and `xtol=0` switches this off ([Section 7.7](#77-tolerances-round-off-and-conservation)).
- **Time.** For $M u' = f(t, u)$, write $t$ as a `fd.Constant` in the forms of `rhs` and pass it as `time=`. `fd.IRK` sets it to the stage time $t_n + c_i \Delta t$ before every evaluation of $f$ and of its Jacobian, and to $t_{n+1}$ after the step. `t0=` is the initial time, and `irk.t` the current one. A callable `rhs` may take `(t, U)` instead of `U`, and its `jacobian` likewise.
- **Boundary data.** The data on the conditions built into $V$ are the values given with the space, `left=` and `right=`, each a number, a tuple, or a callable of $t$ ([Section 4](#4-boundary-conditions)). The same keywords given to `fd.IRK` replace them. The field is $u = u_g(t) + P c$ with the lift $u_g(t)$ = `V.lift(left=g_L(t), right=g_R(t))`, and the stages solve for the coefficients $c$ in $V$,
  $$M c' = f\left(t, u_g(t) + P c\right) - M u_g'(t).$$
  The unknown `u` is then a Function of `V.unconstrained`, `M` must be a rank-2 `Form` (it acts on $u_g'$), and the right-hand side must be forms in a Function of `V.unconstrained`, as for `fd.newton` ([Section 7.4](#74-boundary-data-and-coupled-systems)). $u_g'$ comes from the data by the complex step $\mathrm{Im}\, g(t + ih)/h$, exact to round-off for data written with NumPy, and by a central difference for a callable that does not accept a complex $t$. The data hold exactly at every stage and every step.

```python
V = fd.SplineSpace(grid, 3, bc="dirichlet",
                   left=lambda t: np.cos(t), right=lambda t: 2*np.cos(t) + np.sin(t))   # data in time
u, v = fd.TrialFunction(V), fd.TestFunction(V)
w = fd.Function(V.unconstrained)                     # the state carries the data
t = fd.Constant(0.0)
M = fd.form(u*v*dx)
src = fd.exp(-t) * fd.sin(np.pi * fd.x)              # a source f(x, t), an expression in fd.x and t
R = fd.form((-D(w)*D(v) + src*v) * dx)               # u_t = u_xx + f(x, t)
irk = fd.IRK(M, R, dt, "radau", 3, time=t)          # the boundary data come from V
un = fd.Function(V.unconstrained).interpolate(u0)
for n in range(nsteps):
    irk.step(un)                                     # un(0) = cos(t), un(1) = 2 cos(t) + sin(t) exactly
```

- **Systems.** `M` may be a matrix or form on a `ProductSpace` $W$. The state is then a `ProductFunction` of $W$ and the forms of `rhs` are written in its components, as in [Section 6.10](#610-interior-facets-ds-jump-avg). The stages live on one `ProductSpace` holding every field of $W$ once per stage, numbered by position, so each Newton iteration is still one banded solve. Boundary data are given per field, `left=[g_0, None]`, with `None` for a field without data. The state is then a `ProductFunction` of `W.unconstrained`, and the data of each field hold exactly.

  With a time-dependent right-hand side the methods keep their classical order, $2s$ for Gauss and $2s - 1$ for Radau IIA. With time-dependent boundary data a stiff problem shows the usual order reduction of Runge-Kutta methods. For the heat equation above on four cubic elements, Gauss with 2 stages approaches order 4 as $\Delta t$ falls, Radau IIA with 2 stages keeps order 3, and Radau IIA with 3 stages gives about 4.2 instead of 5.

For the KdV solitary wave with 400 cubic elements and $\Delta t = 0.05$ there are two Newton iterations per step, $\|F\| = 2.4\cdot 10^{-2} \to 3.0\cdot 10^{-7} \to 1.7\cdot 10^{-13}$, and a full cycle takes 1.9 s.

**`fd.newton_system(F, J, x0)`** is Newton's method for any algebraic system $F(x) = 0$. `J(x)` returns the Jacobian as a `Matrix` (factored with `backend`), a `LinearSolver` (used as it is, so returning the same one every time is the simplified method), a SciPy sparse matrix, or a dense array. A SciPy matrix is factored by FEMd's sparse LU and a dense array by its dense LU with partial pivoting, so neither goes through SciPy or NumPy. It has the same line search, stopping test and `NewtonInfo` as `fd.newton`, and returns `(x, info)`. It also takes the step test `xtol=` that `fd.IRK` uses (default 0, off). The iteration itself, the norms, the line search and the stopping tests, is one C++ routine, `newton_solve` in `solve/newton.hpp`, shared by `fd.newton`, `fd.newton_system` and the stage equations of `fd.IRK` (`timestep/implicit_rk.hpp`, the class `ImplicitRK`, which also forms the stage values and residuals and the update $u^{n+1} = u^n + \Delta t \sum_i b_i K_i$ in C++). Python supplies the callables: the residual, which assembles the form at the current iterate (with the lift of the boundary data), and the direction, which factors the Jacobian or runs GMRES on it, or hands over the frozen stage factorization.

The pieces stay available for a scheme neither of these covers. For instance the stage Jacobian of `fd.IRK` is written with `fd.block` from `Nf.jacobian(w, w=U)`, the Jacobian of the form at a given coefficient vector.

### 7.7 Tolerances, round-off and conservation

**The residual has a floor.** $\|R\|$ cannot fall below the rounding error of evaluating it, roughly the machine precision times the largest terms of the operator times $\sqrt{n}$. For the KdV stages that floor is about $2 \cdot 10^{-13}$. A tolerance below it makes Newton stall at the floor until `maxiter`. Since convergence is quadratic, one iteration takes $10^{-7}$ to $10^{-13}$, and a tolerance a little above the floor costs nothing extra. In `fd.IRK` the default `xtol` stops on the step instead, so a `tol` below the floor is harmless there.

**Invariants drift with the operators, not with the tolerance.** A scheme that conserves a quantity exactly does so through discrete identities. For KdV with $S_{ij} = (N_j'', N_i')$, the mass needs the column sums of $S$ to vanish and the $L^2$ norm needs $S$ to be skew-symmetric. Tightening the Newton tolerance does not touch these.

**Apply operators in the residual through forms, not assembled matrices.** `S @ U` and the form $(U_{xx}, v_x)$ are equal in exact arithmetic but not in floating point. The entries $S_{ij}$ are $O(h^{-2})$ and each is rounded with an error of about $\varepsilon h^{-2}$, so the identities above hold only to that error, a fixed pattern. A solitary wave has positive coefficients, so in $e^{\mathsf T} S U$ and $U^{\mathsf T} S U$ the pattern adds up with the same sign step after step. The form instead sums $U_{xx} = \sum_j U_j N_j''$ at each quadrature point first, an $O(1)$ number, so the $O(h^{-2})$ values never appear, and its rounding errors multiply $U_{xx}$, which changes sign and cancels. For the solitary wave at 81 positions, one evaluation of the mass identity was off by $1.2 \cdot 10^{-13}$ on average with the matrix (the same sign at 80% of the positions) and by $1.8 \cdot 10^{-17}$ with the form. With the matrix it grows like $h^{-3}$ under refinement ($2 \cdot 10^{-14}$ at 200 elements, $3 \cdot 10^{-11}$ at 1600). With the form it stays below $10^{-15}$.

So write the residual with forms, and keep assembled matrices for Jacobians and preconditioners, where rounding only changes the speed of Newton's method. With `fd.IRK`:

```python
f = fd.form((D(w, 2)*D(v) - w*D(w)*v) * dx)        # both terms in one form in w
irk = fd.IRK(M, f, dt)                              # not [S, (-1.0, Nf)] with S assembled
```

The Jacobian of the linear part of the form is the assembled $S$, so Newton is unchanged (two iterations per step), and the relative drift in mass and $L^2$ norm over one cycle falls from $8.4 \cdot 10^{-13}$ to $9 \cdot 10^{-15}$, for about 10% more time.

When an assembled matrix must be applied in the residual, imposing the identity in floating point also works: `S = 0.5 * (S - S.T)` is exactly skew and `M = 0.5 * (M + M.T)` exactly symmetric. For KdV this gives about $2 \cdot 10^{-14}$. More accurate entries do not help. Building the quadrature tables in coordinates local to each element (tried on 2026-09-25 and reverted) shrank the largest entry of $S + S^{\mathsf T}$ from $1.6 \cdot 10^{-11}$ to $2.9 \cdot 10^{-14}$, and the drift stayed at $10^{-12}$, because what accumulates is the sign-coherent part of the rounding, not its size.

### 7.8 Explicit SSP Runge-Kutta and limiting

**`fd.SSPRK`** advances $M u' = f(t, u)$ explicitly by a strong-stability-preserving Runge-Kutta method, for DG and other hyperbolic problems where an implicit solve does not pay. It takes `M` and `rhs` as `fd.IRK` does (a Matrix, a 2D `SparseMatrix` or a rank-2 form of a space or a `ProductSpace`, and one term, a list of terms, or a callable), needs no Jacobian, and factors `M` once. A rank-1 term without the unknown is a source. On triangles it runs DG with `fd.VertexLimiter` ([Section 12.9](#129-discontinuous-galerkin-on-triangles)). The Legendre DG mass matrix is diagonal, so every solve is a division ([Section 7](#7-solving)).

```python
lim = fd.TVBLimiter(V, M=50.0)                       # optional
rk = fd.SSPRK(fd.form(u*v*dx), R, dt, order=3, limiter=lim)
for n in range(nsteps):
    rk.step(w)                                       # w a Function or ProductFunction, in place
```

| `order`, `stages` | method | SSP coefficient `rk.cfl` |
|---|---|---|
| 1, 1 | forward Euler | 1 |
| 2, 2 | Heun | 1 |
| 3, 3 (default) | Shu-Osher | 1 |
| 3, 4 | four stages | 2 |
| 4, 10 (`order=4`) | Ketcheson, low storage | 6 |

Each method is a convex combination of forward Euler steps. A property forward Euler has for $\Delta t \le \Delta t_{FE}$ (TVD in the means, a maximum principle, positivity) therefore holds for $\Delta t \le$ `rk.cfl` $\cdot \Delta t_{FE}$. For DG of degree $p$ with the Lax-Friedrichs flux, $\Delta t \approx 0.3\,h / \left((2p+1)\alpha\right)$ is a safe start. `time=` and `t0=` work as for `fd.IRK`, and the stages are evaluated at their own times, so a time-dependent source keeps the full order. `limiter=` is any callable, array in and array out, applied after every stage.

**`fd.TVBLimiter(V, M=0.0)`** is the Cockburn-Shu TVB minmod limiter for a `DGSpace` of any degree and either basis. On each element it compares the jumps from the mean to the two end values with the jumps between neighboring means. When their minmod changes an end jump, the element keeps its mean, its slope becomes the minmod of the slope and the two mean jumps, and its higher modes are dropped. The TVB constant skips elements whose end jumps are below $M h^2$, which leaves smooth extrema alone when $M$ is about $|u_{xx}|$ there. `M = 0` is plain minmod, which also clips smooth extrema and costs an order of accuracy at them. The means never change, so the limiter conserves mass. On a `ProductSpace` it limits every DG field component by component (`fields=` picks some) and leaves the other fields alone. Characteristic limiting for systems is not provided. Call it on a Function to limit in place, `lim(u)`, or on an array for a limited copy. `lim.troubled` counts the elements the last call limited. The limiting runs in C++ (`fe/limiters.hpp`), in parallel over the elements, with the same result for any number of threads; so does `fd.SlopeLimiter`.

For Burgers with $\sin(2\pi x)$ on 100 elements and $M = 50$: before the shock the limited and unlimited solutions are identical (errors $2.5\cdot 10^{-3}$, $1.4\cdot 10^{-4}$, $4.3\cdot 10^{-6}$ for $p = 1, 2, 3$), and after it the solution stays in $\left[-0.96, 0.96\right]$ for every degree with the mass conserved to round-off. The Lobatto basis gives the same numbers.

**`fd.SlopeLimiter(V, limiter, reconstruction)`** gives a DG space the reconstructions of finite volume schemes. On every element the mean $W_j$ is kept, the slope is replaced by a limited slope $S_j$ of the means and the higher modes are dropped, so the end values are the reconstructed interface values $W_j + \tfrac12 S_j$ and $W_j - \tfrac12 S_j$. Each of the four limiters works with either reconstruction.

- **TVD2** applies the limiter to the differences of the means, $S_j = \phi(r_j)\, d_{j+1/2} W$ with $r_j = d_{j-1/2} W / d_{j+1/2} W$ and $d_{j+1/2} W = W_{j+1} - W_j$.
- **UNO2** applies it to the differences corrected by the second differences, $S_j = \phi(r_j)\, S_j^+$ with $r_j = S_j^- / S_j^+$, $S_j^\pm = d_{j\pm1/2} W \mp \tfrac12 D_{j\pm1/2} W$, $D_{j+1/2} W = m(D_j W, D_{j+1} W)$, $D_j W = W_{j+1} - 2W_j + W_{j-1}$ and $m$ the minmod function. With minmod this is $S_j = m(S_j^+, S_j^-)$, the reconstruction of Harten and Osher.

| `limiter` | | $\phi(r)$ |
|---|---|---|
| `"minmod"` | minmod | $\max\left(0, \min(1, r)\right)$ |
| `"vanleer"` | Van Leer | $(r + \lvert r\rvert)/(1 + \lvert r\rvert)$ |
| `"mc"` | monotonized central | $\max\left(0, \min\left((1 + r)/2, 2, 2r\right)\right)$ |
| `"vanalbada"` | Van Albada | $(r + r^2)/(1 + r^2)$ |

On a nonuniform grid the same formulas are used with divided differences. With elements of degree 1 and the default `troubled="all"`, the means evolve exactly as the cell averages of a finite volume scheme to round-off. So TVD2 is first order at smooth extrema and UNO2 second order. For $u_t + u_x = 0$ with $\sin 2\pi x$ on 40 to 320 elements, the maximum error decreases at a rate between 1 and 1.6 with TVD2 and at the rate 2 with UNO2, for every limiter.

```python
lim = fd.SlopeLimiter(V, "vanalbada", "uno2")             # every element, as in the finite volume scheme
lim = fd.SlopeLimiter(V, "mc", troubled="tvb", M=50.0)    # TVD2, only on the elements the TVB test flags
rk = fd.SSPRK(fd.form(u*v*dx), R, dt, order=3, limiter=lim)
```

### 7.9 Classical explicit Runge-Kutta

**`fd.ERK(M, rhs, dt, method="rk4")`** advances $M u' = f(t, u)$ by an explicit Runge-Kutta method, for problems that are not stiff. It takes `M` and `rhs` exactly as `fd.IRK` does, but needs no Jacobian and no Newton iteration. Each stage is one evaluation of $f$ and one solve with $M$, factored once. The stage loop runs in C++ (`include/femd/timestep/explicit_rk.hpp`).

```python
M = fd.form((u*v + D(u)*D(v)) * dx)
f = fd.form((-3*w*D(w)*v - (w*D(w, 2) + 0.5*D(w)**2) * D(v)) * dx)   # Camassa-Holm
rk = fd.ERK(M, f, dt, "rk4")
for n in range(nsteps):
    rk.step(w)                                   # w updated in place, rk.t advanced
```

| `method` | stages | order | tableau |
|---|---|---|---|
| `"euler"` | 1 | 1 | forward Euler |
| `"midpoint"`, `"rk2"` | 2 | 2 | explicit midpoint rule |
| `"heun"` | 2 | 2 | explicit trapezoidal rule (the same as `fd.SSPRK(order=2)`) |
| `"ralston"` | 2 | 2 | Ralston's method, the smallest error constant of order 2 |
| `"rk3"` | 3 | 3 | Kutta's third-order method |
| `"rk4"` | 4 | 4 | the classical fourth-order method |
| `"3/8"` | 4 | 4 | Kutta's 3/8 rule |

Any other explicit tableau is given as `(A, b)` or `(A, b, c)`, with `A` strictly lower triangular. `fd.butcher("rk4")` returns the named ones.

- **Mass matrices.** `M` is a `Matrix` (1D, the factored solve then runs in C++), a `SparseMatrix` (2D, sparse Cholesky when symmetric positive definite, the pivoting $LDL^T$ when symmetric otherwise, the sparse LU when nonsymmetric) or a rank-2 `Form`, of a space or a `ProductSpace`. `M=None` with a callable `rhs` integrates the plain ODE $u' = f(t, u)$ on NumPy arrays.
- **Krylov mass solves.** `backend="cg"`, `"gmres"` or `"lgmres"` solves $M k = f$ at every stage iteratively instead of factoring $M$, in 1D and 2D alike. `solver_options=` sets `precond` (`"jacobi"` by default, `"ssor"`, `"ilu0"`, `None` or a preconditioner object), `tol` (the relative residual of every solve, `1e-12`), `maxiter`, `restart` and `k_aug` (GMRES, LGMRES) and `warm_start` (`True`: each solve starts from the previous stage's rate, which differs by $O(\Delta t)$). A mass matrix is spectrally equivalent to its diagonal, so Jacobi-preconditioned CG needs a fixed number of iterations however fine the mesh: 10 per solve for $P_2$ at 40 401 and at 160 801 unknowns. There it matched the sparse Cholesky per step (0.085 s against 0.080 s for RK4 at 160 801 unknowns) with no factorization (0.56 s saved) and no fill, which is what makes it worth having for large meshes and for memory. Warm starting saves about 40% of the iterations. `rk.solver.iterations` lists the iterations of every solve, `rk.solver.info` the last `KrylovInfo`, and a solve that misses `tol` raises with the reason.

  ```python
  rk = fd.ERK(M, f, dt, "rk4", backend="cg")                                   # Jacobi, tol 1e-12
  rk = fd.ERK(M, f, dt, "rk4", backend="cg", solver_options=dict(precond="ilu0", tol=1e-13))
  ```
- **Time.** `time=` is a `fd.Constant` in the forms, set to $t_n + c_i \Delta t$ before each stage. A callable `rhs(t, U)` receives $t$ directly.
- **Stability.** An explicit method is stable only below a step limit set by the fastest mode of $M^{-1} f'$, a CFL condition $\Delta t \lesssim h / |a|$ for advection and $\Delta t \lesssim h^2$ for diffusion. For stiff problems use `fd.IRK` with Radau IIA, and `fd.SSPRK` when a limiter or a strong stability property is needed. Unlike Gauss, the explicit methods do not conserve quadratic invariants exactly.

For Camassa-Holm on 400 cubic elements up to $t = 20$, RK4 costs 0.14 s for 400 steps of $\Delta t = 0.05$ against 1.34 s for Gauss with 2 stages. It is less accurate at the same step, since the steepening peaks bring RK4 close to its stability limit. The error against a fine Gauss solution is $2.7\cdot10^{-1}$, $1.2\cdot10^{-2}$ and $4.2\cdot10^{-4}$ for $\Delta t = 0.05$, $0.025$ and $0.0125$, against $2.7\cdot10^{-3}$ for Gauss at $\Delta t = 0.05$. RK4 at $\Delta t = 0.0125$ is then both cheaper and more accurate.

### 7.10 Eigenproblems

**`fd.eigs(a, m, k=6, sigma=0.0)`** computes the $k$ eigenpairs of $A x = \lambda M x$ nearest the shift $\sigma$, for any space in 1D or 2D. `a` and `m` are bilinear forms, their expressions, or assembled matrices, and `m=None` is the standard problem $A x = \lambda x$, for a 2D `SparseMatrix` only. In 1D pass a matrix as `m`.

```python
V = fd.SplineSpace(np.linspace(0, np.pi, 41), 3, bc="dirichlet")
u, v = fd.TrialFunction(V), fd.TestFunction(V)
r = fd.eigs(fd.D(u) * fd.D(v) * fd.dx, u * v * fd.dx, k=5)
r.values                  # 1, 4, 9, 16, 25 (to 3e-6)
lam, X = r                # the eigenvalues and the coefficient vectors, columns of X
modes = r.functions       # one Function per eigenvector
```

It is shift-invert Arnoldi in C++ (`solve/eigen.hpp`: full reorthogonalization, explicit restarts, the $k$ Ritz pairs of largest magnitude kept from one restart to the next and deflated once converged; in the $M$ inner product for symmetric problems, where it is Lanczos and the values come out real), with the inner solve $(A - \sigma M)^{-1}$ done by FEMd's own factorization of the shifted matrix: banded or cyclic in 1D, sparse Cholesky, the pivoting $LDL^T$ or the sparse LU in 2D. One factorization serves the whole iteration.

- **The shift.** The result is the $k$ eigenvalues nearest $\sigma$, sorted by their distance to it. $\sigma$ must not be an eigenvalue. For an operator with a large kernel, such as curl-curl, put $\sigma$ inside the spectrum of interest, so that the kernel is farther from it than the eigenvalues wanted ([Section 12.8](#128-mixed-and-hcurl-problems)).
- **Symmetry.** `symmetric=None` measures $A$ and $M$. A symmetric problem needs $M$ positive definite. A nonsymmetric one may return complex values, and real ones are returned as real.
- **Checks.** `r.residuals[i]` is the backward error $\|Ax - \lambda Mx\| / ((\|A\|_1 + |\lambda|\,\|M\|_1)\|x\|)$ of each pair. `which="largest"` iterates on $M^{-1}A$ (no shifted factorization) and `"smallest"` is the shift $0$; `tol=` is the relative residual tolerance of a Ritz pair ($10^{-10}$ by default), `maxiter=` the number of restarts (300), `ncv=` the Arnoldi vectors per restart ($\max(2k + 1, 20)$). `vectors=False` skips the eigenvectors, and `backend=` picks the solver of the shifted matrix. The starting vector `v0=` is a fixed pseudo-random one by default, so a call returns the same result every time.
- **Clusters.** An eigenvalue of very high multiplicity near the shift, such as the kernel of curl-curl, slows the iteration and can fill the result with copies of it. Ask for the eigenvalues on the far side of the shift from it.

## 8. A complete example

The BBM equation $u_t - u_{xxt} + u_x + u\,u_x = 0$, RK4:

```python
import numpy as np, femd as fd
from femd import D, dx

V = fd.SplineSpace(np.linspace(-60, 60, 601), 3, bc="periodic")
u, v, w = fd.TrialFunction(V), fd.TestFunction(V), fd.Function(V)

solve = fd.form((u*v + D(u)*D(v)) * dx).assemble().solver()   # (1 - d_xx), factored once
R = fd.form(-(D(w) + w*D(w)) * v * dx)                          # -(u_x + u u_x, v)

def rhs(state):
    w.assign(state); return solve.solve(R.assemble())

un = fd.Function(V).project(lambda x: 1.5 / np.cosh(0.5 * np.sqrt(1/3) * x) ** 2)
dt = 0.01
for n in range(2000):
    k1 = rhs(un); k2 = rhs(un + 0.5*dt*k1); k3 = rhs(un + 0.5*dt*k2); k4 = rhs(un + dt*k3)
    un.assign(un + dt/6 * (k1 + 2*k2 + 2*k3 + k4))
```

## 9. Low-level assembly

The form language sits on three kernels that you can call directly. Terms are `(a, b, coeff)` for $\int c\,(D^a N_i)(D^b N_j)$ and `(a, coeff)` for $\int c\,D^a N_i$, with `coeff` a number or an array of values at the cache nodes.

```python
Q = V.cache(npts=5, nder=1)                        # tables at 5 Gauss points, values and first derivatives
A = fd.assemble(Q, [(0, 0, 1.0), (1, 1, 1.0)])    # (u,v) + (u',v')
b = fd.assemble_vector(Q, [(0, f(Q.nodes()))], V.dim)
s = fd.assemble_scalar(Q, g(Q.nodes()))
wq, wxq = w.at_quad(Q, 0), w.at_quad(Q, 1)        # a known field at the nodes
r = -fd.assemble_vector(Q, [(0, wxq + wq*wxq)], V.dim)
```

Every form ultimately becomes calls like these.

## 10. Two-dimensional meshes

The meshes of this section carry the spaces of [Section 11](#11-spaces-in-two-dimensions). `triangulate` takes a polygon and returns a quality-guaranteed Delaunay triangulation of it: arrays of points and triangles, full neighbour topology, and markers carried through from the input rings.

```python
m = fd.triangulate([(0, 0), (2, 0), (2, 1), (0, 1)],      # outer polygon
                   holes=[fd.circle(1.0, 0.5, 0.2, 24)],
                   min_angle=30, max_area=2e-3, smooth=2)

m.points          # (npoints, 2) float64
m.triangles       # (ntriangles, 3) int32, counter-clockwise
m.point_markers   # (npoints,) int32; 0 interior, otherwise the ring's marker
```

The pipeline is Bowyer-Watson for the initial Delaunay triangulation, midpoint splitting to make the input segments appear as edges, a flood fill to discard the outside, and Ruppert refinement to reach the shape and size targets.

**`min_angle` is in true degrees, and has a ceiling.** Ruppert's algorithm is proved to terminate up to $20.7^\circ$ and in practice reaches about $33^\circ$; $34^\circ$ and above is refused up front. Geometry imposes its own limit: a corner of the input polygon with interior angle $\alpha$ forces a triangle with an angle of at most $\alpha$, so asking for more than the sharpest input corner is refused too, with the corner named. Nothing here runs until it exhausts memory; a target that is merely expensive stops at `max_points` with a message saying which criterion is still failing.

**Corner triangles.** By default the triangle filling a corner of the polygon often has two of its edges on the boundary, so all three of its vertices are constrained and the element carries no free degree of freedom once Dirichlet data is imposed everywhere. `interior_vertex=True` rules that out, by flipping the third edge to reach an interior vertex where that keeps the angle bound and by inserting a vertex where it does not. It halves the ceiling: two triangles must then share each corner, so one of them has an angle of at most $\alpha/2$. On the usual geometry with $90^\circ$ corners the ceiling is $45^\circ$ and this never bites.

The arrays are views onto the C++ mesh that owns them, so they cost nothing to take and must not be written to; `.copy()` one if you want to edit it. Full argument list in [the API reference](reference/meshing-2d.md#j-2d-meshing).

**Markers.** The outer boundary gets the marker `marker` (1 by default) and the holes `marker+1, marker+2, ...` in the order they are given, or the markers in `hole_markers=`. To give the edges of the outer polygon their own markers, list one per edge with `edge_markers=`, where edge $i$ joins vertex $i$ and vertex $i+1$ and the last edge closes the polygon. The holes then get the numbers after the largest edge marker:

```python
m = fd.triangulate([(-15, -6), (20, -6), (20, 6), (-15, 6)],     # bottom, right, top, left
                   holes=[fd.circle(4, 0, 1.0, 48)],             # marker 5
                   edge_markers=[1, 2, 3, 4], max_edge=0.3)
```

`np.unique(m.segment_markers)` lists the markers of a mesh, and `m.remark(rule)` changes them afterwards by a rule on the side midpoints.

### 10.1 Smoothing a mesh you already have

`smooth(mesh, passes)` runs Laplacian smoothing on a mesh without retriangulating it. `triangulate(..., smooth=n)` does the same as its final step, so this is for when you want more passes than you asked for, or want to smooth a mesh that has been refined since.

```python
m  = fd.triangulate(poly, min_angle=30, max_area=1e-3)     # smooth=0 by default
m2 = fd.smooth(m, 5)                                        # a new mesh, m untouched
```

Every interior vertex moves to the centroid of its one-ring, but only if the move leaves every incident triangle positive in area and makes neither the worst quality nor the worst angle of its fan worse. Smoothing therefore cannot tangle the mesh or undo the angle bound refinement established. Boundary vertices do not move, so the area is exact and the numbering is unchanged.

Three to five passes is the useful range: the worst quality reaches its final value by the third, and the vertex positions are at machine precision by twenty-five. On a mesh from `bisect_triangles` or `split_triangles`, pass `restore_delaunay=False` so the flip back to Delaunay does not undo their structure. `restore_delaunay(mesh)` does that flip on its own when you do want it.

**Inside `triangulate` the size bounds hold too.** A vertex moving to its centroid enlarges the triangles on the far side, and before 2026-09-29 `smooth=` left triangles up to a third larger than `max_area` and edges up to 13% longer than `max_edge`. There a move is now also rejected when it makes a triangle of the fan larger than `max_area`, or an edge at the vertex longer than `max_edge`. The Delaunay flips that end each pass can still enlarge a triangle, since the quadrilateral of a flip is fixed but its diagonal is not, so `triangulate` refines once more after smoothing. That adds one or two points on a mesh of a thousand and restores every guarantee: `min_angle`, `max_area`, `max_edge`, the Delaunay property, and with `interior_vertex=True` the interior vertex of every triangle. `fd.smooth` on its own knows no size bounds and keeps only the shape guards.

### 10.2 Refining a mesh you already have

Three schemes take a mesh and a set of triangles, and return a refined copy. They differ in what they guarantee, and the difference decides which is usable in an adaptive loop. Refining the same corner of a $30^\circ$ mesh repeatedly:

| level | `split_triangles` | | `bisect_triangles` | | `refine_triangles` | |
|---|---|---|---|---|---|---|
| | min angle | local $h$ | min angle | local $h$ | min angle | local $h$ |
| 0 | 31.21 | 0.1250 | 31.21 | 0.1250 | 31.21 | 0.1250 |
| 1 | 17.90 | **0.1250** | 26.70 | 0.0955 | 30.03 | 0.0955 |
| 2 | 6.34 | **0.1250** | 26.70 | 0.0785 | 30.01 | 0.0437 |
| 3 | 2.12 | **0.1250** | 26.70 | 0.0487 | 30.00 | 0.0264 |

`split_triangles` is the barycentric $1 \to 3$ split, `trichotomy` in the C meshgen. It is conforming on its own and touches nothing outside the triangle, which makes it the right answer to "split this one element". It is the wrong answer to "refine this region": each child keeps a whole edge of its parent, so the element **diameter never shrinks**, and the interpolation error of a finite element is governed by the diameter, not the area. Meanwhile the angles fall by about two thirds per level.

`bisect_triangles` is Rivara longest-edge bisection. It halves the longest edge, propagating to the neighbour whose longest edge it also is, and so on along the longest-edge propagation path; halving the terminal edge splits both its triangles at once, so no hanging node is ever created and no closure step is needed. It is non-degenerate: every angle it can produce is at least half the smallest angle of the mesh it started from, however many levels are applied, which is what the plateau at $26.70^\circ$ above shows. Growth is about $2\times$ per level, so refinement stays tightly local. The result is conforming but not Delaunay.

`refine_triangles` inserts circumcentres through the same Ruppert machinery `triangulate` uses, so it is the only one that keeps the mesh constrained Delaunay **and** keeps the quality guarantee. It refines hardest of the three, because each insertion can cascade through the quality loop.

All three **return a refined copy** and leave the mesh passed in untouched, because the arrays handed out are borrowed views and refining in place would leave any array you were still holding pointing at freed memory. Vertex numbering is append-only, so a solution vector indexed by vertex can be extended rather than re-projected:

```python
bad = eta > tol                              # an error indicator per triangle
m2 = fd.bisect_triangles(m, bad)             # or a list of indices
u2 = np.concatenate([u, np.zeros(m2.npoints - m.npoints)])   # old dofs keep their index
```

### 10.3 Structured meshes

`triangulate` decides where the vertices go. When you want to decide, build a structured mesh: a grid of $n_x \times n_y$ cells, each cut into triangles by a fixed rule.

```python
m = fd.rectangle_mesh(0, 0, 2, 1, 40, 20)                       # [0,2] x [0,1], 1600 triangles
m = fd.rectangle_mesh(0, 0, 2, 1, 40, 20, diagonal="crossed")   # 4 triangles per cell

arc = lambda r: (lambda s: np.c_[r*np.cos(s*np.pi/2), r*np.sin(s*np.pi/2)])
m = fd.mapped_mesh(bottom=[(1, 0), (2, 0)], right=arc(2.0),     # a quarter annulus
                   top=[(0, 1), (0, 2)], left=arc(1.0), nx=8, ny=24)
```

**Diagonals.** `"right"` cuts every cell along the `/` diagonal and `"left"` along `\`. Both are the classical test meshes, and both carry a directional bias that shows in anisotropic problems. `"alternate"` switches between them cell by cell like a checkerboard, which removes the bias. `"crossed"` adds a vertex at the center of each cell and makes four triangles, a mesh with the full symmetry of the grid. On square cells every pattern has all angles at least $45^\circ$.

**Curved regions.** `mapped_mesh` takes four sides meeting at four corners and fills the region by transfinite (Coons) interpolation of the unit square. A side is a callable $s \mapsto (x, y)$ on $[0, 1]$, an array of points, or two points for a straight side. The boundary vertices are the side samples exactly, and a straight-sided region gets its bilinear map, so a rectangle comes out as the exact tensor grid of the samples. A map that folds, because the sides cross or a cell is too distorted for the chosen diagonal, is refused with the first bad cell named. More cells or another diagonal usually cures the second case.

**Numbering and markers.** Grid vertex $(i, j)$ is point `j*(nx+1) + i`, so the grid is stored row by row from the bottom, and `crossed` puts the cell centers after the grid. Triangles follow cell by cell in the same order. This makes it easy to go back and forth between a mesh vector and a 2D array, `u.reshape(ny+1, nx+1)`. The four sides get the markers `(1, 2, 3, 4)` for bottom, right, top and left, or the `markers=` you pass. Each corner carries the marker of the side leaving it counter-clockwise, so $(0,0)$ is bottom and $(n_x, 0)$ right.

The result is an ordinary `Mesh2D`. `smooth`, `bisect_triangles`, `refine_triangles` and `validate` all work on it. Smoothing leaves a uniform grid where it is, since every vertex already sits at the centroid of its ring. Refinement keeps the numbering append-only as usual, but the grid structure is of course lost where it refines.

**A mesh from somewhere else.** `fd.mesh_from_arrays(points, triangles)` turns any triangulation, for instance one read from a file, into a `Mesh2D`. It builds the neighbor links, re-orients clockwise triangles, constrains every boundary edge and marks the boundary vertices. It refuses overlapping or folded input, edges shared by more than two triangles, degenerate triangles and unused points. The structured generators are built on it.

### 10.4 Removing vertices

`remove_vertices(mesh, vertices)` coarsens a mesh by taking vertices out. Each removed vertex leaves a hole, the polygon of its neighbors, and the hole is refilled with triangles on those neighbors. No other vertex moves and none is added.

```python
P = np.asarray(m.points)
far = np.hypot(P[:, 0] - 0.5, P[:, 1] - 0.5) > 0.6          # a boolean mask, or a list of indices
m2 = fd.remove_vertices(m, far, min_angle=20)
u2 = u[m2.report["vertex_map"] >= 0]                           # a nodal vector carries over
```

**What the fill is.** Each hole is refilled by Delaunay ear clipping. On a Delaunay mesh, which is what `triangulate` returns, the result is exactly the Delaunay triangulation of the vertices that remain, since deleting a vertex changes nothing outside the triangles around it. On a mesh that is not Delaunay, for instance after `bisect_triangles`, the fill picks the valid ears with the largest angles instead. Either way no triangle is ever inverted or degenerate.

**Which vertices can go.** Any interior vertex, unless it ends an interior segment. On the boundary, only a vertex in the middle of a straight side goes. Its two segments must carry the same marker, and its own marker must be that of the side (or of both its neighbors along the side). The two segments then merge into one, so the boundary and the area stay exactly as they were. Corners never go, and neither does a vertex that a marker singles out. `boundary=False` keeps every boundary vertex. A vertex that cannot go is kept, and `report["kept_reasons"]` says why. Removing one vertex can make a neighbor removable, so the list is swept until a sweep removes nothing.

**Quality.** Removing many neighboring vertices at once produces large triangles joined to small ones, and without a bound nothing stops that from making slivers. Removing everything outside a disc around a hole took a $28^\circ$ mesh down to $2.1^\circ$. `min_angle=`, `max_edge=` and `max_area=` refuse each removal whose new triangles would break them. With `min_angle=20` the same request kept 35 more vertices and ended at $20.4^\circ$.

**Numbering.** The removed points are dropped and the others renumbered in their original order. `report["vertex_map"]` gives, for every old vertex, its new index or $-1$. Because the surviving vertices do not move, a $P_1$ nodal vector restricts to the coarse mesh exactly by dropping entries, `u[vertex_map >= 0]`. This is interpolation onto the coarse mesh, not $L^2$ projection.

### 10.5 Quadrilateral meshes

A `QuadMesh` holds points, counter-clockwise convex quads, neighbours across each edge, and marked boundary segments, like `Mesh2D`.

| function | mesh |
|---|---|
| `rectangle_quad_mesh(x0, y0, x1, y1, nx, ny, markers=(1, 2, 3, 4))` | $n_x \times n_y$ rectangles, numbered as `rectangle_mesh` |
| `mapped_quad_mesh(bottom, right, top, left, nx, ny, markers=...)` | a four-sided region with curved sides, by transfinite interpolation (sides as in `mapped_mesh`) |
| `quadrangulate(boundary, holes, max_area=, max_edge=, min_angle=28, smooth=3, ...)` | any polygonal domain with holes, all quads |
| `quad_mesh_from_triangles(mesh, smooth=3)` | the all-quad mesh of any triangle mesh, for instance one refined locally |
| `quad_mesh_from_arrays(points, quads, segments=None, segment_markers=None, ...)` | from arrays, clockwise quads re-oriented, non-convex ones refused |

`quadrangulate` triangulates the domain with `triangulate`, splits every triangle into three quads at its edge midpoints and centroid, and smooths the interior vertices. Every quad of the split is convex, and a smoothing move that would break that is halved and then dropped, so the result is always valid. `max_area` and `max_edge` refer to the quads, and the triangles are made three times larger in area (twice as long in edge) to match. With the default `min_angle` the quad angles lie between about 30 and 145 degrees. Side markers carry over, and `m.remark(rule)` splits a ring into sides as `Mesh2D.remark` does for triangles ([Section 12.5](#125-systems-vector-fields-and-several-unknowns)).

## 11. Spaces in two dimensions

On a two-dimensional mesh `LagrangeSpace(mesh, k)` is the space of continuous piecewise polynomials of degree $k$: the $P_k$ element on the triangles of a `Mesh2D`, and the $Q_k$ element on the quadrilaterals of a `QuadMesh`. Everything in Sections [5](#5-functions) to [7](#7-solving) that is not tied to an interval carries over. Functions carry their coefficients, forms are written as inner products and compiled once, Jacobians are symbolic, and boundary data enter through a lift. Boundary conditions, the form language, the solvers and time stepping on these spaces are in [Section 12](#12-finite-elements-in-two-dimensions).

### 11.1 $P_k$ elements on triangles

With a `Mesh2D`, `LagrangeSpace(mesh, k)` is the classical $P_k$ element on the triangles, for any degree $k \ge 1$. The object it returns is a `LagrangeSpace2D`, the class that holds the 2D options, and `fd.FunctionSpace(mesh, "lagrange", k)` gives the same.

The nodes are the vertices, $k-1$ points on every edge and $(k-1)(k-2)/2$ points inside every triangle, so a triangle holds $(k+1)(k+2)/2$ of them. By default they sit on the uniform lattice. `nodes="lobatto"` moves them to Warburton's warp-and-blend points. These coincide with the equispaced points for $k \le 2$, put the 1D Gauss-Lobatto points on every edge, and keep the interpolation well conditioned at high degree. The basis is nodal, so the coefficients of a Function are its values at `V.dof_coordinates()` and `V.interpolate(f)` needs no solve. Internally each basis function is a combination of the orthonormal Dubiner polynomials, which keeps the evaluation accurate for any degree (tested to $k = 10$).

The option is the same on triangles and quads. The space, and so the Galerkin solution, is the same either way, which the tests check to $10^{-11}$. What changes is the basis. The uniform one is worse conditioned and interpolates worse at high degree: at $k = 8$ the $Q_k$ mass matrix on a $4 \times 4$ mesh with Dirichlet conditions has condition number 11 500 against 68, and at $k = 10$ the interpolation error of $1/(1 + 25 r^2)$ is 2.5 times larger. The choice matters from about $k = 5$, and for $k \le 2$ the two node sets are identical. The uniform nodes are the default, so pass `nodes="lobatto"` for high degree. The 1D `LagrangeSpace` has the same option ([Section 3.2](#32-lagrangespacegrid-degree-bcfree-nodesequispaced-leftnone-rightnone)).

| numbering | rule |
|---|---|
| vertex $v$ | node $v$ |
| node $j$ of edge $g$ | node $n_v + g(k-1) + j$, counted from the lower vertex index of the edge |
| interior node $j$ of triangle $c$ | node $n_v + n_e(k-1) + c\,(k-1)(k-2)/2 + j$ |

Two triangles that share an edge read its nodes in opposite directions. The reference edge nodes are symmetric, so both see the same points and the basis is continuous across the edge.

### 11.2 $Q_k$ elements on quadrilaterals

`fd.LagrangeSpace(quadmesh, k)` is the space of continuous $Q_k$ elements on a quadrilateral mesh. It is used exactly as the triangle space: the same forms, systems and vector spaces, Dirichlet elimination by marker, `ds(marker)`, the solvers and Newton. Interior-facet terms (`dS`) are implemented for triangles only.

The nodes are the tensor product of $k+1$ points in each direction, equally spaced by default and the Gauss-Lobatto points with `nodes="lobatto"`. That is $(k+1)^2$ per quad: the vertices, $k-1$ on each edge (the same points as on a triangle edge) and $(k-1)^2$ inside. The basis is the product of the 1D Lagrange polynomials on those points, so it is nodal without a Vandermonde inverse. With Lobatto nodes it is also well conditioned at high degree. The geometry is bilinear, so the cells need not be parallelograms. The Jacobian then varies inside a cell, and the quadrature cache stores $J^{-1}$ at every point. The mapped space contains every polynomial of total degree $k$ on any convex quad, and all of $Q_k$ on parallelograms, which is what the approximation needs.

Quadrature is the tensor Gauss rule. On quads the degree inference counts the degree **per variable**, where a derivative lowers the degree in one variable only, so a $Q_1$ stiffness matrix gets two points per direction and has no spurious (hourglass) modes.

### 11.3 Vector fields and systems

`fd.VectorFunctionSpace(mesh, k, ...)` is a vector field, one `LagrangeSpace(mesh, k, ...)` per component with the same conditions, and `fd.ProductSpace(V, Q, ...)` joins spaces on one 2D mesh into a system (it returns a `ProductSpace2D`). The unknowns are numbered block by block. Test, trial and known functions of a system unpack **by block**, so a vector block arrives as a vector expression and a scalar block as a scalar one.

```python
V = fd.VectorFunctionSpace(m, 2, dirichlet=[1, 3, 4])     # P2 velocity, one P2 space per component
Q = fd.LagrangeSpace(m, 1)                                 # P1 pressure
W = fd.ProductSpace(V, Q)                                  # the pair (u, p), numbered u first
```

**Known fields.** `fd.Function(W)` is a `ProductFunction`, and on a `VectorFunctionSpace` a `VectorFunction`, which is also a vector expression (`dot(grad(w), w)`, `div(w)`, `w[0]`). `u, p = w` unpacks it by block for forms, `w.split()` gives copies as Functions, `w.at(points)` evaluates a vector field and `w.plot()` draws its magnitude. `w.project([u0, p0])` and `w.interpolate([u0, p0])` set it from per-block data (`W.project` is the $L^2$ projection, and with a slip condition it projects the velocity components together).

The unpacked `u` and `p` are symbols, not arrays. They hold no coefficients of their own and point back to `w`, so a form built from them reads `w.vector` each time it is assembled. Whatever changes `w.vector` in place (`fd.newton(R, w)`, `rk.step(w)`, `w.assign(...)`, `w.vector[:] = ...`) is seen by every form that contains `u` or `p`, with no need to unpack again:

```python
w = fd.Function(W)
eta, vel = w                                   # symbols for forms, tied to w
mass = fd.form(eta * fd.dx)
rk.step(w)                                     # w.vector is overwritten in place
mass.assemble()                                # the mass of the new state
snapshot = w.split()[0].vector.copy()          # split() copies: later steps leave it alone
```

### 11.4 Raviart-Thomas, Nédélec and broken $P_k$

Three more families live on the triangles of a `Mesh2D`. They are what mixed methods and electromagnetics need.

| family | call | conformity | per triangle |
|---|---|---|---|
| Raviart-Thomas $RT_k$, $k \ge 1$ | `fd.FunctionSpace(m, "RT", k)` or `fd.RTSpace(m, k)` | $H(\mathrm{div})$, normal component continuous | $k(k+2)$ |
| Nédélec, first kind $N1curl_k$, $k \ge 1$ | `fd.FunctionSpace(m, "N1curl", k)` or `fd.N1curlSpace(m, k)` | $H(\mathrm{curl})$, tangential component continuous | $k(k+2)$ |
| broken $P_k$, $k \ge 0$ | `fd.FunctionSpace(m, "DG", k)` or `fd.DGSpace(m, k)` | none, every triangle its own | $(k+1)(k+2)/2$ |

The degree follows UFL. $RT_k$ and $N1curl_k$ contain $P_{k-1}^2$ and lie in $P_k^2$, so $RT_1$ and $N1curl_1$ are the lowest order, with one function per edge. $RT_k = P_{k-1}^2 + \mathbf{x}\,\tilde P_{k-1}$ and $N1curl_k = P_{k-1}^2 + (-y, x)\,\tilde P_{k-1}$, with $\tilde P_{k-1}$ the homogeneous polynomials of degree $k-1$. In 2D the second is the first turned by 90 degrees. The reference basis is the dual of the degrees of freedom, built by inverting a generalized Vandermonde matrix on the Dubiner polynomials as for $P_k$, and it is mapped to each triangle by the contravariant Piola map $\varphi = J\hat\varphi/\det J$ ($RT$) or the covariant one $\varphi = J^{-T}\hat\varphi$ ($N1curl$).

**Degrees of freedom.** On each edge, $k$ of them: the normal ($RT$) or tangential ($N1curl$) component times the edge length, at the $k$ Gauss-Legendre points of the edge. The direction is that of the global edge, from its lower to its higher vertex index, and the normal is that direction turned clockwise. A triangle whose own edge runs the other way reads the points in reverse and takes minus the global function. Inside each triangle (for $k \ge 2$) there are $k(k-1)$ moments against the polynomials of degree $k-2$. On the space these $k$ values are equivalent to the moments of the trace against $P_{k-1}$ on the edge. These are the coefficients of a Function. `V.interpolate(f)` is the canonical interpolant: it projects the trace on each edge onto $P_{k-1}$ (a Gauss rule of $k+3$ points) and takes the interior moments with a rule exact to degree $2k+4$. It reproduces $RT_k$ and $N1curl_k$ exactly, and it commutes with the derivative, so $\mathrm{div}$ of the $RT$ interpolant is the $L^2$ projection of $\mathrm{div}\, u$ onto $DG_{k-1}$ (the curl likewise for $N1curl$). A divergence-free field therefore stays divergence-free to quadrature accuracy.

**Vector arguments.** The test, trial and known functions of $RT$ and $N1curl$ are vectors of two components, so `fd.div(u)`, `fd.curl(u)`, `fd.dot(u, v)`, `u[0]` and `fd.grad(u)` act on them as on a `VectorFunctionSpace`. `fd.curl(u)` of a 2D vector is the scalar $\partial_x u_y - \partial_y u_x$, and of a scalar $s$ the vector $(\partial_y s, -\partial_x s)$ (`fd.rot` is the same). `fd.Function(V)` is a `VectorElementFunction`: `w.at(points)` gives $(n, 2)$ values, `w.at(points, what="div")` or `"curl"` the derivative, and `w.plot(what=...)` draws the magnitude with arrows, or the divergence or curl.

```python
V = fd.RTSpace(m, 2)
w = fd.Function(V).interpolate(lambda x, y: (1 + 2*x - y, 3*y + x))
w.at([[0.3, 0.4]]), w.at([[0.3, 0.4]], what="div")     # [[1.2, 1.5]] and [5.0]
fd.form(fd.div(w) * fd.dx).assemble()                   # 5.0, the integral over the unit square
```

**Essential conditions.** $u\cdot n$ is essential in $H(\mathrm{div})$ and $u \times n$ in $H(\mathrm{curl})$. `dirichlet=[markers]`, `bc="dirichlet"` and `dirichlet={marker: data}` eliminate the edge degrees of freedom on those sides, as for Lagrange nodes. The data is a vector field `lambda x, y: (gx, gy)`, whose normal or tangential part is used, or a scalar `g(x, y)`, the normal component $u\cdot n$ with the outward normal ($RT$) or the tangential component $u\cdot t$ with the boundary positively oriented, the domain on the left ($N1curl$). That is counter-clockwise on the outer boundary and clockwise around a hole. An interior side has no outward normal, so it takes vector data only. The edge values are $L^2$ projections of the data, so the flux or circulation through every side is exact. `V.lift(...)`, `fd.solve` and `fd.newton` use it as they use Dirichlet data. Natural data enter through `ds`, with `fd.FacetNormal()`.

**Broken $P_k$.** `fd.DGSpace(m, k)` on a `Mesh2D` returns a `DGSpace2D`: the nodal $P_k$ basis of `LagrangeSpace2D` on every triangle with nothing shared, so a Function is two-valued on every edge. It has no condition built in, and boundary data enter weakly through `ds`. Forms couple its triangles through interior facets (`dS`, `jump`, `avg`, `n('-')`), for DG schemes ([Section 12.9](#129-discontinuous-galerkin-on-triangles)), and it is also the scalar partner of mixed methods ($RT_{k} \times DG_{k-1}$).

All three join systems through `fd.ProductSpace(V, Q)`, where an $RT$ or $N1curl$ block unpacks as a vector and a DG block as a scalar ([Section 12.8](#128-mixed-and-hcurl-problems)). They are written to VTK with every cell's points apart, so the jumps show as they are. Quadrilaterals and curved edges are not supported for these families yet. Periodic sides are supported for `DGSpace` only ([Section 12.9](#129-discontinuous-galerkin-on-triangles)). The tests check the duality of the basis, the continuity of the normal or tangential component across every edge to $10^{-14}$ on distorted meshes with random vertex numbering, the interpolation and its commuting property, the exact sequence ($\nabla P_k \subset N1curl_k$, $\mathrm{rot}\, P_k \subset RT_k$), and the convergence rates below. They also compare the reference spaces with those of Basix for $k \le 5$, and the discrete spectra of div-div and curl-curl with scikit-fem's elements on the same mesh (agreement to $10^{-15}$), when those packages are installed.

## 12. Finite elements in two dimensions

A Poisson problem with mixed boundary conditions on a rectangle:

```python
import numpy as np, femd as fd

m = fd.rectangle_mesh(0, 0, 2, 1, 32, 16)          # markers 1, 2, 3, 4: bottom, right, top, left
V = fd.LagrangeSpace(m, 3, dirichlet=[1, 4])     # u given on the bottom and the left
u, v = fd.TrialFunction(V), fd.TestFunction(V)

f = 3 * fd.exp(fd.x) * fd.sin(2 * fd.y)            # g, h, g1, g4 below are data of the same kind
a = fd.dot(fd.grad(u), fd.grad(v)) * fd.dx + 3 * u * v * fd.ds(3)       # Robin on the top
L = f * v * fd.dx + g * v * fd.ds(2) + h * v * fd.ds(3)                 # Neumann on the right
uh = fd.solve(a, L, dirichlet={1: g1, 4: g4})      # numbers or callables g(x, y)
uh.plot(mesh=True)
```

### 12.1 Boundary conditions

A side of the mesh is named by its segment marker, the one `triangulate`, `rectangle_mesh` and `mapped_mesh` assign ([Section 10](#10-two-dimensional-meshes)). A Dirichlet condition is **built in by elimination**, as the 1D spaces build theirs in. The nodes on the named sides are not degrees of freedom of `V`, so `V.dim` counts only the free ones, matrices are square on them and stay symmetric positive definite for an elliptic form.

| constructor | condition |
|---|---|
| `LagrangeSpace(m, k)` | none built in. Every side is natural (homogeneous Neumann unless a `ds` term says otherwise). |
| `LagrangeSpace(m, k, bc="dirichlet")` | Dirichlet on every exterior side, whatever its marker. |
| `LagrangeSpace(m, k, dirichlet=[1, 3])` | Dirichlet on the sides with markers 1 and 3. |
| `LagrangeSpace(m, k, periodic=[(4, 2), (1, 3)])` | side 2 identified with side 4, side 3 with side 1 (below). |
| `LagrangeSpace(m, k, dirichlet={1: 0.0, 3: g})` | Dirichlet on sides 1 and 3, with their data. |
| `LagrangeSpace(m, k, bc="dirichlet", data=g)` | Dirichlet on every exterior side, with the data $g$. |

Data are carried by a lift, a Function of `V.unconstrained` that holds the boundary values on the eliminated nodes and zero elsewhere.

```python
ug = V.lift(0.5)                             # the same value on every Dirichlet side
ug = V.lift(lambda x, y: x * y)              # g(x, y), vectorized over arrays
ug = V.lift(fd.x * fd.y)                     # the same, as an expression in fd.x and fd.y
ug = V.lift({1: 0.0, 3: g3})                 # per marker, sides left out get zero data
```

Data given with the space, `dirichlet={marker: data}` or `data=` (anything `V.lift` takes), are used by every `fd.solve`, `fd.newton` and `fd.IRK` on the space and by `V.lift()`. Data given to a call, `fd.solve(a, L, dirichlet=...)`, apply to that call only and replace them. A callable $g(x, y, t)$ or $g(t)$ is data in time for `fd.IRK`. An expression in `fd.x`, `fd.y`, numbers and Constants, such as the inflow profile `(4*y*(1 - y), 0.0)` of a vector block, is data constant in time, and `V.interpolate` takes one too.

The solution of $a(u, v) = L(v)$ with this data is $u = u_0 + u_g$ with $u_0$ in $V$ solving $a(u_0, v) = L(v) - a(u_g, v)$, exactly as in [Section 4.1](#41-lifting-step-by-step). `fd.solve(a, L, dirichlet=...)` does those steps and returns a Function of `V.unconstrained`. Without `dirichlet=` it returns a Function of `V` with homogeneous data. Where two sides with different data meet, the corner takes the data of the larger marker.

**Periodic conditions** identify two sides. `periodic=(a, b)` makes side `b` a copy of side `a`, and a list of pairs combines several, so a rectangle from `rectangle_mesh` is doubly periodic with

```python
V = fd.LagrangeSpace(m, 3, periodic=[(4, 2), (1, 3)])       # left = right, bottom = top
V = fd.LagrangeSpace(m, 2, dirichlet=[1, 3], periodic=(4, 2)) # a channel, periodic in x
```

The two sides must match node for node after a translation, which is found from the sides themselves, so any parallelogram works (`V.periodic_info` gives the shift of each pair). The structured meshes, `rectangle_quad_mesh`, and `triangulate` on a rectangle all qualify. A mesh refined near one side only does not, and is refused with the node that has no partner. The nodes of side `b` are no degrees of freedom of their own. They share those of side `a`, the four corners of a doubly periodic rectangle become one node, and assembly adds the element contributions of identified nodes, so forms, solvers, `fd.IRK` and `fd.ERK` need nothing new. `V.dim` counts one node per class, $N^2 k^2$ for $P_k$ on an $N \times N$ torus. A side may not be both periodic and Dirichlet, but the two combine on different sides, and a class touching a Dirichlet side is eliminated. `VectorFunctionSpace(m, k, periodic=...)` passes it to every component, and it combines with `slip=` on the other sides. A doubly periodic operator such as $-\Delta$ has the constants in its kernel, so it needs a zeroth-order term or a mean-value constraint to be solvable. The tests find the rates $k + 1$ in $L^2$ for $k = 1, 2, 3$ on triangles and quads, and Gauss IRK advection on the torus keeps the mass and the $L^2$ norm to round-off.

Neumann and Robin data are natural, written with `ds`.

| measure | integrates over |
|---|---|
| `ds` | every exterior side |
| `ds(2)` | the sides with marker 2 |
| `ds((2, 3))` | the sides with marker 2 or 3 |

`fd.FacetNormal()` is the outward unit normal on those sides, so a flux is `fd.dot(q, fd.FacetNormal()) * v * fd.ds`.

### 12.2 The form language in 2D

| symbol | meaning |
|---|---|
| `fd.x`, `fd.y` | the coordinates |
| `fd.Dx(e, 0)`, `fd.Dx(e, 1)` | $\partial e / \partial x$, $\partial e / \partial y$ |
| `fd.grad(e)` | $(\partial_x e, \partial_y e)$, a vector expression |
| `fd.div(w)` | $\partial_x w_0 + \partial_y w_1$ |
| `fd.dot(a, b)`, `fd.inner(a, b)` | $a \cdot b$ for vectors, $ab$ for scalars |
| `fd.as_vector([a, b])` | a vector from two scalar expressions |
| `fd.curl(e)`, `fd.rot(e)` | $\partial_x e_1 - \partial_y e_0$ for a vector, $(\partial_y e, -\partial_x e)$ for a scalar |
| `fd.abs`, `fd.sign`, `fd.max_value(a, b)`, `fd.min_value(a, b)` | for numerical fluxes, with symbolic derivatives |
| `e('-')`, `e('+')`, `fd.jump(e)`, `fd.jump(e, n)`, `fd.avg(e)` | the two traces on an interior facet and their combinations, in `dS` terms on triangles ([Section 12.9](#129-discontinuous-galerkin-on-triangles)) |
| `fd.FacetNormal()` | the outward normal in `ds` terms, `n('-')` and `n('+')` in `dS` terms |

`D(u)` belongs to 1D meshes and is refused on a 2D space with a pointer to `grad` and `Dx`. A vector is not an integrand. Reduce it to a scalar with `dot`, `inner` or a component first. Only first derivatives enter a form on a 2D space, broken $P_k$ included, so a form with $\partial_x^2 u$ is refused. Integrate it by parts.

A Function is evaluated with `u.at(points)` for an `(n, 2)` array, or `u.at(x, y)` with arrays of any shape. `deriv="x"` or `deriv="y"` gives a derivative, and points outside the mesh give NaN. `u.plot()` draws it, and `V.triangulation(u)` returns a matplotlib `Triangulation` with the values for custom plots. A Function of a space on another mesh can enter a form. It is then evaluated at the quadrature points by point location.

The quadrature degree is inferred from the polynomial degree of every term, as in 1D, with a non-polynomial function counted as degree $k+1$ ([Section 6.7](#67-quadrature)). Low degrees use symmetric rules (1, 3 and 7 points, exact to degrees 1, 2 and 5). Higher degrees use a collapsed Gauss rule, exact at any degree. `dx(8)` or `ds(2, quad_degree=6)` override the inference.

### 12.3 Matrices and solvers

A bilinear form on a 2D space assembles into a `SparseMatrix`, FEMd's own compressed sparse row store. The pattern is built once per pair of spaces and shared, so a mass and a stiffness matrix of one space hold one copy of the indices and their sum adds two value arrays. `A.tocsr()` is a SciPy view of the same arrays, with no copy.

| solver | call | use |
|---|---|---|
| sparse Cholesky | `A.solver()` | the default for a symmetric matrix with a positive diagonal. FEMd's own $LDL^T$ with AMD ordering, in C++, factored once. |
| pivoting $LDL^T$ | `A.solver()`, `A.solver("ldlt")` | the default for a symmetric matrix that is not positive definite (a saddle point such as Stokes, Helmholtz), and the fallback when the Cholesky meets a pivot $\le 0$. FEMd's own multifrontal $LDL^T$ with $1\times1$ and $2\times2$ pivots, in C++. |
| sparse LU | `A.solver()`, `A.solver("lu")` | the default for a nonsymmetric matrix (convection-diffusion, the Navier-Stokes Jacobian): FEMd's own multifrontal LU with threshold pivoting, in C++, real or complex. Options `ordering=` (`"auto"`, `"ata"`, `"amd"`, `"natural"`) and `pivot_tol=` (0.1). |
| SuperLU | `A.solver("superlu")` | SciPy's LU, kept for comparison. SciPy's solver is looked up only here, so if a SciPy update breaks it, only this backend fails. |
| CG | `fd.cg(A, b, M=A.preconditioner("ilu0"))` | symmetric positive definite systems. Runs in C++. |
| GMRES, LGMRES | `fd.gmres(A, b, M=...)` | nonsymmetric systems. Runs in C++ with a native `A` and `M`. |
| iterative as a solver | `A.solver("cg", M="ssor", tol=1e-12)` | the same, behind the `solve()` interface |
| dense LU | `A.solver("dense")` | small problems and tests |

The preconditioners `"jacobi"`, `"ssor"` (with `omega`) and `"ilu0"` are built in C++ from the matrix and are applied without a round trip through Python. For a symmetric matrix SSOR and ILU(0) are symmetric, so either may precondition CG. ILU(0) of a $P_2$ or higher stiffness matrix is not guaranteed positive, and `P.min_pivot` reports the smallest pivot it met. SSOR and ILU(0) work on the matrix reordered by reverse Cuthill-McKee (`ordering="rcm"`, the default), which brings the entries they drop closer to the diagonal. On unstructured meshes ILU(0) then needs 30 to 45 per cent fewer CG iterations, and SSOR 10 to 20 per cent. `ordering="natural"` keeps the numbering of the space.

**Lumped mass.** `V.mass_matrix(lumped=True)` is a diagonal `SparseMatrix`. On a $Q_k$ space with `nodes="lobatto"` it is the tensor Gauss-Lobatto rule at the nodes, the entries of `u*v*dx(scheme="lobatto")`, whose off-diagonal entries are exactly zero on any quadrilateral mesh. On the other Lagrange and DG spaces it is the row sums of the mass matrix. $RT$ and $N1curl$ spaces have no lumped mass. These are positive for $P_1$ and for $Q_k$, but $P_k$ triangles from $k = 2$ are refused, since their vertex rows sum to zero. Systems (`VectorFunctionSpace`, `ProductSpace`) are lumped field by field, also with a slip condition. The eigenvalues of $-\Delta$ keep their order $2k$ ([Section 6.7](#67-quadrature)).

On the unit square with $P_2$ and $h = 1/150$ (89 401 unknowns) assembly takes 0.02 s, the sparse Cholesky 0.17 s, SuperLU 0.5 s and CG with ILU(0) 0.35 s (207 iterations, against 353 in the natural numbering), measured on the Linux VM.

**The sparse Cholesky.** `sparse/cholesky.hpp` factors $P A P^T = L D L^T$ with $L$ unit lower triangular, in three phases. The ordering $P$ is an approximate minimum degree ordering (`sparse/amd.hpp`, after Amestoy, Davis and Duff), with supervariables, mass elimination, aggressive absorption, dense rows held back, and a post-order of the assembly tree. The symbolic phase finds the elimination tree and the column counts of $L$. The numeric phase is up-looking, one row of $L$ per sparse triangular solve. No square roots are taken and no pivoting is done, which a symmetric positive definite matrix never needs. `S.factor` holds the factorization, with `nnz_L`, `flops`, `min_pivot`, `permutation()` and `refactor(K)` for new values on the same pattern. It is a `SparsePreconditioner`, so `fd.cg(A, b, M=S)`, and `fd.ERK` on a `SparseMatrix`, apply it in C++ without a Python round trip.

| BBM-BBM matrix (205 531 unknowns, P2, slip) | factor | one solve | stored factor |
|---|---|---|---|
| sparse Cholesky (AMD) | 0.81 s | 19 ms | $L$: 13.4 M entries |
| SuperLU (symmetric mode) | 6.6 s | 32 ms | $L$ and $U$: 27.3 M entries |

`positive_definite=False` (`A.solver("cholesky", positive_definite=False)`) accepts negative pivots and refuses only a zero one. That factors symmetric quasi-definite matrices, such as $\begin{pmatrix} K & G \\ G^T & -C \end{pmatrix}$ with $K$ and $C$ positive definite, but an indefinite matrix may then be factored inaccurately. The Stokes saddle point, with its zero pressure block, needs pivoting and goes to the pivoting $LDL^T$.

**The pivoting $LDL^T$.** A symmetric matrix that is not positive definite is factored by `sparse/ldlt.hpp`, a multifrontal $LDL^T$ after Duff and Reid. It uses the AMD ordering and the elimination tree of the Cholesky, and eliminates each supernode in a dense front. $D$ has $1\times1$ and $2\times2$ blocks, chosen by a threshold test that bounds the growth of the factors (`threshold=0.01` by default), so the factorization is stable for any nonsingular symmetric matrix. A variable that fails the test is delayed to the parent front. `S.factor.inertia` gives the numbers of positive, negative and zero eigenvalues. For Stokes these are the velocity and the pressure unknowns, which checks that the discrete problem is well posed. On a Taylor-Hood channel with 179 501 unknowns it factors in 1 s, where SuperLU takes 18 s.

**The sparse LU.** A nonsymmetric matrix is factored by `sparse/lu.hpp` with threshold pivoting: the pivot of a column is an entry at least `pivot_tol` (0.1) times the largest entry of the column, the diagonal when it passes and the largest candidate otherwise, which bounds the growth of the factors by $1/\texttt{pivot\_tol}$ per step. The default method (`method="frontal"`, `sparse/lu_frontal.hpp`) is multifrontal, after Duff and Reid and MA41: the pattern of $A + A^T$ is ordered by AMD, its elimination tree is grouped into supernodes, and each supernode's front, a dense square matrix on its variables, the rows and columns below them and the variables its children could not eliminate, is assembled from $A$ and the children's contribution blocks and eliminated in panels with dense updates. The pivot rows are chosen among the fully summed rows of the front only, so the fill is that of the symmetric pattern, and a column with no acceptable pivot there is delayed, with its row, to the parent front, where more of it is summed. `S.factor` reports `method`, `ordering`, `nnz_L`, `nnz_U`, `off_diagonal_pivots`, `delayed_pivots`, `fronts`, `max_front`, `min_pivot` and `max_pivot`. The alternative `method="columns"` is the left-looking column algorithm of Gilbert and Peierls, whose pivots may move anywhere; it orders the columns by AMD of $A + A^T$ when the diagonal is full and nonzero and by AMD of the pattern of $A^T A$ otherwise (`ordering="auto"`, the bound of Gilbert and Ng behind COLAMD), and suits a matrix whose pattern is far from symmetric. On the complex stage matrix of the vortex-shedding problem with 17 051 unknowns the frontal method fills 2.0 million entries against 3.3 for the column method and 4.5 for SuperLU, and factors in 0.08 s against 0.42 and 0.25, so a Radau step with one factorization went from 0.47 s to 0.12 s. The complex systems of `fd.IRK` go through it as two real matrices on one pattern. The solves run in parallel through the elimination tree of the pattern of $L + U$ ([Section 1.1](#11-openmp)): `S.factor` reports `subtrees`, `top_nodes`, `top_chains`, `top_levels`, `top_steps` and `top_fraction`, and `tree_contained`, which is `True` when the factors lie in the tree (always, by construction, and the plain sweeps would take over otherwise).

### 12.4 Accuracy

The tests solve $-\Delta u = f$ with $u = \sin \pi x \sin \pi y$ for $k = 1$ to $4$ and observe the optimal rates, $k+1$ in $L^2$ and $k$ in $H^1$. The same rates hold with Dirichlet data on two sides, Neumann data on a third and Robin data on the fourth, and for $u - \Delta u = f$ with a flux written through the normal. On a curved domain the boundary is the polygon of the mesh, which limits the accuracy to second order whatever $k$ is. Curved (isoparametric) elements are the next step.

**Quadrilaterals.** $-\Delta u = f$ with the rates $k+1$ in $L^2$ for $k = 1$ to $4$ on rectangles, on smoothly distorted non-parallelogram quads, and on `quadrangulate` meshes with holes (1.98, 3.11, 4.07 for $k = 1, 2, 3$ in the notebook). Taylor-Hood $Q_2/Q_1$ Stokes converges at 3 (velocity) and 2 (pressure). The cylinder benchmark of [Section 12.5](#125-systems-vector-fields-and-several-unknowns) on a quad mesh from the same refined triangulation gives $c_D = 5.5763$, $c_L = 0.01058$ and $\Delta p = 0.1175$.

### 12.5 Systems: vector fields and several unknowns

A system is written as on paper. Taylor-Hood elements for the Stokes equations, on the spaces of [Section 11.3](#113-vector-fields-and-systems):

```python
V = fd.VectorFunctionSpace(m, 2, dirichlet=[1, 3, 4])     # P2 velocity
Q = fd.LagrangeSpace(m, 1)                                 # P1 pressure
W = fd.ProductSpace(V, Q)                                  # Taylor-Hood
(u, p), (v, q) = fd.TrialFunctions(W), fd.TestFunctions(W)
a = (nu * fd.inner(fd.grad(u), fd.grad(v)) - p * fd.div(v) - q * fd.div(u)) * fd.dx
L = fd.dot(f, v) * fd.dx + fd.dot(g, v) * fd.ds(2)         # f, g vector expressions
w = fd.solve(a, L, dirichlet=[u_data, None])              # one entry per block
uh, ph = w.split()
```

**Vector and matrix expressions.** `grad` of a vector is the matrix $\nabla u_{ij} = \partial u_i / \partial x_j$, so `dot(grad(u), b)` is $(b \cdot \nabla) u$ and the convective term $(u\cdot\nabla)u \cdot v$ is `dot(dot(grad(u), u), v)`.

| expression | meaning |
|---|---|
| `grad(u)` | $\partial u_i / \partial x_j$ (a matrix) for a vector $u$ |
| `nabla_grad(u)` | its transpose, $\partial u_j / \partial x_i$ |
| `div(u)`, `div(A)` | $\partial_i u_i$, and the row divergence $\partial_j A_{ij}$ (so `div(grad(u))` is $\Delta u$) |
| `dot(a, b)` | vector-vector, matrix-vector, vector-matrix or matrix-matrix product |
| `inner(A, B)` | full contraction, $A : B$ for matrices |
| `outer(a, b)`, `transpose(A)`, `A.T`, `sym(A)`, `skew(A)`, `tr(A)`, `Identity(2)` | the usual |
| `as_vector([...])`, `as_matrix([[...], [...]])`, `u[i]`, `A[i, j]` | building and indexing |

Linear elasticity then reads as on paper:

```python
eps = lambda z: fd.sym(fd.grad(z))
sigma = lambda z: 2 * mu * eps(z) + lam * fd.tr(eps(z)) * fd.Identity(2)
a = fd.inner(sigma(u), eps(v)) * fd.dx
```

**Data.** A vector block takes a pair (one entry per component, each anything `LagrangeSpace.lift` takes), a callable returning the pair, or a dict `{marker: pair or callable}`. A system takes one entry per block, `None` for a block without data. `W.lift(...)`, `W.interpolate(...)`, `fd.solve(..., dirichlet=...)` and `fd.newton(..., dirichlet=...)` all read it this way. The data can also be given with the spaces, `VectorFunctionSpace(m, 2, dirichlet={3: (1.0, 0.0), 1: (0.0, 0.0)})` or `data=`, and a `ProductSpace` collects the data of its blocks.

**Newton.** A residual in $w = (u, p)$ of `W.unconstrained` (which carries the boundary data) is solved by `fd.newton(R, w, dirichlet=[u_data, None])`, with the exact Jacobian taken symbolically from the form, and the Newton-Krylov modes of [Section 7.5](#75-newton-krylov) work as well. Three checks:

| problem | result |
|---|---|
| Stokes, Taylor-Hood $P_2/P_1$, $N = 4 \ldots 32$ | rates 2.98 ($u$, $L^2$), 2.00 ($u$, $H^1$), 2.09 ($p$, $L^2$) |
| Navier-Stokes, Kovasznay flow, $Re = 40$ | Newton in 5 iterations, $\|R\|$ from $1$ to $7\cdot10^{-13}$, velocity rate 3 |
| 2D-1 cylinder benchmark (Schäfer and Turek), $Re = 20$, 74 826 unknowns | $c_D = 5.5764$ (ref. 5.5795), $c_L = 0.01061$ (0.010619), $\Delta p = 0.1178$ (0.11752), Newton in 7.7 s |

The benchmark computes the force on the cylinder from the residual, as the reaction of the discrete momentum equation at the cylinder nodes, which is much more accurate than integrating the traction over the polygonal surface. A pressure determined only up to a constant (velocity given on the whole boundary) makes the system singular. The pivoting $LDL^T$ still factors it and returns one solution, whose pressure carries an arbitrary constant. Subtract its mean afterwards, or keep one side natural, as in these examples. Pressure pinning and a mean-value constraint are not built in yet.

`Mesh2D.remark(rule)` gives a mesh new side markers from a rule on the side midpoints, which is how the benchmark splits the single outer ring of `triangulate` into walls, inlet and outlet.

**Slip walls.** On a slip wall the fluid slides but does not cross it, $u\cdot n = 0$, with the tangential traction natural. `VectorFunctionSpace(m, k, slip=[markers])` imposes it strongly:

```python
V = fd.VectorFunctionSpace(m, 2, dirichlet=[4], slip=[1, 3])   # u . n = 0 on sides 1 and 3
W = fd.ProductSpace(V, fd.LagrangeSpace(m, 1))                 # the condition carries into W
```

At every node on a slip side only the tangential component is a degree of freedom, $u_i = a_i t_i$, so $u\cdot n = 0$ holds exactly at the nodes. The coefficients of the system are $x_B = Z x$ with $Z$ sparse with orthonormal columns (`W.slip_matrix`, a `RectMatrix`). A form is assembled without the condition and reduced to $Z^T A Z$, a residual to $Z^T r$, both in C++ (the nodes, normals and $Z$ in `fe/slip.hpp`, the reduction in one pass over the rows by `triple_product`), so `solve`, `newton`, `ERK`, `lift`, `interpolate`, `split` and `plot` work unchanged. A free-slip wall needs no boundary term, and a prescribed tangential traction $g_t$ enters as `dot(g, v) * ds(marker)` (only $g\cdot t$ counts).

The normal at a node inside an edge is the edge's normal. At a vertex between two slip edges it is the length-weighted mean of their normals, the mass-conserving normal $n_i \propto \int\varphi_i n\, ds$, which makes $\int_\Gamma u_h\cdot n\, ds = 0$ exact. Where the two normals differ by more than `corner_angle` (45 degrees by default) the vertex is a corner, and both components are fixed, $u = 0$. A Dirichlet side wins at the nodes it shares with a slip side. `W.slip_info` lists the nodes, their normals and the corners.

| problem | result |
|---|---|
| Stokes P2/P1, channel tilted by 30 degrees, slip walls, $N = 4 \ldots 32$ | rates 3.00 ($u$), 2.00 ($\nabla u$), 1.96 ($p$), the same errors as Nitsche's method |
| free-slip channel, uniform inflow | the plug flow to $10^{-14}$ |
| slip on all four sides (four corners fixed), P2 | rate 2.99 |
| quarter annulus, slip on both arcs | $O(h^2)$ for P1, P2, and P3 with `slip_normal="smooth"` |

On a curved wall the mesh sees a polygon, and no method that uses only the mesh can beat $O(h^2)$ there. For $k \ge 3$ the edge's normal at the nodes inside an edge is off by $O(h)$ and the rate drops to 1. `slip_normal="smooth"` blends the vertex normals along the edge instead, which restores $O(h^2)$ but gives up the exact mass conservation. The two options coincide on straight walls. The weak alternative, Nitsche's method, needs only the form language.

### 12.6 Implicit time stepping

`fd.IRK` ([Section 7.6](#76-implicit-runge-kutta-time-stepping-and-algebraic-systems)) works on 2D spaces and systems with the same call. It dispatches on the mass matrix: a `SparseMatrix`, or a rank-2 `Form`, on a 2D space. `fd.ERK` ([Section 7.9](#79-classical-explicit-runge-kutta)) and `fd.SSPRK` ([Section 7.8](#78-explicit-ssp-runge-kutta-and-limiting)) need nothing new either.

```python
V = fd.LagrangeSpace(m, 3, bc="dirichlet")
U = V.unconstrained
u, v = fd.TrialFunction(V), fd.TestFunction(V)
w = fd.Function(U)
t = fd.Constant(0.0)
R = fd.form((-fd.dot(fd.grad(w), fd.grad(v)) + f * v) * fd.dx)       # f may contain t
g = lambda x, y, t: (1 + x**2 + y**2) * np.cos(t)                       # data in time
irk = fd.IRK(fd.form(u * v * fd.dx), R, dt, "radau", 3, time=t, dirichlet=g)
un = fd.Function(U)
for n in range(N):
    irk.step(un)
```

**Boundary data.** `dirichlet=` takes anything `V.lift` takes (a number, a callable, a dict per marker, a pair for a vector block, a list per block of a system). A callable $g(x, y, t)$ or $g(t)$ makes the data depend on time, while $g(x, y)$ and numbers are constant. The state is a Function of `V.unconstrained`, $u = u_g(t) + Pc$, and the stages solve $M c' = f(t, u_g + Pc) - M u_g'(t)$ as in 1D. $u_g'$ comes from the data by the complex step in $t$, or by a fourth-order central difference when a callable refuses a complex $t$. The data hold exactly at every step. Time-dependent data need `M` as a form, since it acts on $u_g'$. Constant data also accept an assembled `SparseMatrix`.

**Newton.** The stages are stacked, $K = [K_1; \dots; K_s]$, and `newton=` chooses how the stage system is solved.

| `newton` | stage matrix | factorizations |
|---|---|---|
| `"simplified"` (default in 2D) | $f'$ frozen at $(t_n, u^n)$, and $I \otimes M - \Delta t\, A \otimes f'$ split by the eigenvalues of $A$ into systems $M - \Delta t \lambda_k f'$ of size $n$, complex for a conjugate pair (one per pair) | $\lceil s/2 \rceil$ per step (1 for Gauss 2 and Radau 2, 2 for Radau 3) |
| `"frozen"` | the same, kept from step to step, renewed at once when Newton fails and at the next step when it slows down (`refresh=`, below) | once for a linear problem with a fixed step |
| `"exact"` | $\delta_{ij} M - \Delta t\, a_{ij} f'(t_i, U_i)$, one $sn \times sn$ matrix in the CSR store | one per iteration |

The splitting is Butcher's transformation, as in Hairer and Wanner's RADAU5. A 2D factorization of the $sn$ system costs about $s^3$ times one of size $n$, so the simplified modes are much cheaper while converging to the same stages. For Taylor-Green Navier-Stokes with P2/P1 on a $32 \times 32$ mesh (9153 unknowns), Radau 2 takes 2.5 s per step with `"exact"`, 0.36 s with `"simplified"` and 0.28 s with `"frozen"`, and Radau 3 takes 10.3 s, 0.48 s and 0.39 s. `irk.factorizations` counts them. With `"frozen"`, `refresh="auto"` (the default) renews the factorization at the next step when a step took more than half of `maxiter` iterations, or when a step on an older factorization took more than 1.5 times the iterations of the last new one (at least 2 more). The first test matters when even a new factorization converges slowly: an older one would then fail, and a failed step spends `maxiter` iterations before it refactors and starts again. `refresh=k` renews it after any step with more than `k` iterations, the rule of FEMd 0.1.3 with `k = 6`. On the flow of [example15](../examples/example15.ipynb) (Radau 2, 17 546 unknowns, 150 steps from the Stokes start, 4 threads in a Linux VM) `"auto"` takes 17.5 s against 21.0 s for the earlier rule and 19.1 s for `"simplified"`. The real systems go through the default `SparseSolver` (sparse Cholesky when symmetric positive definite, as for the heat equation), the complex ones through FEMd's sparse LU (`sparse/lu.hpp`).

**Singular mass.** For Stokes and Navier-Stokes the mass form $(u, v)$ has no pressure block, and the stage equations are differential-algebraic (index 2). They are solved as they stand. Use Radau IIA: Gauss methods reach only order 2 in the velocity there. The first Newton guess is $M^{-1} f$ when $M$ factors, and zero when it does not. A mass matrix with a zero on its diagonal (the pressure rows) is taken as singular without trying to factor it, which took 1.6 s for 17 546 unknowns.

```python
W = fd.ProductSpace(fd.VectorFunctionSpace(m, 2, dirichlet=[1, 3, 4]), fd.LagrangeSpace(m, 1))
(u, p), (v, q) = fd.TrialFunctions(W), fd.TestFunctions(W)
w = fd.Function(W.unconstrained)
uw, pw = w
R = fd.form((-nu * fd.inner(fd.grad(uw), fd.grad(v)) - fd.dot(fd.dot(fd.grad(uw), uw), v)
             + pw * fd.div(v) + q * fd.div(uw)) * fd.dx + fd.dot(g, v) * fd.ds(2))
irk = fd.IRK(fd.form(fd.dot(u, v) * fd.dx), R, dt, "radau", 2, time=t, dirichlet=[Uc, None])
```

**Accuracy.** The heat equation with a cubic solution (exact in P3) and data $g(x, y, t)$ on the whole boundary gives the temporal rates of the 1D test of [Section 7.6](#76-implicit-runge-kutta-time-stepping-and-algebraic-systems), with the same order reduction from data in time: Gauss 1 1.9, Gauss 2 3.7, Radau 2 2.8, Radau 3 4.0. The Taylor-Green vortex with Radau 2 converges at about 2.8 in the velocity, Radau 3 at 4.2.

### 12.7 Files: VTK output and Gmsh import

**VTK.** `fd.write_vtk` writes Functions on a 2D mesh to a VTK XML unstructured grid (`.vtu`), which ParaView and VisIt open directly.

```python
fd.write_vtk("flow.vtu", w)                          # a system: w_0 (velocity, a vector), w_1 (pressure)
fd.write_vtk("fields.vtu", uh, ("speed", s))         # several fields; (name, Function) renames one
fd.write_vtk("mesh.vtu", mesh=m)                     # the mesh alone, with its sides and their markers
series = fd.VTKSeries("results/run.pvd")             # a time series
for n in range(N):
    irk.step(w)
    series.write(irk.t, w)                           # results/run_000000.vtu, ...; run.pvd rewritten
```

- **High order without loss.** A field of degree $k > 1$ is written on VTK's Lagrange cells of order $k$ (triangle 69, quadrilateral 70), so ParaView draws the element polynomial itself, not a linear sub-triangulation. The points are the equispaced lattice of every cell, shared between neighbors, so a continuous field stays continuous. For Lobatto nodes the values are evaluated at that lattice. `degree=` sets the order. The default is the highest degree among the fields, so a Taylor-Hood pair is written on order 2 and the $P_1$ pressure exactly. A field of degree 0 alone needs `degree=1`.
- **What is written.** A scalar Function is a scalar array, a `VectorFunction` a 3-component vector ($z = 0$), a `ProductFunction` one array per block (`w_0`, `w_1`, ...). `cell_data={name: array}` adds per-cell values, such as an error indicator. A mesh read by `read_gmsh` also gets its region tags as `cell_marker`. `segments=True` adds the sides as line cells with their markers as `marker`, the default for the mesh alone, which shows at a glance which marker each side carries. A `VectorElementFunction` ($RT$, $N1curl$) is a 3-component vector, and with a discontinuous field among them (DG, $RT$, $N1curl$) every cell keeps its own points, so jumps show as they are.
- **Format.** Binary (base64) by default, `binary=False` for ASCII. Periodic spaces are written on the mesh as it is. `VTKSeries` writes one file per call and rewrites the `.pvd` collection each time, so a run that stops early still leaves a readable series. Open the `.pvd` for the time slider.

The output was checked with VTK 9.7: probing every file at arbitrary points reproduces `u.at(points)` to the probe's tolerance for $P_1$ to $P_4$ and $Q_2$, $Q_3$, and the integrated area of mapped quads is exact.

**Gmsh.** `fd.read_gmsh("domain.msh")` reads the ASCII formats 4.1 (Gmsh's default) and 2.2.

```python
m = fd.read_gmsh("channel.msh")                      # Mesh2D for triangles, QuadMesh for quads
side = m.report["physical_names"]                    # {"inlet": 1, "outlet": 2, "walls": 3, ...}
V = fd.VectorFunctionSpace(m, 2, dirichlet=[side["inlet"], side["walls"]])
```

- **Markers.** Each line element becomes a segment whose marker is the tag of its physical group, or of its curve when it belongs to none. These are the markers `dirichlet=`, `slip=`, `periodic=` and `ds(marker)` name, and `report["physical_names"]` maps the group names to them. Boundary edges without a line element get `marker=` (0). The physical (or surface) tag of every cell is in `report["cell_markers"]`, in the cells' order, and `report["region_names"]` names the 2D groups.
- **Cells.** Triangles give a `Mesh2D`, quadrilaterals (`Mesh.RecombineAll = 1`) a `QuadMesh`. A file with both is refused. Only the nodes the cells use are kept, in their original order.
- **Higher order.** Second- and higher-order elements are read by their corners, so curved sides become straight, and a warning gives how far the dropped nodes were from them. Isoparametric elements would use them ([Section 12.4](#124-accuracy)).
- **Refused.** Binary files (save with `Mesh.Binary = 0`), 3D elements, and meshes that are not planar.

### 12.8 Mixed and H(curl) problems

**Mixed Poisson and Darcy flow.** With $\sigma = \nabla u$, the Poisson problem $-\Delta u = f$ becomes a saddle point problem for the flux $\sigma \in RT_k$ and $u \in DG_{k-1}$. The Dirichlet value of $u$ is now natural and enters through `ds`, while a prescribed flux $\sigma\cdot n$ is essential and is built into the $RT$ space.

```python
V, Q = fd.FunctionSpace(m, "RT", k), fd.FunctionSpace(m, "DG", k - 1)
W = fd.ProductSpace(V, Q)
(s, u), (t, v) = fd.TrialFunctions(W), fd.TestFunctions(W)
n = fd.FacetNormal()
a = (fd.dot(s, t) + u * fd.div(t) + fd.div(s) * v) * fd.dx
L = -f * v * fd.dx + g * fd.dot(t, n) * fd.ds(2)         # u = g on side 2
w = fd.solve(a, L)                                        # sigma . n = 0 elsewhere unless built in
sh, uh = w.split()
```

The matrix is symmetric and indefinite, so `solver()` picks the pivoting sparse $LDL^T$. On the unit square with $u = \sin\pi x \sin\pi y$, $u$, $\sigma$ and $\mathrm{div}\,\sigma$ converge in $L^2$ at the optimal rate $k$ for $k = 1, 2, 3$ (rates 1.00, 2.00, 3.00 between $16\times16$ and $32\times32$ squares). With the flux given on two sides, as a vector field or as $\sigma\cdot n$, and $u$ on the other two, the rates are the same. At $RT_2 \times DG_1$ on $32 \times 32$ squares (16 512 unknowns) assembly takes 0.014 s and the factorization 0.022 s. A nonlinear coefficient, such as $\sigma/(1+u^2)$, goes to `fd.newton(R, w)` as on any system.

**Maxwell eigenvalues.** The curl-curl eigenproblem $\nabla\times\nabla\times E = \lambda E$ on $[0,\pi]^2$ with $E\times n = 0$ has the eigenvalues $m^2 + n^2$, $m, n \ge 0$ not both zero, and a kernel made of all gradients. Nodal elements produce spurious eigenvalues here, and Nédélec elements do not.

```python
E = fd.FunctionSpace(m, "N1curl", 2, bc="dirichlet")
u, v = fd.TrialFunction(E), fd.TestFunction(E)
r = fd.eigs(fd.curl(u) * fd.curl(v) * fd.dx, fd.dot(u, v) * fd.dx, k=12, sigma=5.5)
r.values            # 1, 1, 2, 4, 4, 5, 5, 8, 9, 9, 10, 10 to 5e-4 on 16 x 16 squares
```

$\sigma = 5.5$ sits inside the spectrum of interest, so the kernel (the eigenvalue 0, of large multiplicity) is farther from the shift than every eigenvalue below 11 ([Section 7.10](#710-eigenproblems)). The error decreases as $h^{2k}$ (0.37 to 0.094 for $k = 1$, and $6.9\cdot10^{-3}$ to $4.6\cdot10^{-4}$ for $k = 2$ from $8\times8$ to $16\times16$ squares), and no spurious value appears. With free sides the nonzero eigenvalues are $m^2+n^2$ with $m, n \ge 1$ ($2, 5, 5, 8, \dots$).

### 12.9 Discontinuous Galerkin on triangles

`fd.DGSpace(m, k)` on a `Mesh2D` is broken $P_k$, $k \ge 0$ ([Section 11.4](#114-raviart-thomas-nédélec-and-broken-p_k)). Forms couple its triangles through the interior-facet measure `dS`, as in 1D ([Section 6.10](#610-interior-facets-ds-jump-avg)). On a facet between the triangles $K^-$ and $K^+$, `u('-')` and `u('+')` are the two traces, `fd.jump(u)` is $u^- - u^+$, `fd.avg(u)` is $(u^- + u^+)/2$, and `n('-')` is the unit normal pointing out of $K^-$, with `n('+')` $= -$`n('-')`. `fd.jump(u, n)` is UFL's $u^- n^- + u^+ n^+$. In a `dS` term the test and trial functions, broken fields, derivatives and the normal must carry a side. A continuous field may appear unrestricted. Which triangle of a facet is `'-'` is a fixed choice of the mesh (the lower cell index), and a flux written with `n('-')` and `jump` is the same whichever it is.

`fd.abs`, `fd.sign`, `fd.max_value` and `fd.min_value` complete the language for numerical fluxes. Their derivatives are taken symbolically ($|z|' = \mathrm{sign}\, z$), so `jacobian`, `fd.newton` and `fd.IRK` work with them. The upwind flux for $u_t + \nabla\cdot(\mathbf{b} u) = 0$ is

```python
V = fd.DGSpace(m, 2, periodic=[(4, 2), (1, 3)])          # doubly periodic unit square
u, v, n = fd.TrialFunction(V), fd.TestFunction(V), fd.FacetNormal()
b = fd.as_vector([1.0, 0.5])
bn = fd.dot(b, n('-'))
a = (u * fd.dot(b, fd.grad(v)) * fd.dx
     - (bn * fd.avg(u) + 0.5 * fd.abs(bn) * fd.jump(u)) * fd.jump(v) * fd.dS)
rk = fd.SSPRK(fd.form(u * v * fd.dx), fd.form(a), dt, order=3)
```

With $u_0 = \sin 2\pi x \sin 2\pi y$ up to $t = 0.25$, the $L^2$ error converges at rate $k + 1$ (1.99, 2.99 and 3.98 for $k = 1, 2, 3$ between $16 \times 16$ and $32 \times 32$ squares), and the mass is conserved to $10^{-17}$. On distorted meshes with inflow data through `ds` (`fd.min_value(dot(b, n), 0) * g * v * ds`, outflow `fd.max_value(dot(b, n), 0) * u * v * ds`) the rate is the same. The mass matrix is block diagonal, so its sparse Cholesky factor has no fill and `fd.SSPRK` (which takes a 2D `SparseMatrix` or form) costs one assembly and one cheap solve per stage. `fd.IRK` works too.

**Periodic sides.** `periodic=[(a, b)]` joins every edge of side $b$ to the edge of side $a$ it is a translate of, as interior facets with the '+' side moved by the shift. Unlike `LagrangeSpace` (which identifies nodes) no degree of freedom is shared. `V.n_periodic_facets` counts them, and the vertex patches of the limiter are joined across the sides too.

**`fd.VertexLimiter(V)`** is the vertex-based limiter of Kuzmin (2010), for `SSPRK(..., limiter=lim)`. On each triangle the linear part of the solution, its $L^2$ projection onto $P_1$, is scaled about the cell mean by the largest $\alpha \in [0, 1]$ that keeps its three vertex values between the smallest and the largest mean of the triangles around each vertex. For $P_1$ this is Kuzmin's limiter. For $k \ge 2$ a triangle with $\alpha < 1$ is replaced by its limited linear part, and the others keep every mode. The means never change, so mass is conserved, and there is no parameter. Like any such limiter it also clips smooth extrema, and a linear field near the boundary, where the error is then of first order. On a `ProductSpace` it limits every DG field one by one (`fields=` picks some). Limiting in characteristic variables is not provided. Advecting a square of height 1 on $32 \times 32$ squares, the unlimited $P_2$ solution reaches $[-0.11, 1.19]$, and the limited one keeps every mean in $[0, 1]$, with the mass conserved to $10^{-15}$. `V.cell_means(c)` gives the means. The limiting runs in C++ (`fe/limiters.hpp`), in parallel over the triangles.

**Shallow water.** The equations $U_t + \nabla\cdot F(U) = 0$ for $U = (h, hu, hv)$ take a `ProductSpace` of three DG fields, the local Lax-Friedrichs (Rusanov) flux and reflecting walls through the pressure on `ds`:

```python
Q = fd.DGSpace(m, 1)
W = fd.ProductSpace(Q, Q, Q)
w = fd.Function(W)
h, hu, hv = w
tests = fd.TestFunctions(W)
n = fd.FacetNormal()
p = 0.5 * g * h * h
F = [fd.as_vector([hu, hv]), fd.as_vector([hu*hu/h + p, hu*hv/h]), fd.as_vector([hv*hu/h, hv*hv/h + p])]
nm = n('-')
lam = lambda s: fd.abs((hu(s)*nm[0] + hv(s)*nm[1]) / h(s)) + fd.sqrt(g * h(s))
alpha = fd.max_value(lam('-'), lam('+'))                 # the largest wave speed on the facet
R = fd.form(sum(fd.dot(Fi, fd.grad(vi)) for Fi, vi in zip(F, tests)) * fd.dx
            - sum((fd.dot(fd.avg(Fi), nm) + 0.5*alpha*fd.jump(Ui)) * fd.jump(vi)
                  for Fi, Ui, vi in zip(F, (h, hu, hv), tests)) * fd.dS
            - p * (n[0]*tests[1] + n[1]*tests[2]) * fd.ds)
M = fd.form(sum(a * b for a, b in zip(fd.TrialFunctions(W), tests)) * fd.dx)
rk = fd.SSPRK(M, R, dt, order=3, limiter=fd.VertexLimiter(W))
```

For the dam break $h = 2$ for $x < 5$ and $h = 1$ beyond, in the channel $[0, 10] \times [0, 1]$ up to $t = 0.5$, the depth matches Stoker's exact solution with an $L^1$ error of $0.117$, $0.060$ and $0.030$ on $50 \times 5$, $100 \times 10$ and $200 \times 20$ squares (first order, as a shock allows), stays in $[1, 2]$ without over- or undershoots, and conserves the mass to $10^{-13}$. The transverse momentum, which the diagonals of the mesh excite, decreases at the same rate. Each step on the finest mesh (8 000 triangles, 24 000 unknowns) takes 0.13 s. Field values and shared subexpressions such as the wave speed are evaluated once per assembly. The tests also check the Jacobian of a nonlinear facet residual against finite differences. The facet assembly adds its contributions in facet order, so its result does not depend on the number of threads (1 and 4 threads agreed bit for bit on 19 040 facets).

