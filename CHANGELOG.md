# Changes

Each version is in `include/femd/version.hpp`, and `fd.__version__` shows the installed one.

## 0.1.3 (2026-10-05)

- `SparseMatrix @ x` with an array `x` now returns an array, as `A.matvec(x)` does. A Function still gives a Function. Wrapping the product in a Function cost up to a fifth of the matvec, and `x @ (A @ x)` failed on vector spaces. Code that reads `(A @ x).vector` with an array `x` should use `A @ x` directly.
- Faster assembly of 2D forms that read known fields. The value and the first derivatives of a scalar field are computed in one pass over the quadrature points (`Cache2D.at_points_multi`), each field is evaluated once per assembly instead of once per component of a vector test space, and `Cache2D.at_points` no longer copies the coefficients or the result. The assembled numbers are unchanged bit for bit. The Navier-Stokes vector field of `ns_exact_dissipation.ipynb` assembles in 5.0 ms instead of 7.1 ms.
- `fd.ddot(x, y)`, `Matrix.inner(x, y=None)` and `SparseMatrix.inner(x, y=None)`: dot products and $y^{\mathsf T}Ax$ in C++ without BLAS, as ordered chunked sums with the same bits for any number of threads. The OpenBLAS of Anaconda's NumPy on Apple silicon threads a dot product of 16130 entries across 10 cores and takes 500 µs for it instead of 3 µs, which made the time loop of the Navier-Stokes notebook five times slower than its parts. `detail::ordered_sum` takes the parallel threshold as an argument.

## 0.1.2 (2026-10-04)

- `examples/example9.ipynb`, convection-diffusion in a recirculating flow, GMRES against LGMRES.
- `examples/example10.ipynb`, solitary waves of the Whitham equation, complex-step Newton-Krylov against Newton with a dense Jacobian.
- `examples/example11.ipynb`, the inviscid Burgers equation with DG elements, before and after the shock.
- `fd.SlopeLimiter(V, limiter, reconstruction)`, the slope limiters minmod, Van Leer, monotonized central and Van Albada, each with the TVD2 or the UNO2 reconstruction, for DG spaces in 1D, as in Dutykh, Katsaounis and Mitsotakis (2011). With degree 1 it reproduces their finite volume schemes. `fd.limited_slope` gives the slopes. `examples/example12.ipynb` compares them.
- `examples/example13.ipynb`, the heat equation in 2D with $P_2$ elements, backward Euler and Radau IIA.
- Dirichlet data and `interpolate` on 2D spaces take expressions in `fd.x`, `fd.y`, numbers and Constants, for instance `dirichlet={4: (4*y*(H - y)/H**2, 0.0)}`. Before, an expression there was called as a restriction and failed.
- `examples/example14.ipynb`, the steady Navier-Stokes equations, flow around a cylinder (DFG benchmark 2D-1) with Taylor-Hood elements and Newton's method.
- `examples/example15.ipynb`, vortex shedding behind a cylinder (DFG benchmark 2D-2), Radau IIA with `fd.IRK` from the Stokes flow.
- `examples/example16.ipynb`, output for ParaView, a time series of the heat equation in a plate with a hole (replaces `examples/old/vtk_paraview.ipynb`).
- `examples/example17.ipynb`, a mesh from Gmsh with named sides and regions, heat conduction through a plate with an inclusion (replaces `examples/old/gmsh_laplace.ipynb`). The GitHub build installs `gmsh` (and on Linux the OpenGL libraries it needs) to run it.
- Fix: plotting a vector field with `refine=` failed (the second component was drawn on the default sub-triangulation).
- Fix: `fd.lgmres` returned NaN when `maxiter` ended the iteration in the middle of a restart cycle. It now returns the best iterate from the columns it formed.

## 0.1.1 (2026-10-03)

- `SparseMatrix.ordering("rcm" | "amd" | "natural")`, `SparseMatrix.bandwidth(p)` and `SparseMatrix.permuted(p)`, so the C++ orderings are reachable without the internal module. `examples/example8.ipynb` uses them.
- `examples/example3.ipynb` runs without FFTW, using the cyclic banded solver for $M$, and the OpenMP job of the GitHub build installs FFTW.

## 0.1.0 (2026-10-02)

First public release, under the GNU General Public License, version 3 or later.

- 1D: B-spline, Lagrange and DG spaces with built-in Dirichlet, Neumann, clamped, periodic and Robin conditions, and banded, cyclic and FFT solvers.
- 2D: Delaunay mesh generation with refinement and coarsening, structured meshes and Gmsh import. $P_k$ and $Q_k$ Lagrange elements, vector and mixed spaces, Raviart-Thomas, Nédélec and DG elements on triangles. Dirichlet, natural, periodic and slip conditions. VTK output.
- A form language shared by 1D and 2D, with symbolic Jacobians.
- Sparse Cholesky and pivoting $LDL^T$, CG, GMRES and LGMRES, Newton and Newton-Krylov, implicit and explicit Runge-Kutta, SSP methods with limiters, and sparse eigenvalue solvers. Optional OpenMP.
