# Changes

Each version is in `include/femd/version.hpp`, and `fd.__version__` shows the installed one.

## Unreleased

- `examples/example9.ipynb`, convection-diffusion in a recirculating flow, GMRES against LGMRES.
- `examples/example10.ipynb`, solitary waves of the Whitham equation, complex-step Newton-Krylov against Newton with a dense Jacobian.
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
