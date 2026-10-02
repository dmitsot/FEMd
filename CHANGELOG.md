# Changes

Each version is in `include/femd/version.hpp`, and `fd.__version__` shows the installed one.

## 0.1.0 (2026-10-02)

First public release, under the GNU General Public License, version 3 or later.

- 1D: B-spline, Lagrange and DG spaces with built-in Dirichlet, Neumann, clamped, periodic and Robin conditions, and banded, cyclic and FFT solvers.
- 2D: Delaunay mesh generation with refinement and coarsening, structured meshes and Gmsh import. $P_k$ and $Q_k$ Lagrange elements, vector and mixed spaces, Raviart-Thomas, Nédélec and DG elements on triangles. Dirichlet, natural, periodic and slip conditions. VTK output.
- A form language shared by 1D and 2D, with symbolic Jacobians.
- Sparse Cholesky and pivoting $LDL^T$, CG, GMRES and LGMRES, Newton and Newton-Krylov, implicit and explicit Runge-Kutta, SSP methods with limiters, and sparse eigenvalue solvers. Optional OpenMP.
