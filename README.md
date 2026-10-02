# FEMd

![FEMd](docs/mascot/femd_logo.jpg)

**Version 0.1.0** · [Changes](CHANGELOG.md) · [MIT License](LICENSE)

**FEMd** is a Finite Element Library for problems in one and two space dimensions. Its core is header-only C++17 and is used from Python. You give a domain, a polynomial degree and boundary conditions, write the equations in weak form, and FEMd assembles and solves them. The form language follows FEniCS, so FEniCS users will find the workflow familiar.

- **1D.** B-splines of any degree and continuity, Lagrange and DG elements. Dirichlet, Neumann, clamped, periodic and Robin conditions are built into the basis. Banded, cyclic and FFT solvers.
- **2D.** A Delaunay mesh generator with a guaranteed minimum angle, holes, markers, local refinement and coarsening, structured meshes and Gmsh import. Continuous $P_k$ elements on triangles and $Q_k$ on quadrilaterals of any degree, vector and mixed spaces (Stokes, Navier-Stokes), Raviart-Thomas, Nédélec and DG elements. Dirichlet, natural, periodic and slip conditions. VTK output for ParaView.
- **Solvers.** Symbolic Jacobians, sparse Cholesky and pivoting $LDL^T$, CG, GMRES and LGMRES with preconditioners, Newton and Newton-Krylov, implicit and explicit Runge-Kutta, SSP methods with limiters, and sparse eigenvalue solvers. Optional OpenMP.

## Installation

You need a C++17 compiler, Python 3.10 or later, NumPy and SciPy. Matplotlib is optional and is used by the plots and the examples.

```bash
git clone https://github.com/dmitsot/FEMd      # or Code > Download ZIP
cd FEMd
pip install .
```

`pip` builds the C++ extension, which takes a minute or two, and installs the package `femd`. Run it from the Python environment your notebooks use.

- **FFTW (optional).** If FFTW 3 is found at build time (`brew install fftw`, or `apt install libfftw3-dev`), the FFT solver backend is enabled. `fd.has_fftw()` tells you.
- **OpenMP (optional).** `pip install . -C cmake.define.FEMD_OPENMP=ON`. On macOS run `brew install libomp` first. See Section 1.1 of the manual.
- **Version.** `fd.__version__` gives the installed version, and [CHANGELOG.md](CHANGELOG.md) lists what changed in each one.
- **Updating.** After downloading a new version, run `pip install .` again and restart the Python kernel.

The library has been developed on macOS. It is built and tested on Linux and macOS at every change.

## Quick start

Solve $-u'' + u = f$ on $(0, 1)$ with $u(0) = 0$, $u(1) = 1$ and cubic splines on 20 elements.

```python
import numpy as np
import femd as fd
from femd import D, dx

V = fd.SplineSpace(np.linspace(0, 1, 21), 3, bc="dirichlet", left=0.0, right=1.0)
u, v = fd.TrialFunction(V), fd.TestFunction(V)
f = (np.pi**2 + 1) * fd.sin(np.pi * fd.x) + fd.x

uh = fd.solve(D(u) * D(v) * dx + u * v * dx, f * v * dx)
print(uh(0.5))          # about 1.5, the exact solution is x + sin(pi x)
```

The same in 2D, Poisson's equation with $P_3$ elements on a triangulated square.

```python
m = fd.rectangle_mesh(0, 0, 1, 1, 32, 32)
V = fd.LagrangeSpace(m, 3, bc="dirichlet")
u, v = fd.TrialFunction(V), fd.TestFunction(V)
f = 2 * np.pi**2 * fd.sin(np.pi * fd.x) * fd.sin(np.pi * fd.y)
uh = fd.solve(fd.dot(fd.grad(u), fd.grad(v)) * fd.dx, f * v * fd.dx)
```

## Examples

The notebooks in [`examples/`](examples) run as they are after installation.

| notebook | problem |
|---|---|
| [example0](examples/example0.ipynb) | $-u'' + u = f$ with Dirichlet conditions, cubic splines, errors |
| [example1](examples/example1.ipynb) | the same with a Dirichlet and a Neumann condition |
| [example2](examples/example2.ipynb) | $-u'' + u^3 = f$ with Newton's method and Newton-Krylov |
| [example3](examples/example3.ipynb) | the BBM equation, a solitary wave with periodic cubic splines, RK4 and a Gauss method |
| [example4](examples/example4.ipynb) | the Bona-Smith system, a solitary wave reflected by a wall ($\eta_x = 0$, $u = 0$), a system of two fields |
| [example5](examples/example5.ipynb) | $-\Delta u + u = f$ in 2D on a square with a hole, $P_2$ elements, Dirichlet outside and free on the hole |
| [example6](examples/example6.ipynb) | the BBM-BBM system in 2D, a solitary wave in a channel with a cylinder, $P_1$ for $\eta$ and $P_2$ for $u$ with slip walls |
| [example7](examples/example7.ipynb) | the same problem with OpenMP, timing on 1, 2, 4, ... threads |

## Documentation

- [Manual](docs/manual.md), the main ideas and worked examples in 1D and 2D.
- [API reference](docs/reference/index.md), every function and class.

## License

FEMd is released under the [MIT License](LICENSE).

## Author

FEMd was designed and developed by D. Mitsotakis (Victoria University of Wellington). The Delaunay mesh generator was developed in collaboration with T. Katsaounis.
