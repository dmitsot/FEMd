"""Small helpers shared by the modules of the package."""
from __future__ import annotations

import numpy as np


def _vec(c) -> np.ndarray:
    """Contiguous float64 view of an array or a Function's coefficients."""
    return np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1))


def _real_linear(f, c):
    """Apply a real linear map f to c, also when c is complex: f(Re c) + i f(Im c).  The C++
    kernels behind f take float64 only; the complex step (fd.csnewton) passes complex vectors."""
    a = getattr(c, "vector", c)
    if np.iscomplexobj(a):
        a = np.asarray(a).reshape(-1)
        return f(np.ascontiguousarray(a.real)) + 1j * f(np.ascontiguousarray(a.imag))
    return f(c)


def _coeff(c):
    if np.isscalar(c):
        return float(c)
    return _vec(c)
