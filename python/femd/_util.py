"""Small helpers shared by the modules of the package."""
from __future__ import annotations

import numpy as np


def _norm2(x) -> float:
    """The 2-norm of a real or complex vector (or a Function), in C++ without BLAS."""
    from . import _femd as _C
    x = np.asarray(getattr(x, "vector", x)).reshape(-1)
    if np.iscomplexobj(x):
        return float(_C.norm2(np.ascontiguousarray(x.real, dtype=np.float64), np.ascontiguousarray(x.imag, dtype=np.float64)))
    return float(_C.norm2(np.ascontiguousarray(x, dtype=np.float64)))


def _by_columns(f, v, nrows):
    """f applied to a vector, or to each column of a 2-D array (complex by parts): the products of
    a matrix with several vectors through its C++ kernel."""
    v = np.asarray(v)
    if np.iscomplexobj(v):
        return _by_columns(f, v.real, nrows) + 1j * _by_columns(f, v.imag, nrows)
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return f(np.ascontiguousarray(v))
    if v.shape[1] == 0:
        return np.zeros((nrows, 0))
    return np.column_stack([f(np.ascontiguousarray(v[:, k])) for k in range(v.shape[1])])


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
