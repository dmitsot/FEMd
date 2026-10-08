"""
femd.forms -- the form language (design doc, Section 15).

The user writes inner products; the library turns them into a number, a vector
or a matrix.  Every term reduces to

    integral c(x) (D^a N_i)(D^b N_j) dx

with c sampled at the quadrature nodes, so this module is a parser, a
canonicalizer and a dispatcher onto the three kernels.  No code generation.

    u, v, w = TrialFunction(V), TestFunction(V), Function(V, "w")
    A = form((u*v + D(u)*D(v)) * dx).assemble()          # Matrix, constant, cached
    R = form(-(D(w) + w*D(w)) * v * dx)                   # rank 1 in v
    r = R.assemble(w=wn)                                  # vector
    J = R.jacobian("w", w=wn)                             # Matrix, d r / d w

Rank = number of unresolved argument slots: TestFunction -> row, TrialFunction
-> column, Function -> a named slot filled at assembly time.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np

from . import _femd as _C

# ============================================================================ expressions
class Expr:
    """Base of the expression tree.  Arithmetic builds Sum / Prod / Pow nodes.

    An expression that is a LINEAR combination of Functions of one space converts to a
    NumPy array on demand (np.asarray(u - v), np.abs(u + 0.5*k)), so concrete Functions
    can be compared and combined in NumPy code without leaving the form language."""
    def __array__(self, dtype=None, copy=None):
        fs = [a for a in collect_arguments(self, []) if isinstance(a, Function)]
        if not fs:
            raise TypeError(f"{self!r} contains no Function and has no array value")
        v = _linear_combination(self, fs[0].space)
        return v if dtype is None else v.astype(dtype)
    # NumPy scalars (np.float64(2.0) * v) would otherwise try to turn the expression into
    # an array; a high priority makes NumPy defer to the reflected operators below, while
    # np.asarray(u - w) and ufuncs on linear combinations keep working through __array__.
    __array_priority__ = 1000

    def __call__(self, side):
        """e('-') or e('+'): the expression restricted to one side of an interior facet (dS)."""
        return restrict(self, side)

    def __add__(self, o):  return Sum([self, _lift(o)])
    def __radd__(self, o): return Sum([_lift(o), self])
    def __sub__(self, o):  return Sum([self, Prod([Const(-1.0), _lift(o)])])
    def __rsub__(self, o): return Sum([_lift(o), Prod([Const(-1.0), self])])
    def __neg__(self):     return Prod([Const(-1.0), self])
    def __pos__(self):     return self
    def __mul__(self, o):
        if isinstance(o, Measure):
            return Integral(self, o)
        if isinstance(o, ListTensor):
            return o.__rmul__(self)
        return Prod([self, _lift(o)])
    def __rmul__(self, o): return Prod([_lift(o), self])
    def __truediv__(self, o):
        o = _lift(o)
        if has_arguments(o, ("test", "trial")):
            raise TypeError("cannot divide by an expression containing a test or trial function")
        return Prod([self, Pow(o, -1)])
    def __rtruediv__(self, o): return Prod([_lift(o), Pow(self, -1)])
    def __pow__(self, n):
        if not isinstance(n, (int, float)):
            raise TypeError("exponent must be a number")
        return Pow(self, n)


def _lift(o) -> Expr:
    if isinstance(o, Expr):
        return o
    if isinstance(o, ListTensor):
        raise TypeError(f"a vector {o!r} where a scalar is needed: use dot(), inner() or a component")
    if isinstance(o, (int, float, np.floating, np.integer)):
        return Const(float(o))
    raise TypeError(f"cannot use {o!r} in a form")


class Const(Expr):
    def __init__(self, value: float): self.value = float(value)
    def __repr__(self): return f"{self.value:g}"


class Spatial(Expr):
    """A coordinate: x (index 0) or, on a 2D mesh, y (index 1)."""
    def __init__(self, index: int = 0): self.index = int(index)
    def __repr__(self): return "xy"[self.index]


x = Spatial(0)
y = Spatial(1)


class Normal(Expr):
    """A component of the unit normal: outward on a boundary facet (ds terms in 2D), and on an
    interior facet (dS) n('-') points out of the '-' cell, n('+') = -n('-') out of the '+' one."""
    def __init__(self, index: int, side=None): self.index, self.side = int(index), side
    def __repr__(self): return f"n[{self.index}]" + ("" if self.side is None else f"('{self.side}')")


class Constant(Expr):
    """A number in a form whose value is read when the form is assembled.

        t  = fd.Constant(0.0)                      # named "t" after the variable
        f  = fd.form(fd.sin(t) * fd.cos(fd.x) * v * dx)
        t.assign(0.5); b = f.assemble()            # uses t = 0.5
        b = f.assemble(t=0.7)                      # this call only, t keeps its value

    A plain number is fixed when the form is compiled; a Constant can change between
    assemblies, like the coefficients of a Function.  It is the time in a
    non-autonomous right-hand side (fd.IRK(..., time=t)), or any parameter swept
    without recompiling.  A form that contains a Constant is not cached.
    """

    def __init__(self, value=0.0, name=None):
        if name is None:
            try:
                name = _infer_name(depth=2, callee="Constant")
            except Exception:
                name = None
        self.name = name or "c"
        self.value = _scalar(value)

    def assign(self, value):
        """Set the value.  Returns self."""
        self.value = _scalar(value)
        return self

    def __float__(self): return float(self.value)
    def __repr__(self): return self.name


def _scalar(v):
    v = v.value if isinstance(v, Constant) else v
    if np.iscomplexobj(v):
        return complex(v)
    return float(v)


class Argument(Expr):
    """Test / trial function, or a known field (a Function or its derivative).

    side is None, or '-' / '+' for the restriction to the left / right element of an
    interior facet, which only a dS term may contain.

    comp is None for a scalar space, or the component (0 or 1) of a vector-valued element
    (Raviart-Thomas, Nedelec), whose test and trial functions are vectors of two Arguments."""
    def __init__(self, kind, space, name=None, field=None, product=None, deriv=0, owner=None, side=None, comp=None):
        if getattr(space, "tdim", 1) == 2 and not isinstance(deriv, tuple):
            if deriv != 0:
                raise TypeError("on a 2D space derivatives are partial: use grad(u), Dx(u, 0) or Dx(u, 1), not D(u)")
            deriv = (0, 0)
        self.kind, self.space, self.name, self.field, self.product, self.deriv = kind, space, name, field, product, deriv
        self.owner = owner            # the Function / ProductFunction whose vector supplies the values
        self.side = side
        self.comp = comp

    def __repr__(self):
        base = {"test": "v", "trial": "u", "func": self.name}[self.kind]
        if self.field is not None:
            base += f"[{self.field}]"
        if self.comp is not None:
            base += f"_{'xy'[self.comp]}"
        if isinstance(self.deriv, tuple):
            if any(self.deriv):
                base = "D" + "x" * self.deriv[0] + "y" * self.deriv[1] + f"({base})"
        else:
            base = base if self.deriv == 0 else f"D({base},{self.deriv})"
        return base if self.side is None else f"{base}('{self.side}')"

    def slot(self):
        return (id(self.owner) if self.owner is not None else self.name, self.field, self.deriv, self.side, self.comp)

    def code(self):
        """The 2D kernel's derivative code: 3 comp + (0 value, 1 d/dx, 2 d/dy)."""
        d = self.deriv if isinstance(self.deriv, tuple) else (self.deriv, 0)
        return 3 * (self.comp or 0) + (0 if d[0] + d[1] == 0 else (1 if d[0] else 2))

    def same_field(self, other):
        """Same known field (ignoring derivative order)."""
        return (self.owner is other.owner if self.owner is not None else self.name == other.name) and self.field == other.field


class Sum(Expr):
    def __init__(self, terms): self.terms = list(terms)
    def __repr__(self): return "(" + " + ".join(map(repr, self.terms)) + ")"


class Prod(Expr):
    def __init__(self, factors): self.factors = list(factors)
    def __repr__(self): return "*".join(map(repr, self.factors))


class Pow(Expr):
    def __init__(self, base, n): self.base, self.n = base, n
    def __repr__(self): return f"({self.base!r})**{self.n}"


class Func(Expr):
    """Pointwise elementary function of an expression free of test/trial functions."""
    TABLE = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "tanh": np.tanh, "sqrt": np.sqrt,
             "sinh": np.sinh, "cosh": np.cosh, "sech": lambda z: _sech(z), "abs": lambda z: _abs(z),
             "sign": lambda z: np.sign(np.real(z))}

    def __init__(self, name, arg):
        if has_arguments(arg, ("test", "trial")):
            raise TypeError(f"{name}() of a test or trial function is not a valid (multilinear) form")
        self.name, self.arg = name, arg

    def __repr__(self): return f"{self.name}({self.arg!r})"


def sin(e):  return Func("sin", _lift(e))
def cos(e):  return Func("cos", _lift(e))
def exp(e):  return Func("exp", _lift(e))
def log(e):  return Func("log", _lift(e))
def tanh(e): return Func("tanh", _lift(e))
def sqrt(e): return Func("sqrt", _lift(e))
def sinh(e): return Func("sinh", _lift(e))
def cosh(e): return Func("cosh", _lift(e))
def sech(e): return Func("sech", _lift(e))
def sign(e): return Func("sign", _lift(e))


def abs_(e):
    """fd.abs (named abs_ here so that the builtin stays available in this module)."""
    return Func("abs", _lift(e))


abs_.__doc__ = """|e|, pointwise (fd.abs).  Its derivative is sign(e), so a Newton or Jacobian of a form with
fd.abs works away from e = 0 (an upwind or Rusanov flux)."""


def max_value(a, b):
    """max(a, b) pointwise, as (a + b + |a - b|) / 2."""
    a, b = _lift(a), _lift(b)
    return 0.5 * (a + b + Func("abs", a - b))


def min_value(a, b):
    """min(a, b) pointwise, as (a + b - |a - b|) / 2."""
    a, b = _lift(a), _lift(b)
    return 0.5 * (a + b - Func("abs", a - b))


def _abs(z):
    """|z| for real z; for complex z (the complex-step derivative) sign(Re z) z, whose imaginary part
    is then the derivative's."""
    z = np.asarray(z)
    if np.iscomplexobj(z):
        return np.sign(z.real) * z
    return np.abs(z)


def _sech(z):
    """1 / cosh(z), without overflow for large real |z| (2 e^{-|z|} / (1 + e^{-2|z|})); a complex argument
    (the complex-step derivative) is taken as 1 / cosh(z), whose imaginary part is then exact."""
    z = np.asarray(z)
    if np.iscomplexobj(z):
        return 1.0 / np.cosh(z)
    e = np.exp(-np.abs(z))
    return 2.0 * e / (1.0 + e * e)


_SIDES = ("-", "+")


def restrict(e, side):
    """The expression e on one side of an interior facet: '-' the left element, '+' the right one.

    e('-') and e('+') do the same.  Constants and x are single-valued and pass through."""
    if side not in _SIDES:
        raise ValueError(f"a restriction is '-' (the left element) or '+' (the right one), got {side!r}")
    if isinstance(e, ListTensor):
        return ListTensor([restrict(c, side) for c in e.components])
    if isinstance(e, Argument):
        if e.side is not None and e.side != side:
            raise ValueError(f"{e!r} is already restricted to the other side")
        return Argument(e.kind, e.space, e.name, e.field, e.product, e.deriv, e.owner, side, e.comp)
    if isinstance(e, Normal):
        if e.side is not None and e.side != side:
            raise ValueError(f"{e!r} is already restricted to the other side")
        return Normal(e.index, side)
    if isinstance(e, Sum):  return Sum([restrict(t, side) for t in e.terms])
    if isinstance(e, Prod): return Prod([restrict(f, side) for f in e.factors])
    if isinstance(e, Pow):  return Pow(restrict(e.base, side), e.n)
    if isinstance(e, Func): return Func(e.name, restrict(e.arg, side))
    if isinstance(e, (Const, Constant, Spatial)):
        return e
    if isinstance(e, (int, float, np.floating, np.integer)):
        return Const(float(e))
    raise TypeError(f"cannot restrict {e!r}")


def jump(e, n=None):
    """e('-') - e('+'): the value on the '-' side minus that on the '+' side.  In 1D this is the jump
    times the outward normal of the left element (+1), so an upwind flux for a > 0 is  a * u('-')
    and a Lax-Friedrichs flux is  avg(F(u)) + 0.5 * alpha * jump(u).  In 2D, with n('-') the normal
    out of the '-' cell, the same flux reads  dot(avg(F(u)), n('-')) + 0.5 * alpha * jump(u).

    jump(e, n) with the facet normal n (2D) is UFL's: e('-') n('-') + e('+') n('+'), a vector for a
    scalar e, and dot(e('-'), n('-')) + dot(e('+'), n('+')), a scalar, for a vector e."""
    if n is None:
        return restrict(e, "-") - restrict(e, "+")
    n = as_vector(n)
    if isinstance(e, ListTensor) or isinstance(e, (list, tuple)):
        e = as_vector(e)
        return dot(restrict(e, "-"), restrict(n, "-")) + dot(restrict(e, "+"), restrict(n, "+"))
    return restrict(e, "-") * restrict(n, "-") + restrict(e, "+") * restrict(n, "+")


def avg(e):
    """(e('-') + e('+')) / 2 (componentwise for a vector)."""
    return 0.5 * (restrict(e, "-") + restrict(e, "+"))


def D(e, k: int = 1):
    """k-th derivative.  Linear: distributes over sums and scalar factors; on an Argument it bumps deriv."""
    if k == 0:
        return e
    if isinstance(e, ListTensor):
        raise TypeError("D() of a vector: take the derivative of a component, or use div()")
    if isinstance(e, Argument):
        if isinstance(e.deriv, tuple):
            raise TypeError(f"D({e!r}): on a 2D space derivatives are partial; use grad(), Dx(u, 0) or Dx(u, 1)")
        return Argument(e.kind, e.space, e.name, e.field, e.product, e.deriv + k, e.owner, e.side, e.comp)
    if isinstance(e, Sum):
        return Sum([D(t, k) for t in e.terms])
    if isinstance(e, Prod):
        consts = [f for f in e.factors if isinstance(f, Const)]
        rest = [f for f in e.factors if not isinstance(f, Const)]
        if not rest:
            return Const(0.0)
        if len(rest) == 1:
            return Prod(consts + [D(rest[0], k)])
        if k == 1:                                    # product rule
            out = []
            for i, f in enumerate(rest):
                out.append(Prod(consts + rest[:i] + [D(f, 1)] + rest[i + 1:]))
            return Sum(out)
        return D(D(e, 1), k - 1)
    if isinstance(e, (Const, Constant)):
        return Const(0.0)
    if isinstance(e, Spatial):
        if e.index != 0:
            raise TypeError("D() is the derivative along a 1D mesh; y has none (use Dx(e, 1) in 2D)")
        return Const(1.0) if k == 1 else Const(0.0)
    if isinstance(e, Normal):
        raise TypeError("the facet normal cannot be differentiated")
    if isinstance(e, Pow):
        if k == 1:
            return Prod([Const(float(e.n)), Pow(e.base, e.n - 1), D(e.base, 1)])
        return D(D(e, 1), k - 1)
    if isinstance(e, Func):
        if k == 1:
            return Prod([_dfunc(e), D(e.arg, 1)])
        return D(D(e, 1), k - 1)
    raise TypeError(f"cannot differentiate {e!r}")


def _dfunc(f: Func) -> Expr:
    a = f.arg
    return {"sin": lambda: Func("cos", a), "cos": lambda: Prod([Const(-1.0), Func("sin", a)]),
            "exp": lambda: f, "log": lambda: Pow(a, -1),
            "tanh": lambda: Sum([Const(1.0), Prod([Const(-1.0), Pow(Func("tanh", a), 2)])]),
            "sqrt": lambda: Prod([Const(0.5), Pow(a, -0.5)]),
            "sinh": lambda: Func("cosh", a), "cosh": lambda: Func("sinh", a),
            "sech": lambda: Prod([Const(-1.0), Func("sech", a), Func("tanh", a)]),
            "abs": lambda: Func("sign", a), "sign": lambda: Const(0.0)}[f.name]()


# ---- 2D: partial derivatives and small vectors ------------------------------------
def Dx(e, i: int):
    """Partial derivative d e / d x_i (i = 0 for x, 1 for y) on a 2D mesh.  Linear, with the
    product and chain rules on coefficients, as D() is in 1D."""
    if i not in (0, 1):
        raise ValueError(f"Dx(e, i): i is 0 (x) or 1 (y), got {i!r}")
    if isinstance(e, ListTensor):
        return ListTensor([Dx(c, i) for c in e.components])
    if isinstance(e, (int, float, np.floating, np.integer)):
        return Const(0.0)
    if isinstance(e, Argument):
        if not isinstance(e.deriv, tuple):
            if i != 0:
                raise TypeError(f"Dx({e!r}, 1): a function on a 1D mesh has no y-derivative")
            return D(e, 1)
        d = list(e.deriv); d[i] += 1
        return Argument(e.kind, e.space, e.name, e.field, e.product, tuple(d), e.owner, e.side, e.comp)
    if isinstance(e, Sum):
        return Sum([Dx(t, i) for t in e.terms])
    if isinstance(e, Prod):
        consts = [f for f in e.factors if isinstance(f, Const)]
        rest = [f for f in e.factors if not isinstance(f, Const)]
        if not rest:
            return Const(0.0)
        if len(rest) == 1:
            return Prod(consts + [Dx(rest[0], i)])
        return Sum([Prod(consts + rest[:j] + [Dx(f, i)] + rest[j + 1:]) for j, f in enumerate(rest)])
    if isinstance(e, (Const, Constant)):
        return Const(0.0)
    if isinstance(e, Spatial):
        return Const(1.0 if e.index == i else 0.0)
    if isinstance(e, Normal):
        raise TypeError("the facet normal cannot be differentiated")
    if isinstance(e, Pow):
        return Prod([Const(float(e.n)), Pow(e.base, e.n - 1), Dx(e.base, i)])
    if isinstance(e, Func):
        return Prod([_dfunc(e), Dx(e.arg, i)])
    raise TypeError(f"cannot differentiate {e!r}")


class ListTensor:
    """A small vector or matrix of scalar expressions: grad(u), as_vector([a, b]), the normal n,
    a vector test or trial function, grad of a vector (a 2 x 2 matrix).

    A vector holds scalar expressions, a matrix holds vectors (its rows).  Supports +, -, scalar
    * and /, indexing (A[i], A[i, j]), len, .T, and dot / inner / div / tr / sym.  It is not an
    integrand by itself: reduce it to a scalar (inner(grad(u), grad(v)), dot(f, v), b[0]*v, ...)."""
    __array_priority__ = 1001

    def __init__(self, components):
        comps = [c if isinstance(c, ListTensor) else (as_vector(c) if isinstance(c, (list, tuple)) else _lift(c))
                 for c in components]
        if comps and len({isinstance(c, ListTensor) for c in comps}) > 1:
            raise TypeError("a tensor mixes scalars and vectors among its entries")
        if comps and isinstance(comps[0], ListTensor) and len({(c.rank, len(c)) for c in comps}) > 1:
            raise TypeError("the rows of a matrix must have one length")
        self.components = comps

    @property
    def rank(self):
        return 1 + (self.components[0].rank if self.components and isinstance(self.components[0], ListTensor) else 0)

    @property
    def shape(self):
        return (len(self),) if self.rank == 1 else (len(self), len(self.components[0]))

    def __len__(self): return len(self.components)
    def __call__(self, side):
        """t('-') or t('+'): the tensor restricted to one side of an interior facet (dS)."""
        return restrict(self, side)
    def __getitem__(self, i):
        if isinstance(i, tuple):
            out = self
            for k in i:
                out = out.components[k]
            return out
        return self.components[i]
    def __iter__(self): return iter(self.components)
    def __repr__(self): return "[" + ", ".join(map(repr, self.components)) + "]"

    def _same(self, o, what):
        o = as_vector(o) if isinstance(o, (list, tuple)) else o
        if not isinstance(o, ListTensor) or o.shape != self.shape:
            raise TypeError(f"{what}: both operands must be tensors of the same shape")
        return o

    def __add__(self, o):
        if isinstance(o, (int, float)) and o == 0:
            return self                              # sum() starts from 0
        o = self._same(o, "tensor + tensor")
        return ListTensor([a + b for a, b in zip(self.components, o.components)])
    __radd__ = __add__
    def __sub__(self, o):
        o = self._same(o, "tensor - tensor")
        return ListTensor([a - b for a, b in zip(self.components, o.components)])
    def __rsub__(self, o):
        return self._same(o, "tensor - tensor") - self
    def __neg__(self): return ListTensor([-a for a in self.components])
    def __mul__(self, o):
        if isinstance(o, ListTensor):
            raise TypeError("tensor * tensor is ambiguous: use dot(a, b), inner(a, b) or outer(a, b)")
        if isinstance(o, Measure):
            raise TypeError("the integrand is a vector or matrix: reduce it to a scalar first (dot, inner, a component)")
        return ListTensor([a * o for a in self.components])
    def __rmul__(self, o):
        if isinstance(o, ListTensor):
            raise TypeError("tensor * tensor is ambiguous: use dot(a, b), inner(a, b) or outer(a, b)")
        return ListTensor([o * a for a in self.components])
    def __truediv__(self, o): return ListTensor([a / o for a in self.components])

    @property
    def T(self):
        """The transpose of a matrix."""
        return transpose(self)


def as_vector(components):
    """A vector expression from a list of scalar expressions (or numbers)."""
    return components if isinstance(components, ListTensor) else ListTensor(list(components))


def as_matrix(rows):
    """A matrix expression from a list of rows, each a list of scalar expressions."""
    return rows if isinstance(rows, ListTensor) else ListTensor([as_vector(r) for r in rows])


def _rank(e):
    return e.rank if isinstance(e, ListTensor) else 0


def _tdim_of(e):
    if isinstance(e, ListTensor):
        dims = {_tdim_of(c) for c in e.components}
        dims.discard(None)
        return dims.pop() if len(dims) == 1 else 2
    args = collect_arguments(e, []) if isinstance(e, Expr) else []
    dims = {getattr(a.space, "tdim", 1) for a in args}
    if len(dims) > 1:
        raise TypeError("an expression mixes functions on 1D and 2D meshes")
    return dims.pop() if dims else 2


def grad(e):
    """The gradient.  Of a scalar on a 2D mesh: the vector (Dx(e, 0), Dx(e, 1)).  Of a vector u:
    the matrix grad(u)[i, j] = d u_i / d x_j, so dot(grad(u), b) = (b . nabla) u.  On a 1D mesh
    grad(e) is D(e)."""
    if isinstance(e, ListTensor):
        if e.rank != 1:
            raise TypeError("grad() of a matrix is not supported")
        return ListTensor([grad(c) for c in e.components])
    e = _lift(e)
    if _tdim_of(e) == 1:
        return D(e, 1)
    return ListTensor([Dx(e, 0), Dx(e, 1)])


def nabla_grad(e):
    """grad(e) transposed for a vector: nabla_grad(u)[i, j] = d u_j / d x_i (the other convention)."""
    g = grad(e)
    return transpose(g) if isinstance(g, ListTensor) and g.rank == 2 else g


def div(v):
    """The divergence: of a vector, Dx(v[0], 0) + Dx(v[1], 1); of a matrix, the vector of the
    divergences of its rows, div(A)_i = sum_j d A_ij / d x_j (so div(grad(u)) is the Laplacian)."""
    v = as_vector(v)
    if v.rank == 2:
        return ListTensor([div(row) for row in v.components])
    return Sum([Dx(c, i) for i, c in enumerate(v.components)])


def curl(v):
    """The curl in 2D.  Of a vector v: the scalar rot v = d v_y / dx - d v_x / dy (the H(curl)
    derivative of a Nedelec function).  Of a scalar s: the vector (d s / dy, -d s / dx)."""
    if isinstance(v, ListTensor) or isinstance(v, (list, tuple)):
        v = as_vector(v)
        if v.rank != 1 or len(v) != 2:
            raise TypeError("curl(): a 2D vector, or a scalar")
        return Dx(v[1], 0) - Dx(v[0], 1)
    s = _lift(v)
    return ListTensor([Dx(s, 1), -Dx(s, 0)])


rot = curl


def dot(a, b):
    """a . b.  vector . vector: sum_i a_i b_i.  matrix . vector: (A b)_i = sum_j A_ij b_j.
    vector . matrix: sum_i a_i A_ij.  matrix . matrix: the matrix product.  scalars: a b."""
    a = as_vector(a) if isinstance(a, (list, tuple)) else a
    b = as_vector(b) if isinstance(b, (list, tuple)) else b
    ra, rb = _rank(a), _rank(b)
    if ra == 0 and rb == 0:
        return _lift(a) * _lift(b)
    if ra == 0 or rb == 0:
        raise TypeError("dot(a, b): a scalar and a tensor; use a * b for scaling")
    if ra == 1 and rb == 1:
        if len(a) != len(b):
            raise TypeError(f"dot(a, b): lengths {len(a)} and {len(b)} differ")
        return Sum([p * q for p, q in zip(a.components, b.components)])
    if ra == 2 and rb == 1:
        return ListTensor([dot(row, b) for row in a.components])
    if ra == 1 and rb == 2:
        return dot(transpose(b), a)
    return ListTensor([ListTensor([dot(row, col) for col in transpose(b).components]) for row in a.components])


def inner(a, b):
    """The full contraction: a b for scalars, sum_i a_i b_i for vectors, sum_ij A_ij B_ij (A : B)
    for matrices."""
    ra, rb = _rank(a), _rank(b)
    if ra != rb:
        raise TypeError(f"inner(a, b): ranks {ra} and {rb} differ")
    if ra == 0:
        return _lift(a) * _lift(b)
    if len(a) != len(b):
        raise TypeError("inner(a, b): shapes differ")
    return Sum([inner(p, q) for p, q in zip(a.components, b.components)])


def outer(a, b):
    """The matrix a b^T of two vectors."""
    a, b = as_vector(a), as_vector(b)
    return ListTensor([ListTensor([p * q for q in b.components]) for p in a.components])


def transpose(A):
    if _rank(A) != 2:
        raise TypeError("transpose() of something that is not a matrix")
    n, m = A.shape
    return ListTensor([ListTensor([A.components[i].components[j] for i in range(n)]) for j in range(m)])


def sym(A):
    """(A + A^T) / 2, the symmetric part (sym(grad(u)) is the strain rate)."""
    return 0.5 * (A + transpose(A))


def skew(A):
    """(A - A^T) / 2."""
    return 0.5 * (A - transpose(A))


def tr(A):
    """The trace of a matrix."""
    if _rank(A) != 2:
        raise TypeError("tr() of something that is not a matrix")
    return Sum([A.components[i].components[i] for i in range(min(A.shape))])


def Identity(d=2):
    """The d x d identity matrix, as an expression."""
    return ListTensor([ListTensor([1.0 if i == j else 0.0 for j in range(d)]) for i in range(d)])


def FacetNormal(mesh=None):
    """The outward unit normal (n[0], n[1]) on the exterior facets, for ds terms on a 2D mesh."""
    return ListTensor([Normal(0), Normal(1)])


n = FacetNormal()


def _with_deriv(e, deriv):
    """e differentiated by a derivative index: an int (1D) or a tuple (2D)."""
    if isinstance(deriv, tuple):
        for i, k in enumerate(deriv):
            for _ in range(k):
                e = Dx(e, i)
        return e
    return D(e, deriv)


def _order(deriv):
    return sum(deriv) if isinstance(deriv, tuple) else deriv


def _arg_degree(a):
    """Polynomial degree of a test, trial or known function in the sense the quadrature rule is chosen:
    total degree on triangles and intervals, per-variable degree on quadrilaterals, where a derivative
    lowers the degree in one variable only (so it does not lower the per-variable maximum)."""
    if getattr(a.space, "nverts", 3) == 4:
        return a.space.degree
    return max(a.space.degree - _order(a.deriv), 0)


# ---- constructors ------------------------------------------------------------
import ast as _ast
import inspect as _inspect
import linecache as _linecache
import textwrap as _textwrap


def _infer_name(depth: int = 2, callee: str = "Function"):
    """Name of the variable the caller assigns this call to, read from the source.

    Handles  w = fd.Function(V),  w: T = fd.Function(V)  and tuple unpacking
    u, w, z = fd.TrialFunction(V), fd.Function(V), fd.Function(V)  (Python >= 3.11
    knows the column of the executing call, so several Functions on one line are
    told apart).  Returns None when no source is available (bare REPL), and raises
    when the call is nested inside a larger expression, where no name is implied.
    """
    callee_name = callee
    frame = _inspect.currentframe()
    try:
        for _ in range(depth):
            frame = frame.f_back
            if frame is None:
                return None
        code, lineno = frame.f_code, frame.f_lineno
        lines = _linecache.getlines(code.co_filename, frame.f_globals)
        if not lines:
            return None
        # column of the executing instruction (3.11+), to disambiguate several calls on a line
        col = None
        if hasattr(code, "co_positions"):
            try:
                pos = list(code.co_positions())[frame.f_lasti // 2]
                if pos[0] is not None: lineno, col = pos[0], pos[2]
            except Exception:
                col = None
        for lo, hi in ((lineno, lineno), (lineno - 1, lineno), (lineno, lineno + 1), (lineno - 2, lineno + 2), (lineno - 5, lineno + 5), (lineno - 12, lineno + 12)):
            lo, hi = max(1, lo), min(len(lines), hi)
            src = "".join(lines[lo - 1:hi])
            try:
                tree = _ast.parse(_textwrap.dedent(src))
            except SyntaxError:
                continue
            indent = len(src) - len(src.lstrip(" \t")) if src.strip() else 0
            calls = []
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Call):
                    f = node.func
                    callee = f.attr if isinstance(f, _ast.Attribute) else getattr(f, "id", None)
                    if callee == callee_name:
                        calls.append(node)
            if not calls:
                return None
            if col is not None:
                # first-line columns were shifted by the dedent; later lines were not
                def matches(c):
                    L = c.lineno + lo - 1
                    if L != lineno: return False
                    shift = indent if c.lineno == 1 else 0
                    return abs(c.col_offset + shift - col) <= 1
                hits = [c for c in calls if matches(c)]
                if len(hits) != 1:
                    if len(calls) == 1: hits = calls
                    else: return None
                call = hits[0]
            elif len(calls) == 1:
                call = calls[0]
            else:
                raise ValueError("several Function(...) calls on one line need Python >= 3.11 or explicit names")
            # the assignment that owns this call: match the call, or a method chain on it,
            # against the target(s), element-wise for tuple unpacking
            def base_of(v):
                while isinstance(v, _ast.Call) and isinstance(v.func, _ast.Attribute) and v is not call:
                    v = v.func.value
                return v
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.Assign, _ast.AnnAssign)):
                    targets = node.targets if isinstance(node, _ast.Assign) else [node.target]
                    if len(targets) != 1: continue
                    tgt, val = targets[0], node.value
                    pairs = []
                    if isinstance(val, _ast.Tuple) and isinstance(tgt, _ast.Tuple) and len(val.elts) == len(tgt.elts):
                        pairs = list(zip(tgt.elts, val.elts))
                    else:
                        pairs = [(tgt, val)]
                    for t_el, v_el in pairs:
                        if base_of(v_el) is call:
                            if isinstance(t_el, _ast.Name):
                                return t_el.id
                            if isinstance(t_el, _ast.Tuple):                 # (w1, w2) = Functions(P)
                                return "_".join(el.id for el in t_el.elts if isinstance(el, _ast.Name)) or None
            raise ValueError(f"{callee_name}(...) is not the right-hand side of an assignment here, so no name is implied; pass name=...")
        return None
    finally:
        del frame


def _is_system_2d(V):
    return getattr(V, "tdim", 1) == 2 and hasattr(V, "fields")


def _is_vector_element(V):
    """A vector-valued finite element (Raviart-Thomas, Nedelec): one space, two components."""
    return getattr(V, "tdim", 1) == 2 and getattr(V, "ncomp", 1) == 2 and not hasattr(V, "fields")


def _element_vector(kind, V, field=None, product=None, name=None, owner=None):
    return ListTensor([Argument(kind, V, name=name, field=field, product=product, owner=owner, comp=c) for c in (0, 1)])


def _vector_argument(kind, V):
    if not getattr(V, "is_vector", False):
        raise TypeError(f"{'TestFunction' if kind == 'test' else 'TrialFunction'}(W) on a system of several fields: "
                        f"use {'TestFunctions' if kind == 'test' else 'TrialFunctions'}(W), which gives one per block")
    return ListTensor([Argument(kind, f, field=i, product=V) for i, f in enumerate(V.fields)])


def TestFunction(V):
    """The test function of V: a scalar Argument, or a vector (ListTensor) on a VectorFunctionSpace."""
    if _is_vector_element(V):
        return _element_vector("test", V)
    return _vector_argument("test", V) if _is_system_2d(V) else Argument("test", V)


def TrialFunction(V):
    """The trial function of V: a scalar Argument, or a vector (ListTensor) on a VectorFunctionSpace."""
    if _is_vector_element(V):
        return _element_vector("trial", V)
    return _vector_argument("trial", V) if _is_system_2d(V) else Argument("trial", V)


class Function(Argument):
    """A known field in a space: an Argument that CARRIES its coefficient vector.

        u = fd.Function(V)                 # zero, named "u" after the variable
        u.project(f)  /  u.interpolate(f)  # set the coefficients from a callable
        u(x), u.derivative(x, k)           # point evaluation
        u.vector                           # the NumPy coefficient array (writable)
        u + v, a*u, u - v, u.assign(w)     # linear algebra between Functions of one space -> Function
        u * v, D(u), fd.sin(u)             # inside a form: expression nodes

    In a form the vector is read at assembly time, so R.assemble() needs no
    arguments.  A keyword override by name, R.assemble(u=other_array), is still
    honored.  np.asarray(u) is the vector, so u drops into NumPy code unchanged.
    """

    def __new__(cls, V=None, name=None, vector=None, _infer=True):
        # a system (ProductSpace in 1D or 2D, VectorFunctionSpace) or a vector element: the known
        # field is a ProductFunction / VectorFunction / VectorElementFunction
        if cls is Function and (hasattr(V, "fields") or _is_vector_element(V)):
            if isinstance(name, (np.ndarray, list, tuple, ListTensor, Expr)) and vector is None:
                name, vector = None, name
            if name is None and _infer:
                name = _infer_name(depth=2)
            if _is_vector_element(V):
                return VectorElementFunction(V, name or "w", vector)
            return _product_function(V, name or "w", vector)
        return super().__new__(cls)

    def __init__(self, V, name=None, vector=None, _infer=True):
        if isinstance(name, (np.ndarray, list, tuple, Expr)) and vector is None:
            name, vector = None, name
        if name is None and _infer:
            name = _infer_name(depth=2)
        if name is None:
            warnings.warn("Function(V): no source to read the variable name from, using name='w'; "
                          "pass name=... if you have more than one Function", stacklevel=2)
            name = "w"
        super().__init__("func", V, name=name, owner=self)
        if vector is None: self.vector = np.zeros(V.dim)
        elif isinstance(vector, Expr): self.vector = np.array(_as_vector(vector, V), dtype=np.float64)
        else: self.vector = np.array(vector, dtype=np.float64).reshape(-1)
        if self.vector.shape != (V.dim,):
            raise ValueError(f"Function: vector length {self.vector.size} != V.dim = {V.dim}")

    # A Function supplies its own values, so its owner is itself.  Stored as an attribute
    # that would be a reference cycle, which refcounting never frees: every A @ x and
    # every solve() result would then wait for the cyclic garbage collector, holding its
    # array and defeating the allocator's reuse in a time-stepping loop.
    @property
    def owner(self):
        return self

    @owner.setter
    def owner(self, value):
        if value is not None and value is not self:
            raise AttributeError("a Function is its own owner")

    @classmethod
    def _adopt(cls, V, name, vector):
        """Wrap a fresh float64 array of length V.dim that the caller hands over: no copy,
        no checks.  Internal, for results the library has just computed."""
        if cls is Function and _is_vector_element(V):
            return VectorElementFunction(V, name, vector, _copy=False)
        f = cls.__new__(cls)
        Argument.__init__(f, "func", V, name=name, owner=f)
        f.vector = vector
        return f

    # ---- values --------------------------------------------------------------
    def __array__(self, dtype=None, copy=None):
        return self.vector if dtype is None else self.vector.astype(dtype)
    def __len__(self): return self.vector.size
    def __getitem__(self, i): return self.vector[i]
    def __setitem__(self, i, v): self.vector[i] = v
    def copy(self, name=None): return Function(self.space, name or self.name, self.vector.copy(), _infer=False)

    def assign(self, other):
        """u.assign(v) copies coefficients from a Function or an array, or evaluates a linear
        combination of Functions of this space such as  u0 + 0.5*dt*k1.  Returns self."""
        self.vector[:] = _as_vector(other, self.space)
        return self

    def project(self, f, npts=None):
        """L2 projection of a callable (or values at the cache nodes) into the space.  Returns self."""
        self.vector[:] = self.space.project(f, npts)
        return self

    def interpolate(self, f):
        """Interpolation at the degrees of freedom' coordinates: nodes for Lagrange, Greville
        abscissae for splines (banded collocation solve).  Returns self."""
        self.vector[:] = self.space.interpolate(f)
        return self

    def project_to(self, W, check: bool = True):
        """Exact L2 projection of this Function onto the space W, a new Function in W.

        Exact for any two grids (the mixed mass matrix is integrated on their union), and
        conservative, int u_W = int u, whenever constants lie in W.  With check=True a
        warning is issued if u violates a condition built into W.  For a transfer that is
        repeated, build T = W.transfer_matrix(u.space) once and call T(u)."""
        from .transfer import Transfer
        return Transfer(W, self.space, "project")(self, check=check, _stacklevel=4)

    def interpolate_to(self, W, check: bool = True):
        """Interpolant of this Function in the space W, collocated at W.dof_coordinates()."""
        from .transfer import Transfer
        return Transfer(W, self.space, "interpolate")(self, check=check, _stacklevel=4)

    def __call__(self, x, k: int = 0):
        """u(x) or u(x, k) = d^k u / dx^k at the points x.  u('-') and u('+') are the restrictions
        to the left and right element of an interior facet, for dS terms."""
        if isinstance(x, str):
            return restrict(self, x)
        if getattr(self.space, "tdim", 1) == 2:
            raise TypeError("on a 2D space call u.at(points) or u.at(x, y) (u(x, y) would be ambiguous with a "
                            "derivative order); u('-') / u('+') restrict")
        return self.space.evaluate(self.vector, x, k)

    def at(self, x, y=None, deriv=0):
        """Values at points on a 2D mesh: u.at(points) for an (n, 2) array, or u.at(x, y) with arrays
        (the result then has their shape).  deriv: 0, 'x' or 'y'.  NaN outside the mesh."""
        if getattr(self.space, "tdim", 1) != 2:
            return self.space.evaluate(self.vector, x, 0 if deriv == 0 else deriv)
        return self.space.evaluate(self.vector, x, y, deriv)

    def plot(self, ax=None, **kw):
        """Plot the Function (2D spaces: V.plot(u, ...))."""
        if not hasattr(self.space, "plot"):
            raise TypeError("plot(): only Functions on a 2D space can plot themselves; plot u(x) against x in 1D")
        return self.space.plot(self, ax=ax, **kw)
    def derivative(self, x, k: int = 1): return self.space.evaluate(self.vector, x, k)
    def at_quad(self, cache, k: int = 0): return self.space.at_quad(cache, self.vector, k)

    @property
    def shape(self): return self.vector.shape

    # ---- arithmetic -------------------------------------------------------------
    # u + v, a*u, u - v, -u build EXPRESSIONS (as in FEniCS), never new Functions, so a
    # Function inside a form is always read live.  Evaluate a linear combination with
    # w.assign(u + 0.5*dt*k) or fd.Function(V, expr).  In-place ops mutate the vector.
    # With a NumPy array operand (which can never enter a form) the result is an array.
    def __add__(self, o):  return self.vector + o if isinstance(o, np.ndarray) else super().__add__(o)
    def __radd__(self, o): return o + self.vector if isinstance(o, np.ndarray) else super().__radd__(o)
    def __sub__(self, o):  return self.vector - o if isinstance(o, np.ndarray) else super().__sub__(o)
    def __rsub__(self, o): return o - self.vector if isinstance(o, np.ndarray) else super().__rsub__(o)
    def __mul__(self, o):  return self.vector * o if isinstance(o, np.ndarray) else super().__mul__(o)
    def __rmul__(self, o): return o * self.vector if isinstance(o, np.ndarray) else super().__rmul__(o)
    def __iadd__(self, o): self.vector += _as_vector(o, self.space); return self
    def __isub__(self, o): self.vector -= _as_vector(o, self.space); return self
    def __imul__(self, a): self.vector *= float(a); return self
    def __itruediv__(self, a): self.vector /= float(a); return self

    def __repr__(self):
        return f"Function({self.name!r}, {self.space!r})"


def _as_vector(o, V):
    """Coefficient vector of: a Function of V, an array, or a LINEAR combination of Functions of V."""
    if isinstance(o, (Function, VectorElementFunction)):
        if o.space is not V: raise ValueError("Functions live in different spaces")
        return o.vector
    if isinstance(o, ListTensor):
        raise ValueError("a vector expression is not a coefficient vector; use V.project()")
    if isinstance(o, Expr):
        return _linear_combination(o, V)
    v = np.asarray(o, dtype=np.float64).reshape(-1)
    if v.shape != (V.dim,): raise ValueError(f"vector length {v.size} != V.dim = {V.dim}")
    return v


def _linear_combination(e, V):
    """Evaluate sum_i a_i u_i for Functions u_i of V; raise if e is not of that form."""
    if isinstance(e, Function):
        if e.space is not V: raise ValueError("Functions live in different spaces")
        return e.vector.copy()
    if isinstance(e, (Const, Constant)):
        raise ValueError("a constant is not in the space; use project()")
    if isinstance(e, Sum):
        out = np.zeros(V.dim)
        for t in e.terms: out += _linear_combination(t, V)
        return out
    if isinstance(e, Prod):
        scal, funcs = 1.0, []
        for f in e.factors:
            if isinstance(f, (Const, Constant)): scal *= float(f.value)
            elif isinstance(f, Function): funcs.append(f)
            elif isinstance(f, Prod) or isinstance(f, Sum): funcs.append(f)
            else: raise ValueError(f"{e!r} is not a linear combination of Functions; use project()")
        if len(funcs) != 1: raise ValueError(f"{e!r} is not linear in the Functions; use project()")
        return scal * _linear_combination(funcs[0], V)
    raise ValueError(f"{e!r} is not a linear combination of Functions; use project()")


class ProductFunction:
    """A known field on a ProductSpace: one global vector, per-field components usable in forms.

    On a 2D mesh it unpacks by BLOCK: for W = ProductSpace(VectorFunctionSpace(m, 2), Q),
    u, p = w gives the velocity as a vector expression and the pressure as a scalar one."""
    def __init__(self, P, name="w", vector=None):
        self.product, self.name = P, name
        self.vector = np.zeros(P.dim) if vector is None else np.array(vector, dtype=np.float64).reshape(-1)
        if self.vector.shape != (P.dim,):
            raise ValueError(f"ProductFunction: vector length {self.vector.size} != dim = {P.dim}")
        self.components = tuple(_element_vector("func", f, field=i, product=P, name=name, owner=self)
                                if getattr(f, "ncomp", 1) == 2 else
                                Argument("func", f, name=name, field=i, product=P, owner=self) for i, f in enumerate(P.fields))
    def __array__(self, dtype=None, copy=None): return self.vector if dtype is None else self.vector.astype(dtype)

    @property
    def space(self):
        return self.product

    @property
    def blocks(self):
        """Per block of a 2D system: a vector expression for a vector block, a scalar one otherwise."""
        if not _is_system_2d(self.product):
            return self.components
        return tuple(ListTensor([self.components[i] for i in idx]) if kind == "vector" else self.components[idx[0]]
                     for kind, idx in self.product.blocks)

    def __iter__(self): return iter(self.blocks)
    def __getitem__(self, i): return self.blocks[i]
    def __len__(self): return len(self.blocks)

    def split(self):
        """Per-field Functions (copies of the current coefficients); per block on a 2D system."""
        parts = self.product.split(self.vector)
        if not _is_system_2d(self.product):
            return tuple(Function(f, f"{self.name}[{i}]", part, _infer=False)
                         for i, (f, part) in enumerate(zip(self.product.fields, parts)))
        out = []
        for b, (kind, idx) in enumerate(self.product.blocks):
            if kind == "vector":
                B = self.product.block_space(b)
                out.append(VectorFunction(B, f"{self.name}[{b}]", np.concatenate([parts[i] for i in idx])))
            elif kind == "element":
                out.append(VectorElementFunction(self.product.fields[idx[0]], f"{self.name}[{b}]", parts[idx[0]]))
            else:
                out.append(Function(self.product.fields[idx[0]], f"{self.name}[{b}]", parts[idx[0]], _infer=False))
        return tuple(out)

    def assign(self, other):
        self.vector[:] = np.asarray(getattr(other, "vector", other), dtype=np.float64).reshape(-1)
        return self

    def project(self, f, degree=None):
        """L2 projection of per-field data onto the space, one entry per field (per block on a 2D
        system, see ProductSpace2D.project).  In 1D each entry is anything Function.project takes,
        an expression in fd.x or a callable, or None for zero; degree is then the number of Gauss
        points (npts).  Returns self."""
        if hasattr(self.product, "project"):
            self.vector[:] = self.product.project(f, degree)
        else:
            self.vector[:] = self._per_field(f, lambda g, V: Function(V, "_", None, _infer=False).project(g, degree))
        return self

    def interpolate(self, f):
        """Nodal interpolation of per-field data (per block on a 2D system), one entry per field,
        None for zero.  Returns self."""
        if hasattr(self.product, "interpolate"):
            self.vector[:] = self.product.interpolate(f)
        else:
            self.vector[:] = self._per_field(f, lambda g, V: Function(V, "_", None, _infer=False).interpolate(g))
        return self

    def _per_field(self, f, make):
        fields = self.product.fields
        if not isinstance(f, (list, tuple)) or len(f) != len(fields):
            raise ValueError(f"expected a list with one entry per field ({len(fields)}), got {f!r}")
        return self.product.gather([np.zeros(V.dim) if g is None else make(g, V).vector
                                    for g, V in zip(f, fields)])

    def copy(self, name=None):
        return _product_function(self.product, name or self.name, self.vector.copy())

    def __repr__(self): return f"ProductFunction({self.name!r}, {self.product!r})"


class VectorFunction(ProductFunction, ListTensor):
    """A known vector field on a VectorFunctionSpace: a ProductFunction that is also a vector
    expression, so dot(grad(w), w), div(w) and w[0] enter forms directly."""
    def __init__(self, P, name="w", vector=None):
        ProductFunction.__init__(self, P, name, vector)

    def __iter__(self): return iter(self.components)
    def __getitem__(self, i): return ListTensor.__getitem__(self, i)
    def __len__(self): return len(self.components)

    def split(self):
        """The components as scalar Functions (copies of the coefficients)."""
        return tuple(Function(f, f"{self.name}[{i}]", part, _infer=False)
                     for i, (f, part) in enumerate(zip(self.product.fields, self.product.split(self.vector))))

    def at(self, x, y=None):
        """Values at points: an (n, 2) array for (n, 2) points, or (..., 2) for x and y arrays."""
        parts = self.product.split(self.vector)
        vals = [f.evaluate(p, x, y) for f, p in zip(self.product.fields, parts)]
        return np.stack(vals, axis=-1)

    def plot(self, ax=None, **kw):
        return self.product.plot(self, ax=ax, **kw)

    def __repr__(self): return f"VectorFunction({self.name!r}, {self.product!r})"


def _element_combination(e, V):
    """The coefficients of a LINEAR combination of VectorElementFunctions of V (w1 + 0.5*w2), read
    from its first component and checked on the second."""
    if isinstance(e, VectorElementFunction):
        if e.space is not V:
            raise ValueError("Functions live in different spaces")
        return e.vector.copy()
    if not (isinstance(e, ListTensor) and e.rank == 1 and len(e) == 2):
        raise ValueError(f"{e!r} is not a linear combination of Functions of this space; use V.project()")

    def comb(x, c):
        if isinstance(x, Argument) and x.kind == "func" and isinstance(x.owner, VectorElementFunction) \
                and x.comp == c and not any(x.deriv if isinstance(x.deriv, tuple) else (x.deriv,)):
            if x.owner.space is not V:
                raise ValueError("Functions live in different spaces")
            return x.owner.vector.copy()
        if isinstance(x, Sum):
            return sum((comb(t, c) for t in x.terms), np.zeros(V.dim))
        if isinstance(x, Prod):
            scal, rest = 1.0, []
            for f in x.factors:
                if isinstance(f, (Const, Constant)):
                    scal *= float(f.value)
                else:
                    rest.append(f)
            if len(rest) != 1:
                raise ValueError(f"{e!r} is not linear in the Functions; use V.project()")
            return scal * comb(rest[0], c)
        raise ValueError(f"{e!r} is not a linear combination of Functions of this space; use V.project()")
    v0, v1 = comb(e[0], 0), comb(e[1], 1)
    if not np.allclose(v0, v1, rtol=0, atol=1e-14 * max(1.0, np.abs(v0).max())):
        raise ValueError(f"{e!r} mixes components; it is not a combination of whole Functions")
    return v0


class VectorElementFunction(ListTensor):
    """A known field of a vector-valued element space (Raviart-Thomas, Nedelec): fd.Function(V)
    returns one.  It is a vector expression, so div(w), curl(w), dot(w, v) and w[0] enter forms,
    and it carries its coefficient vector as a Function does.

        w = fd.Function(V)              # zero, named after the variable
        w.interpolate(lambda x, y: (y, -x))
        w.at(points)                    # (n, 2) values; w.at(points, what="div") the divergence
        w.vector                        # the coefficients (normal or tangential moments)"""

    def __init__(self, V, name="w", vector=None, _copy=True):
        self.space, self.name = V, name
        if vector is None:
            self.vector = np.zeros(V.dim)
        elif isinstance(vector, (Expr, ListTensor)):
            self.vector = _element_combination(vector, V)
        else:
            self.vector = np.array(vector, dtype=np.float64).reshape(-1) if _copy else vector
        if self.vector.shape != (V.dim,):
            raise ValueError(f"Function: vector length {self.vector.size} != V.dim = {V.dim}")
        ListTensor.__init__(self, [Argument("func", V, name=name, owner=self, comp=c) for c in (0, 1)])

    @property
    def owner(self):
        return self

    def __array__(self, dtype=None, copy=None):
        return self.vector if dtype is None else self.vector.astype(dtype)

    def copy(self, name=None):
        return VectorElementFunction(self.space, name or self.name, self.vector.copy())

    def assign(self, other):
        """Copy coefficients from a Function of this space or an array.  Returns self."""
        if isinstance(other, VectorElementFunction):
            if other.space is not self.space:
                raise ValueError("Functions live in different spaces")
            other = other.vector
        self.vector[:] = np.asarray(other, dtype=np.float64).reshape(-1)
        return self

    def interpolate(self, f):
        """The canonical interpolant (edge and interior moments) of f(x, y) -> (fx, fy).  Returns self."""
        self.vector[:] = self.space.interpolate(f)
        return self

    def project(self, f, degree=None):
        """L2 projection of f(x, y) -> (fx, fy) (or a vector expression).  Returns self."""
        self.vector[:] = self.space.project(f, degree)
        return self

    def at(self, x, y=None, what="value"):
        """Values at points: (n, 2) for "value", or one number per point for "div", "curl", "x", "y"."""
        return self.space.evaluate(self.vector, x, y, what)

    def plot(self, ax=None, **kw):
        return self.space.plot(self, ax=ax, **kw)

    def __iadd__(self, o): self.vector += np.asarray(getattr(o, "vector", o), dtype=np.float64); return self
    def __isub__(self, o): self.vector -= np.asarray(getattr(o, "vector", o), dtype=np.float64); return self
    def __imul__(self, a): self.vector *= float(a); return self

    def __repr__(self): return f"Function({self.name!r}, {self.space!r})"


def _product_function(P, name, vector=None):
    if getattr(P, "is_vector", False):
        return VectorFunction(P, name, vector)
    return ProductFunction(P, name, vector)


def TestFunctions(P):
    """The test functions of a system, one per field (1D) or one per block (2D: vector or scalar)."""
    if _is_system_2d(P):
        return tuple(ListTensor([Argument("test", P.fields[i], field=i, product=P) for i in idx]) if kind == "vector"
                     else _element_vector("test", P.fields[idx[0]], field=idx[0], product=P) if kind == "element"
                     else Argument("test", P.fields[idx[0]], field=idx[0], product=P) for kind, idx in P.blocks)
    return tuple(Argument("test", f, field=i, product=P) for i, f in enumerate(P.fields))


def TrialFunctions(P):
    """The trial functions of a system, one per field (1D) or one per block (2D: vector or scalar)."""
    if _is_system_2d(P):
        return tuple(ListTensor([Argument("trial", P.fields[i], field=i, product=P) for i in idx]) if kind == "vector"
                     else _element_vector("trial", P.fields[idx[0]], field=idx[0], product=P) if kind == "element"
                     else Argument("trial", P.fields[idx[0]], field=idx[0], product=P) for kind, idx in P.blocks)
    return tuple(Argument("trial", f, field=i, product=P) for i, f in enumerate(P.fields))


def Functions(P, name: str | None = None, vector=None):
    """A ProductFunction: one global coefficient vector, iterable into per-field components for forms
    (per block on a 2D system).  (w1, w2) = fd.Functions(P)  unpacks the components."""
    if name is None:
        name = _infer_name(depth=2, callee="Functions") or "w"
    return _product_function(P, name, vector)


# ---- measures -------------------------------------------------------------------
class Measure:
    def __init__(self, kind, where=None, quad_degree=None, scheme="gauss"):
        self.kind, self.where, self.quad_degree, self.scheme = kind, where, quad_degree, scheme

    def __call__(self, *args, quad_degree=None, scheme=None):
        if self.kind == "dS":
            if args or quad_degree is not None or scheme is not None:
                raise TypeError("dS takes no arguments: it is every interior facet (and the seam of a periodic space)")
            return self
        if self.kind == "dx":
            if args:
                quad_degree = args[0]
            scheme = "gauss" if scheme is None else str(scheme).lower()
            if scheme not in ("gauss", "lobatto"):
                raise ValueError(f"dx: scheme must be 'gauss' or 'lobatto', got {scheme!r}")
            if quad_degree is not None and scheme == "lobatto" and int(quad_degree) < 2:
                raise ValueError("dx(n, scheme='lobatto'): a Gauss-Lobatto rule needs at least 2 points")
            return Measure("dx", None, quad_degree, scheme)
        if scheme is not None:
            raise TypeError("scheme= is a quadrature option of dx; ds is a point evaluation in 1D")
        where = args[0] if args else None
        if len(args) > 1:
            where = tuple(args)
        if isinstance(where, (int, np.integer)):
            return Measure("ds", (int(where),), quad_degree)
        if isinstance(where, (tuple, list)) and where and all(isinstance(m, (int, np.integer)) for m in where):
            return Measure("ds", tuple(sorted({int(m) for m in where})), quad_degree)
        if where not in (None, "left", "right", "both"):
            raise ValueError("ds takes 'left', 'right' or 'both' on a 1D mesh, and a marker or markers on a 2D one")
        return Measure("ds", where or "both", quad_degree)

    def __rmul__(self, e):
        return Integral(_lift(e), self)


dx = Measure("dx")
ds = Measure("ds", "both")
dS = Measure("dS")          # interior facets: the vertices between elements, plus the seam when periodic


class Integral:
    def __init__(self, expr, measure): self.expr, self.measure = expr, measure
    def __add__(self, o):  return FormExpr([self]) + o
    def __radd__(self, o): return FormExpr([self]) + o
    def __sub__(self, o):  return FormExpr([self]) - o
    def __neg__(self):     return Integral(Prod([Const(-1.0), self.expr]), self.measure)
    def __mul__(self, c):  return Integral(Prod([_lift(c), self.expr]), self.measure)
    __rmul__ = __mul__


class FormExpr:
    def __init__(self, integrals): self.integrals = list(integrals)
    def __add__(self, o):
        if isinstance(o, Integral):  return FormExpr(self.integrals + [o])
        if isinstance(o, FormExpr):  return FormExpr(self.integrals + o.integrals)
        if o == 0:                   return self
        raise TypeError("can only add integrals to a form")
    __radd__ = __add__
    def __sub__(self, o):
        o = o if isinstance(o, FormExpr) else FormExpr([o])
        return FormExpr(self.integrals + [-i for i in o.integrals])
    def __neg__(self): return FormExpr([-i for i in self.integrals])


# ============================================================================ analysis
def has_arguments(e, kinds) -> bool:
    if isinstance(e, Argument): return e.kind in kinds
    if isinstance(e, Sum):      return any(has_arguments(t, kinds) for t in e.terms)
    if isinstance(e, Prod):     return any(has_arguments(f, kinds) for f in e.factors)
    if isinstance(e, Pow):      return has_arguments(e.base, kinds)
    if isinstance(e, Func):     return has_arguments(e.arg, kinds)
    return False


def collect_arguments(e, out):
    if isinstance(e, Argument): out.append(e)
    elif isinstance(e, Sum):    [collect_arguments(t, out) for t in e.terms]
    elif isinstance(e, Prod):   [collect_arguments(f, out) for f in e.factors]
    elif isinstance(e, Pow):    collect_arguments(e.base, out)
    elif isinstance(e, Func):   collect_arguments(e.arg, out)
    return out


def collect_constants(e, out):
    if isinstance(e, Constant): out.append(e)
    elif isinstance(e, Sum):    [collect_constants(t, out) for t in e.terms]
    elif isinstance(e, Prod):   [collect_constants(f, out) for f in e.factors]
    elif isinstance(e, Pow):    collect_constants(e.base, out)
    elif isinstance(e, Func):   collect_constants(e.arg, out)
    return out


def expand(e):
    """Sum-of-products over the test/trial structure.  Pointwise subtrees stay opaque.
    Returns a list of factor lists."""
    if not has_arguments(e, ("test", "trial")):
        return [[e]]
    if isinstance(e, Argument):
        return [[e]]
    if isinstance(e, Sum):
        return [t for term in e.terms for t in expand(term)]
    if isinstance(e, Prod):
        out = [[]]
        for f in e.factors:
            out = [a + b for a in out for b in expand(f)]
        return out
    if isinstance(e, Pow):
        if e.n == 1: return expand(e.base)
        raise TypeError(f"{e!r}: a form must be linear in each test and trial function")
    raise TypeError(f"cannot expand {e!r}")


class Term:
    """coeff * D^a(test) * D^b(trial), either argument possibly absent."""
    def __init__(self, test, trial, coeff, measure):
        self.test, self.trial, self.coeff, self.measure = test, trial, coeff, measure

    @property
    def rank(self): return (self.test is not None) + (self.trial is not None)

    def __repr__(self):
        parts = [repr(self.coeff)] + [repr(a) for a in (self.test, self.trial) if a is not None]
        tail = {"dx": "*dx", "dS": "*dS"}.get(self.measure.kind, f"*ds({self.measure.where})")
        return "*".join(parts) + tail


def canonicalize(fe: FormExpr):
    terms = []
    for integral in fe.integrals:
        for factors in expand(integral.expr):
            tests  = [f for f in factors if isinstance(f, Argument) and f.kind == "test"]
            trials = [f for f in factors if isinstance(f, Argument) and f.kind == "trial"]
            rest   = [f for f in factors if not (isinstance(f, Argument) and f.kind in ("test", "trial"))]
            if len(tests) > 1 or len(trials) > 1:
                raise TypeError(f"term {factors!r} is not linear in the test/trial functions")
            if trials and not tests:
                raise TypeError("a term with a trial function but no test function has no row index")
            coeff = Prod(rest) if len(rest) != 1 else rest[0]
            if not rest:
                coeff = Const(1.0)
            terms.append(Term(tests[0] if tests else None, trials[0] if trials else None, coeff, integral.measure))
    # a zero constant term (sum() starts from 0, and 0 * dx is rank 0) would only spoil the rank
    nonzero = [t for t in terms if not (t.test is None and t.trial is None and _trivially_zero(t.coeff))]
    return nonzero if nonzero else terms


def _trivially_zero(e):
    """A coefficient that is zero by its structure: 0, a product with a factor 0, a sum of zeros."""
    if isinstance(e, Const):
        return e.value == 0.0
    if isinstance(e, Prod):
        return any(_trivially_zero(f) for f in e.factors)
    if isinstance(e, Sum):
        return all(_trivially_zero(t) for t in e.terms)
    return False


def poly_degree(e):
    """Polynomial degree per element of a pointwise expression, or None if not polynomial."""
    if isinstance(e, (Const, Constant)): return 0
    if isinstance(e, Spatial):  return 1
    if isinstance(e, Normal):   return 0
    if isinstance(e, Argument): return _arg_degree(e)
    if isinstance(e, Sum):
        ds_ = [poly_degree(t) for t in e.terms]
        return None if any(d is None for d in ds_) else max(ds_, default=0)
    if isinstance(e, Prod):
        ds_ = [poly_degree(f) for f in e.factors]
        return None if any(d is None for d in ds_) else sum(ds_)
    if isinstance(e, Pow):
        d = poly_degree(e.base)
        if d is None or not float(e.n).is_integer() or e.n < 0: return None
        return int(e.n) * d
    return None


_default_quad_degree = None


def set_quadrature_degree(degree=None):
    """Use one quadrature rule in every form created from now on, instead of inferring it per form.

    `degree` is what `dx(degree)` would set on each measure of the form: on a 2D mesh the degree of
    exactness of the rule on the cells, the sides and the interior facets, on a 1D mesh the number of
    Gauss points per element (exact to degree 2n - 1). A measure with its own `quad_degree=` still
    wins, and `dx(scheme="lobatto")` is not affected. `set_quadrature_degree(None)` returns to the
    inferred rules. Forms made earlier keep their rule."""
    global _default_quad_degree
    if degree is None:
        _default_quad_degree = None
        return
    n = int(degree)
    if n != degree or n < 1:
        raise ValueError(f"set_quadrature_degree: degree must be a positive integer or None, got {degree!r}")
    _default_quad_degree = n


def quadrature_degree():
    """The degree set by `set_quadrature_degree`, or None when the rules are inferred per form."""
    return _default_quad_degree


def _measure_degree(measure):
    """The degree a measure asks for: its own quad_degree, else the global default, else None."""
    if measure.quad_degree is not None:
        return int(measure.quad_degree)
    return _default_quad_degree


def quad_degree_estimate(e, kmax):
    """Degree of the polynomial that stands in for a pointwise expression when choosing the quadrature
    rule. Polynomial parts count exactly. A non-polynomial expression of the data alone (sin x,
    exp(-x**2) * sin y, ...) counts as degree kmax, and so does a non-polynomial function of a field
    (sin u, sqrt(w)). The callers pass kmax = k + 1 for a space of degree k: the interpolant of degree k
    plus the leading term of its remainder, which is a polynomial of degree k + 1 on each element. So
    f v is integrated exactly for degree 2k + 1 and the error norm (w - exact)**2 for degree 2k + 2,
    which is what the square of the leading error term needs."""
    d = poly_degree(e)
    if d is not None: return d
    if not collect_arguments(e, []): return kmax           # data alone: one function to interpolate
    if isinstance(e, Argument): return _arg_degree(e)
    if isinstance(e, Sum):
        return max((quad_degree_estimate(t, kmax) for t in e.terms), default=0)
    if isinstance(e, Prod):
        return sum(quad_degree_estimate(f, kmax) for f in e.factors)
    if isinstance(e, Pow):
        if float(e.n).is_integer() and e.n >= 0:
            return int(e.n) * quad_degree_estimate(e.base, kmax)
        return kmax
    return kmax


def diff(e, slot):
    """Symbolic derivative of a pointwise expression w.r.t. the Function slot (name, field, k)."""
    if isinstance(e, (Const, Constant, Spatial, Normal)):
        return Const(0.0)
    if isinstance(e, Argument):
        return Const(1.0) if e.kind == "func" and e.slot() == slot else Const(0.0)
    if isinstance(e, Sum):
        parts = [diff(t, slot) for t in e.terms]
        parts = [p for p in parts if not _is_zero(p)]
        return Const(0.0) if not parts else (parts[0] if len(parts) == 1 else Sum(parts))
    if isinstance(e, Prod):
        out = []
        for i, f in enumerate(e.factors):
            d = diff(f, slot)
            if not _is_zero(d):
                out.append(Prod(e.factors[:i] + [d] + e.factors[i + 1:]))
        return Const(0.0) if not out else (out[0] if len(out) == 1 else Sum(out))
    if isinstance(e, Pow):
        d = diff(e.base, slot)
        if _is_zero(d): return Const(0.0)
        return Prod([Const(float(e.n)), Pow(e.base, e.n - 1), d])
    if isinstance(e, Func):
        d = diff(e.arg, slot)
        if _is_zero(d): return Const(0.0)
        return Prod([_dfunc(e), d])
    raise TypeError(f"cannot differentiate {e!r}")


def _is_zero(e): return isinstance(e, Const) and e.value == 0.0


# ============================================================================ evaluation
def _source_vector(a, values, owner_values=None):
    """The coefficient vector a known field reads: an internal override by id(owner), a keyword value
    by name, or the owner's vector."""
    if owner_values and a.owner is not None and id(a.owner) in owner_values:
        vals = owner_values[id(a.owner)]
    elif a.name in values:
        vals = values[a.name]
        vals = vals.vector if isinstance(vals, (Function, ProductFunction)) else vals
    elif a.owner is not None:
        vals = a.owner.vector
    else:
        raise KeyError(f"no value supplied for Function '{a.name}'")
    return np.asarray(vals)


class _Context:
    """Where pointwise expressions get their numbers: quadrature nodes of the caches, or one endpoint."""
    def __init__(self, xs, values, caches=None, point=None, node_space=None, owner_values=None):
        self.xs, self.values, self.caches, self.point = xs, values, caches, point
        self.node_space = node_space      # the space whose quadrature nodes xs are
        self.owner_values = owner_values  # {id(Function or ProductFunction): vector}, internal overrides

    def func_values(self, a: Argument):
        """The values of a known field at the context's points, computed once per field, derivative,
        side and component in one assembly (a flux names the same traces many times)."""
        memo = self.__dict__.setdefault("_memo", {})
        key = a.slot()
        v = memo.get(key)
        if v is None:
            v = self._func_values(a)
            memo[key] = v
        return v

    def _source(self, a: Argument):
        """The coefficient vector of the known field a reads: an override, a value, or the owner's."""
        return _source_vector(a, self.values, self.owner_values)

    def _func_values(self, a: Argument):
        vals = self._source(a)
        if np.iscomplexobj(vals):
            # every map from coefficients to values is real and linear: take the parts apart
            return self._real_values(a, vals.real) + 1j * self._real_values(a, vals.imag)
        return self._real_values(a, vals)

    def _real_values(self, a, vals):
        if a.product is not None:
            vals = a.product.split(np.ascontiguousarray(vals, dtype=np.float64))[a.field]
        vals = np.ascontiguousarray(vals, dtype=np.float64)
        if self.point is not None:
            return a.space.evaluate(vals, [self.point], a.deriv)
        if self.node_space is None or _same_mesh(a.space, self.node_space):
            return a.space.at_quad(self.caches[id(a.space)], vals, a.deriv)
        return a.space.evaluate(vals, self.xs, a.deriv)     # another grid: evaluate at these nodes


class _FacetContext(_Context):
    """Values on the interior facets: xs are the facet points, and a Function is read on the side
    it is restricted to ('-' when unrestricted, which a single-valued field allows)."""
    def __init__(self, xs, values, facet_cache, owner_values=None):
        super().__init__(xs, values, None, owner_values=owner_values)
        self._facet_cache = facet_cache           # space -> FacetCache

    def _real_values(self, a, vals):
        if a.product is not None:
            vals = a.product.split(np.ascontiguousarray(vals, dtype=np.float64))[a.field]
        F = self._facet_cache(a.space)
        side = 0 if (a.side or "-") == "-" else 1
        return np.asarray(_C.facet_values(F, np.ascontiguousarray(vals, dtype=np.float64), side, a.deriv))


class _Context2D(_Context):
    """Values at the points of a 2D cache (cells or boundary facets)."""
    def __init__(self, form_, Q, key, node_space, values, owner_values=None):
        super().__init__(np.asarray(Q.x()), values, None, owner_values=owner_values)
        self.ys = np.asarray(Q.y())
        self.normals = (np.asarray(Q.normal_x()), np.asarray(Q.normal_y())) if Q.on_facets else None
        self._form, self._Q, self._key, self._node_space = form_, Q, key, node_space

    def _func_values(self, a: Argument):
        """A scalar field on the form's mesh: the value and the derivatives the form reads of it are
        computed together, in one pass over the points (Cache2D.at_points_multi), and kept for the
        other slots of the same field."""
        Q = self._fast_cache(a)
        if Q is None:
            return super()._func_values(a)
        key = (a.slot()[0], a.field, a.side)
        fields = self.__dict__.setdefault("_fields", {})
        got = fields.setdefault(key, {})
        code = a.code()
        if code not in got:
            need = getattr(self._form, "_codes2d", {}).get(key, ())
            codes = sorted(({code} | set(need)) - set(got))
            vals = self._source(a)
            if np.iscomplexobj(vals):
                rows = self._multi(Q, a, vals.real, codes) + 1j * self._multi(Q, a, vals.imag, codes)
            else:
                rows = self._multi(Q, a, vals, codes)
            got.update(zip(codes, rows))
        return got[code]

    def _fast_cache(self, a):
        if a.comp is not None or getattr(a.space, "tdim", 1) != 2 or a.space.mesh is not self._node_space.mesh:
            return None
        Q = self._form._cache2d(a.space, self._key)
        if getattr(Q, "ncodes", 0) != 3 or not hasattr(Q, "at_points_multi"):
            return None
        return Q

    @staticmethod
    def _multi(Q, a, vals, codes):
        if a.product is not None:
            vals = a.product.split(np.ascontiguousarray(vals, dtype=np.float64))[a.field]
        return Q.at_points_multi(np.ascontiguousarray(vals, dtype=np.float64), codes)

    def _real_values(self, a, vals):
        if a.product is not None:
            vals = a.product.split(np.ascontiguousarray(vals, dtype=np.float64))[a.field]
        vals = np.ascontiguousarray(vals, dtype=np.float64)
        code = a.code()
        if getattr(a.space, "tdim", 1) != 2:
            raise TypeError(f"{a!r} lives on a 1D mesh and cannot enter a form on a 2D one")
        if a.space.mesh is self._node_space.mesh:
            return np.asarray(self._form._cache2d(a.space, self._key).at_points(vals, code))
        pts = np.column_stack([self.xs, self.ys])
        if a.comp is not None:
            return np.asarray(a.space.evaluate(vals, pts, what=code))
        return np.asarray(a.space.evaluate(vals, pts, deriv=code))


class _FacetContext2D(_Context):
    """Values at the points of the interior facets of a 2D mesh (dS): a Function is read on the side
    it is restricted to ('-' when it is not, which a single-valued field allows), and n('+') = -n('-')."""
    def __init__(self, form_, F, values, owner_values=None):
        super().__init__(np.asarray(F.x()), values, None, owner_values=owner_values)
        self.ys = np.asarray(F.y())
        self.normals = (np.asarray(F.normal_x()), np.asarray(F.normal_y()))
        self.facet = True
        self._form, self._F = form_, F

    def _real_values(self, a, vals):
        if a.product is not None:
            vals = a.product.split(np.ascontiguousarray(vals, dtype=np.float64))[a.field]
        vals = np.ascontiguousarray(vals, dtype=np.float64)
        if getattr(a.space, "tdim", 1) != 2:
            raise TypeError(f"{a!r} lives on a 1D mesh and cannot enter a form on a 2D one")
        if a.space.mesh is not self._F_space_mesh():
            raise ValueError(f"dS: {a!r} lives on another mesh; every field of a dS term must be on the form's mesh")
        side = 1 if a.side == "+" else 0
        return np.asarray(self._form._cache2d(a.space, ("dS",)).at_points(vals, side, a.code()))

    def _F_space_mesh(self):
        return self._form._facet_mesh


def _pattern_2d(T, R, facets=False):
    """The CSR pattern of matrices with test space T and trial space R, each a LagrangeSpace2D or a
    2D ProductSpace (cached on T).  facets=True adds the couplings through interior facets (dS)."""
    T = getattr(T, "_base", None) or T                 # a slip condition: the pattern before reduction
    R = getattr(R, "_base", None) or R
    cache = T.__dict__.setdefault("_pattern", {})
    P = cache.get(("dS", id(R)) if facets else id(R))
    if P is None and facets:
        tf, to = (list(T.fields), list(T.offsets[:-1])) if hasattr(T, "fields") else ([T], [0])
        rf, ro = (list(R.fields), list(R.offsets[:-1])) if hasattr(R, "fields") else ([R], [0])
        P = _C.facet_pattern(tf, [int(o) for o in to], T.dim, rf, [int(o) for o in ro], R.dim)
        cache[("dS", id(R))] = P
    if P is None:
        tf, to = (list(T.fields), list(T.offsets[:-1])) if hasattr(T, "fields") else ([T], [0])
        rf, ro = (list(R.fields), list(R.offsets[:-1])) if hasattr(R, "fields") else ([R], [0])
        if len(tf) == 1 and len(rf) == 1 and not hasattr(T, "fields") and not hasattr(R, "fields"):
            P = _C.cell_pattern(T, R)
        else:
            P = _C.product_pattern(tf, to, T.dim, rf, ro, R.dim)
        cache[id(R)] = P
    return P


def _has_normal(e):
    if isinstance(e, Normal): return True
    if isinstance(e, Sum):  return any(_has_normal(t) for t in e.terms)
    if isinstance(e, Prod): return any(_has_normal(f) for f in e.factors)
    if isinstance(e, Pow):  return _has_normal(e.base)
    if isinstance(e, Func): return _has_normal(e.arg)
    return False


class _RectAccum:
    """Triplets of a rectangular form while it is assembled; finish() gives the RectMatrix."""
    def __init__(self, T, R):
        self.T, self.R, self.parts, self.extra = T, R, [], []

    def add_entry(self, i, j, v):
        self.extra.append((i, j, v))

    def finish(self):
        from . import RectMatrix
        rows = [np.asarray(p[0], dtype=np.int32) for p in self.parts]
        cols = [np.asarray(p[1], dtype=np.int32) for p in self.parts]
        vals = [np.asarray(p[2], dtype=np.float64) for p in self.parts]
        if self.extra:
            e = np.array(self.extra, dtype=np.float64)
            rows.append(e[:, 0].astype(np.int32)); cols.append(e[:, 1].astype(np.int32)); vals.append(e[:, 2])
        if rows:
            r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        else:
            r = c = np.zeros(0, dtype=np.int32); v = np.zeros(0)
        K = _C.csr_from_triplets(int(self.T.dim), int(self.R.dim), np.ascontiguousarray(r), np.ascontiguousarray(c),
                                 np.ascontiguousarray(v))
        return RectMatrix(K, self.T, self.R)


def _same_mesh(A, B):
    """Same elements, so the two spaces' caches of one rule have the same nodes."""
    if A is B:
        return True
    if A.nelem != B.nelem:
        return False
    return np.array_equal(A.grid, B.grid)


def evaluate(e, ctx: _Context):
    if isinstance(e, Const):   return np.full_like(ctx.xs, e.value)
    if isinstance(e, Constant):
        v = ctx.values.get(e.name, e.value)
        v = _scalar(v)
        return np.full(ctx.xs.shape, v, dtype=complex if isinstance(v, complex) else np.float64)
    if isinstance(e, Spatial):
        if e.index == 0: return ctx.xs
        ys = getattr(ctx, "ys", None)
        if ys is None:
            raise TypeError("y appears in a form on a 1D mesh")
        return ys
    if isinstance(e, Normal):
        nrm = getattr(ctx, "normals", None)
        if nrm is None:
            raise TypeError("the facet normal n appears outside a ds or dS term on a 2D mesh")
        if getattr(ctx, "facet", False):
            if e.side is None:
                raise TypeError("in a dS term the normal is two-valued: use n('-') (out of the '-' cell) or n('+')")
            return nrm[e.index] if e.side == "-" else -nrm[e.index]
        if e.side is not None:
            raise TypeError("the restrictions n('-') and n('+') belong in dS terms")
        return nrm[e.index]
    if isinstance(e, Argument):
        if e.kind != "func":
            raise TypeError("internal: test/trial function inside a coefficient")
        return ctx.func_values(e)
    if isinstance(e, (Sum, Prod, Pow, Func)):
        # a subexpression shared by several terms (a flux's wave speed) is evaluated once per context
        memo = ctx.__dict__.setdefault("_ememo", {})
        v = memo.get(id(e))
        if v is not None and v[0] is e:
            return v[1]
        if isinstance(e, Sum):
            r = sum(evaluate(t, ctx) for t in e.terms)
        elif isinstance(e, Prod):
            r = None
            for f in e.factors:
                fv = evaluate(f, ctx)
                r = fv if r is None else r * fv
            r = np.ones_like(ctx.xs) if r is None else (r if np.ndim(r) else np.full_like(ctx.xs, r))
        elif isinstance(e, Pow):
            r = evaluate(e.base, ctx) ** e.n
        else:
            r = Func.TABLE[e.name](evaluate(e.arg, ctx))
        memo[id(e)] = (e, r)
        return r
    raise TypeError(f"cannot evaluate {e!r}")


# ============================================================================ compiled integrands
_COMPILED = True


def set_compiled_integrands(on=True):
    """Assemble forms in C++ (default): the coefficients by the Integrand programs (in complex arithmetic
    for complex values, the complex-step derivative) and the whole loop by FormAssembler1D /
    FormAssembler2D.  False: the Python driver with the NumPy evaluation of the coefficients."""
    global _COMPILED
    _COMPILED = bool(on)


def compiled_integrands():
    return _COMPILED


class _Program:
    """A coefficient expression compiled for _C.Integrand (forms/integrand.hpp): a register program
    whose inputs are the values of the known fields at the points (one per slot, from
    ctx.func_values) and whose constants are the Const values and the current values of the
    Constants.  compile() returns None for an expression the evaluator does not cover, and
    evaluate() returns None when the values are complex (the complex-step derivative) or the
    context lacks what the program reads; the NumPy evaluate() then takes over."""

    _FUNCS = {"sin": 9, "cos": 10, "exp": 11, "log": 12, "tanh": 13, "sqrt": 14, "sinh": 15, "cosh": 16, "sech": 17,
              "abs": 18, "sign": 19}                               # the Op numbers of forms/integrand.hpp
    CONST, X, Y, NX, NY, INPUT, ADD, MUL, POW = range(9)

    def __init__(self):
        self.code, self.consts, self.inputs, self.input_keys, self.nreg, self.memo = [], [], [], {}, 0, {}
        self.normal_sides = set()               # the restrictions the normal carries; checked against the context

    @classmethod
    def compile(cls, expr):
        P = cls()
        try:
            P._emit(expr)
        except NotImplementedError:
            return None
        P.prog = _C.Integrand(np.asarray(P.code, dtype=np.int32).ravel(), max(P.nreg, 1), len(P.consts), len(P.inputs))
        return P

    def _reg(self):
        self.nreg += 1
        return self.nreg - 1

    def _const(self, c):
        self.consts.append(c)
        return len(self.consts) - 1

    def _emit(self, e):
        key = id(e)
        got = self.memo.get(key)
        if got is not None and got[0] is e:
            return got[1]
        r = self._emit_node(e)
        self.memo[key] = (e, r)
        return r

    def _emit_node(self, e):
        if isinstance(e, Const):
            r = self._reg(); self.code.append((self.CONST, r, self._const(float(e.value)), 0)); return r
        if isinstance(e, Constant):
            r = self._reg(); self.code.append((self.CONST, r, self._const(e), 0)); return r
        if isinstance(e, Spatial):
            r = self._reg(); self.code.append((self.X if e.index == 0 else self.Y, r, 0, 0)); return r
        if isinstance(e, Normal):
            self.normal_sides.add(e.side)
            r = self._reg(); self.code.append((self.NX if e.index == 0 else self.NY, r, -1 if e.side == "+" else 1, 0)); return r
        if isinstance(e, Argument):
            if e.kind != "func":
                raise NotImplementedError
            k = e.slot()
            if k not in self.input_keys:
                self.input_keys[k] = len(self.inputs)
                self.inputs.append(e)
            r = self._reg(); self.code.append((self.INPUT, r, self.input_keys[k], 0)); return r
        if isinstance(e, Sum):
            if not e.terms:
                r = self._reg(); self.code.append((self.CONST, r, self._const(0.0), 0)); return r
            acc = self._emit(e.terms[0])
            for t in e.terms[1:]:
                b = self._emit(t); r = self._reg(); self.code.append((self.ADD, r, acc, b)); acc = r
            return acc
        if isinstance(e, Prod):
            if not e.factors:
                r = self._reg(); self.code.append((self.CONST, r, self._const(1.0), 0)); return r
            acc = self._emit(e.factors[0])
            for f in e.factors[1:]:
                b = self._emit(f); r = self._reg(); self.code.append((self.MUL, r, acc, b)); acc = r
            return acc
        if isinstance(e, Pow):
            n = e.n
            if isinstance(n, Expr) or np.iscomplexobj(n):
                raise NotImplementedError
            a = self._emit(e.base); r = self._reg(); self.code.append((self.POW, r, a, self._const(float(n)))); return r
        if isinstance(e, Func):
            op = self._FUNCS.get(e.name)
            if op is None:
                raise NotImplementedError
            a = self._emit(e.arg); r = self._reg(); self.code.append((op, r, a, 0)); return r
        raise NotImplementedError

    def evaluate(self, ctx):
        npts = int(np.size(ctx.xs))
        if self.prog.needs_normal:
            # the NumPy evaluation raises for a normal that does not fit the measure: leave those to it
            if getattr(ctx, "normals", None) is None:
                return None
            if getattr(ctx, "facet", False) and None in self.normal_sides:
                return None
            if not getattr(ctx, "facet", False) and any(sd is not None for sd in self.normal_sides):
                return None
        consts = []
        for c in self.consts:
            v = _scalar(ctx.values.get(c.name, c.value)) if isinstance(c, Constant) else c
            if isinstance(v, complex):
                return None
            consts.append(v)
        inputs = []
        for a in self.inputs:
            v = np.asarray(ctx.func_values(a))
            if np.iscomplexobj(v) or v.shape != (npts,):
                return None
            inputs.append(np.ascontiguousarray(v, dtype=np.float64))
        xs = ys = nx = ny = None
        if self.prog.needs_coordinates:
            xs = np.ascontiguousarray(ctx.xs, dtype=np.float64).reshape(-1)
            ys = np.ascontiguousarray(ctx.ys, dtype=np.float64).reshape(-1)
        if self.prog.needs_normal:
            nx = np.ascontiguousarray(ctx.normals[0], dtype=np.float64).reshape(-1)
            ny = np.ascontiguousarray(ctx.normals[1], dtype=np.float64).reshape(-1)
        return self.prog.evaluate(npts, xs, ys, nx, ny, inputs, np.asarray(consts, dtype=np.float64))


class _Fallback(Exception):
    """Raised while compiling a 2D form when a term needs the NumPy evaluation."""


class _Compiled2D:
    """A 2D form (one list of terms) compiled for _C.FormAssembler2D (forms/assembler_2d.hpp): the
    point sets, the known fields read at each of them and the terms with their Integrand programs
    are described once; assemble() then hands the current coefficient vectors and constants to
    C++, which evaluates the fields, the coefficients and the kernels with no Python in the loop
    (and without the GIL).  compile() returns None when a term needs the NumPy path: an expression
    the Integrand does not cover, a field on another mesh, a normal that does not fit the measure
    (the NumPy path raises the proper error)."""

    def __init__(self, form_, terms):
        self.form, self.terms, self.rank = form_, terms, terms[0].rank
        rank = self.rank
        T = form_.test_space if rank else next(iter(form_.spaces.values()))
        R = form_.trial_space if rank == 2 else None
        self.T0, self.R0 = T, R
        self.ZT = getattr(T, "_Z", None) if rank else None            # slip: assemble without it, then reduce
        self.ZR = getattr(R, "_Z", None) if rank == 2 else None
        if self.ZT is not None:
            T = T._base
        if self.ZR is not None:
            R = R._base
        self.T, self.R = T, R
        self.facets = any(t.measure.kind == "dS" for t in terms)
        groups = _group_terms_2d(terms, T)
        progs = {}
        for t in terms:
            P = form_._program(t)
            if P is None:
                raise _Fallback
            progs[id(t)] = P
        self.sources, self.consts = [], []                              # Arguments naming the vectors; floats or Constants
        src_index, fields, field_order, plan = {}, {}, [], []
        side = lambda a: 1 if a.side == "+" else 0                       # noqa: E731

        def source(a):
            k = (a.slot()[0], id(a.product) if a.product is not None else None)
            if k not in src_index:
                src_index[k] = len(self.sources)
                self.sources.append(a)
            if a.product is not None:
                return src_index[k], int(a.product.offsets[a.field]), int(a.product.fields[a.field].dim)
            return src_index[k], 0, int(a.space.dim)

        for key, (Ts, toff), (Rs, roff), ts in groups.values():
            Q = form_._cache2d(Ts, key)
            if (Q.nf if key[0] == "dS" else Q.nent) == 0:
                continue
            QS = form_._cache2d(Rs, key) if rank == 2 else None
            gterms = []
            for t in ts:
                P = progs[id(t)]
                if P.prog.needs_normal:
                    if key[0] == "dS" and None in P.normal_sides:
                        raise _Fallback
                    if key[0] != "dS" and any(sd is not None for sd in P.normal_sides):
                        raise _Fallback
                inputs = []
                for a in P.inputs:
                    if getattr(a.space, "tdim", 1) != 2:
                        raise _Fallback
                    vec, off, dim = source(a)
                    if key[0] == "dS":
                        if a.space.mesh is not form_._facet_mesh:
                            raise _Fallback
                        fk = ("dS", id(a.space), vec, off, side(a))
                        Qa = form_._cache2d(a.space, ("dS",))
                    else:
                        if a.space.mesh is not Ts.mesh:
                            raise _Fallback
                        fk = (key, id(a.space), vec, off)
                        Qa = form_._cache2d(a.space, key)
                    if fk not in fields:
                        fields[fk] = (Qa, vec, off, dim, side(a), [])
                        field_order.append(fk)
                    codes = fields[fk][5]
                    if a.code() not in codes:
                        codes.append(a.code())
                    inputs.append((fk, a.code()))
                cidx = list(range(len(self.consts), len(self.consts) + len(P.consts)))
                self.consts.extend(P.consts)
                a_, b_ = (t.test.code(), side(t.test)) if t.test is not None else (0, 0)
                c_, d_ = (t.trial.code(), side(t.trial)) if t.trial is not None else (0, 0)
                gterms.append((P.prog, inputs, cidx, a_, c_, b_, d_))
            plan.append((key, Q, QS, int(toff), int(roff), gterms))

        A = _C.FormAssembler2D(rank, len(self.sources), len(self.consts))
        first = {}
        for fk in field_order:
            Qa, vec, off, dim, sd, codes = fields[fk]
            if fk[0] == "dS":
                first[fk] = A.add_facet_field(Qa, vec, off, dim, sd, codes)
            else:
                first[fk] = A.add_field(Qa, vec, off, dim, codes)
        for key, Q, QS, toff, roff, gterms in plan:
            g = A.add_facet_group(Q, QS, toff, roff) if key[0] == "dS" else A.add_group(Q, QS, toff, roff)
            for prog, inputs, cidx, a_, c_, b_, d_ in gterms:
                A.add_term(g, prog, [first[fk] + fields[fk][5].index(code) for fk, code in inputs], cidx, a_, c_, b_, d_)
        self.A = A
        self._fields = fields                                   # keeps the caches referenced

    @classmethod
    def compile(cls, form_, terms):
        try:
            return cls(form_, terms)
        except _Fallback:
            return None

    def vectors(self, values, owner_values=None, cplx=False):
        out = []
        for a in self.sources:
            v = np.asarray(_source_vector(a, values, owner_values)).reshape(-1)
            if not cplx:
                v = v.astype(np.float64, copy=False)
            if a.product is not None and getattr(a.product, "_Z", None) is not None:
                v = a.product._z(v)                                  # slip: the fields before Z
            out.append(v if cplx else np.ascontiguousarray(v, dtype=np.float64))
        return out

    def constants(self, values, cplx=False):
        return np.asarray([_scalar(values.get(c.name, c.value)) if isinstance(c, Constant) else c for c in self.consts],
                          dtype=np.complex128 if cplx else np.float64)

    def assemble_complex(self, values, owner_values=None):
        """Rank 0 or 1 with complex values (the complex-step derivative), in C++."""
        vre, vim = _complex_parts(self.vectors(values, owner_values, cplx=True))
        k = self.constants(values, cplx=True)
        got = self.A.assemble_complex(vre, vim, np.ascontiguousarray(k.real), np.ascontiguousarray(k.imag),
                                      int(self.T.dim) if self.rank == 1 else 0)
        if self.rank == 0:
            return complex(got[0], got[1])
        out = np.asarray(got[0]) + 1j * np.asarray(got[1])
        return self.T0._zt(out) if self.ZT is not None else out

    def assemble(self, values, owner_values=None):
        from .sparse import SparseMatrix
        vecs, consts = self.vectors(values, owner_values), self.constants(values)
        T, R, T0, R0 = self.T, self.R, self.T0, self.R0
        if self.rank == 0:
            return float(self.A.assemble_scalar(vecs, consts))
        if self.rank == 1:
            out = self.A.assemble_vector(vecs, consts, int(T.dim))
            return T0._zt(out) if self.ZT is not None else out
        K = _C.CSRMatrix(_pattern_2d(T, R, facets=self.facets), False)
        self.A.assemble_matrix(vecs, consts, K)
        if self.ZT is not None or self.ZR is not None:
            return _slip_reduce(K, T0, R0, self.ZT is not None, self.ZR)
        if T is R:
            K.symmetric = bool(K.asymmetry() <= 1e-13)
        return SparseMatrix(K, R, T)


class _Compiled1D:
    """A 1D form (one list of terms) compiled for _C.FormAssembler1D (forms/assembler_1d.hpp): the
    point sets (quadrature nodes of dx, Gauss or Lobatto; the interior facets of dS; the end points
    of ds), the known fields read at each of them and the terms in the order of the form, each with
    its Integrand program and its kernel, are described once; assemble() hands the current
    coefficient vectors and constants to C++, which evaluates the fields and coefficients and
    calls the kernels term after term as the Python driver did.  compile() returns None when the
    Python driver is needed: an expression the Integrand does not cover, a normal or y (the NumPy
    path raises the error), ds on a periodic space or dS across grids (errors again).  A rectangular
    form (test and trial in two spaces on one grid) gives a RectMatrix on the CSR store."""

    _Y = 2                                                       # the Y instruction of forms/integrand.hpp

    def __init__(self, form_, terms):
        f = form_
        self.form, self.terms = f, terms
        rank = self.rank = terms[0].rank
        T = f.test_space if rank else None
        is_prod = bool(rank) and hasattr(T, "fields") and not hasattr(T, "degree")
        R = f.trial_space if rank == 2 else None
        rect = rank == 2 and R is not T
        if rect and (is_prod or (hasattr(R, "fields") and not hasattr(R, "degree")) or not _same_mesh(T, R)):
            raise _Fallback                                      # a mismatch the Python driver reports
        self.T, self.R, self.is_prod, self.rect = T, R, is_prod, rect
        progs = []
        for t in terms:
            P = f._program(t)
            if P is None or P.prog.needs_normal or any(int(op) == self._Y for op, *_ in P.code):
                raise _Fallback
            progs.append(P)
            if t.measure.kind == "ds":
                ns = t.test.space if t.test is not None else next(iter(f.spaces.values()))
                if ns.bc.is_periodic:
                    raise _Fallback
            elif t.measure.kind == "dS":
                ns = t.test.space if t.test is not None else next(iter(f.spaces.values()))
                if any(not _same_mesh(sp_, ns) for sp_ in f.spaces.values()):
                    raise _Fallback

        self.sources, src_index = [], {}
        self.consts = []
        gidx = (lambda P_, k: np.ascontiguousarray(f._global_indices(P_)[k], dtype=np.int32))      # noqa: E731
        empty = np.zeros(0, dtype=np.int32)

        def source(a):
            k = (a.slot()[0], id(a.product) if a.product is not None else None)
            if k not in src_index:
                src_index[k] = len(self.sources)
                self.sources.append(a)
            return src_index[k], (gidx(a.product, a.field) if a.product is not None else empty)

        # the plan: point sets, fields and terms, in the order of the form
        sets, set_order, fields, field_order, plan = {}, [], {}, [], []
        side = lambda a: 0 if a.side == "-" else 1               # noqa: E731  (_add_facet's rule for test/trial)
        fside = lambda a: 0 if (a.side or "-") == "-" else 1     # noqa: E731  (_FacetContext's rule for fields)

        def point_set(key, make):
            if key not in sets:
                sets[key] = make
                set_order.append(key)
            return key

        for t, P in zip(terms, progs):
            kind = t.measure.kind
            ns = t.test.space if t.test is not None else next(iter(f.spaces.values()))
            if kind == "dx":
                rule = ("lobatto", f._lobatto_points(t)) if t.measure.scheme == "lobatto" else None
                cache = (lambda sp_, rule=rule: f._cache(sp_, rule))                              # noqa: E731
                keys = [point_set(("dx", rule, id(ns)), ("cells", cache(ns)))]
            elif kind == "dS":
                keys = [point_set(("dS", id(ns)), ("facets", f._facet_cache(ns)))]
            else:
                mesh = ns.mesh
                pts = {"left": [mesh.a], "right": [mesh.b], "both": [mesh.a, mesh.b]}[t.measure.where]
                keys = [point_set(("pt", float(xp)), ("point", float(xp))) for xp in pts]
            for key in keys:
                inputs = []
                for a in P.inputs:
                    vec, mp = source(a)
                    fk = (key, vec, a.field, a.deriv, a.side, id(a.space))
                    if fk not in fields:
                        if kind == "dx":
                            if _same_mesh(a.space, ns):
                                fields[fk] = ("quad", a.space, cache(a.space), vec, mp, int(a.deriv))
                            else:
                                fields[fk] = ("point", a.space, vec, mp, int(a.deriv))
                        elif kind == "dS":
                            fields[fk] = ("facet", a.space, f._facet_cache(a.space), vec, mp, fside(a), int(a.deriv))
                        else:
                            fields[fk] = ("point", a.space, vec, mp, int(a.deriv))
                        field_order.append(fk)
                    inputs.append(fk)
                cidx = list(range(len(self.consts), len(self.consts) + len(P.consts)))
                self.consts.extend(P.consts)
                plan.append((key, kind, t, P, inputs, cidx, rule if kind == "dx" else None))

        n = int(T.dim) if rank else 0
        A = _C.FormAssembler1D(rank, len(self.sources), len(self.consts), n, int(R.dim) if rect else -1)
        set_index = {}
        for key in set_order:
            what, obj = sets[key]
            set_index[key] = A.add_cells(obj) if what == "cells" else (A.add_facets(obj) if what == "facets" else A.add_point(obj))
        field_index = {}
        for fk in field_order:
            d = fields[fk]
            ps = set_index[fk[0]]
            if d[0] == "quad":
                field_index[fk] = A.add_quad_field(ps, d[1], d[2], d[3], d[4], d[5])
            elif d[0] == "facet":
                field_index[fk] = A.add_facet_field(ps, d[1], d[2], d[3], d[4], d[5], d[6])
            else:
                field_index[fk] = A.add_point_field(ps, d[1], d[2], d[3], d[4])
        for key, kind, t, P, inputs, cidx, rule in plan:
            ps, ins, prog = set_index[key], [field_index[fk] for fk in inputs], P.prog
            a_ = int(t.test.deriv) if t.test is not None else 0
            b_ = int(t.trial.deriv) if t.trial is not None else 0
            if kind == "dx":
                if rank == 2 and is_prod:
                    A.add_cell_term(ps, prog, ins, cidx, a_, b_, T, int(t.test.field), int(t.trial.field), f._cache(t.trial.space, rule))
                elif rank == 1 and is_prod:
                    A.add_cell_term(ps, prog, ins, cidx, a_, 0, T, int(t.test.field), 0, None)
                elif rect:
                    A.add_cell_term(ps, prog, ins, cidx, a_, b_, None, 0, 0, f._cache(t.trial.space, rule))
                else:
                    A.add_cell_term(ps, prog, ins, cidx, a_, b_, None, 0, 0, None)
            elif kind == "dS":
                st = side(t.test) if t.test is not None else 0
                ss = side(t.trial) if t.trial is not None else 0
                if rank == 1:
                    nt = int(t.test.space.dim) if is_prod else n
                    mp = gidx(T, t.test.field) if is_prod else empty
                    A.add_facet_term(ps, prog, ins, cidx, st, 0, a_, 0, nt, mp, None, None, 0, 0, False)
                elif rank == 2:
                    FS = f._facet_cache(t.trial.space)
                    if is_prod:
                        clear = t.test.field != t.trial.field or t.test.deriv != t.trial.deriv or t.test.side != t.trial.side
                        A.add_facet_term(ps, prog, ins, cidx, st, ss, a_, b_, 0, empty, FS, T, int(t.test.field),
                                         int(t.trial.field), bool(clear))
                    else:
                        A.add_facet_term(ps, prog, ins, cidx, st, ss, a_, b_, 0, empty, FS, None, 0, 0, False)
                else:
                    A.add_facet_term(ps, prog, ins, cidx, 0, 0, 0, 0, 0, empty, None, None, 0, 0, False)
            else:
                Ts = t.test.space if t.test is not None else None
                Ss = t.trial.space if t.trial is not None else None
                mT = gidx(T, t.test.field) if (is_prod and t.test is not None) else empty
                mS = gidx(T, t.trial.field) if (is_prod and t.trial is not None) else empty
                A.add_point_term(ps, prog, ins, cidx, Ts, a_, mT, Ss, b_, mS)
        self.A = A
        self._keep = (sets, fields)                              # the caches stay referenced

    @classmethod
    def compile(cls, form_, terms):
        try:
            return cls(form_, terms)
        except (_Fallback, TypeError):
            return None

    def assemble_complex(self, values, owner_values=None):
        """Rank 0 or 1 with complex values (the complex-step derivative), in C++."""
        vre, vim = _complex_parts([np.asarray(_source_vector(a, values, owner_values)).reshape(-1) for a in self.sources])
        k = np.asarray([_scalar(values.get(c.name, c.value)) if isinstance(c, Constant) else c for c in self.consts],
                       dtype=np.complex128)
        got = self.A.assemble_complex(vre, vim, np.ascontiguousarray(k.real), np.ascontiguousarray(k.imag))
        if self.rank == 0:
            return complex(got[0], got[1])
        return np.asarray(got[0]) + 1j * np.asarray(got[1])

    def assemble(self, values, owner_values=None):
        vecs = [np.ascontiguousarray(_source_vector(a, values, owner_values), dtype=np.float64).reshape(-1) for a in self.sources]
        consts = np.asarray([_scalar(values.get(c.name, c.value)) if isinstance(c, Constant) else c for c in self.consts],
                            dtype=np.float64)
        if self.rank == 0:
            return float(self.A.assemble_scalar(vecs, consts))
        if self.rank == 1:
            return self.A.assemble_vector(vecs, consts)
        from . import Matrix, RectMatrix
        T = self.T
        if self.rect:
            return RectMatrix(self.A.assemble_rect(vecs, consts), T, self.R)
        if self.is_prod:
            out = Matrix(_C.AssembledMatrix(T.dim, T.uband), T.periodic, T)
        else:
            out = Matrix(_C.AssembledMatrix(T.dim, T.uband), T.bc.is_periodic, self.R)
        self.A.assemble_matrix(vecs, consts, out._K)
        return out


def _complex_parts(vecs):
    """The real parts of the coefficient vectors, and their imaginary parts (None for a real vector)."""
    re, im = [], []
    for v in vecs:
        v = np.asarray(v).reshape(-1)
        if np.iscomplexobj(v):
            re.append(np.ascontiguousarray(v.real, dtype=np.float64))
            im.append(np.ascontiguousarray(v.imag, dtype=np.float64))
        else:
            re.append(np.ascontiguousarray(v, dtype=np.float64))
            im.append(None)
    return re, im


def _slip_reduce(K, T0, R0, reduce_rows, ZR):
    """Z^T K Z of a matrix assembled without the slip condition, in C++ (CSRMatrix products), with
    the entries that come out exactly zero dropped; symmetric measured when test and trial agree."""
    from .sparse import SparseMatrix
    if reduce_rows and ZR is not None:
        M = K.triple_product(T0._ZT, ZR)                    # one pass, exact zeros dropped
    else:
        M = (T0._ZT.multiply(K) if reduce_rows else K.multiply(ZR)).pruned(0.0)
    S = SparseMatrix(M, R0, T0)
    S.symmetric = S.is_symmetric() if T0 is R0 else False
    return S


def _group_terms_2d(terms, T):
    """The terms of a 2D form grouped by point set (dx, Lobatto dx, ds(where), dS) and by the test
    and trial blocks (field space and offset) they fill: {group key: (key, (Ts, toff), (Rs, roff), terms)}."""
    def field(a):                                   # (field space, offset) of a test / trial Argument
        if a.product is not None:
            return a.product.fields[a.field], a.product.offsets[a.field]
        return a.space, 0

    groups = {}
    for t in terms:
        ft = field(t.test) if t.test is not None else (T, 0)
        fr = field(t.trial) if t.trial is not None else (None, 0)
        if t.measure.kind == "dx" and t.measure.scheme == "lobatto":
            n = t.measure.quad_degree
            if n is None:                            # the nodal rule of the highest degree among test and trial
                n = max(getattr(s_, "degree", 1) for s_ in (ft[0], fr[0]) if s_ is not None) + 1
            key = ("dxL", int(n))
        elif t.measure.kind == "dS":
            key = ("dS",)
        else:
            key = ("dx",) if t.measure.kind == "dx" else ("ds", t.measure.where)
        gk = (key, id(ft[0]), ft[1], id(fr[0]), fr[1])
        groups.setdefault(gk, (key, ft, fr, []))[3].append(t)
    return groups


# ============================================================================ the Form
class Form:
    def __init__(self, fe, space=None):
        if isinstance(fe, Integral):
            fe = FormExpr([fe])
        if not isinstance(fe, FormExpr):
            raise TypeError("form() takes expr*dx, expr*ds, or a sum of those")
        self.terms = canonicalize(fe)
        if not self.terms:
            raise ValueError("empty form")
        ranks = {t.rank for t in self.terms}
        if len(ranks) != 1:
            raise TypeError(f"all terms must have the same rank, got {sorted(ranks)}: {self.terms}")
        self.rank = ranks.pop()

        args = []
        for t in self.terms:
            collect_arguments(t.coeff, args)
            for a in (t.test, t.trial):
                if a is not None: args.append(a)
        funcs = {}
        for a in args:
            if a.kind == "func":
                funcs.setdefault(a.slot()[0], a)
        self._funcs = funcs                                   # slot key -> a representative Argument
        codes2d = {}                                          # (field key, field, side) -> codes read (2D)
        for a in args:
            if a.kind == "func" and a.comp is None:
                k = a.slot()
                codes2d.setdefault((k[0], a.field, a.side), set()).add(a.code())
        self._codes2d = codes2d
        self.functions = sorted({a.name for a in funcs.values()})
        owners = [a.owner for a in funcs.values() if a.owner is not None]
        consts = []
        for t in self.terms:
            collect_constants(t.coeff, consts)
        self._consts = {}
        for c in consts:
            if c.name in self._consts and self._consts[c.name] is not c:
                raise TypeError(f"two different Constants share the name '{c.name}'; give them distinct names")
            self._consts[c.name] = c
        self.constants = sorted(self._consts)
        clash = set(self.constants) & set(self.functions)
        if clash:
            raise TypeError(f"a Constant and a Function share the name(s) {sorted(clash)}")
        self.constant = not funcs and not consts               # nothing that changes: assembled once and cached
        if not self.constant and len({a.name for a in funcs.values()}) < len(funcs):
            names = [a.name for a in funcs.values()]
            dup = sorted({n for n in names if names.count(n) > 1})
            if any(a.owner is None for a in funcs.values()):
                raise TypeError(f"two different Functions share the name {dup}; give them distinct names")
            warnings.warn(f"two distinct Functions share the name {dup}; keyword overrides by name will be ambiguous", stacklevel=3)
        self.trial_space = None
        self.spaces = {}
        if space is not None:
            self.spaces[id(space)] = space
        for a in args:
            self.spaces[id(a.space)] = a.space
        if not self.spaces:
            raise ValueError("a form with no test, trial or Function needs the mesh: form(expr*dx, space=V)")
        tdims = {getattr(sp_, "tdim", 1) for sp_ in self.spaces.values()}
        if len(tdims) > 1:
            raise TypeError("a form mixes spaces on 1D and 2D meshes")
        self.tdim = tdims.pop()
        if self.tdim == 2:
            self._caches = {}
            self._constant_result = None
            self._init_2d(args)
            return
        self._nder = {sid: 1 for sid in self.spaces}
        for a in args:
            self._nder[id(a.space)] = max(self._nder[id(a.space)], a.deriv)

        tests  = [t.test for t in self.terms if t.test is not None]
        trials = [t.trial for t in self.terms if t.trial is not None]
        self.test_space  = self._row_space(tests) if tests else None
        self.trial_space = self._row_space(trials) if trials else None
        self._check_conformity(args)
        self._check_restrictions()
        self._warn_strong_natural(tests, trials)

        # quadrature degree, Section 15.6
        m = 0
        for t in self.terms:
            if t.measure.kind != "dx" or t.measure.scheme == "lobatto":
                continue
            if _measure_degree(t.measure) is not None:
                m = max(m, _measure_degree(t.measure)); continue
            d = poly_degree(t.coeff)
            for a in (t.test, t.trial):
                if a is not None and d is not None:
                    d += max(a.space.degree - a.deriv, 0)
            if d is None:
                # a non-polynomial function of the data or of a field counts as degree p + 1, the
                # interpolant plus the leading term of its remainder (quad_degree_estimate)
                d = quad_degree_estimate(t.coeff, max(s.degree for s in self.spaces.values()) + 1)
                for a in (t.test, t.trial):
                    if a is not None:
                        d += max(a.space.degree - a.deriv, 0)
            m = max(m, d // 2 + 1)
        self.quad_degree = max(m, 1)
        self._caches = {}
        self._constant_result = None

    # ---- 2D -------------------------------------------------------------------------
    def _init_2d(self, args):
        """Checks and quadrature degrees of a form on triangles (LagrangeSpace2D)."""
        for a in args:
            if a.product is not None and not _is_system_2d(a.product):
                raise TypeError("a 1D ProductSpace in a form on a 2D mesh")
        tests  = [t.test for t in self.terms if t.test is not None]
        trials = [t.trial for t in self.terms if t.trial is not None]
        self.test_space  = self._row_space(tests) if tests else None
        self.trial_space = self._row_space(trials) if trials else None
        for t in self.terms:
            k = t.measure.kind
            if k == "ds" and t.measure.where in ("left", "right"):
                raise TypeError(f"ds('{t.measure.where}') is for 1D meshes; on a 2D mesh use ds(marker), ds((m1, m2)) or ds")
            targs = collect_arguments(t.coeff, []) + [a for a in (t.test, t.trial) if a is not None]
            for a in targs:
                if k != "dS" and a.side is not None:
                    raise TypeError(f"{a!r}: restrictions ('-', '+') and jump/avg belong in dS terms")
                if k == "dS" and a.side is None and (a.kind in ("test", "trial") or getattr(a.space, "broken", False)
                                                      or getattr(a.space, "ncomp", 1) > 1 or _order(a.deriv) > 0):
                    what = {"test": "test function", "trial": "trial function"}.get(a.kind, "field")
                    raise TypeError(f"in a dS term the {what} {a!r} is two-valued: restrict it, {a!r}('-') or "
                                    f"{a!r}('+'), or use jump() / avg()")
                if _order(a.deriv) > 1:
                    raise ValueError(f"{a!r}: derivatives of order {_order(a.deriv)} are not available on a C^0 "
                                     "Lagrange space; integrate by parts (only first derivatives enter a form)")
            if k == "dx" and _has_normal(t.coeff):
                    raise TypeError("the facet normal n appears in a dx term; it exists on ds terms only")
            if k == "dx" and t.measure.scheme == "lobatto" and \
                    any(getattr(sp_, "nverts", 3) != 4 for sp_ in self.spaces.values()):
                raise ValueError("dx(scheme='lobatto') on a 2D mesh needs quadrilaterals (Q_k): the Gauss-Lobatto "
                                 "rule is a tensor rule, and triangles have none")
        if self.test_space is not None and self.trial_space is not None and \
                self.test_space.mesh is not self.trial_space.mesh:
            raise ValueError("the test and trial spaces of a form must be on the same mesh object")
        kmax = max(sp_.degree for sp_ in self.spaces.values())
        deg = {"dx": 0, "ds": 0, "dS": 0}
        self._facet_mesh = None
        if any(t.measure.kind == "dS" for t in self.terms):
            meshes = {id(sp_.mesh): sp_.mesh for sp_ in self.spaces.values()}
            if len(meshes) > 1:
                raise ValueError("dS: every space of a form with interior-facet terms must be on one mesh")
            self._facet_mesh = next(iter(meshes.values()))
            if any(getattr(sp_, "nverts", 3) != 3 for sp_ in self.spaces.values()):
                raise ValueError("dS on a 2D mesh is implemented for triangles")
        for t in self.terms:
            k = t.measure.kind
            if k == "dx" and t.measure.scheme == "lobatto":
                continue
            if _measure_degree(t.measure) is not None:
                deg[k] = max(deg[k], _measure_degree(t.measure)); continue
            d = poly_degree(t.coeff)
            arg_d = sum(_arg_degree(a) for a in (t.test, t.trial) if a is not None)
            if d is None:
                d = quad_degree_estimate(t.coeff, kmax + 1)  # non-polynomial parts count as degree kmax + 1
            deg[k] = max(deg[k], arg_d + d)
        self.quad_degree = max(deg["dx"], 1)
        self.facet_degree = max(deg["ds"], 1)
        self.interior_degree = max(deg["dS"], 1)

    def _cache2d(self, space, key):
        ck = ("2d", id(space), key)
        Q = self._caches.get(ck)
        if Q is None:
            if key[0] == "dx":
                Q = space.cache(self.quad_degree)
            elif key[0] == "dxL":
                Q = space.cache(rule="lobatto", npts=key[1])
            elif key[0] == "dS":
                Q = _C.InteriorFacetCache2D(space, self.interior_degree)
            else:
                where = key[1]
                Q = space.boundary_cache(self.facet_degree, None if where == "both" else list(where))
            self._caches[ck] = Q
        return Q

    def _program(self, t):
        """The Integrand program of a term's coefficient (cached per term), or None when the
        expression is not covered."""
        progs = self.__dict__.setdefault("_programs", {})
        got = progs.get(id(t))
        if got is None or got[0] is not t:
            got = progs[id(t)] = (t, _Program.compile(t.coeff))
        return got[1]

    def _coefficient(self, t, ctx):
        """The coefficient of a term at the points of a 2D context: the compiled program when there is
        one and it applies, the NumPy evaluation otherwise."""
        if _COMPILED:
            P = self._program(t)
            if P is not None:
                c = P.evaluate(ctx)
                if c is not None:
                    return c
        return np.asarray(evaluate(t.coeff, ctx))

    def _compiled_2d(self, terms):
        """The _Compiled2D of a term list (cached by the list's identity), or None."""
        comps = self.__dict__.setdefault("_compiled", {})
        got = comps.get(id(terms))
        if got is None or got[0] is not terms:
            got = comps[id(terms)] = (terms, _Compiled2D.compile(self, terms))
        return got[1]

    def _compiled_1d(self, terms):
        """The _Compiled1D of a term list (cached by the list's identity), or None."""
        comps = self.__dict__.setdefault("_compiled1d", {})
        got = comps.get(id(terms))
        if got is None or got[0] is not terms:
            got = comps[id(terms)] = (terms, _Compiled1D.compile(self, terms))
        return got[1]

    def _assemble_terms_2d(self, terms, values, owner_values=None):
        from .sparse import SparseMatrix
        rank = terms[0].rank
        cplx = any(np.iscomplexobj(getattr(v, "vector", v)) for v in values.values()) or \
            any(np.iscomplexobj(v) for v in (owner_values or {}).values()) or \
            any(isinstance(c.value, complex) for c in self._consts.values())
        if cplx and rank == 2:
            raise TypeError("complex field values are supported for rank 0 and rank 1 forms only")
        if _COMPILED:
            comp = self._compiled_2d(terms)
            if comp is not None:                                # the whole form in C++
                return comp.assemble_complex(values, owner_values) if cplx else comp.assemble(values, owner_values)
        T = self.test_space if rank else next(iter(self.spaces.values()))
        R = self.trial_space if rank == 2 else None
        T0, R0 = T, R
        ZT = getattr(T, "_Z", None) if rank else None          # slip: assemble without it, then reduce
        ZR = getattr(R, "_Z", None) if rank == 2 else None
        if ZT is not None:
            T = T._base
        if ZR is not None:
            R = R._base
        groups = _group_terms_2d(terms, T)
        if rank == 2:
            K = _C.CSRMatrix(_pattern_2d(T, R, facets=any(t.measure.kind == "dS" for t in terms)), False)
        out = np.zeros(T.dim, dtype=np.complex128 if cplx else np.float64) if rank == 1 else 0.0
        side = lambda a: 1 if a.side == "+" else 0                               # noqa: E731
        ctxs = {}                                # one context per point set: field values computed once
        fields_by_key = {}                       # field values shared by the contexts of one mesh and rule
        for key, (Ts, toff), (Rs, roff), ts in groups.values():
            Q = self._cache2d(Ts, key)
            if key[0] == "dS":
                if Q.nf == 0:
                    continue
                ctx = ctxs.get(("dS",))
                if ctx is None:
                    ctx = ctxs[("dS",)] = _FacetContext2D(self, Q, values, owner_values)
                npts = Q.nf * Q.nq
                cs = []
                for t in ts:
                    c = self._coefficient(t, ctx)
                    cs.append(np.broadcast_to(c, (npts,)) if c.ndim == 0 or c.shape != (npts,) else c)
                if rank == 0:
                    for c in cs:
                        if np.iscomplexobj(c):
                            out += _C.assemble_facet_scalar_2d(Q, np.ascontiguousarray(c.real)) + \
                                1j * _C.assemble_facet_scalar_2d(Q, np.ascontiguousarray(c.imag))
                        else:
                            out += _C.assemble_facet_scalar_2d(Q, np.ascontiguousarray(c, dtype=np.float64))
                elif rank == 1:
                    a = [t.test.code() for t in ts]
                    sd = [side(t.test) for t in ts]
                    sl = slice(toff, toff + Ts.dim)
                    if any(np.iscomplexobj(c) for c in cs):
                        out[sl] += _C.assemble_facet_vector_2d(Q, sd, a, [np.ascontiguousarray(np.real(c), dtype=np.float64) for c in cs])
                        out[sl] += 1j * _C.assemble_facet_vector_2d(Q, sd, a, [np.ascontiguousarray(np.imag(c), dtype=np.float64) for c in cs])
                    else:
                        out[sl] += _C.assemble_facet_vector_2d(Q, sd, a, [np.ascontiguousarray(c, dtype=np.float64) for c in cs])
                else:
                    QS = self._cache2d(Rs, key)
                    _C.assemble_facet_matrix_2d(Q, QS, [side(t.test) for t in ts], [side(t.trial) for t in ts],
                                                [t.test.code() for t in ts], [t.trial.code() for t in ts],
                                                [np.ascontiguousarray(c, dtype=np.float64) for c in cs], K, toff, roff)
                continue
            if Q.nent == 0:
                continue
            ck = (key, id(Ts))
            ctx = ctxs.get(ck)
            if ctx is None:
                ctx = ctxs[ck] = _Context2D(self, Q, key, Ts, values, owner_values)
                # the values of a field depend on its own cache only, so contexts on the same mesh
                # and point set (the components of a vector space) share them
                ctx._fields = fields_by_key.setdefault((key, id(Ts.mesh)), {})
            npts = Q.nent * Q.nq
            cs = []
            for t in ts:
                c = self._coefficient(t, ctx)
                cs.append(np.broadcast_to(c, (npts,)) if c.ndim == 0 or c.shape != (npts,) else c)
            if rank == 0:
                for c in cs:
                    if np.iscomplexobj(c):
                        out += _C.assemble_scalar_2d(Q, np.ascontiguousarray(c.real)) + \
                            1j * _C.assemble_scalar_2d(Q, np.ascontiguousarray(c.imag))
                    else:
                        out += _C.assemble_scalar_2d(Q, np.ascontiguousarray(c, dtype=np.float64))
            elif rank == 1:
                a = [t.test.code() for t in ts]
                sl = slice(toff, toff + Ts.dim)
                if any(np.iscomplexobj(c) for c in cs):
                    out[sl] += _C.assemble_vector_2d(Q, a, [np.ascontiguousarray(np.real(c), dtype=np.float64) for c in cs])
                    out[sl] += 1j * _C.assemble_vector_2d(Q, a, [np.ascontiguousarray(np.imag(c), dtype=np.float64) for c in cs])
                else:
                    out[sl] += _C.assemble_vector_2d(Q, a, [np.ascontiguousarray(c, dtype=np.float64) for c in cs])
            else:
                QS = self._cache2d(Rs, key)
                _C.assemble_matrix_2d(Q, QS, [t.test.code() for t in ts], [t.trial.code() for t in ts],
                                      [np.ascontiguousarray(c, dtype=np.float64) for c in cs], K, toff, roff)
        if rank == 2:
            if ZT is not None or ZR is not None:
                return _slip_reduce(K, T0, R0, ZT is not None, ZR)
            if T is R:
                K.symmetric = bool(K.asymmetry() <= 1e-13)
            return SparseMatrix(K, R, T)
        if rank == 1 and ZT is not None:
            out = T0._zt(out)
        return out

    # ---- setup helpers ------------------------------------------------------
    @staticmethod
    def _row_space(args):
        prods = {id(a.product): a.product for a in args if a.product is not None}
        plain = {id(a.space): a.space for a in args if a.product is None}
        if len(prods) > 1 or (prods and plain) or len(plain) > 1:
            raise TypeError("all test functions (and all trial functions) of a form must come from one space or one ProductSpace")
        return next(iter(prods.values())) if prods else next(iter(plain.values()))

    def _check_restrictions(self):
        """Restrictions ('-', '+') belong in dS terms, and there the test and trial functions need one."""
        for t in self.terms:
            args = collect_arguments(t.coeff, []) + [a for a in (t.test, t.trial) if a is not None]
            if t.measure.kind != "dS":
                bad = [a for a in args if a.side is not None]
                if bad:
                    raise TypeError(f"{bad[0]!r}: the restrictions ('-', '+') and jump/avg belong in dS terms, not in "
                                    f"{'dx' if t.measure.kind == 'dx' else 'ds'} terms")
                continue
            for a in args:
                if a.side is None and (a.kind in ("test", "trial") or a.space.broken or a.deriv > 0):
                    what = {"test": "test function", "trial": "trial function"}.get(a.kind, "field")
                    raise TypeError(f"in a dS term the {what} {a!r} is two-valued: restrict it, {a!r}('-') or "
                                    f"{a!r}('+'), or use jump() / avg()")

    @staticmethod
    def _check_conformity(args):
        for a in args:
            p, cont = a.space.degree, getattr(a.space, "continuity", 0)
            if getattr(a.space, "broken", False):
                continue                                  # element-wise derivatives, zero above the degree
            if a.deriv > p:
                raise ValueError(f"D({a!r}) of order {a.deriv} exceeds the degree {p} of its space")
            if a.deriv > cont + 1:
                raise ValueError(f"D(.,{a.deriv}) is not conforming on a C^{cont} space (needs C^{a.deriv - 1}); "
                                 f"raise the continuity or integrate by parts")

    def _warn_strong_natural(self, tests, trials):
        if self.rank != 2 or not trials:
            return
        symmetric = all(t.test.deriv == t.trial.deriv and t.test.field == t.trial.field for t in self.terms if t.measure.kind == "dx")
        if not symmetric:
            return
        dx_terms = [t for t in self.terms if t.measure.kind == "dx"]
        if not dx_terms:
            return
        k = max(t.test.deriv for t in dx_terms)
        for a in tests:
            bc = a.space.bc
            if bc.is_periodic: continue
            for side, b in (("left", bc.left), ("right", bc.right)):
                if b.name != "free" and b.max_order >= k >= 1:
                    warnings.warn(f"'{b.name}' at the {side} end builds in a derivative of order {b.max_order}, which is a "
                                  f"NATURAL condition for this operator of order {2*k}; it has been imposed strongly. "
                                  f"Use 'free' for the classical Galerkin treatment.", stacklevel=4)

    def _facet_cache(self, space):
        key = ("facet", id(space))
        if key not in self._caches:
            self._caches[key] = _C.FacetCache(space, self._nder.get(id(space), 1))
        return self._caches[key]

    def _cache(self, space, rule=None):
        """The quadrature cache of a space: the form's Gauss rule, or rule = ("lobatto", n)."""
        if rule is None:
            key = id(space)
            if key not in self._caches:
                self._caches[key] = _C.QuadratureCache(space, self.quad_degree, self._nder.get(key, 1))
            return self._caches[key]
        key = (id(space), rule)
        if key not in self._caches:
            self._caches[key] = _C.QuadratureCache(space, rule[1], self._nder.get(id(space), 1), True)
        return self._caches[key]

    def _lobatto_points(self, t):
        """Points of the Gauss-Lobatto rule of a dx(scheme='lobatto') term: n when given, otherwise
        p + 1 for the highest degree p among the term's test and trial spaces (all the form's spaces
        for a rank-0 term), the nodal rule of a Lobatto basis of that degree."""
        if t.measure.quad_degree is not None:
            return int(t.measure.quad_degree)
        sp = [a.space for a in (t.test, t.trial) if a is not None] or list(self.spaces.values())
        return max(2, max(s.degree for s in sp) + 1)

    # ---- assembly ---------------------------------------------------------------
    def assemble(self, **values):
        """Rank 0 -> float, rank 1 -> ndarray, rank 2 -> Matrix.

        Known fields are read from the Function objects in the form, and Constants from
        their current value.  A keyword ``name=array_or_Function`` overrides that Function's
        vector for this call, and ``name=number`` a Constant's value.
        A complex override gives a complex vector or number (rank 0 and 1 only), which is
        what the complex-step derivative Im R(u + i h v) / h needs."""
        from . import Matrix
        for k in values:
            if k not in self.functions and k not in self.constants:
                raise KeyError(f"'{k}' is not a Function or Constant of this form "
                               f"(Functions {self.functions}, Constants {self.constants})")
        missing = [a.name for a in self._funcs.values() if a.owner is None and a.name not in values]
        if missing:
            raise KeyError(f"missing values for Function(s) {missing}")
        if self.constant and self._constant_result is not None:
            return self._constant_result
        result = self._assemble_terms(self.terms, values, Matrix)
        if self.constant:
            self._constant_result = result
        return result

    def derivative(self, wrt, space=None, part="all"):
        """The Jacobian of a rank-1 form as a rank-2 FORM, d R / d wrt, not yet assembled.

            J = R.derivative(u)            # J.assemble() is R.jacobian(u)
            Jv = J.action(dv).assemble()   # the Jacobian-vector product, matrix-free

        space=: the trial space of the result.  By default it is the space of `wrt`.  For u
        in V.unconstrained (a field carrying boundary data) pass space=V: the perturbations
        then satisfy V's homogeneous conditions, and the Jacobian is square on V.
        part="constant" keeps only the terms whose coefficient does not depend on any
        Function: the linear, constant-coefficient part, a natural preconditioner."""
        target, jterms = self._derivative_terms(wrt, space)
        if part == "constant":
            jterms = [t for t in jterms if not collect_arguments(t.coeff, [])]
            if not jterms:
                raise ValueError(f"the Jacobian with respect to '{target.name}' has no constant-coefficient part")
        elif part != "all":
            raise ValueError("derivative(part=...): 'all' or 'constant'")
        return Form(FormExpr([Integral(Prod([t.coeff, t.test, t.trial]), t.measure) for t in jterms]))

    def _derivative_terms(self, wrt, space=None):
        if self.rank != 1:
            raise TypeError("jacobian() and derivative() are defined for rank-1 forms (residuals)")
        if isinstance(wrt, str):
            cands = [a for a in self._funcs.values() if a.name == wrt]
            if not cands: raise KeyError(f"'{wrt}' is not a Function of this form")
            if len(cands) > 1: raise KeyError(f"name '{wrt}' is ambiguous here; pass the Function object")
            target = cands[0]
        else:
            owner = getattr(wrt, "owner", wrt)
            cands = [a for a in self._funcs.values() if a.owner is owner]
            if not cands: raise KeyError("that Function does not appear in this form")
            target = cands[0]
        # a ProductFunction is differentiated with respect to EVERY one of its fields
        if target.product is None:
            match = target.same_field
        else:
            match = lambda a: a.owner is target.owner          # noqa: E731
        jterms, seen = [], set()
        for t in self.terms:
            for a in collect_arguments(t.coeff, []):
                if a.kind == "func" and match(a):
                    slot = a.slot()
                    if (id(t), slot) in seen:
                        continue
                    seen.add((id(t), slot))
                    dc = diff(t.coeff, slot)
                    if _is_zero(dc):
                        continue
                    tspace, tprod = a.space, a.product
                    if space is not None and a.product is None:
                        if not (space is a.space or getattr(space, "unconstrained", None) is a.space):
                            raise ValueError("derivative(space=W): W must be the space of the field, or a space "
                                             "whose unconstrained companion it is")
                        tspace = space
                    elif space is not None:
                        fs = getattr(space, "fields", None)
                        if fs is None or len(fs) != len(a.product.fields) or \
                                not (fs[a.field] is a.space or getattr(fs[a.field], "unconstrained", None) is a.space):
                            raise ValueError("derivative(space=W): for a ProductFunction, W must be its ProductSpace, "
                                             "or the ProductSpace whose unconstrained companion it lives on")
                        tspace, tprod = fs[a.field], space
                    trial = Argument("trial", tspace, field=a.field, product=tprod, deriv=a.deriv, side=a.side, comp=a.comp)
                    jterms.append(Term(t.test, trial, dc, t.measure))
        if not jterms:
            raise ValueError(f"the form does not depend on '{target.name}'")
        return target, jterms

    def jacobian(self, wrt, space=None, **values):
        """d(assemble)/d(coefficients of the Function `wrt`) for a rank-1 form -> Matrix.
        `wrt` is the Function object or its name; space= as in derivative()."""
        from . import Matrix
        # the terms of one (wrt, space) are built once, so their compiled programs and assembler are reused
        jcache = self.__dict__.setdefault("_jterms", {})
        wobj = wrt if isinstance(wrt, str) else getattr(wrt, "owner", wrt)
        jkey = (wobj if isinstance(wobj, str) else id(wobj), id(space))
        got = jcache.get(jkey)
        if got is None or (not isinstance(wobj, str) and got[0] is not wobj) or got[1] is not space:
            target, jterms = self._derivative_terms(wrt, space)
            got = jcache[jkey] = (wobj, space, target, jterms)
        target, jterms = got[2], got[3]
        saved = self.trial_space
        self.trial_space = self._row_space([jt.trial for jt in jterms])
        try:
            M = self._assemble_terms(jterms, values, Matrix)
        finally:
            self.trial_space = saved
        if hasattr(M, "_K"):
            M.symmetric = False
        return M

    def action(self, w):
        """The linear form a(w, v) of a bilinear form a(u, v): the trial function replaced by
        the Function w, which may live in any space on the interval (for instance the lift
        V.lift(...) in V.unconstrained).  a.action(w).assemble() is A @ w.vector when w is in
        the trial space, and a(w, v) for the test functions of V in general."""
        if self.rank != 2:
            raise TypeError("action() is defined for bilinear (rank-2) forms")
        prod = isinstance(w, ProductFunction)
        if not isinstance(w, (Function, VectorElementFunction)) and not prod:
            raise TypeError("action(w): w must be a Function, or a ProductFunction for a form on a ProductSpace")
        integrals = []
        for t in self.terms:
            if (t.trial.product is not None) != prod:
                raise TypeError("action(w): a form on a ProductSpace acts on a ProductFunction, a form on one "
                                "space on a Function")
            if prod:
                k = t.trial.field
                fk = w.product.fields[k] if k < len(w.product.fields) else None
                if len(w.product.fields) != len(t.trial.product.fields) or \
                        not (fk is t.trial.space or getattr(t.trial.space, "unconstrained", None) is fk):
                    raise ValueError("action(w): w must live on the form's ProductSpace, or on its unconstrained companion")
                wk = w.components[k]
                wt = _with_deriv(wk[t.trial.comp] if t.trial.comp is not None else wk, t.trial.deriv)
            else:
                if (t.trial.comp is not None) != isinstance(w, VectorElementFunction):
                    raise TypeError("action(w): w must be a Function of the form's trial space")
                wt = _with_deriv(w[t.trial.comp] if t.trial.comp is not None else w, t.trial.deriv)
            if t.trial.side is not None:
                wt = restrict(wt, t.trial.side)
            integrals.append(Integral(Prod([t.coeff, t.test, wt]), t.measure))
        return Form(FormExpr(integrals))

    def _assemble_owner(self, owner_values):
        """Assemble with some Functions' vectors replaced, keyed by id(Function): internal."""
        from . import Matrix
        return self._assemble_terms(self.terms, {}, Matrix, owner_values)

    def _assemble_terms(self, terms, values, Matrix, owner_values=None):
        if getattr(self, "tdim", 1) == 2:
            return self._assemble_terms_2d(terms, values, owner_values)
        rank = terms[0].rank
        cplx = any(np.iscomplexobj(getattr(v, "vector", v)) for v in values.values()) or \
            any(np.iscomplexobj(v) for v in (owner_values or {}).values()) or \
            any(isinstance(c.value, complex) for c in self._consts.values())
        if cplx and rank == 2:
            raise TypeError("complex field values are supported for rank 0 and rank 1 forms only")
        if _COMPILED:
            comp = self._compiled_1d(terms)
            if comp is not None:                                # the whole form in C++
                return comp.assemble_complex(values, owner_values) if cplx else comp.assemble(values, owner_values)
        T = self.test_space if rank else None
        is_prod = rank and hasattr(T, "fields") and not hasattr(T, "degree")
        caches = {sid: self._cache(s) for sid, s in self.spaces.items()}
        if is_prod:
            for f in T.fields: caches[id(f)] = self._cache(f)
        if rank == 2:
            R = self.trial_space
            if is_prod != (hasattr(R, "fields") and not hasattr(R, "degree")):
                raise TypeError("test and trial functions must both be plain or both be product-space arguments")
            if is_prod and R is not T:
                raise TypeError("test and trial ProductSpaces must be the same object")
            if is_prod:
                out = Matrix(_C.AssembledMatrix(T.dim, T.uband), T.periodic, T)
            elif R is not T:
                if not _same_mesh(T, R):
                    raise ValueError("a form with test and trial functions in different spaces needs both on the same "
                                     "grid; for a mass-type coupling between grids use fd.mixed_mass(W, V), which "
                                     "integrates exactly on the union of the two grids")
                out = _RectAccum(T, R)
            else:
                out = Matrix(_C.AssembledMatrix(T.dim, T.uband), T.bc.is_periodic, R)
        elif rank == 1:
            out = np.zeros(T.dim, dtype=np.complex128 if cplx else np.float64)
        else:
            out = [0.0]                      # mutable scalar accumulator

        for t in terms:
            if t.measure.kind == "dx" and t.measure.scheme == "lobatto":
                rule = ("lobatto", self._lobatto_points(t))
                lc = {sid: self._cache(s, rule) for sid, s in self.spaces.items()}
                if is_prod:
                    for f in T.fields: lc[id(f)] = self._cache(f, rule)
                self._add_interior(t, values, lc, out, is_prod, owner_values)
            elif t.measure.kind == "dx":
                self._add_interior(t, values, caches, out, is_prod, owner_values)
            elif t.measure.kind == "dS":
                self._add_facet(t, values, out, is_prod, owner_values)
            else:
                self._add_boundary(t, values, out, is_prod, owner_values)
        if isinstance(out, _RectAccum):
            return out.finish()
        return out[0] if rank == 0 else out

    def _add_interior(self, t, values, caches, out, is_prod, owner_values=None):
        space_for_nodes = t.test.space if t.test is not None else next(iter(self.spaces.values()))
        Q = caches[id(space_for_nodes)]
        ctx = _Context(Q.nodes(), values, caches, node_space=space_for_nodes, owner_values=owner_values)
        c = np.asarray(evaluate(t.coeff, ctx))
        if t.rank < 2:
            # the kernels are real and linear in the coefficient: a complex one is two real calls
            if t.rank == 0:
                kern = lambda cc: _C.assemble_scalar(Q, cc)                                   # noqa: E731
            elif is_prod:
                kern = lambda cc: _C.assemble_block_vector(self.test_space, t.test.field,     # noqa: E731
                                                          caches[id(t.test.space)], t.test.deriv, cc)
            else:
                kern = lambda cc: _C.assemble_vector(Q, t.test.deriv, cc, self.test_space.dim)  # noqa: E731
            if np.iscomplexobj(c):
                val = kern(np.ascontiguousarray(c.real)) + 1j * kern(np.ascontiguousarray(c.imag))
            else:
                val = kern(np.ascontiguousarray(c, dtype=np.float64))
            if t.rank == 0:
                out[0] += val
            else:
                out += val
            return
        c = np.ascontiguousarray(c, dtype=np.float64)
        if is_prod:
            out.add_block(self.test_space, t.test.field, t.trial.field, caches[id(t.test.space)], caches[id(t.trial.space)],
                          t.test.deriv, t.trial.deriv, c)
        elif isinstance(out, _RectAccum):
            out.parts.append(_C.assemble_rect(caches[id(t.test.space)], caches[id(t.trial.space)],
                                              t.test.deriv, t.trial.deriv, c))
        else:
            out.add(caches[id(t.test.space)], t.test.deriv, t.trial.deriv, c)

    def _global_indices(self, P):
        """Per field, the product-space index of each field dof (cached)."""
        key = ("gidx", id(P))
        if key not in self._caches:
            parts = P.split(np.arange(P.dim, dtype=np.float64))
            self._caches[key] = [np.rint(np.asarray(q)).astype(np.int32) for q in parts]
        return self._caches[key]

    def _add_facet(self, t, values, out, is_prod=False, owner_values=None):
        space = t.test.space if t.test is not None else next(iter(self.spaces.values()))
        FT = self._facet_cache(space)
        for s in self.spaces.values():
            if not _same_mesh(s, space):
                raise ValueError("dS: every space in the form must be on the same grid")
        ctx = _FacetContext(np.asarray(FT.points(0)), values, self._facet_cache, owner_values=owner_values)
        c = np.asarray(evaluate(t.coeff, ctx))
        if c.ndim == 0:
            c = np.full(FT.nf, c)
        side = lambda a: 0 if a.side == "-" else 1            # noqa: E731
        if t.rank == 0:
            out[0] += c.sum()
            return
        if t.rank == 1:
            n = t.test.space.dim if is_prod else self.test_space.dim
            kern = lambda cc: _C.facet_vector(FT, side(t.test), t.test.deriv, np.ascontiguousarray(cc), n)  # noqa: E731
            val = (kern(c.real) + 1j * kern(c.imag)) if np.iscomplexobj(c) else kern(np.asarray(c, dtype=np.float64))
            if is_prod:
                out[self._global_indices(self.test_space)[t.test.field]] += val
            else:
                out += val
            return
        FS = self._facet_cache(t.trial.space)
        c = np.ascontiguousarray(c, dtype=np.float64)
        if is_prod:
            r, cc, v = _C.facet_triplets(FT, FS, side(t.test), side(t.trial), t.test.deriv, t.trial.deriv, c)
            out._K.add_coo(self.test_space, t.test.field, t.trial.field, np.ascontiguousarray(r, dtype=np.int32),
                           np.ascontiguousarray(cc, dtype=np.int32), np.ascontiguousarray(v, dtype=np.float64))
            if t.test.field != t.trial.field or t.test.deriv != t.trial.deriv or t.test.side != t.trial.side:
                out._K.symmetric = False
        elif isinstance(out, _RectAccum):
            out.parts.append(_C.facet_triplets(FT, FS, side(t.test), side(t.trial), t.test.deriv, t.trial.deriv, c))
        else:
            _C.facet_matrix(out._K, FT, FS, side(t.test), side(t.trial), t.test.deriv, t.trial.deriv, c)

    def _add_boundary(self, t, values, out, is_prod, owner_values=None):
        space = t.test.space if t.test is not None else next(iter(self.spaces.values()))
        mesh = space.mesh
        if space.bc.is_periodic:
            raise ValueError("ds has no meaning on a periodic space")
        pts = {"left": [mesh.a], "right": [mesh.b], "both": [mesh.a, mesh.b]}[t.measure.where]
        for xp in pts:
            ctx = _Context(np.array([xp]), values, None, point=xp, owner_values=owner_values)
            c = np.asarray(evaluate(t.coeff, ctx))[0]
            c = complex(c) if np.iscomplexobj(c) else float(c)
            if t.rank == 0:
                out[0] += c
                continue
            rows = t.test.space.basis_matrix([xp], t.test.deriv).tocoo()
            gi = (lambda j: self.test_space.global_index(t.test.field, j)) if is_prod else (lambda j: j)
            if t.rank == 1:
                for j, v in zip(rows.col, rows.data):
                    out[gi(j)] += c * v
                continue
            cols = t.trial.space.basis_matrix([xp], t.trial.deriv).tocoo()
            gj = (lambda j: self.test_space.global_index(t.trial.field, j)) if is_prod else (lambda j: j)
            add = out.add_entry if isinstance(out, _RectAccum) else out._K.add_entry
            for i, vi in zip(rows.col, rows.data):
                for j, vj in zip(cols.col, cols.data):
                    add(gi(i), gj(j), c * vi * vj)

    def __repr__(self):
        return f"Form(rank={self.rank}, terms={self.terms}, functions={self.functions}, quad_degree={self.quad_degree})"


def form(fe, space=None) -> Form:
    """Compile expr*dx (+ expr*ds ...) into a Form.  `space` is needed only when the
    expression contains no test, trial or Function (a pure integral of data)."""
    return Form(fe, space)
