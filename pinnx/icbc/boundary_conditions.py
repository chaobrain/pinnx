"""Boundary conditions."""

import numbers
from typing import Callable

import brainstate as bst
import numpy as np

from pinnx import grad
from pinnx import utils
from pinnx.utils.sampler import BatchSampler
from .base import ICBC, npfunc_range_autocache

__all__ = [
    "BC",
    "DirichletBC",
    "Interface2DBC",
    "NeumannBC",
    "OperatorBC",
    "PeriodicBC",
    "PointSetBC",
    "PointSetOperatorBC",
    "RobinBC",
]


class BC(ICBC):
    """
    Boundary condition base class.

    Args:
        geom: A ``pinnx.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array([on_boundary(x[i], on[i]) for i in range(len(x))])
        self.component = component
        self.boundary_normal = npfunc_range_autocache(utils.return_tensor(self.geom.boundary_normal))

    def filter(self, X):
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, approx: Callable, X, inputs, beg, end):
        dydx = grad.jacobian(lambda x: approx(x)[self.component], inputs)[beg:end]
        n = self.boundary_normal(X, beg, end, None)
        return np.sum(dydx * n, 1, keepdims=True)


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if np.ndim(values) == 2 and np.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component: self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        aux_var = None
        values = self.func(X, beg, end, aux_var)
        return self.normal_derivative(approx, X, inputs, beg, end) - values


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        return self.normal_derivative(approx, X, inputs, beg, end) - self.func(X[beg:end], outputs[beg:end])


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError("PeriodicBC only supports derivative_order 0 or 1.")

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component: self.component + 1]
            yright = outputs[mid:end, self.component: self.component + 1]
        else:
            dydx = grad.jacobian(lambda x: approx(x)[self.component], inputs)[..., self.component_x]
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        return yleft - yright


class OperatorBC(BC):
    """General operator boundary conditions: func(inputs, outputs, X) = 0.

    Args:
        geom: ``Geometry``.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors,
            respectively; `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    Warning:
        If you use `X` in `func`, then do not set ``num_test`` when you define
        ``pinnx.data.PDE`` or ``pinnx.data.TimePDE``, otherwise DeepXDE would throw an
        error. In this case, the training points will be used for testing, and this will
        not affect the network training and training loss. This is a bug of DeepXDE,
        which cannot be fixed in an easy way for all backends.
    """

    def __init__(self, geom, func, on_boundary):
        super().__init__(geom, on_boundary, 0)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        return self.func(inputs, outputs, X)[beg:end]


class PointSetBC(BC):
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        component: Integer or a list of integers. The output components satisfying this BC.
            List of integers only supported for the backend PyTorch.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'pinnx.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(self, points, values, component=0, batch_size=None, shuffle=True):
        self.points = np.array(points, dtype=bst.environ.dftype())
        self.values = np.asarray(values, dtype=bst.environ.dftype())
        self.component = component
        self.batch_size = batch_size

        if batch_size is not None:  # batch iterator and state
            self.batch_sampler = BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        if self.batch_size is not None:
            if isinstance(self.component, numbers.Number):
                return (outputs[beg:end, self.component: self.component + 1]
                        - self.values[self.batch_indices])
            return outputs[beg:end, self.component] - self.values[self.batch_indices]
        if isinstance(self.component, numbers.Number):
            return outputs[beg:end, self.component: self.component + 1] - self.values
        return outputs[beg:end, self.component] - self.values


class PointSetOperatorBC(BC):
    """General operator boundary conditions for a set of points.

    Compare the function output, func, (that associates with `points`)
        with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are
            known and used for training.
        values: An array of values which output of function should fulfill.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of
            `inputs`. `inputs` and `outputs` are the network input and output
            tensors, respectively; `X` are the NumPy array of the `inputs`.
    """

    def __init__(self, points, values, func):
        self.points = np.array(points, dtype=bst.environ.dftype())
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError("PointSetOperatorBC should output 1D values")
        self.values = np.asarray(values, dtype=bst.environ.dftype())
        self.func = func

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        return self.func(inputs, outputs, X)[beg:end] - self.values


class Interface2DBC:
    """2D interface boundary condition.

    This BC applies to the case with the following conditions:
    (1) the network output has two elements, i.e., output = [y1, y2],
    (2) the 2D geometry is ``pinnx.geometry.Rectangle`` or ``pinnx.geometry.Polygon``, which has two edges of the same length,
    (3) uniform boundary points are used, i.e., in ``pinnx.data.PDE`` or ``pinnx.data.TimePDE``, ``train_distribution="uniform"``.
    For a pair of points on the two edges, compute <output_1, d1> for the point on the first edge
    and <output_2, d2> for the point on the second edge in the n/t direction ('n' for normal or 't' for tangent).
    Here, <v1, v2> is the dot product between vectors v1 and v2;
    and d1 and d2 are the n/t vectors of the first and second edges, respectively.
    In the normal case, d1 and d2 are the outward normal vectors;
    and in the tangent case, d1 and d2 are the outward normal vectors rotated 90 degrees clockwise.
    The points on the two edges are paired as follows: the boundary points on one edge are sampled clockwise,
    and the points on the other edge are sampled counterclockwise. Then, compare the sum with 'values',
    i.e., the error is calculated as <output_1, d1> + <output_2, d2> - values,
    where 'values' is the argument `func` evaluated on the first edge.

    Args:
        geom: a ``pinnx.geometry.Rectangle`` or ``pinnx.geometry.Polygon`` instance.
        func: the target discontinuity between edges, evaluated on the first edge,
            e.g., ``func=lambda x: 0`` means no discontinuity is wanted.
        on_boundary1: First edge func. (x, Geometry.on_boundary(x)) -> True/False.
        on_boundary2: Second edge func. (x, Geometry.on_boundary(x)) -> True/False.
        direction (string): "normal" or "tangent".
    """

    def __init__(self, geom, func, on_boundary1, on_boundary2, direction="normal"):
        self.geom = geom
        self.func = npfunc_range_autocache(utils.return_tensor(func))
        self.on_boundary1 = lambda x, on: np.array([on_boundary1(x[i], on[i]) for i in range(len(x))])
        self.on_boundary2 = lambda x, on: np.array([on_boundary2(x[i], on[i]) for i in range(len(x))])
        self.direction = direction
        self.boundary_normal = npfunc_range_autocache(utils.return_tensor(self.geom.boundary_normal))

    def collocation_points(self, X):
        on_boundary = self.geom.on_boundary(X)
        X1 = X[self.on_boundary1(X, on_boundary)]
        X2 = X[self.on_boundary2(X, on_boundary)]
        # Flip order of X2 when pinnx.geometry.Polygon is used
        if self.geom.__class__.__name__ == "Polygon":
            X2 = np.flip(X2, axis=0)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end, approx: Callable):
        mid = beg + (end - beg) // 2
        if not mid - beg == end - mid:
            raise RuntimeError(
                "There is a different number of points on each edge,\n\
                this is likely because the chosen edges do not have the same length."
            )
        aux_var = None
        values = self.func(X, beg, mid, aux_var)
        if np.ndim(values) == 2 and np.shape(values)[1] != 1:
            raise RuntimeError("BC function should return an array of shape N by 1")
        left_n = self.boundary_normal(X, beg, mid, None)
        right_n = self.boundary_normal(X, mid, end, None)
        if self.direction == "normal":
            left_side = outputs[beg:mid, :]
            right_side = outputs[mid:end, :]
            left_values = np.sum(left_side * left_n, 1, keepdims=True)
            right_values = np.sum(right_side * right_n, 1, keepdims=True)

        elif self.direction == "tangent":
            # Tangent vector is [n[1],-n[0]] on edge 1
            left_side1 = outputs[beg:mid, 0:1]
            left_side2 = outputs[beg:mid, 1:2]
            right_side1 = outputs[mid:end, 0:1]
            right_side2 = outputs[mid:end, 1:2]
            left_values_1 = np.sum(left_side1 * left_n[:, 1:2], 1, keepdims=True)
            left_values_2 = np.sum(-left_side2 * left_n[:, 0:1], 1, keepdims=True)
            left_values = left_values_1 + left_values_2
            right_values_1 = np.sum(right_side1 * right_n[:, 1:2], 1, keepdims=True)
            right_values_2 = np.sum(-right_side2 * right_n[:, 0:1], 1, keepdims=True)
            right_values = right_values_1 + right_values_2

        return left_values + right_values - values
