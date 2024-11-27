"""Boundary conditions."""

import numbers
from typing import Callable, Dict

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from pinnx import utils
from pinnx.nn.model import Model
from pinnx.utils.sampler import BatchSampler
from .base import ICBC

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
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
    """

    def __init__(
        self,
        on_boundary: Callable[[Dict, np.array], np.array],
    ):
        self.on_boundary = lambda x, on: jax.vmap(on_boundary)(x, on)

    @utils.check_not_none('geometry')
    def filter(self, X):
        """
        Filter the collocation points for boundary conditions.

        Args:
            X: Collocation points.

        Returns:
            Filtered collocation points.
        """
        positions = self.on_boundary(X, self.geometry.on_boundary(X))
        return jax.tree.map(lambda x: x[positions], X)

    def collocation_points(self, X):
        """
        Return the collocation points for boundary conditions.

        Args:
            X: Collocation points.

        Returns:
            Collocation points for boundary conditions.
        """
        return self.filter(X)

    def normal_derivative(self, inputs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Compute the normal derivative of the output.
        """
        # first order derivative
        assert isinstance(self.problem.approximator, Model), ("Normal derivative is only supported "
                                                              "for Sequential approximator.")
        dydx = self.problem.approximator.jacobian(inputs)

        # boundary normal
        n = self.geometry.boundary_normal(inputs)

        return jax.tree.map(lambda x, y: x * y, dydx, n)


class DirichletBC(BC):
    """
    Dirichlet boundary conditions: ``y(x) = func(x)``.

    Args:
        func: A function that takes an array of points and returns an array of values.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    """

    def __init__(
        self,
        func: Callable[[Dict], Dict] | Dict,
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func if callable(func) else lambda x: func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        values = self.func(bc_inputs)
        errors = dict()
        for component in values.keys():
            errors[component] = bc_outputs[component] - values[component]
        return errors


class NeumannBC(BC):
    """
    Neumann boundary conditions: ``dy/dn(x) = func(x)``.

    Args:
        func: A function that takes an array of points and returns an array of values.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
    """

    def __init__(
        self,
        func: Callable[[Dict], Dict],
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        values = self.func(bc_inputs)
        normals = self.normal_derivative(bc_inputs)
        return jax.tree.map(lambda x, y: x - y, normals, values)


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(
        self,
        func: Callable[[Dict, Dict], Dict],
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        values = self.func(bc_inputs, bc_outputs)
        normals = self.normal_derivative(bc_inputs)
        return jax.tree.map(lambda x, y: x - y, normals, values)


class PeriodicBC(BC):
    """
    Periodic boundary conditions on component_x.
    """

    def __init__(
        self,
        component_x,
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
        derivative_order: int = 0,
    ):
        super().__init__(on_boundary)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError("PeriodicBC only supports derivative_order 0 or 1.")

    @utils.check_not_none('geometry')
    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geometry.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, bc_inputs, bc_outputs, **kwargs):
        mid = bc_inputs.shape[0] // 2
        if self.derivative_order == 0:
            yleft = bc_outputs[:mid, self.component: self.component + 1]
            yright = bc_outputs[mid:, self.component: self.component + 1]
        else:
            dydx = grad.jacobian(lambda x: approx(x)[self.component], bc_inputs)[..., self.component_x]
            yleft = dydx[:mid]
            yright = dydx[mid:]
        return yleft - yright


class OperatorBC(BC):
    """General operator boundary conditions: func(inputs, outputs, X) = 0.

    Args:
        geometry: ``Geometry``.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors,
            respectively; `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    Warning:
        If you use `X` in `func`, then do not set ``num_test`` when you define
        ``pinnx.problem.PDE`` or ``pinnx.problem.TimePDE``, otherwise DeepXDE would throw an
        error. In this case, the training points will be used for testing, and this will
        not affect the network training and training loss. This is a bug of DeepXDE,
        which cannot be fixed in an easy way for all backends.
    """

    def __init__(
        self,
        func: Callable[[Dict, Dict], Dict],
        component: str,
        on_boundary: Callable[[Dict, np.array], np.array] = lambda x, on: on,
    ):
        super().__init__(on_boundary, component)
        self.func = func

    def error(self, bc_inputs, bc_outputs, **kwargs):
        return self.func(bc_inputs, bc_outputs)


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

    def __init__(
        self,
        points,
        values,
        component: int = 0,
        batch_size: int = None,
        shuffle: bool = True
    ):
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

    def error(self, bc_inputs, bc_outputs, **kwargs):
        if self.batch_size is not None:
            if isinstance(self.component, numbers.Number):
                return bc_outputs[:, self.component: self.component + 1] - self.values[self.batch_indices]
            else:
                return bc_outputs[:, self.component] - self.values[self.batch_indices]
        if isinstance(self.component, numbers.Number):
            return bc_outputs[:, self.component: self.component + 1] - self.values
        else:
            return bc_outputs[:, self.component] - self.values


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

    def __init__(
        self,
        points,
        values,
        func: Callable[[Dict, Dict], Dict]
    ):
        self.points = np.array(points, dtype=bst.environ.dftype())
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError("PointSetOperatorBC should output 1D values")
        self.values = np.asarray(values, dtype=bst.environ.dftype())
        self.func = func

    def collocation_points(self, X):
        return self.points

    def error(self, bc_inputs, bc_outputs, **kwargs):
        return self.func(bc_inputs, bc_outputs) - self.values


class Interface2DBC(BC):
    """2D interface boundary condition.

    This BC applies to the case with the following conditions:
    (1) the network output has two elements, i.e., output = [y1, y2],
    (2) the 2D geometry is ``pinnx.geometry.Rectangle`` or ``pinnx.geometry.Polygon``, which has two edges of the same length,
    (3) uniform boundary points are used, i.e., in ``pinnx.problem.PDE`` or ``pinnx.problem.TimePDE``, ``train_distribution="uniform"``.
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
        geometry: a ``pinnx.geometry.Rectangle`` or ``pinnx.geometry.Polygon`` instance.
        func: the target discontinuity between edges, evaluated on the first edge,
            e.g., ``func=lambda x: 0`` means no discontinuity is wanted.
        on_boundary1: First edge func. (x, Geometry.on_boundary(x)) -> True/False.
        on_boundary2: Second edge func. (x, Geometry.on_boundary(x)) -> True/False.
        direction (string): "normal" or "tangent".
    """

    def __init__(
        self,
        func: Callable[[Dict], Dict],
        on_boundary1: Callable[[Dict, np.array], np.array],
        on_boundary2: Callable[[Dict, np.array], np.array],
        direction: str = "normal"
    ):
        self.func = utils.return_tensor(func)
        self.on_boundary1 = lambda x, on: np.array([on_boundary1(x[i], on[i]) for i in range(len(x))])
        self.on_boundary2 = lambda x, on: np.array([on_boundary2(x[i], on[i]) for i in range(len(x))])
        self.direction = direction

    @utils.check_not_none('geometry')
    def collocation_points(self, X):
        on_boundary = self.geometry.on_boundary(X)
        X1 = X[self.on_boundary1(X, on_boundary)]
        X2 = X[self.on_boundary2(X, on_boundary)]
        # Flip order of X2 when pinnx.geometry.Polygon is used
        if self.geometry.__class__.__name__ == "Polygon":
            X2 = np.flip(X2, axis=0)
        return np.vstack((X1, X2))

    @utils.check_not_none('geometry')
    def error(self, bc_inputs, bc_outputs, **kwargs):
        mid = bc_inputs.shape[0] // 2
        if bc_inputs.shape[0] % 2 != 0:
            raise RuntimeError("There is a different number of points on each edge,\n "
                               "this is likely because the chosen edges do not have the same length.")
        aux_var = None
        values = self.func(bc_inputs[: mid])
        if np.ndim(values) == 2 and np.shape(values)[1] != 1:
            raise RuntimeError("BC function should return an array of shape N by 1")
        left_n = self.geometry.boundary_normal(bc_inputs[: mid])
        right_n = self.geometry.boundary_normal(bc_inputs[: mid])
        if self.direction == "normal":
            left_side = bc_outputs[:mid, :]
            right_side = bc_outputs[mid:, :]
            left_values = u.math.sum(left_side * left_n, 1, keepdims=True)
            right_values = u.math.sum(right_side * right_n, 1, keepdims=True)

        elif self.direction == "tangent":
            # Tangent vector is [n[1],-n[0]] on edge 1
            left_side1 = bc_outputs[:mid, 0:1]
            left_side2 = bc_outputs[:mid, 1:2]
            right_side1 = bc_outputs[mid:, 0:1]
            right_side2 = bc_outputs[mid:, 1:2]
            left_values_1 = u.math.sum(left_side1 * left_n[:, 1:2], 1, keepdims=True)
            left_values_2 = u.math.sum(-left_side2 * left_n[:, 0:1], 1, keepdims=True)
            left_values = left_values_1 + left_values_2
            right_values_1 = u.math.sum(right_side1 * right_n[:, 1:2], 1, keepdims=True)
            right_values_2 = u.math.sum(-right_side2 * right_n[:, 0:1], 1, keepdims=True)
            right_values = right_values_1 + right_values_2

        else:
            raise ValueError("Invalid direction, must be 'normal' or 'tangent'.")

        return left_values + right_values - values
