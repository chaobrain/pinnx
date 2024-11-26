from typing import Literal, Union, Dict

import brainstate as bst
import brainunit as u
import numpy as np

from pinnx.utils import isclose
from pinnx.utils.sampling import sample
from .base import Geometry

__all__ = [
    'Interval',
]


class Interval(Geometry):
    """
    A class for 1D interval geometry.
    """

    def __init__(
        self,
        name: str,
        l: bst.typing.ArrayLike,
        r: bst.typing.ArrayLike,
    ):
        super().__init__(1, (u.math.array([l]), u.math.array([r])), r - l, [name])
        self.l, self.r = l, r
        assert isinstance(name, str), "name must be a string."
        self.name = name

    def inside(
        self,
        x: Dict[str, bst.typing.ArrayLike]
    ):
        return u.math.logical_and(self.l <= x[self.name], x[self.name] <= self.r).flatten()

    def on_boundary(
        self,
        x: Dict[str, bst.typing.ArrayLike]
    ):
        return u.math.logical_or(isclose(x[self.name], self.l),
                                 isclose(x[self.name], self.r))

    def distance2boundary(
        self,
        x: Dict[str, bst.typing.ArrayLike],
        dirn
    ):
        return (x[self.name] - self.l) if dirn < 0 else (self.r - x[self.name])

    def mindist2boundary(
        self,
        x: Dict[str, bst.typing.ArrayLike]
    ):
        return u.math.minimum(u.math.amin(x[self.name] - self.l),
                              u.math.amin(self.r - x[self.name]))

    def boundary_constraint_factor(
        self,
        x,
        smoothness: Literal["C0", "C0+", "Cinf"] = "C0+",
        where: Union[None, Literal["left", "right"]] = None,
    ):
        """
        Compute the hard constraint factor at x for the boundary.

        This function is used for the hard-constraint methods in Physics-Informed Neural Networks (PINNs).
        The hard constraint factor satisfies the following properties:

        - The function is zero on the boundary and positive elsewhere.
        - The function is at least continuous.

        In the ansatz `boundary_constraint_factor(x) * NN(x) + boundary_condition(x)`, when `x` is on the boundary,
        `boundary_constraint_factor(x)` will be zero, making the ansatz be the boundary condition, which in
        turn makes the boundary condition a "hard constraint".

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the geometry. Note that `x` should be a tensor type
                of backend (e.g., `tf.Tensor` or `torch.Tensor`), not a numpy array.
            smoothness (string, optional): A string to specify the smoothness of the distance function,
                e.g., "C0", "C0+", "Cinf". "C0" is the least smooth, "Cinf" is the most smooth.
                Default is "C0+".

                - C0
                The distance function is continuous but may not be non-differentiable.
                But the set of non-differentiable points should have measure zero,
                which makes the probability of the collocation point falling in this set be zero.

                - C0+
                The distance function is continuous and differentiable almost everywhere. The
                non-differentiable points can only appear on boundaries. If the points in `x` are
                all inside or outside the geometry, the distance function is smooth.

                - Cinf
                The distance function is continuous and differentiable at any order on any
                points. This option may result in a polynomial of HIGH order.

            where (string, optional): A string to specify which part of the boundary to compute the distance,
                e.g., "left", "right". If `None`, compute the distance to the whole boundary. Default is `None`.

        Returns:
            A tensor of a type determined by the backend, which will have a shape of (n, 1).
            Each element in the tensor corresponds to the computed distance value for the respective point in `x`.
        """

        if where not in [None, "left"]:
            raise ValueError("where must be None or left")
        if smoothness not in ["C0", "C0+", "Cinf"]:
            raise ValueError("smoothness must be one of C0, C0+, Cinf")

        # To convert self.l and self.r to tensor,
        # and avoid repeated conversion in the loop
        l_tensor = self.l
        r_tensor = self.r

        dist_l = dist_r = None
        if where != "right":
            dist_l = u.math.abs((x[self.name] - l_tensor) / (r_tensor - l_tensor) * 2)
        if where != "left":
            dist_r = u.math.abs((x[self.name] - r_tensor) / (r_tensor - l_tensor) * 2)

        if where is None:
            if smoothness == "C0":
                return u.math.minimum(dist_l, dist_r)
            if smoothness == "C0+":
                return dist_l * dist_r
            return u.math.square(dist_l * dist_r)
        if where == "left":
            if smoothness == "Cinf":
                dist_l = u.math.square(dist_l)
            return dist_l
        if smoothness == "Cinf":
            dist_r = u.math.square(dist_r)
        return dist_r

    def boundary_normal(self, x):
        normal = (-isclose(x[self.name], self.l).astype(bst.environ.dftype()) +
                  isclose(x[self.name], self.r).astype(bst.environ.dftype()))
        return u.math.expand_dims(normal, axis=-1)

    def uniform_points(self, n, boundary=True) -> Dict[str, bst.typing.ArrayLike]:
        if boundary:
            r = u.math.linspace(self.l, self.r, num=n, dtype=bst.environ.dftype())
        else:
            r = u.math.linspace(self.l, self.r, num=n + 1, endpoint=False, dtype=bst.environ.dftype())[1:]
        return {self.name: r}

    def random_points(self, n, random="pseudo") -> Dict[str, bst.typing.ArrayLike]:
        x = sample(n, 1, random)[..., 0]
        x = ((self.r - self.l) * x + self.l).astype(bst.environ.dftype())
        return {self.name: x}

    def uniform_boundary_points(self, n) -> Dict[str, bst.typing.ArrayLike]:
        if n == 1:
            r = u.math.array([self.l]).astype(bst.environ.dftype())
        else:
            xl = u.math.full((n // 2,), self.l).astype(bst.environ.dftype())
            xr = u.math.full((n - n // 2,), self.r).astype(bst.environ.dftype())
            r = u.math.vstack((xl, xr))
        return {self.name: r}

    def random_boundary_points(self, n, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        if n == 2:
            r = u.math.array([self.l, self.r]).astype(bst.environ.dftype())
        else:
            r = u.math.where(bst.random.rand(n) < 0.5, self.l, self.r).astype(bst.environ.dftype())
        return {self.name: r}

    def periodic_point(self, x, component=0) -> Dict[str, bst.typing.ArrayLike]:
        tmp = x[self.name]
        tmp = u.math.where(isclose(tmp, self.l), self.r, self.l)
        return {self.name: tmp}

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Args:
            dirn: -1 (left), or 1 (right), or 0 (both direction).
            dist2npt: A function which converts distance to the number of extra
                points (not including x).
            shift: The number of shift.
        """

        def background_points_left():
            dx = x[0] - self.l
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] - np.arange(-shift, n - shift + 1, dtype=bst.environ.dftype()) * h
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + np.arange(-shift, n - shift + 1, dtype=bst.environ.dftype()) * h
            return pts[:, None]

        return (
            background_points_left()
            if dirn < 0
            else background_points_right()
            if dirn > 0
            else np.vstack((background_points_left(), background_points_right()))
        )
