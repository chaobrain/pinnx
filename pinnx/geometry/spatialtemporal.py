# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Dict

import brainstate as bst
import brainunit as u
import jax.tree

from pinnx.utils import subdict, merge_dict
from .base import AbstractGeometry, Geometry
from .timedomain import TimeDomain

__all__ = [
    'GeometryXTime',
]


class GeometryXTime(AbstractGeometry):
    """
    A class for spatio-temporal domain.
    """

    def __init__(
        self,
        geometry: Geometry,
        timedomain: TimeDomain
    ):
        assert isinstance(geometry, Geometry), 'geometry must be an instance of Geometry.'
        self.geometry = geometry

        assert isinstance(timedomain, TimeDomain), 'timedomain must be an instance of TimeDomain.'
        self.timedomain = timedomain

        super().__init__(geometry.dim + timedomain.dim, geometry.names + timedomain.names)

    def inside(self, x):
        """
        Check if points are inside the spatio-temporal domain.

        Args:
            x: A 2D array of shape (n, dim), where `n` is the number of points and
                `dim` is the dimension of the spatio-temporal domain.

        Returns:
            A 1D boolean array of shape (n,) indicating whether the points are inside the domain.
        """
        return u.math.logical_and(self.geometry.inside(subdict(x, self.geometry.names)),
                                  self.timedomain.inside(subdict(x, self.timedomain.names)))

    def on_boundary(self, x):
        return self.geometry.on_boundary(subdict(x, self.geometry.names))

    def on_initial(self, x):
        return self.timedomain.on_initial(subdict(x, self.timedomain.names))

    def boundary_normal(self, x):
        _n = self.geometry.boundary_normal(subdict(x, self.geometry.names))
        return u.math.hstack([_n, u.math.zeros((len(_n), 1))])

    def uniform_points(self, n, boundary=True) -> Dict[str, bst.typing.ArrayLike]:
        """
        Uniform points on the spatio-temporal domain.

        Geometry volume ~ bbox.
        Time volume ~ diam.
        """
        x = self.geometry.uniform_points(n, boundary=boundary)
        t = self.timedomain.uniform_points(n, boundary=boundary)
        t = jax.tree.map(bst.random.permutation, t)
        return merge_dict(x, t)

    def random_points(self, n, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        x = self.geometry.random_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = jax.tree.map(bst.random.permutation, t)
        return merge_dict(x, t)

    def uniform_boundary_points(self, n) -> Dict[str, bst.typing.ArrayLike]:
        """Uniform boundary points on the spatio-temporal domain.

        Geometry surface area ~ bbox.
        Time surface area ~ diam.
        """
        x = self.geometry.uniform_boundary_points(n)
        t = self.timedomain.uniform_boundary_points(n)
        return merge_dict(x, t)

    def random_boundary_points(self, n, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        x = self.geometry.random_boundary_points(n, random=random)
        t = self.timedomain.random_points(n, random=random)
        t = jax.tree.map(bst.random.permutation, t)
        return merge_dict(x, t)

    def uniform_initial_points(self, n) -> Dict[str, bst.typing.ArrayLike]:
        x = self.geometry.uniform_points(n, True)
        t = {'t': u.math.full(n, self.timedomain.t0, dtype=bst.environ.dftype())}
        return merge_dict(x, t)

    def random_initial_points(self, n, random: str = "pseudo") -> Dict[str, bst.typing.ArrayLike]:
        x = self.geometry.random_points(n, random=random)
        t = {'t': u.math.full(n, self.timedomain.t0, dtype=bst.environ.dftype())}
        return merge_dict(x, t)

    def periodic_point(self, x, component) -> Dict[str, bst.typing.ArrayLike]:
        xp = self.geometry.periodic_point(subdict(x, self.geometry.names), component)
        xp[self.timedomain.name] = x[self.timedomain.name]
        return xp
