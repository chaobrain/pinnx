# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
from typing import Optional, Dict

import brainstate as bst
from pinnx.geometry import AbstractGeometry
from pinnx.problem.base import Problem


class ICBC(abc.ABC):
    """
    Base class for initial and boundary conditions.
    """

    # A ``pinnx.geometry.Geometry`` instance.
    geometry: Optional[AbstractGeometry]
    problem: Optional[Problem]

    def apply_geometry(self, geom: AbstractGeometry):
        assert isinstance(geom, AbstractGeometry), 'geometry must be an instance of AbstractGeometry.'
        self.geometry = geom

    def apply_problem(self, problem: Problem):
        assert isinstance(problem, Problem), 'problem must be an instance of Problem.'
        self.problem = problem

    @abc.abstractmethod
    def filter(self, X):
        """
        Filters the input data.
        """
        pass

    @abc.abstractmethod
    def collocation_points(self, X):
        """
        Returns the collocation points.
        """
        pass

    @abc.abstractmethod
    def error(self, inputs, outputs, **kwargs) -> Dict[str, bst.typing.ArrayLike]:
        """
        Returns the loss for each component at the initial or boundary conditions.
        """
