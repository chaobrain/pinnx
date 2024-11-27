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


from pinnx.utils import isclose
from .geometry_1d import Interval
import brainstate as bst

__all__ = [
    'TimeDomain',
]


class TimeDomain(Interval):
    """
    A class for 1D time domain.
    """

    def __init__(
        self,
        name: str,
        t0: bst.typing.ArrayLike,
        t1: bst.typing.ArrayLike,

    ):
        super().__init__(name, t0, t1)
        self.t0 = t0
        self.t1 = t1

    def on_initial(self, t):
        return isclose(t[self.name], self.t0).flatten()
