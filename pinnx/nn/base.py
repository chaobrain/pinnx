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


from typing import Optional, Callable

import brainstate as bst
import jax.tree


class NN(bst.nn.Module):
    """Base class for all neural network modules."""

    def __init__(
        self,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.regularization = None
        self._input_transform = input_transform
        self._output_transform = output_transform

    def apply_feature_transform(self, transform):
        """Compute the features by applying a transform to the network inputs, i.e.,
        ``features = transform(inputs)``. Then, ``outputs = network(features)``.
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        ``outputs = transform(inputs, outputs)``.
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        n_param = 0
        for key, val in self.states(bst.ParamState).items():
            n_param += [v.size for v in jax.tree_leaves(val)]
        return n_param
