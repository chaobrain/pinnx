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

import brainunit as u
import numpy as np

__all__ = [
    "array_to_dict",
    "dict_to_array",
]


def array_to_dict(x, *names):
    """
    Convert args to a dictionary.
    """
    if x.shape[-1] != len(names):
        raise ValueError("The number of columns of x must be equal to the number of names.")

    return {key: x[..., i] for i, key in enumerate(names)}


def dict_to_array(d):
    """
    Convert a dictionary to an array.

    Args:
        d (dict): The dictionary.

    Returns:
        ndarray: The array.
    """
    keys = tuple(d.keys())
    if isinstance(d[keys[0]], np.ndarray):
        return np.stack([d[key] for key in keys], axis=-1)
    else:
        return u.math.stack([d[key] for key in keys], axis=-1)
