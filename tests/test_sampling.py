# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import numpy as np
import pytest

from pinnx.utils.sampling import sample


@pytest.mark.parametrize("sampler", ["pseudo", "LHS", "Halton", "Hammersley", "Sobol"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sample_shape_and_range(sampler, dim):
    x = sample(50, dim, sampler)
    assert isinstance(x, np.ndarray)
    assert x.shape == (50, dim)
    assert np.all(x >= 0) and np.all(x < 1)


@pytest.mark.parametrize("sampler", ["Halton", "Hammersley", "Sobol"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_quasirandom_is_deterministic_and_skips_special_points(sampler, dim):
    x = sample(64, dim, sampler)
    np.testing.assert_array_equal(x, sample(64, dim, sampler))
    assert not np.any(np.all(x == 0, axis=1))
    if dim >= 3:
        # [0.5, ..., 0.5] breaks Hypersphere sampling
        assert not np.any(np.all(x == 0.5, axis=1))
    assert len(np.unique(x, axis=0)) == len(x)


def test_hammersley_first_coordinate_is_uniform_grid():
    x = sample(4, 2, "Hammersley")
    np.testing.assert_allclose(x[:, 0], np.arange(1, 5) / 5)


def test_unknown_sampler_message():
    with pytest.raises(ValueError, match="foo sampling is not available"):
        sample(4, 2, "foo")
