# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import pytest

import pinnx

pytest.importorskip("sklearn")


def test_grf_kl():
    space = pinnx.fnspace.GRF_KL(num_eig=4, N=50)
    feats = space.random(3)
    assert space.eval_batch(feats, np.linspace(0, 1, 7)[:, None]).shape == (3, 7)


@pytest.mark.parametrize("cls", [pinnx.fnspace.GRF, pinnx.fnspace.GRF_KL, pinnx.fnspace.GRF2D])
def test_unknown_kernel(cls):
    with pytest.raises(ValueError, match="Unknown kernel"):
        cls(kernel="foo")
