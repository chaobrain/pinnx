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


import saiunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import pinnx
from pinnx import utils
from pinnx.utils import losses


@pytest.mark.parametrize(
    "a",
    [np.array([0.0, 1e-9, 1.0]), jnp.array([0.0, 1e-9, 1.0]), [0.0, 1e-9, 1.0]],
)
def test_isclose_accepts_numpy_jax_and_lists(a):
    np.testing.assert_array_equal(np.asarray(utils.isclose(a, 0)), [True, True, False])


@pytest.mark.parametrize("a", [1, 1.0, np.float32(1.0)])
def test_isclose_accepts_scalars(a):
    assert bool(utils.isclose(a, 1))


def test_isclose_quantity():
    x = u.math.asarray([0.0, 1.0]) * u.meter
    np.testing.assert_array_equal(np.asarray(utils.isclose(x, 0 * u.meter)), [True, False])


def test_standardize_returns_scaler():
    x_train = np.array([[0.0, 10.0], [2.0, 30.0]])
    x_test = np.array([[1.0, 20.0]])
    scaler, xs_train, xs_test = utils.standardize(x_train, x_test)
    np.testing.assert_allclose(xs_train.mean(axis=0), 0)
    np.testing.assert_allclose(xs_train.std(axis=0), 1)
    np.testing.assert_allclose(xs_test, [[0.0, 0.0]])
    np.testing.assert_allclose(scaler.transform(x_test), xs_test)
    np.testing.assert_allclose(scaler.inverse_transform(xs_train), x_train)


def test_standardize_1d_and_constant_feature():
    scaler, xs_train, xs_test = utils.standardize(np.ones(4), np.ones(2))
    assert xs_train.shape == (4,) and xs_test.shape == (2,)
    np.testing.assert_allclose(xs_train, 0)


def test_dataset_standardize():
    problem = pinnx.problem.DataSet(
        X_train={"x": np.array([0.0, 2.0])},
        y_train={"y": np.array([0.0, 1.0])},
        X_test={"x": np.array([1.0])},
        y_test={"y": np.array([0.5])},
        standardize=True,
    )
    np.testing.assert_allclose(problem.train_x["x"], [-1.0, 1.0])
    np.testing.assert_allclose(problem.test_x["x"], [0.0])
    np.testing.assert_allclose(problem.scaler_x["x"].transform(np.array([2.0])), [1.0])


@pytest.mark.parametrize("name", ["sin", "cos", "tanh", "relu", "sigmoid", "silu"])
def test_get_activation(name):
    f = utils.get_activation(name)
    np.testing.assert_allclose(f(jnp.array([0.5])), getattr(u.math, name)(jnp.array([0.5])), rtol=1e-6)


def test_get_activation_unknown():
    with pytest.raises(ValueError, match="foo"):
        utils.get_activation("foo")


def test_mean_l2_relative_error_is_relative():
    y_true = {"y": jnp.array([3.0, 4.0])}
    y_pred = {"y": jnp.array([3.0, 4.0]) * 1.1}
    np.testing.assert_allclose(losses.mean_l2_relative_error(y_true, y_pred)["y"], 0.1, rtol=1e-5)


def test_softmax_cross_entropy_argument_order():
    labels = {"y": jnp.array([[0.0, 1.0]])}
    logits = {"y": jnp.array([[0.0, 10.0]])}
    loss = losses.softmax_cross_entropy(labels, logits)["y"]
    assert float(loss) < 1e-3
