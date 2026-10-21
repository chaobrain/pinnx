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


import os

import braintools
import brainstate
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pinnx


def _poisson(solution=True):
    net = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=None),
        pinnx.nn.FNN([1, 8, 1], "tanh"),
        pinnx.nn.ArrayToDict(y=None),
    )

    def pde(x, y):
        return -net.hessian(x)["y"]["x"]["x"] - 1.0

    geom = pinnx.geometry.Interval(0, 1).to_dict_point("x")
    sol = (lambda x: {"y": 0.5 * x["x"] * (1 - x["x"])}) if solution else None
    return pinnx.problem.PDE(
        geom, pde, [pinnx.icbc.DirichletBC({"y": 0.0})], net,
        num_domain=8, num_boundary=2, solution=sol, num_test=10,
    )


def test_save_restore_round_trip(tmp_path):
    trainer = pinnx.Trainer(_poisson())
    trainer.compile(braintools.optim.Adam(1e-2)).train(iterations=2, display_every=1)
    path = trainer.save(str(tmp_path / "model"))
    assert os.path.exists(path)
    saved = {k: jax.tree.map(np.asarray, v.value) for k, v in trainer.params.items()}
    trainer.train(iterations=3, display_every=1)
    trainer.restore(path)
    for k, v in trainer.params.items():
        jax.tree.map(np.testing.assert_allclose, jax.tree.map(np.asarray, v.value), saved[k])


def test_compile_string_metric_and_no_solution():
    trainer = pinnx.Trainer(_poisson())
    trainer.compile(braintools.optim.Adam(1e-3), metrics="l2 relative error").train(iterations=1)
    assert len(trainer.train_state.metrics_test) == 1

    trainer = pinnx.Trainer(_poisson(solution=False))
    trainer.compile(braintools.optim.Adam(1e-3), metrics=["l2 relative error"]).train(iterations=1)
    assert trainer.train_state.metrics_test == []


def test_compile_returns_trainer_when_measuring():
    trainer = pinnx.Trainer(_poisson())
    assert trainer.compile(braintools.optim.Adam(1e-3), measture_train_step_compile_time=True) is trainer
    assert trainer.train_step_compile_time > 0


def test_tuple_of_external_variables():
    a, b = brainstate.ParamState(1.0), brainstate.ParamState(2.0)
    trainer = pinnx.Trainer(_poisson(), external_trainable_variables=(a, b))
    assert sum(1 for k in trainer.params if k[0] == "external_trainable_variable") == 2


def test_trainer_batch_size_is_used():
    X = (np.random.rand(6, 3), np.random.rand(5, 1))
    y = np.random.rand(6, 5)
    net = pinnx.nn.DeepONetCartesianProd([3, 8, 8], [1, 8, 8], "tanh")
    problem = pinnx.problem.TripleCartesianProd(X, y, X, y, approximator=net)
    seen = []
    orig = problem.train_next_batch
    problem.train_next_batch = lambda batch_size=None: seen.append(batch_size) or orig(batch_size)
    trainer = pinnx.Trainer(problem, batch_size=4)
    trainer.compile(braintools.optim.Adam(1e-3)).train(iterations=1)
    assert seen and all(b == 4 for b in seen)


def test_model_checkpoint_save_better_only(tmp_path):
    trainer = pinnx.Trainer(_poisson())
    ckpt = pinnx.callbacks.ModelCheckpoint(str(tmp_path / "m"), save_better_only=True, period=1)
    trainer.compile(braintools.optim.Adam(1e-3)).train(iterations=2, display_every=1, callbacks=[ckpt])
    assert any(f.endswith(".msgpack") for f in os.listdir(tmp_path))


def test_movie_dumper_with_model(tmp_path):
    trainer = pinnx.Trainer(_poisson())
    dumper = pinnx.callbacks.MovieDumper(str(tmp_path / "movie"), [0.0], [1.0], num_points=5, period=1)
    trainer.compile(braintools.optim.Adam(1e-3)).train(iterations=2, display_every=1, callbacks=[dumper])
    assert np.asarray(dumper.y[0]).shape == (5,)


def test_metrics_on_1d_arrays():
    y = jnp.array([1.0, 2.0, 3.0])
    assert float(pinnx.metrics.mean_l2_relative_error(y, y)) == 0.0
    assert float(pinnx.metrics.accuracy(jnp.array([0, 1]), jnp.array([0, 0]))) == 0.5


def test_num_trainable_parameters():
    assert pinnx.nn.FNN([2, 3, 1], "tanh").num_trainable_parameters() == 2 * 3 + 3 + 3 + 1


def test_pod_deeponet_call():
    basis = jnp.ones((5, 2))
    out = pinnx.nn.PODDeepONet(basis, [3, 4, 2], "tanh")((jnp.ones((6, 3)), jnp.ones((5, 1))))
    assert out.shape == (6, 5)
    out = pinnx.nn.PODDeepONet(basis, [3, 4, 4], "tanh", layer_sizes_trunk=[1, 4, 2])(
        (jnp.ones((6, 3)), jnp.ones((5, 1))))
    assert out.shape == (6, 5)


@pytest.mark.parametrize("op", ["mul", "add", "cat"])
def test_mionet_cartesian_prod(op):
    net = pinnx.nn.MIONetCartesianProd(
        [3, 4], [2, 4], [1, 4], "tanh", braintools.init.KaimingUniform(),
        output_merge_operation=op, layer_sizes_output_merger=[8 if op == "cat" else 4, 4, 1],
    )
    out = net((jnp.ones((6, 3)), jnp.ones((6, 2)), jnp.ones((5, 1))))
    assert out.shape == (6, 5)


def test_pod_mionet_with_trunk_and_merger():
    net = pinnx.nn.PODMIONet(
        jnp.ones((5, 2)), [3, 4], [2, 4], "tanh", braintools.init.KaimingUniform(),
        layer_sizes_trunk=[1, 4, 2], layer_sizes_merger=[4, 4, 4],
    )
    assert net((jnp.ones((6, 3)), jnp.ones((6, 2)), jnp.ones((5, 1)))).shape == (6, 5)


def test_array_to_dict_accepts_unit_valued_quantity():
    layer = pinnx.nn.ArrayToDict(u=u.meter, k=1 / u.second)
    out = layer(jnp.ones((2, 2)))
    assert out["k"].unit == u.Hz
    with pytest.raises(TypeError):
        pinnx.nn.ArrayToDict(k=2 * u.second)
