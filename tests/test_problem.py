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


import braintools
import brainstate
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import COO

import pinnx


def _net(nin=1, nout=1, xs=("x",), ys=("y",)):
    return pinnx.nn.Model(
        pinnx.nn.DictToArray(**{k: None for k in xs}),
        pinnx.nn.FNN([nin, 8, nout], "tanh"),
        pinnx.nn.ArrayToDict(**{k: None for k in ys}),
    )


def _finite(tree):
    return all(np.all(np.isfinite(np.asarray(u.get_magnitude(x)))) for x in jax.tree.leaves(tree))


def _train(problem, iterations=3, **kw):
    trainer = pinnx.Trainer(problem, **kw)
    trainer.compile(braintools.optim.Adam(1e-3)).train(iterations=iterations, display_every=1)
    assert _finite(trainer.train_state.loss_train)
    assert _finite(trainer.train_state.loss_test)
    return trainer


def test_pde_none_trains_on_bc_only():
    net = _net()
    geom = pinnx.geometry.Interval(0, 1).to_dict_point("x")
    bc = pinnx.icbc.DirichletBC({"y": 1.0})
    problem = pinnx.problem.PDE(geom, None, [bc], net, num_domain=8, num_boundary=2)
    assert len(problem.train_x["x"]) == 2  # no PDE points when there is no PDE
    _train(problem)


def test_data_losses_weights_and_list_loss_fn():
    problem = pinnx.problem.DataSet(
        X_train={"x": np.zeros(2)}, y_train={"a": np.ones(2), "b": np.ones(2)},
        X_test={"x": np.zeros(2)}, y_test={"a": np.ones(2), "b": np.ones(2)},
        loss_fn=["MSE", "MAE"], loss_weights=[1.0, 10.0],
    )
    targets = {"a": jnp.ones(2), "b": jnp.ones(2)}
    outputs = {"a": jnp.zeros(2), "b": jnp.full(2, 3.0)}
    losses = problem.losses(None, outputs, targets)
    np.testing.assert_allclose(float(losses["a"]), 1.0)
    np.testing.assert_allclose(float(losses["b"]), 20.0)


def test_quadruple_dataset_batch_shape():
    n = 10
    X = (np.random.rand(n, 3), np.random.rand(n, 2), np.random.rand(n, 1))
    problem = pinnx.problem.QuadrupleDataset(X, np.random.rand(n, 1), X, np.random.rand(n, 1))
    x, y = problem.train_next_batch(4)
    assert [len(a) for a in x] == [4, 4, 4] and y.shape == (4, 1)


def test_triple_cartesian_prod_tuple_batch():
    X = (np.random.rand(6, 3), np.random.rand(5, 1))
    y = np.random.rand(6, 5)
    problem = pinnx.problem.TripleCartesianProd(X, y, X, y)
    (xb, xt), yb = problem.train_next_batch((3, 2))
    assert xb.shape == (3, 3) and xt.shape == (2, 1) and yb.shape == (3, 2, 1)


def test_quadruple_cartesian_prod_init():
    X = (np.random.rand(4, 3), np.random.rand(4, 2), np.random.rand(5, 1))
    y = np.random.rand(4, 5)
    problem = pinnx.problem.QuadrupleCartesianProd(X, y, X, y)
    loss = problem.losses(None, jnp.zeros((4, 5)), jnp.ones((4, 5)))
    np.testing.assert_allclose(float(loss), 1.0)


def test_mf_dataset_standardize():
    x_lo, x_hi = np.random.rand(6, 1), np.random.rand(3, 1)
    problem = pinnx.problem.MfDataSet(
        X_lo_train=x_lo, X_hi_train=x_hi,
        y_lo_train=np.random.rand(6, 1), y_hi_train=np.random.rand(3, 1),
        X_hi_test=x_hi, y_hi_test=np.random.rand(3, 1), standardize=True,
    )
    np.testing.assert_allclose(problem.X_lo_train.mean(), 0, atol=1e-6)
    losses = problem.losses_train(None, [jnp.zeros((9, 1))] * 2, [jnp.ones((9, 1))] * 2)
    assert len(losses) == 2


def test_mf_func_init():
    geom = pinnx.geometry.Interval(0, 1)
    problem = pinnx.problem.MfFunc(geom, lambda x: x, lambda x: 2 * x, 4, 2, 5)
    loss_lo, loss_hi = problem.losses(None, [jnp.zeros((6, 1))] * 2, [jnp.ones((6, 1))] * 2)
    np.testing.assert_allclose(float(loss_lo), 1.0)


def _fpde_problem(alpha, meshtype):
    net = _net()

    def fpde(x, y, int_mat):
        if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
            rowcols = np.asarray(int_mat[0], dtype=np.int32).T
            int_mat = COO((int_mat[1], rowcols[0], rowcols[1]), shape=int_mat[2])
        return int_mat @ y["y"] - 1.0

    geom = pinnx.geometry.Interval(0, 1).to_dict_point("x")
    bc = pinnx.icbc.DirichletBC({"y": 0.0})
    if meshtype == "static":
        return pinnx.problem.FPDE(geom, fpde, alpha, bc, [11], net, meshtype="static", num_boundary=2)
    return pinnx.problem.FPDE(geom, fpde, alpha, bc, [8], net, num_domain=10, num_boundary=2)


@pytest.mark.parametrize("meshtype", ["static", "dynamic"])
def test_fpde_trains(meshtype):
    _train(_fpde_problem(1.5, meshtype))


@pytest.mark.parametrize("meshtype", ["static", "dynamic"])
def test_fpde_learnable_alpha_gets_gradient(meshtype):
    problem = _fpde_problem(brainstate.ParamState(1.5), meshtype)

    def total(alpha):
        int_mat = problem.get_int_matrix(True, alpha=alpha)
        values = int_mat[1] if isinstance(int_mat, tuple) else int_mat
        return jnp.sum(jnp.asarray(values))

    assert abs(float(jax.grad(total)(jnp.asarray(1.5)))) > 0
    _train(problem, external_trainable_variables=[problem.alpha])


def test_fpde_resample_resets_int_matrix():
    problem = _fpde_problem(1.5, "dynamic")
    problem.int_mat_train = "stale"
    problem.resample_train_points()
    assert problem.int_mat_train is None


def _ide_problem(constraints):
    net = _net()

    def ide(x, y, int_mat):
        lhs = int_mat @ y["y"]
        return lhs + (net.jacobian(x)["y"]["x"] - 1.0)[: len(lhs)]

    geom = pinnx.geometry.TimeDomain(0, 1).to_dict_point("x")
    return pinnx.problem.IDE(geom, ide, constraints, 4, net, num_domain=8, num_boundary=2, num_test=6)


def test_ide_trains_and_test_loss_has_no_bc_slicing():
    problem = _ide_problem(pinnx.icbc.IC(lambda x: {"y": 0.0 * x["x"]}))
    _train(problem)
    with brainstate.environ.context(fit=False):
        out = problem.approximator(problem.test_x)
        errors = problem.call_pde_errors(problem.test_x, out)
    assert len(errors) == 6


def test_ide_without_constraints():
    _train(_ide_problem([]))
