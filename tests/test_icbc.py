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


import braintools
import brainunit as u
import jax
import numpy as np
import pytest

import pinnx


def _train(problem, iterations=3):
    trainer = pinnx.Trainer(problem)
    trainer.compile(braintools.optim.Adam(1e-3)).train(iterations=iterations, display_every=1)
    for loss in jax.tree.leaves(trainer.train_state.loss_train):
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(loss))))
    return trainer


def _net(xs, ys, units_x=None, units_y=None):
    units_x = units_x or {k: None for k in xs}
    units_y = units_y or {k: None for k in ys}
    return pinnx.nn.Model(
        pinnx.nn.DictToArray(**units_x),
        pinnx.nn.FNN([len(xs), 8, len(ys)], "tanh"),
        pinnx.nn.ArrayToDict(**units_y),
    )


def _poisson_1d(bcs, net):
    def pde(x, y):
        return -net.hessian(x)["y"]["x"]["x"] - 1.0

    geom = pinnx.geometry.Interval(-1, 1).to_dict_point("x")
    return pinnx.problem.PDE(geom, pde, bcs, net, num_domain=16, num_boundary=2)


def test_dirichlet_constant_value():
    net = _net(["x"], ["y"])
    _train(_poisson_1d([pinnx.icbc.DirichletBC({"y": 0.0})], net))


def test_periodic_bc_derivative_order_1():
    net = _net(["x"], ["y"])

    def boundary_r(x, on_boundary):
        return u.math.logical_and(on_boundary, pinnx.utils.isclose(x["x"], 1))

    bcs = [
        pinnx.icbc.DirichletBC({"y": 0.0}),
        pinnx.icbc.PeriodicBC("y", "x", boundary_r, derivative_order=1),
    ]
    _train(_poisson_1d(bcs, net))


def test_neumann_bc_with_units():
    net = _net(["x"], ["y"], {"x": u.meter}, {"y": u.meter})

    def pde(x, y):
        return net.hessian(x)["y"]["x"]["x"] * u.meter

    geom = pinnx.geometry.Interval(0, 1).to_dict_point(x=u.meter)
    bcs = [
        pinnx.icbc.DirichletBC(lambda x: {"y": 0.0 * u.meter}),
        pinnx.icbc.NeumannBC(lambda x: {"y": 0.0 * u.meter / u.meter}),
    ]
    _train(pinnx.problem.PDE(geom, pde, bcs, net, num_domain=8, num_boundary=2))


@pytest.mark.parametrize("direction", ["normal", "tangent"])
def test_interface2d_bc(direction):
    net = _net(["x", "y"], ["u", "v"])

    def pde(x, y):
        return y["u"] * 0.0

    def left(x, on):
        return u.math.logical_and(on, pinnx.utils.isclose(x["x"], 0.0))

    def right(x, on):
        return u.math.logical_and(on, pinnx.utils.isclose(x["x"], 1.0))

    geom = pinnx.geometry.Rectangle([0.0, 0.0], [1.0, 1.0]).to_dict_point("x", "y")
    bc = pinnx.icbc.Interface2DBC(lambda x: 0.0, left, right, direction=direction)
    problem = pinnx.problem.PDE(
        geom, pde, [bc], net, num_domain=8, num_boundary=16, train_distribution="uniform"
    )
    trainer = _train(problem)

    # errors pair points on the two edges: <y1, n_left> + <y2, n_right>
    x = bc.collocation_points(problem.train_x_all)
    n = len(x["x"])
    assert n > 0 and n % 2 == 0
    np.testing.assert_allclose(np.asarray(x["x"][: n // 2]), 0.0)
    np.testing.assert_allclose(np.asarray(x["x"][n // 2:]), 1.0)
    y = trainer.problem.approximator(x)
    err = jax.tree.leaves(bc.error(x, y))[0]
    if direction == "normal":
        expected = -y["u"][: n // 2] + y["u"][n // 2:]
    else:
        expected = y["v"][: n // 2] - y["v"][n // 2:]
    # corners carry averaged vertex normals; compare points strictly inside the edges
    yl, yr = np.asarray(x["y"][: n // 2]), np.asarray(x["y"][n // 2:])
    interior = (yl > 0) & (yl < 1) & (yr > 0) & (yr < 1)
    assert interior.any()
    np.testing.assert_allclose(
        np.asarray(err)[interior], np.asarray(expected)[interior], rtol=1e-5, atol=1e-6
    )
