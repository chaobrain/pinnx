"""Backend supported: tensorflow.compat.v1, paddle

Implementation of the Poisson 1D example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import brainstate as bst
import brainunit as u
import numpy as np

import pinnx

A = 2
B = 50


def pde(neu, x):
    x = pinnx.array_to_dict(x, "x")
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), "y")
    hessian = pinnx.grad.hessian(approx, x)
    x = x["x"]
    dy_xx = hessian["y"]["x"]["x"]
    return (
        dy_xx
        + (np.pi * A) ** 2 * u.math.sin(np.pi * A * x)
        + 0.1 * (np.pi * B) ** 2 * u.math.sin(np.pi * B * x)
    )


def func(x):
    return np.sin(np.pi * A * x) + 0.1 * np.sin(np.pi * B * x)


geom = pinnx.geometry.Interval(0, 1)
bc = pinnx.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)
data = pinnx.data.PDE(
    geom,
    pde,
    bc,
    1280,
    2,
    train_distribution="pseudo",
    solution=func,
    num_test=10000,
)

layer_size = [1] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = pinnx.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])

model = pinnx.Model(data, net)
model.compile(
    bst.optim.Adam(1e-3),
    metrics=["l2 relative error"],
    decay=("inverse time", 2000, 0.9),
)

pde_residual_resampler = pinnx.callbacks.PDEPointResampler(period=1)
model.train(iterations=20000, callbacks=[pde_residual_resampler])

pinnx.saveplot(model.loss_history, model.train_state, issave=True, isplot=True)
