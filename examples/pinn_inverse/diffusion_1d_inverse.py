import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate as bst
import brainunit as u
import numpy as np

import pinnx as pinnx

C = bst.ParamState(2.0)


def pde(neu, x):
    x = pinnx.array_to_dict(x, "x", "t")
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), "y")
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    dy_t = jacobian["y"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    return (
        dy_t
        - C.value * dy_xx
        + u.math.exp(-x['t'])
        * (u.math.sin(np.pi * x['x']) -
           np.pi ** 2 * u.math.sin(np.pi * x['x']))
    )


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc = pinnx.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = pinnx.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
observe_y = pinnx.icbc.PointSetBC(observe_x, func(observe_x), component=0)

data = pinnx.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=observe_x,
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
net = pinnx.nn.FNN(layer_size, activation)

model = pinnx.Model(data, net)

model.compile(
    bst.optim.Adam(0.001),
    metrics=["l2 relative error"],
    external_trainable_variables=C
)
variable = pinnx.callbacks.VariableValue(C, period=1000)
losshistory, train_state = model.train(iterations=50000, callbacks=[variable])

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
