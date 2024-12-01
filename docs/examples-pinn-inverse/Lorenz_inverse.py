import brainstate as bst
import numpy as np
import optax

import pinnx


def gen_traindata():
    data = np.load("../../docs/dataset/Lorenz.npz")
    return data["t"], data["y"]


C1 = bst.ParamState(1.0)
C2 = bst.ParamState(1.0)
C3 = bst.ParamState(1.0)


def Lorenz_system(neu, x):
    """
    Lorenz system.

    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (15 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3
    """
    approx = lambda x: pinnx.array_to_dict(neu(x), ["y1", 'y2', 'y3'])
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)
    y1, y2, y3 = y['y1'], y['y2'], y['y3']
    dy1_x = jacobian['y1']
    dy2_x = jacobian['y2']
    dy3_x = jacobian['y3']
    return [
        dy1_x - C1.value * (y2 - y1),
        dy2_x - y1 * (C2.value - y3) + y2,
        dy3_x - y1 * y2 + C3.value * y3,
    ]


def boundary(_, on_initial):
    return on_initial


geom = pinnx.geometry.TimeDomain(0, 3)

# Initial conditions
# TODO: component parameter is not supported in DirichletBC
ic1 = pinnx.icbc.IC(geom, lambda X: -8, boundary, component=0)
ic2 = pinnx.icbc.IC(geom, lambda X: 7, boundary, component=1)
ic3 = pinnx.icbc.IC(geom, lambda X: 27, boundary, component=2)

# Get the train data
observe_t, ob_y = gen_traindata()
observe_y0 = pinnx.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = pinnx.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = pinnx.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

data = pinnx.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

net = pinnx.nn.FNN([1] + [40] * 3 + [3], "tanh")
model = pinnx.Trainer(data, net)

external_trainable_variables = [C1, C2, C3]
variable = pinnx.callbacks.VariableValue(
    external_trainable_variables,
    period=600,
    filename="variables.dat"
)

# train adam
model.compile(
    bst.optim.Adam(0.001),
    external_trainable_variables=external_trainable_variables
)
losshistory, train_state = model.train(iterations=20000, callbacks=[variable])

model.compile(
    bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)),
    external_trainable_variables=external_trainable_variables
)
losshistory, train_state = model.train(10000, callbacks=[variable])

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
