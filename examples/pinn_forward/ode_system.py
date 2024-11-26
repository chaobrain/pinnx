import brainstate as bst
import numpy as np

import pinnx


def ode_system(neu, x):
    """ODE system.
    dy1/dx = y2
    dy2/dx = -y1
    """
    x = pinnx.array_to_dict(x, ["x"])
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), ["y1", "y2"])
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)

    y1, y2 = y['y1'], y['y2']
    dy1_x = jacobian["y1"]["x"]
    dy2_x = jacobian["y2"]["x"]
    return [dy1_x - y2, dy2_x + y1]


def func(x):
    """
    y1 = sin(x)
    y2 = cos(x)
    """
    return np.hstack((np.sin(x), np.cos(x)))


geom = pinnx.geometry.TimeDomain(0, 10)
ic1 = pinnx.icbc.IC(geom, lambda x: 0, lambda _, on_initial: on_initial, component=0)
ic2 = pinnx.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial, component=1)
data = pinnx.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)

net = pinnx.nn.FNN([1] + [50] * 3 + [2], "tanh")

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
