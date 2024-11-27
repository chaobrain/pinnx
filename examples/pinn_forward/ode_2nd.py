import brainstate as bst
import brainunit as u
import numpy as np

import pinnx


def ode(neu, t):
    dy_dt, y = pinnx.grad.jacobian(neu, t, return_value=True)
    d2y_dt2 = pinnx.grad.hessian(neu, t)
    dy_dt = u.math.squeeze(dy_dt)
    y = u.math.squeeze(y)
    d2y_dt2 = u.math.squeeze(d2y_dt2)
    t = u.math.squeeze(t)
    return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t


def func(t):
    return 50 / 81 + t * 5 / 9 - 2 * np.exp(t) + (31 / 81) * np.exp(9 * t)


geom = pinnx.geometry.TimeDomain(0, 0.25)


def boundary_l(t, on_initial):
    return on_initial and pinnx.utils.isclose(t[0], 0)


def bc_func1(inputs, outputs, X):
    return outputs + 1


def bc_func2(inputs, outputs, X):
    return pinnx.grad.jacobian(lambda x: net(x)[0], inputs) - 2


ic1 = pinnx.icbc.IC(geom, lambda x: -1, lambda _, on_initial: on_initial)
ic2 = pinnx.icbc.OperatorBC(geom, bc_func2, boundary_l)

data = pinnx.data.TimePDE(geom, ode, [ic1, ic2], 16, 2, solution=func, num_test=500)
layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh")

model = pinnx.Trainer(data, net)
model.compile(
    bst.optim.Adam(0.001), metrics=["l2 relative error"], loss_weights=[0.01, 1, 1]
)
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
