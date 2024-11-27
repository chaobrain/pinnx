import brainstate as bst
import brainunit as u
import numpy as np

import pinnx


def pde(neu, x):
    dy_xx = pinnx.grad.hessian(neu, x)
    dy_xx = u.math.squeeze(dy_xx)
    x = u.math.squeeze(x)
    return -dy_xx - np.pi ** 2 * u.math.sin(np.pi * x)


def boundary_l(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], -1)


def boundary_r(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 1)


def func(x):
    return np.sin(np.pi * x)


geom = pinnx.geometry.Interval(-1, 1)
bc1 = pinnx.icbc.DirichletBC(geom, func, boundary_l)
bc2 = pinnx.icbc.PeriodicBC(geom, 0, boundary_r)
data = pinnx.data.PDE(geom, pde, [bc1, bc2], 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh")

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
