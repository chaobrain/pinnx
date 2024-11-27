import brainstate as bst

import pinnx


def pde(nwu, x):
    dy_xx = pinnx.grad.hessian(nwu, x)
    return dy_xx - 2


def boundary_l(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], -1)


def boundary_r(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 1)


def func(x):
    return (x + 1) ** 2


geom = pinnx.geometry.Interval(-1, 1)
bc_l = pinnx.icbc.DirichletBC(geom, func, boundary_l)
bc_r = pinnx.icbc.RobinBC(geom, lambda X, y: y, boundary_r)
data = pinnx.data.PDE(geom, pde, [bc_l, bc_r], 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
net = pinnx.nn.FNN(layer_size, activation)

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
