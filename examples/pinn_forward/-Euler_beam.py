import brainstate as bst

import pinnx


def ddy(x, y):
    return pinnx.grad.hessian(y, x)


def dddy(x, y):
    return pinnx.grad.jacobian(ddy(x, y), x)


def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = pinnx.grad.hessian(dy_xx, x)
    return dy_xxxx + 1


def boundary_l(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 1)


def func(x):
    return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4


geom = pinnx.geometry.Interval(0, 1)

bc1 = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = pinnx.icbc.NeumannBC(geom, lambda x: 0, boundary_l)
bc3 = pinnx.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
bc4 = pinnx.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

data = pinnx.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=10,
    num_boundary=2,
    solution=func,
    num_test=100,
)
layer_size = [1] + [20] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh", bst.init.KaimingUniform())

model = pinnx.Model(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
