import brainstate as bst
import brainunit as u

import pinnx


def pde(x, y):
    dy_xxxx = net.gradient(x, order=4)['y']['x']['x']['x']['x']
    return dy_xxxx + 1


def boundary_l(x, on_boundary):
    return u.math.logical_and(on_boundary, pinnx.utils.isclose(x['x'], 0))


def boundary_r(x, on_boundary):
    return u.math.logical_and(on_boundary, pinnx.utils.isclose(x['x'], 1))


def func(x):
    x = x['x']
    y = -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4
    return {'y': y}


geom = pinnx.geometry.Interval(0, 1).to_dict_point('x')

bc1 = pinnx.icbc.DirichletBC(lambda x: {'y': 0}, boundary_l)
bc2 = pinnx.icbc.NeumannBC(lambda x: {'y': 0}, boundary_l)
bc3 = pinnx.icbc.OperatorBC(lambda x, y: net.hessian(x), boundary_r)
bc4 = pinnx.icbc.OperatorBC(lambda x, y: net.gradient(x, order=3), boundary_r)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

data = pinnx.problem.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    net,
    num_domain=100,
    num_boundary=20,
    solution=func,
    num_test=100,
)

trainer = pinnx.Trainer(data)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
