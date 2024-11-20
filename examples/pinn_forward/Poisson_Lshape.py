import brainstate as bst
import optax

import pinnx


def pde(neu, x):
    x = pinnx.array_to_dict(x, "x", "y")
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), "u")
    hessian = pinnx.grad.hessian(approx, x)
    dy_xx = hessian["u"]["x"]["x"]
    dy_yy = hessian["u"]["y"]["y"]
    return -dy_xx - dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary


geom = pinnx.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = pinnx.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
net = pinnx.nn.FNN([2] + [50] * 4 + [1], "tanh")
model = pinnx.Model(data, net)

model.compile(
    bst.optim.Adam(1e-3),
)
model.train(iterations=50000)
model.compile(
    bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)),
)
losshistory, train_state = model.train(10000)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
