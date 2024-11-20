import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import brainstate as bst
import numpy as np
import optax

import pinnx


def gen_testdata():
    data = np.load("../dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(net, x):
    x = pinnx.array_to_dict(x, 'x', 't')
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), 'y')
    jacobian, u = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)
    dy_x = jacobian['y']['x']
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t + u['y'] * dy_x - 0.01 / np.pi * dy_xx


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 0.99)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc = pinnx.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda _, on_boundary: on_boundary
)
ic = pinnx.icbc.IC(
    geomtime,
    lambda x: -np.sin(np.pi * x[:, 0:1]),
    lambda _, on_initial: on_initial
)

data = pinnx.data.TimePDE(
    geomtime, pde, [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160
)
net = pinnx.nn.FNN([2] + [20] * 3 + [1], "tanh", bst.init.KaimingUniform())
model = pinnx.Model(data, net)

model.compile(bst.optim.Adam(1e-3))
model.train(iterations=15000)
model.compile(
    bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None))
)
losshistory, train_state = model.train(2000, display_every=200)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
