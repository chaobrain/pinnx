"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
"""

import brainstate as bst
import braintools
import brainunit as u
import numpy as np
import optax
from scipy.io import loadmat

import pinnx


def gen_testdata():
    data = loadmat("../dataset/Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

d = 0.001


def pde(net, x):
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), 'y')

    x = pinnx.array_to_dict(x, 'x', 't')
    jacobian, out = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t - d * dy_xx - 5 * (out['y'] - out['y'] ** 3)


# Hard restraints on initial + boundary conditions
# Backend tensorflow.compat.v1 or tensorflow
def output_transform(x, y):
    return x[..., 0:1] ** 2 * u.math.cos(np.pi * x[..., 0:1]) + x[..., 1:2] * (1 - x[..., 0:1] ** 2) * y


data = pinnx.data.TimePDE(
    geomtime,
    pde,
    [],
    num_domain=8000,
    num_boundary=400,
    num_initial=800
)
net = pinnx.nn.FNN(
    [2] + [20] * 3 + [1],
    activation="tanh",
    kernel_initializer=bst.init.KaimingUniform(),
    output_transform=output_transform
)
model = pinnx.Model(data, net)

model.compile(bst.optim.Adam(lr=1e-3))
model.train(iterations=10000)

model.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))
losshistory, train_state = model.train(2000, display_every=200)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=data.pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", braintools.metric.l2_norm(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
