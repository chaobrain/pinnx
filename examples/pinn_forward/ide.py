import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import pinnx


def ide(net, x, int_mat):
    """int_0^x y(t)dt"""
    lhs2, y = pinnx.grad.jacobian(net, x, return_value=True)
    lhs2 = u.math.squeeze(lhs2)
    lhs1 = int_mat @ y
    lhs1 = u.math.squeeze(lhs1)
    rhs = 2 * np.pi * u.math.cos(2 * np.pi * x) + u.math.sin(np.pi * x) ** 2 / np.pi
    rhs = u.math.squeeze(rhs)
    return lhs1 + (lhs2 - rhs)[: len(lhs1)]


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return np.sin(2 * np.pi * x)


geom = pinnx.geometry.TimeDomain(0, 1)
ic = pinnx.icbc.IC(geom, func, lambda _, on_initial: on_initial)

quad_deg = 16
data = pinnx.data.IDE(geom, ide, ic, quad_deg, num_domain=16, num_boundary=2)

net = pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh")

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001))
model.train(iterations=10000)

X = geom.uniform_points(100, True)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X, y_true, "-")
plt.plot(X, y_pred, "o")
plt.show()
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
