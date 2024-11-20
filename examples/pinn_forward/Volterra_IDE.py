import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
import optax

import pinnx


def ide(neu, x, int_mat):
    jacobian, y = pinnx.grad.jacobian(neu, x, return_value=True)
    jacobian = u.math.squeeze(jacobian)
    y = u.math.squeeze(y)
    rhs = u.math.matmul(int_mat, y)
    return (jacobian + y)[: len(rhs)] - rhs


def kernel(x, s):
    return np.exp(s - x)


def func(x):
    return np.exp(-x) * np.cosh(x)


geom = pinnx.geometry.TimeDomain(0, 5)
ic = pinnx.icbc.IC(geom, func, lambda _, on_initial: on_initial)

quad_deg = 20
data = pinnx.data.IDE(
    geom,
    ide,
    ic,
    quad_deg,
    kernel=kernel,
    num_domain=10,
    num_boundary=2,
    train_distribution="uniform",
)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
net = pinnx.nn.FNN(layer_size, activation)

model = pinnx.Model(data, net)
model.compile(
    bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)),
)
model.train(5000, display_every=200)

X = geom.uniform_points(100)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X, y_true, "-")
plt.plot(X, y_pred, "o")
plt.show()
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
