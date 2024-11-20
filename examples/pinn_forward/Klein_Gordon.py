import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.interpolate import griddata

import pinnx


def pde(net, x):
    alpha, beta, gamma, k = -1, 0, 1, 2
    x = pinnx.array_to_dict(x, "x", "t")
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), "y")
    hessian, y = pinnx.grad.hessian(approx, x, return_value=True)

    dy_tt = hessian["y"]["t"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    x, t = x['x'], x['t']
    y = y['y']
    return (
        dy_tt
        + alpha * dy_xx
        + beta * y
        + gamma * (y ** k)
        + x * u.math.cos(t)
        - (x ** 2) * (u.math.cos(t) ** 2)
    )


def func(x):
    return x[:, 0:1] * np.cos(x[:, 1:2])


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 10)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc = pinnx.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic_1 = pinnx.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
ic_2 = pinnx.icbc.OperatorBC(
    geomtime,
    lambda x, y, _: pinnx.grad.jacobian(net, x)[:, 0, 1],
    lambda _, on_initial: on_initial,
)

data = pinnx.data.TimePDE(
    geomtime,
    pde,
    [bc, ic_1, ic_2],
    num_domain=30000,
    num_boundary=1500,
    num_initial=1500,
    solution=func,
    num_test=6000,
)

layer_size = [2] + [40] * 2 + [1]
net = pinnx.nn.FNN(layer_size, "tanh")

model = pinnx.Model(data, net)
model.compile(
    bst.optim.Adam(bst.optim.InverseTimeDecayLR(1e-3, 3000, 0.9)),
    metrics=["l2 relative error"],
)
model.train(iterations=20000)

model.compile(
    bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)),
    metrics=["l2 relative error"]
)
losshistory, train_state = model.train(2000, display_every=200)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

x = np.linspace(-1, 1, 256)
t = np.linspace(0, 10, 256)
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
prediction = model.predict(X_star, operator=None)

v = griddata(X_star, prediction[:, 0], (X, T), method="cubic")

fig, ax = plt.subplots()
ax.set_title("Results")
ax.set_ylabel("Prediction")
ax.imshow(
    v.T,
    interpolation="nearest",
    cmap="viridis",
    extent=(0, 10, -1, 1),
    origin="lower",
    aspect="auto",
)
plt.show()
