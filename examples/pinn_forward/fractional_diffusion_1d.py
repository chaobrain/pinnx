import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import pinnx

alpha = 1.8


def fpde(net, x, int_mat):
    """
    du/dt + (D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)
    """
    x = pinnx.array_to_dict(x, ['x', 't'])
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), ["y"])
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)
    y = y['y']

    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        ini_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = -(ini_mat @ y)
    else:
        lhs = -u.math.matmul(int_mat, y)
    dy_t = jacobian['y']['t']
    x, t = x['x'], x['t']
    rhs = -dy_t - u.math.exp(-t) * (
        x ** 3 * (1 - x) ** 3
        + gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
        - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
        + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
        - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    )
    return lhs - rhs[..., len(lhs)]


def func(x):
    x, t = x[:, :-1], x[:, -1:]
    return np.exp(-t) * x ** 3 * (1 - x) ** 3


geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc = pinnx.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = pinnx.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

data_type = 'static'
data_type = 'dynamic'

if data_type == 'static':
    # Static auxiliary points
    data = pinnx.data.TimeFPDE(
        geomtime,
        fpde,
        alpha,
        [bc, ic],
        [52],
        meshtype="static",
        num_domain=400,
        solution=func,
    )
else:
    # Dynamic auxiliary points
    data = pinnx.data.TimeFPDE(
        geomtime,
        fpde,
        alpha,
        [bc, ic],
        [100],
        num_domain=20,
        num_boundary=1,
        num_initial=1,
        solution=func,
        num_test=50,
    )


def out_transform(x, y):
    x = pinnx.array_to_dict(x, ['x', 't'])
    return x['x'] * (1 - x['x']) * x['t'] * y + x['x'] ** 3 * (1 - x['x']) ** 3


net = pinnx.nn.FNN([2] + [20] * 4 + [1], "tanh", bst.init.KaimingUniform())
net.apply_output_transform(out_transform)

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(1e-3))
losshistory, train_state = model.train(iterations=10000)
pinnx.saveplot(losshistory, train_state, issave=False, isplot=True)

X = geomtime.random_points(1000)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
