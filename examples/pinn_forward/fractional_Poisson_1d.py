import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import pinnx

alpha = 1.5


# Backend tensorflow.compat.v1
def fpde(net, x, int_mat):
    """(D_{0+}^alpha + D_{1-}^alpha) u(x) = f(x)"""
    y = net(x)

    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        ini_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = ini_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    rhs = (
        gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
        - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
        + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
        - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    )
    # lhs /= 2 * np.cos(alpha * np.pi / 2)
    # rhs = gamma(alpha + 2) * x
    return lhs - rhs[: len(lhs)]


def func(x):
    # return x * (np.abs(1 - x**2)) ** (alpha / 2)
    return x ** 3 * (1 - x) ** 3


geom = pinnx.geometry.Interval(0, 1)
bc = pinnx.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

data_type = 'static'

if data_type == 'static':
    # Static auxiliary points
    data = pinnx.data.FPDE(
        geom,
        fpde,
        alpha,
        bc,
        [101],
        meshtype="static",
        solution=func
    )

else:

    # Dynamic auxiliary points
    data = pinnx.data.FPDE(
        geom,
        fpde,
        alpha,
        bc,
        [100],
        meshtype="dynamic",
        num_domain=20,
        num_boundary=2,
        solution=func,
        num_test=100
    )

net = pinnx.nn.FNN([1] + [20] * 4 + [1], "tanh", bst.init.KaimingUniform())
net.apply_output_transform(lambda x, y: x * (1 - x) * y)

model = pinnx.Model(data, net)

model.compile(bst.optim.Adam(1e-3))
losshistory, train_state = model.train(iterations=10000)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
