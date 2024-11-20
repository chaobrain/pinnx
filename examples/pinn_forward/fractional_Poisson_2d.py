import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import pinnx

alpha = 1.8


# Backend tensorflow.compat.v1
def fpde(net, x, int_mat):
    """
    \int_theta D_theta^alpha u(x)
    """
    y = net(x)
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        ini_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = ini_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    lhs = lhs[:, 0]
    lhs *= gamma((1 - alpha) / 2) * gamma((2 + alpha) / 2) / (2 * np.pi ** 1.5)
    x = x[: len(lhs)]
    rhs = (
        2 ** alpha
        * gamma(2 + alpha / 2)
        * gamma(1 + alpha / 2)
        * (1 - (1 + alpha / 2) * u.math.sum(x ** 2, axis=1))
    )
    return lhs - rhs


def func(x):
    return (np.abs(1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (
        1 + alpha / 2
    )


geom = pinnx.geometry.Disk([0, 0], 1)
bc = pinnx.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

data = pinnx.data.FPDE(
    geom, fpde, alpha, bc, [8, 100], num_domain=100, num_boundary=1, solution=func
)

net = pinnx.nn.FNN([2] + [20] * 4 + [1], "tanh", bst.init.KaimingUniform())
net.apply_output_transform(
    lambda x, y: (1 - u.math.sum(x ** 2, axis=1, keepdims=True)) * y
)

model = pinnx.Model(data, net)
model.compile(bst.optim.Adam(1e-3))
losshistory, train_state = model.train(iterations=20000)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

X = geom.random_points(1000)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
