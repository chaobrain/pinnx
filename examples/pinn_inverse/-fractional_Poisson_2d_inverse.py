import brainstate as bst
import brainunit as u
import jax
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import pinnx

alpha0 = 1.8
alpha = bst.ParamState(1.5)


def fpde(net, x, int_mat):
    r"""
    \int_theta D_theta^alpha u(x)
    """
    y = net(x)

    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        int_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = int_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    lhs = lhs[:, 0]
    lhs *= (-u.math.exp(jax.lax.lgamma((1 - alpha.value) / 2) +
                        jax.lax.lgamma((2 + alpha.value) / 2)) / (2 * np.pi ** 1.5))
    x = x[: len(lhs)]
    rhs = (
        2 ** alpha0
        * gamma(2 + alpha0 / 2)
        * gamma(1 + alpha0 / 2)
        * (1 - (1 + alpha0 / 2) * u.math.sum(x ** 2, axis=1))
    )
    return lhs - rhs


def func(x):
    return (1 - np.linalg.norm(x, axis=1, keepdims=True) ** 2) ** (1 + alpha0 / 2)


geom = pinnx.geometry.Disk([0, 0], 1)
observe_x = geom.random_points(30)
observe_y = pinnx.icbc.PointSetBC(observe_x, func(observe_x))

data = pinnx.data.FPDE(
    geom,
    fpde,
    alpha,
    observe_y,
    [8, 100],
    num_domain=64,
    anchors=observe_x,
    solution=func,
)

net = pinnx.nn.FNN([2] + [20] * 4 + [1], "tanh")
net.apply_output_transform(lambda x, y: (1 - u.math.sum(x ** 2, axis=-1, keepdims=True)) * y)

model = pinnx.Trainer(data, net, external_trainable_variables=[alpha])
model.compile(bst.optim.Adam(1e-3), loss_weights=[1, 100], )
variable = pinnx.callbacks.VariableValue(alpha, period=1000)
losshistory, train_state = model.train(iterations=10000, callbacks=[variable])
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
