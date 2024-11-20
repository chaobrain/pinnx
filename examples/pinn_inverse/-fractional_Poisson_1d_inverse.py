import brainstate as bst
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma

import pinnx

alpha0 = 1.8
alpha = bst.ParamState(1.5)


def fpde(neu, x, int_mat):
    """
    (D_{0+}^alpha + D_{1-}^alpha) u(x)
    """
    y = neu(x)
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        int_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = int_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
    lhs /= 2 * u.math.cos(alpha.value * np.pi / 2)
    rhs = gamma(alpha0 + 2) * x
    return lhs - rhs[: len(lhs)]


def func(x):
    return x * (np.abs(1 - x ** 2)) ** (alpha0 / 2)


geom = pinnx.geometry.Interval(-1, 1)

observe_x = np.linspace(-1, 1, num=20)[:, None]
observe_y = pinnx.icbc.PointSetBC(observe_x, func(observe_x))

data_type = 'static'

# Static auxiliary points

if data_type == 'static':
    data = pinnx.data.FPDE(
        geom,
        fpde,
        alpha,
        observe_y,
        [101],
        meshtype="static",
        anchors=observe_x,
        solution=func,
    )
else:
    # Dynamic auxiliary points
    data = pinnx.data.FPDE(
        geom,
        fpde,
        alpha,
        observe_y,
        [100],
        meshtype="dynamic",
        num_domain=20,
        anchors=observe_x,
        solution=func,
        num_test=100,
    )

net = pinnx.nn.FNN([1] + [20] * 4 + [1], "tanh")
net.apply_output_transform(lambda x, y: (1 - x ** 2) * y)

model = pinnx.Model(data, net)

model.compile(
    bst.optim.Adam(1e-3),
    loss_weights=[1, 100],
    external_trainable_variables=[alpha]
)
variable = pinnx.callbacks.VariableValue(alpha, period=1000)
losshistory, train_state = model.train(iterations=10000, callbacks=[variable])
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
