import brainstate as bst
import brainunit as u
import numpy as np

import pinnx


def pde(net, x):
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), "y")
    x = pinnx.array_to_dict(x, 'x', 't')
    jacobian = pinnx.grad.jacobian(approx, x)
    hessian = pinnx.grad.hessian(approx, x)
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    d = 1
    return (
        dy_t
        - d * dy_xx
        - u.math.exp(-x['t'])
        * (
            3 * u.math.sin(2 * x['x']) / 2
            + 8 * u.math.sin(3 * x['x']) / 3
            + 15 * u.math.sin(4 * x['x']) / 4
            + 63 * u.math.sin(8 * x['x']) / 8
        )
    )


def func(x):
    x = pinnx.array_to_dict(x, 'x', 't')
    return np.exp(-x['t']) * (
        np.sin(x['x'])
        + np.sin(2 * x['x']) / 2
        + np.sin(3 * x['x']) / 3
        + np.sin(4 * x['x']) / 4
        + np.sin(8 * x['x']) / 8
    )


def output_transform(x, y):
    x = pinnx.array_to_dict(x, 'x', 't')
    return (
        x['t'] * (np.pi ** 2 - x['x'] ** 2) * y
        + u.math.sin(x['x'])
        + u.math.sin(2 * x['x']) / 2
        + u.math.sin(3 * x['x']) / 3
        + u.math.sin(4 * x['x']) / 4
        + u.math.sin(8 * x['x']) / 8
    )


geom = pinnx.geometry.Interval(-np.pi, np.pi)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

data = pinnx.data.TimePDE(
    geomtime, pde, [], num_domain=320, solution=func, num_test=80000
)

layer_size = [2] + [30] * 6 + [1]
net = pinnx.nn.FNN(
    layer_size, "tanh", bst.init.KaimingUniform(),
    output_transform=output_transform
)

model = pinnx.Model(data, net)
model.compile(bst.optim.Adam(1e-3), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
