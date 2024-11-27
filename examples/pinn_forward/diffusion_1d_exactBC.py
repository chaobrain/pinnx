import brainstate as bst
import brainunit as u
import numpy as np

import pinnx


def pde(net, x):
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), ["y"])

    x = pinnx.array_to_dict(x, ["x", "t"])
    jac, y = pinnx.grad.jacobian(approx, x, return_value=True)
    hess = pinnx.grad.hessian(approx, x)

    dy_t = jac["y"]["t"]
    dy_xx = hess["y"]["x"]["x"]
    return (
        dy_t
        - dy_xx
        + u.math.exp(-x['t'])
        * (u.math.sin(np.pi * x['x']) -
           np.pi ** 2 * u.math.sin(np.pi * x['x']))
    )


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

data = pinnx.data.TimePDE(geomtime, pde, [], num_domain=40, solution=func, num_test=10000)

layer_size = [2] + [32] * 3 + [1]
net = pinnx.nn.FNN(
    layer_size,
    "tanh",
    bst.init.KaimingUniform(),
    output_transform=lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + u.math.sin(np.pi * x[:, 0:1])
)

model = pinnx.Trainer(data, net)

model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
