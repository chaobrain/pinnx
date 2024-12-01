import brainstate as bst
import brainunit as u
import numpy as np

import pinnx

geom = pinnx.geometry.Interval('x', -1 * u.meter, 1. * u.meter)
timedomain = pinnx.geometry.TimeDomain('t', 0 * u.second, 1 * u.second)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)


bc = pinnx.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = pinnx.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# TODO: The units on both sides of the equation are not equal
def pde(net, x):
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), ["y"])
    x = pinnx.array_to_dict(x, ["x", "t"])
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    dy_t = jacobian["y"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    return (
        dy_t
        - dy_xx
        + u.math.exp(-x['t'])
        * (u.math.sin(u.math.pi * x['x']) -
           u.math.pi ** 2 * u.math.sin(u.math.pi * x['x']))
    )


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])




data = pinnx.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
net = pinnx.nn.FNN(layer_size, 'tanh', bst.init.KaimingUniform())

model = pinnx.Trainer(data, net)

model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
