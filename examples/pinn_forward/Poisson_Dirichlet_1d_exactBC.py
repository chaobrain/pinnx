import brainstate as bst
import brainunit as u
import numpy as np

import pinnx

geom = pinnx.geometry.Interval(0, np.pi)


def pde(nwu, x):
    x = pinnx.array_to_dict(x, "x")
    approx = lambda x: pinnx.array_to_dict(nwu(pinnx.dict_to_array(x)), "y")
    hessian = pinnx.grad.hessian(approx, x)
    x = x["x"]
    dy_xx = hessian["y"]["x"]["x"]
    summation = sum([i * u.math.sin(i * x) for i in range(1, 5)])
    return -dy_xx - summation - 8 * u.math.sin(8 * x)


def func(x):
    summation = sum([np.sin(i * x) / i for i in range(1, 5)])
    return x + summation + np.sin(8 * x) / 8


data = pinnx.data.PDE(geom, pde, [], num_domain=64, solution=func, num_test=400)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, 'tanh')


def output_transform(x, y):
    return x * (np.pi - x) * y + x


net.apply_output_transform(output_transform)

model = pinnx.Model(data, net)
model.compile(
    bst.optim.Adam(bst.optim.InverseTimeDecayLR(0.001, 1000, 0.3)),
    metrics=["l2 relative error"]
)

losshistory, train_state = model.train(iterations=30000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
