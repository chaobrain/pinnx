import brainstate as bst
import brainunit as u
import numpy as np

import pinnx

# General parameters
n = 2
precision_train = 10
precision_test = 30
hard_constraint = True
weights = 100  # if hard_constraint == False
iterations = 5000
parameters = [1e-3, 3, 150, "sin"]

learning_rate, num_dense_layers, num_dense_nodes, activation = parameters


def pde(net, x):
    x = pinnx.array_to_dict(x, ["x", "y"])
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), ["y"])
    hessian, y = pinnx.grad.hessian(approx, x, return_value=True)

    dy_xx = hessian["y"]["x"]["x"]
    dy_yy = hessian["y"]["y"]["y"]

    f = k0 ** 2 * u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])
    return -dy_xx - dy_yy - k0 ** 2 * y['y'] - f


def func(x):
    x = pinnx.array_to_dict(x, ["x", "y"])
    return np.sin(k0 * x['x']) * np.sin(k0 * x['y'])


def transform(x, y):
    x = pinnx.array_to_dict(x, ["x", "y"])
    res = x['x'] * (1 - x['x']) * x['y'] * (1 - x['y'])
    return res * y


def boundary(_, on_boundary):
    return on_boundary


geom = pinnx.geometry.Rectangle([0, 0], [1, 1])
k0 = 2 * np.pi * n
wave_len = 1 / n

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

if hard_constraint:
    bc = []
else:
    bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = pinnx.data.PDE(
    geom,
    pde,
    bc,
    num_domain=nx_train ** 2,
    num_boundary=4 * nx_train,
    solution=func,
    num_test=nx_test ** 2,
)

net = pinnx.nn.FNN([2] + [num_dense_nodes] * num_dense_layers + [1],
                   u.math.sin,
                   bst.init.KaimingUniform())

if hard_constraint:
    net.apply_output_transform(transform)

model = pinnx.Trainer(data, net)

if hard_constraint:
    model.compile(bst.optim.Adam(learning_rate), metrics=["l2 relative error"])
else:
    loss_weights = [1, weights]
    model.compile(
        bst.optim.Adam(learning_rate),
        metrics=["l2 relative error"],
        loss_weights=loss_weights,
    )

losshistory, train_state = model.train(iterations=iterations)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
