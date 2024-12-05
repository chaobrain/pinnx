import brainstate as bst
import brainunit as u
import numpy as np

import pinnx

# General parameters
n = 2
precision_train = 10
precision_test = 30
hard_constraint = True  # True or False
weights = 100  # if hard_constraint == False
iterations = 5000
parameters = [1e-3, 3, 150]

learning_rate, num_dense_layers, num_dense_nodes = parameters


def pde(x, y):
    hessian = net.hessian(x)

    dy_xx = hessian["y"]["x"]["x"]
    dy_yy = hessian["y"]["y"]["y"]

    f = k0 ** 2 * u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])
    return -dy_xx - dy_yy - k0 ** 2 * y['y'] - f


geom = pinnx.geometry.Rectangle([0, 0], [1, 1]).to_dict_point('x', 'y')
k0 = 2 * np.pi * n
wave_len = 1 / n

hx_train = wave_len / precision_train
nx_train = int(1 / hx_train)

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

if hard_constraint:
    bc = []
else:
    bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0})

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None, y=None),
    pinnx.nn.FNN([2] + [num_dense_nodes] * num_dense_layers + [1],
                 u.math.sin,
                 bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(y=None),
)

if hard_constraint:
    def transform(x, y):
        x = pinnx.utils.array_to_dict(x, ["x", "y"], keep_dim=True)
        res = x['x'] * (1 - x['x']) * x['y'] * (1 - x['y'])
        return res * y


    net.approx.apply_output_transform(transform)

problem = pinnx.problem.PDE(
    geom,
    pde,
    bc,
    net,
    num_domain=nx_train ** 2,
    num_boundary=4 * nx_train,
    solution=lambda x: {'y': u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])},
    num_test=nx_test ** 2,
    loss_weights=None if hard_constraint else [1, weights],
)

trainer = pinnx.Trainer(problem)
trainer.compile(bst.optim.Adam(learning_rate), metrics=["l2 relative error"]).train(iterations=iterations)
trainer.saveplot(issave=True, isplot=True)
