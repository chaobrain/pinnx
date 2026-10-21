import braintools
import saiunit as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import pinnx

# General parameters
d = 2
n = 2
k0 = 2 * np.pi * n
precision_train = 10
precision_test = 30
iterations = 10000


def func(x):
    return {'y': u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])}


def transform(x, y):
    x = pinnx.utils.array_to_dict(x, ["x", "y"], keep_dim=True)
    res = x['x'] * (1 - x['x']) * x['y'] * (1 - x['y'])
    return res * y


def create_model(config):
    def pde(x, y):
        hessian = net.hessian(x)
        dy_xx = hessian['y']['x']['x']
        dy_yy = hessian['y']['y']['y']
        f = (d - 1) * k0 ** 2 * u.math.sin(k0 * x['x']) * u.math.sin(k0 * x['y'])
        return -dy_xx - dy_yy - k0 ** 2 * y['y'] - f

    learning_rate, num_dense_layers, num_dense_nodes, activation = config

    geom = pinnx.geometry.Rectangle([0, 0], [1, 1]).to_dict_point('x', 'y')
    k0 = 2 * np.pi * n
    wave_len = 1 / n

    hx_train = wave_len / precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / precision_test
    nx_test = int(1 / hx_test)

    net = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=None, y=None),
        pinnx.nn.FNN(
            [d] + [num_dense_nodes] * num_dense_layers + [1],
            activation,
            braintools.init.KaimingUniform(),
            output_transform=transform,
        ),
        pinnx.nn.ArrayToDict(y=None),
    )

    problem = pinnx.problem.PDE(
        geom,
        pde,
        [],
        net,
        num_domain=nx_train ** d,
        num_boundary=2 * d * nx_train,
        solution=func,
        num_test=nx_test ** d,
    )

    trainer = pinnx.Trainer(problem)
    trainer.compile(braintools.optim.Adam(learning_rate), metrics=["l2 relative error"])
    return trainer


def train_model(model, config):
    model.train(iterations=iterations)
    loss_test = np.asarray(model.loss_history.loss_test)
    test = loss_test.sum(axis=1).ravel()
    error = test.min()
    return error


# HPO setting: a quasi-random (Sobol) search over the hyper-parameter space
learning_rate_range = (1e-4, 5e-2)  # sampled log-uniformly
num_dense_layers_range = (1, 10)
num_dense_nodes_range = (5, 500)
activations = ["sin", "sigmoid", "tanh"]

default_parameters = [1e-3, 4, 50, "sin"]


def sample_configs(m, seed=1234):
    """Draws ``2 ** m`` hyper-parameter configurations from a scrambled Sobol sequence."""
    unit = qmc.Sobol(d=4, seed=seed).random_base2(m)
    log_lr = np.log10(learning_rate_range)
    learning_rates = 10 ** (log_lr[0] + unit[:, 0] * (log_lr[1] - log_lr[0]))
    num_layers = num_dense_layers_range[0] + np.floor(
        unit[:, 1] * (num_dense_layers_range[1] - num_dense_layers_range[0] + 1)).astype(int)
    num_nodes = num_dense_nodes_range[0] + np.floor(
        unit[:, 2] * (num_dense_nodes_range[1] - num_dense_nodes_range[0] + 1)).astype(int)
    acts = np.floor(unit[:, 3] * len(activations)).astype(int)
    return [
        [float(lr), int(nl), int(nn), activations[a]]
        for lr, nl, nn, a in zip(learning_rates, num_layers, num_nodes, acts)
    ]


def fitness(config):
    learning_rate, num_dense_layers, num_dense_nodes, activation = config
    global ITERATION

    print(ITERATION, "it number")
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print()

    # Create the neural network with these hyper-parameters.
    model = create_model(config)
    # possibility to change where we save
    error = train_model(model, config)

    if np.isnan(error):
        error = 10 ** 5

    ITERATION += 1
    return error


ITERATION = 0

configs = [default_parameters] + sample_configs(m=4)
errors = [fitness(config) for config in configs]

best = int(np.argmin(errors))
print("Best hyper-parameters:", configs[best], "test loss:", errors[best])

plt.plot(np.arange(1, len(errors) + 1), np.minimum.accumulate(errors), "o-")
plt.xlabel("Number of calls")
plt.ylabel("Minimum test loss")
plt.yscale("log")
plt.title("Convergence plot")
plt.show()
