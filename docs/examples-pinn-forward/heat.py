import brainstate as bst
import numpy as np
import optax
import brainunit as u

import pinnx

@bst.compile.jit
def heat_eq_exact_solution(x, t):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    return u.math.exp(-(n ** 2 * u.math.pi ** 2 * a * t) / (L ** 2), unit_to_scale=u.becquerel2) * u.math.sin(n * u.math.pi * x / L, unit_to_scale=u.becquerel)  * u.kelvin

@bst.compile.jit
def gen_exact_solution():
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:
    x_dim, t_dim = (256, 201)

    # Bounds of 'x' and 't':
    x_min, t_min = (0 * u.meter, 0.0 * u.second)
    x_max, t_max = (L, 1.0 * u.second)

    # Create tensors:
    t = u.math.linspace(t_min, t_max, num=t_dim).reshape((t_dim, 1))
    x = u.math.linspace(x_min, x_max, num=x_dim).reshape((x_dim, 1))
    usol = u.math.zeros((x_dim, t_dim), unit=u.kelvin).reshape((x_dim, t_dim))

    # Obtain the value of the exact solution for each generated point:
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i, j] = u.math.squeeze(heat_eq_exact_solution(x[i], t[j]))

    return t, x, usol


def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Load the data, Obtain the values for t, x, and the excat solution:
    t, x, exact = gen_exact_solution()
    # Process the data and flatten it out (like labels and features):
    xx, tt = u.math.meshgrid(x, t)
    X = u.math.vstack((u.math.ravel(xx), u.math.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


# Problem parameters:
a = 0.4 * u.meter2 / u.second # Thermal diffusivity
L = 1 * u.meter  # Length of the bar
n = 1 * u.Hz # Frequency of the sinusoidal initial conditions


# Computational geometry:
geom = pinnx.geometry.Interval('x', 0 * u.meter, L)
timedomain = pinnx.geometry.TimeDomain('t', 0 * u.second, 1 * u.second)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

uy = u.kelvin / u.second
# Initial and boundary conditions:
bc = pinnx.icbc.DirichletBC(
    lambda x : {'y': 0. * uy}
)
ic = pinnx.icbc.IC(
    lambda x: {'y': u.math.sin(n * u.math.pi * x['x'][:] / L, unit_to_scale=u.becquerel) * uy},
)


@bst.compile.jit
def pde(x, y):
    """
    Expresses the PDE residual of the heat equation.
    """
    jacobian = approximator.jacobian(x)
    hessian = approximator.hessian(x)
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t - a * dy_xx

approximator = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=u.meter, t=u.second),
    pinnx.nn.FNN(
        [2] + [20] * 3 + [1],
        "tanh",
        bst.init.KaimingUniform()
    ),
    pinnx.nn.ArrayToDict(y=uy)
)

# Define the PDE problem and configurations of the network:
problem = pinnx.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    approximator,
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)

trainer = pinnx.Trainer(problem)

# Build and train the trainer:
trainer.compile(bst.optim.Adam(1e-3))
trainer.train(iterations=10000)
trainer.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))

# Plot/print the results
trainer.saveplot(issave=True, isplot=True)

# X, y_true = gen_testdata()
# y_pred = trainer.predict(X)
# f = trainer.predict(X, operator=pde)
# print("Mean residual:", u.math.mean(np.absolute(f)))
# print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
# np.savetxt("test.dat", u.math.hstack((X, y_true, y_pred)))
