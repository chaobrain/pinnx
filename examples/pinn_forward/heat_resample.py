import brainstate as bst
import numpy as np
import optax

import pinnx as pinnx


def heat_eq_exact_solution(x, t):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    return np.exp(-(n ** 2 * np.pi ** 2 * a * t) / (L ** 2)) * np.sin(n * np.pi * x / L)


def gen_exact_solution():
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:
    x_dim, t_dim = (256, 201)

    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0.0)
    x_max, t_max = (L, 1.0)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # Obtain the value of the exact solution for each generated point:
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = heat_eq_exact_solution(x[i], t[j])

    # Save solution:
    np.savez("heat_eq_data", x=x, t=t, usol=usol)
    # Load solution:
    data = np.load("heat_eq_data.npz")


def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Load the data:
    data = np.load("heat_eq_data.npz")
    # Obtain the values for t, x, and the excat solution:
    t, x, exact = data["t"], data["x"], data["usol"].T
    # Process the data and flatten it out (like labels and features):
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


# Problem parameters:
a = 0.4  # Thermal diffusivity
L = 1  # Length of the bar
n = 1  # Frequency of the sinusoidal initial conditions

# Generate a dataset with the exact solution (if you dont have one):
gen_exact_solution()


def pde(net, x):
    """
    Expresses the PDE residual of the heat equation.
    """
    x = pinnx.array_to_dict(x, ['x', 't'])
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), ['y'])
    jacobian = pinnx.grad.jacobian(approx, x)
    hessian = pinnx.grad.hessian(approx, x)
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t - a * dy_xx


# Computational geometry:
geom = pinnx.geometry.Interval(0, L)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = pinnx.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = pinnx.icbc.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)
pde_resampler = pinnx.callbacks.PDEPointResampler(period=10)

# Define the PDE problem and configurations of the network:
data = pinnx.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)
net = pinnx.nn.FNN([2] + [20] * 3 + [1], "tanh", bst.init.KaimingUniform())
model = pinnx.Trainer(data, net)

# Build and train the trainer:
model.compile(bst.optim.Adam(1e-3))
model.train(iterations=200000, callbacks=[pde_resampler])
model.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))
losshistory, train_state = model.train(callbacks=[pde_resampler])

# Plot/print the results
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
