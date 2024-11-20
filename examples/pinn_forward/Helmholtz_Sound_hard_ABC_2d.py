import brainstate as bst
import numpy as np
from scipy.special import jv, hankel1

import pinnx

# General parameters
weights = 1
iterations = 10000
learning_rate = 1e-3
num_dense_layers = 3
num_dense_nodes = 350
activation = "tanh"

# Problem parameters
k0 = 2
wave_len = 2 * np.pi / k0
length = 2 * np.pi
R = np.pi / 4
n_wave = 20
h_elem = wave_len / n_wave
nx = int(length / h_elem)

# Computational domain
outer = pinnx.geometry.Rectangle([-length / 2, -length / 2], [length / 2, length / 2])
inner = pinnx.geometry.Disk([0, 0], R)

geom = outer - inner


# Exact solution
def sound_hard_circle_deepxde(k0, a, points):
    fem_xx = points[:, 0:1]
    fem_xy = points[:, 1:2]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    n_terms = int(30 + (k0 * a) ** 1.01)

    u_sc = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n - 1, k0 * a) - n / (k0 * a) * jv(n, k0 * a)
        hankel_deriv = n / (k0 * a) * hankel1(n, k0 * a) - hankel1(n + 1, k0 * a)
        u_sc += (
            -((1j) ** (n))
            * (bessel_deriv / hankel_deriv)
            * hankel1(n, k0 * r)
            * np.exp(1j * n * theta)
        ).ravel()
    return u_sc


# Definition of the pde
def pde(net, x):
    x = pinnx.array_to_dict(x, "x", 'y')
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), 'y0', 'y1')

    hessian, y = pinnx.grad.hessian(approx, x, return_value=True)

    y0, y1 = y['y0'], y['y1']
    y0_xx = hessian['y0']['x']['x']
    y0_yy = hessian['y0']['y']['y']
    y1_xx = hessian['y1']['x']['x']
    y1_yy = hessian['y1']['y']['y']

    return [-y0_xx - y0_yy - k0 ** 2 * y0, -y1_xx - y1_yy - k0 ** 2 * y1]


def sol(x):
    result = sound_hard_circle_deepxde(k0, R, x).reshape((x.shape[0], 1))
    real = np.real(result)
    imag = np.imag(result)
    return np.hstack((real, imag))


# Boundary conditions
def boundary(x, on_boundary):
    return on_boundary


def boundary_outer(x, on_boundary):
    return on_boundary and outer.on_boundary(x)


def boundary_inner(x, on_boundary):
    return on_boundary and inner.on_boundary(x)


def func0_inner(x):
    normal = -inner.boundary_normal(x)
    g = 1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * normal[:, 0:1]
    return np.real(-g)


def func1_inner(x):
    normal = -inner.boundary_normal(x)
    g = 1j * k0 * np.exp(1j * k0 * x[:, 0:1]) * normal[:, 0:1]
    return np.imag(-g)


def func0_outer(x, y):
    result = -k0 * y[:, 1:2]
    return result


def func1_outer(x, y):
    result = k0 * y[:, 0:1]
    return result


# ABCs
bc0_inner = pinnx.icbc.NeumannBC(geom, func0_inner, boundary_inner, component=0)
bc1_inner = pinnx.icbc.NeumannBC(geom, func1_inner, boundary_inner, component=1)

bc0_outer = pinnx.icbc.RobinBC(geom, func0_outer, boundary_outer, component=0)
bc1_outer = pinnx.icbc.RobinBC(geom, func1_outer, boundary_outer, component=1)

bcs = [bc0_inner, bc1_inner, bc0_outer, bc1_outer]

loss_weights = [1, 1, weights, weights, weights, weights]

data = pinnx.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=nx ** 2,
    num_boundary=8 * nx,
    num_test=5 * nx ** 2,
    solution=sol,
)
net = pinnx.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [2], activation,
)
model = pinnx.Model(data, net)

model.compile(bst.optim.Adam(learning_rate), loss_weights=loss_weights, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=iterations)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
