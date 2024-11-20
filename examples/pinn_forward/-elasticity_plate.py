"""

Implementation of the linear elasticity 2D example in paper https://doi.org/10.1016/j.cma.2021.113741.
References:
    https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb.
"""
import brainstate as bst
import brainunit as u
import numpy as np

import pinnx

lmbd = 1.0
mu = 0.5
Q = 4.0

geom = pinnx.geometry.Rectangle([0, 0], [1, 1])
BC_type = ["hard", "soft"][0]


def boundary_left(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 1.0)


def boundary_top(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[1], 0.0)


# Exact solutions
def func(x):
    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
    E_xy = 0.5 * (
        np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
        + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
    )

    Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    Sxy = 2 * E_xy * mu

    return np.hstack((ux, uy, Sxx, Syy, Sxy))


# Soft Boundary Conditions
ux_top_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=0)
ux_bottom_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=0)
uy_left_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=1)
uy_bottom_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
uy_right_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=1)
sxx_left_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=2)
sxx_right_bc = pinnx.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=2)
syy_top_bc = pinnx.icbc.DirichletBC(
    geom,
    lambda x: (2 * mu + lmbd) * Q * np.sin(np.pi * x[:, 0:1]),
    boundary_top,
    component=3,
)


# Hard Boundary Conditions
def hard_BC(x, f):
    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]

    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * u.math.sin(np.pi * x[:, 0])
    Sxy = f[:, 4]
    return u.math.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


def fx(x):
    return (
        -lmbd * (
        4 * np.pi ** 2 * u.math.cos(2 * np.pi * x[:, 0:1]) * u.math.sin(np.pi * x[:, 1:2])
        - Q * x[:, 1:2] ** 3 * np.pi * u.math.cos(np.pi * x[:, 0:1])
    )
        -
        mu * (
            np.pi ** 2 * u.math.cos(2 * np.pi * x[:, 0:1]) * u.math.sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * u.math.cos(np.pi * x[:, 0:1])
        )
        - 8 * mu * np.pi ** 2 * u.math.cos(2 * np.pi * x[:, 0:1]) * u.math.sin(np.pi * x[:, 1:2])
    )


def fy(x):
    return (
        lmbd
        * (
            3 * Q * x[:, 1:2] ** 2 * u.math.sin(np.pi * x[:, 0:1])
            - 2 * np.pi ** 2 * u.math.cos(np.pi * x[:, 1:2]) * u.math.sin(2 * np.pi * x[:, 0:1])
        )
        - mu
        * (
            2 * np.pi ** 2 * u.math.cos(np.pi * x[:, 1:2]) * u.math.sin(2 * np.pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * np.pi ** 2 * u.math.sin(np.pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu * x[:, 1:2] ** 2 * u.math.sin(np.pi * x[:, 0:1])
    )


def jacobian(f, x, i, j):
    return pinnx.grad.jacobian(f, x, i=i, j=j)


def pde(net, x):
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), "y")
    x = pinnx.array_to_dict(x, 'x', 'y')

    E_xx = jacobian(f, x, i=0, j=0)
    E_yy = jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian(f, x, i=2, j=0)
    Syy_y = jacobian(f, x, i=3, j=1)
    Sxy_x = jacobian(f, x, i=4, j=0)
    Sxy_y = jacobian(f, x, i=4, j=1)

    momentum_x = Sxx_x + Sxy_y - fx(x)
    momentum_y = Sxy_x + Syy_y - fy(x)

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


if BC_type == "hard":
    bcs = []
else:
    bcs = [
        ux_top_bc,
        ux_bottom_bc,
        uy_left_bc,
        uy_bottom_bc,
        uy_right_bc,
        sxx_left_bc,
        sxx_right_bc,
        syy_top_bc,
    ]

data = pinnx.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=500,
    num_boundary=500,
    solution=func,
    num_test=100,
)

layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
net = pinnx.nn.PFNN(layers, "tanh", bst.init.KaimingUniform())
if BC_type == "hard":
    net.apply_output_transform(hard_BC)

model = pinnx.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=5000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
