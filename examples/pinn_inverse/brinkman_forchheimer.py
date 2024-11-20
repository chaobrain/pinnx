"""
Implementation of Brinkman-Forchheimer equation example in paper https://arxiv.org/pdf/2111.02801.pdf.
"""

import re

import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import pinnx

g = 1
v = 1e-3
e = 0.4
H = 1

v_e = bst.ParamState(0.1)
K = bst.ParamState(0.1)


def sol(x):
    r = (v * e / (1e-3 * 1e-3)) ** 0.5
    return g * 1e-3 / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
    yvals = sol(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def pde(neu, x):
    du_xx, y = pinnx.grad.hessian(neu, x, return_value=True)
    du_xx = u.math.squeeze(du_xx)
    y = u.math.squeeze(y)
    return -v_e.value / e * du_xx + v * y / K.value - g


def output_transform(x, y):
    return x * (1 - x) * y


geom = pinnx.geometry.Interval(0, 1)
ob_x, ob_u = gen_traindata(5)
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u, component=0)

data = pinnx.data.PDE(
    geom,
    pde,
    solution=sol,
    ic_bcs=[observe_u],
    num_domain=100,
    num_boundary=0,
    train_distribution="uniform",
    num_test=500,
)

net = pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh")
net.apply_output_transform(output_transform)

model = pinnx.Model(data, net, external_trainable_variables=[v_e, K])
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])

variable = pinnx.callbacks.VariableValue([v_e, K], period=200, filename="variables1.dat")
losshistory, train_state = model.train(iterations=30000, callbacks=[variable])
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

lines = open("variables1.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = vkinfer.shape
v_etrue = 1e-3
ktrue = 1e-3

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(vkinfer[:, 0].shape) * v_etrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), vkinfer[:, 0], "b--", label="Pred")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(top=1e-1)
plt.legend(frameon=False)
plt.ylabel(r"$\nu_e$")

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(vkinfer[:, 1].shape) * ktrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), vkinfer[:, 1], "b--", label="Pred")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(ymax=1e-1)
plt.legend(frameon=False)
plt.ylabel(r"$K$")

plt.show()
