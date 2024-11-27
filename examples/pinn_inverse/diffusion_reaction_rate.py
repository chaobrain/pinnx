"""
Implementation for the diffusion-reaction system with a space-dependent reaction rate in paper https://arxiv.org/abs/2111.02801.
"""

import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

import pinnx

l = 0.01


def k(x):
    return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)


def fun(x, y):
    return np.vstack((y[1], (k(x) * y[0] + np.sin(2 * np.pi * x)) / l))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])


num = 100
xvals = np.linspace(0, 1, num)
y = np.zeros((2, xvals.size))
res = solve_bvp(fun, bc, xvals, y)


def gen_traindata(num):
    return np.reshape(xvals, (-1, 1)), np.reshape(res.sol(xvals)[0], (-1, 1))


geom = pinnx.geometry.Interval(0, 1)


def pde(neu, x):
    approx = lambda x: pinnx.array_to_dict(neu(x), ['u', 'k'])
    hessian, y = pinnx.grad.hessian(approx, x, return_value=True)
    u_, k = y['u'], y['k']
    du_xx = u.math.squeeze(hessian['u'])
    k = u.math.squeeze(k)
    x = u.math.squeeze(x)
    return l * du_xx - u_ * k - u.math.sin(2 * np.pi * x)


def func(x):
    return 0


ob_x, ob_u = gen_traindata(num)
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u, component=0)
bc = pinnx.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary, component=0)
data = pinnx.data.PDE(
    geom,
    pde,
    ic_bcs=[bc, observe_u],
    num_domain=50,
    num_boundary=8,
    train_distribution="uniform",
    num_test=1000,
)

net = pinnx.nn.PFNN([1, [20, 20], [20, 20], 2], "tanh", bst.init.KaimingUniform())
model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(1e-3))

losshistory, train_state = model.train(iterations=20000)

x = geom.uniform_points(500)
yhat = model.predict(x)
uhat, khat = yhat[:, 0:1], yhat[:, 1:2]

ktrue = k(x)
print("l2 relative error for k: " + str(pinnx.metrics.l2_relative_error(khat, ktrue)))

plt.figure()
plt.plot(x, ktrue, "-", label="k_true")
plt.plot(x, khat, "--", label="k_NN")
plt.legend()
plt.show()

utrue = res.sol(x)[0]
print("l2 relative error for u: " + str(pinnx.metrics.l2_relative_error(uhat, utrue)))

plt.figure()
plt.plot(x, utrue, "-", label="u_true")
plt.plot(x, uhat, "--", label="u_NN")
plt.legend()
plt.show()
