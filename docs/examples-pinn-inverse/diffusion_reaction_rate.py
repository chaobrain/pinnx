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


def gen_traindata():
    x = {'x': xvals}
    y = {'u': res.sol(xvals)[0]}
    return x, y


geom = pinnx.geometry.Interval(0, 1).to_dict_point('x')


def pde(x, y):
    hessian = net.hessian(x, y='u')
    du_xx = hessian["u"]["x"]["x"]
    return l * du_xx - y['u'] * y['k'] - u.math.sin(2 * np.pi * x['x'])


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.PFNN([1, [20, 20], [20, 20], 2], "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(u=None, k=None),
)

ob_x, ob_u = gen_traindata()
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u)
bc = pinnx.icbc.DirichletBC(lambda x: {'u': 0})

problem = pinnx.problem.PDE(
    geom,
    pde,
    constraints=[bc, observe_u],
    approximator=net,
    num_domain=50,
    num_boundary=8,
    train_distribution="uniform",
    num_test=1000,
)

model = pinnx.Trainer(problem)
model.compile(bst.optim.Adam(1e-3)).train(iterations=20000)

x = geom.uniform_points(500)
yhat = model.predict(x)
uhat, khat = yhat['u'], yhat['k']
x = x['x']

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
