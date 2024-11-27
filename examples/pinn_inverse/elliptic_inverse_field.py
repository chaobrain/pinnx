import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import pinnx


def gen_traindata(num):
    # generate num equally-spaced points from -1 to 1
    xvals = np.linspace(-1, 1, num).reshape(num, 1)
    uvals = np.sin(np.pi * xvals)
    return xvals, uvals


def pde(neu, x):
    approx = lambda x: pinnx.array_to_dict(neu(x), ["u", "q"])
    hessian, y = pinnx.grad.hessian(approx, x, return_value=True)
    u_, q = y['u'], y['q']
    q = u.math.squeeze(q)
    du_xx = u.math.squeeze(hessian['u'])
    du_xx = u.math.squeeze(du_xx)
    return -du_xx + q


def sol(x):
    # solution is u(x) = sin(pi*x), q(x) = -pi^2 * sin(pi*x)
    return np.sin(np.pi * x)


geom = pinnx.geometry.Interval(-1, 1)

bc = pinnx.icbc.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)
ob_x, ob_u = gen_traindata(100)
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u, component=0)

data = pinnx.data.PDE(
    geom,
    pde,
    [bc, observe_u],
    num_domain=200,
    num_boundary=2,
    anchors=ob_x,
    num_test=1000,
)

net = pinnx.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", bst.init.KaimingUniform())

model = pinnx.Trainer(data, net)
model.compile(
    bst.optim.Adam(0.0001),
    loss_weights=[1, 100, 1000]
)
losshistory, train_state = model.train(iterations=20000)
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

# view results
x = geom.uniform_points(500)
yhat = model.predict(x)
uhat, qhat = yhat[:, 0:1], yhat[:, 1:2]

utrue = np.sin(np.pi * x)
print("l2 relative error for u: " + str(pinnx.metrics.l2_relative_error(utrue, uhat)))
plt.figure()
plt.plot(x, utrue, "-", label="u_true")
plt.plot(x, uhat, "--", label="u_NN")
plt.legend()

qtrue = -np.pi ** 2 * np.sin(np.pi * x)
print("l2 relative error for q: " + str(pinnx.metrics.l2_relative_error(qtrue, qhat)))
plt.figure()
plt.plot(x, qtrue, "-", label="q_true")
plt.plot(x, qhat, "--", label="q_NN")
plt.legend()

plt.show()
