import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import pinnx


def gen_traindata(num):
    # generate num equally-spaced points from -1 to 1
    xvals = np.linspace(-1, 1, num)
    uvals = np.sin(np.pi * xvals)
    return {'x': xvals}, {'u': uvals}


def pde(x, y):
    du_xx = net.hessian(x, y='u')['u']['x']['x']
    return -du_xx + y['q']


geom = pinnx.geometry.Interval(-1, 1).to_dict_point('x')


def sol(x):
    # solution is u(x) = sin(pi*x), q(x) = -pi^2 * sin(pi*x)
    # return {'u': u.math.sin(u.math.pi * x['x']), }
    return {'u': u.math.sin(u.math.pi * x['x']), 'q': -u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'])}


bc = pinnx.icbc.DirichletBC(sol)
ob_x, ob_u = gen_traindata(100)
observe_u = pinnx.icbc.PointSetBC(ob_x, ob_u)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(u=None, q=None),
)
problem = pinnx.problem.PDE(
    geom,
    pde,
    [bc, observe_u],
    net,
    num_domain=200,
    num_boundary=2,
    anchors=ob_x,
    num_test=1000,
    loss_weights=[1, 100, 1000],
)

model = pinnx.Trainer(problem)
model.compile(bst.optim.Adam(0.0001)).train(iterations=20000)
model.saveplot(issave=True, isplot=True)

# view results
x = geom.uniform_points(500)
yhat = model.predict(x)
uhat, qhat = yhat['u'], yhat['q']
x = x['x']

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
