import brainstate as bst
import numpy as np
import optax
import brainunit as u

import pinnx

geom = pinnx.geometry.Interval('x', -1 * u.meter, 1. * u.meter)
timedomain = pinnx.geometry.TimeDomain('t', 0 * u.second, 0.99 * u.second)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

uy = u.meter / u.second
bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0. * uy})
ic = pinnx.icbc.IC(
    lambda x: {'y': -u.math.sin(u.math.pi * x['x'] / u.meter) * uy}
)

v = 0.01 / np.pi * u.meter ** 2 / u.second

def pde(x, y):
    jacobian = approximator.jacobian(x)
    hessian = approximator.hessian(x)

    dy_x = jacobian['y']['x']
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t + y['y'] * dy_x - v * dy_xx

approximator = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=u.meter, t=u.second),
    pinnx.nn.FNN([2] + [20] * 3 + [1], "tanh", bst.init.KaimingUniform()),
    pinnx.nn.ArrayToDict(y=uy)
)

problem = pinnx.problem.TimePDE(
    geomtime, pde, [bc, ic], approximator, num_domain=2540, num_boundary=80, num_initial=160
)

trainer = pinnx.Trainer(problem)

trainer.compile(bst.optim.Adam(1e-3))
trainer.train(iterations=4000)
# trainer.compile("L-BFGS")
# trainer.train()

def gen_testdata():
    data = np.load("../../docs/dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

X = geomtime.random_points(100000)
err = 1
while err > 0.005:
    f = trainer.predict(X, operator=pde)
    err_eq = u.math.absolute(f)
    err = u.math.mean(err_eq)
    print(f"Mean residual: {err:.2f}")

    x_id = u.math.argmax(err_eq)
    new_point = {key: value[x_id] for key, value in X.items()}
    print("Adding new point:", new_point, "\n")
    problem.add_anchors(new_point)
    # TODO: fix callbacks compat with brainunit
    early_stopping = pinnx.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    trainer.compile(bst.optim.Adam(1e-3))
    trainer.train(iterations=10000, disregard_previous_best=True, callbacks=[early_stopping])
    trainer.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))
    trainer.train(1000, display_every=100)

trainer.saveplot(issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = trainer.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
