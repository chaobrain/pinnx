import brainstate as bst
import numpy as np
import optax

import pinnx


def gen_testdata():
    data = np.load("../dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(neuralnet, x):
    approx = lambda x: pinnx.array_to_dict(neuralnet(pinnx.dict_to_array(x)), 'y')
    x = pinnx.array_to_dict(x, 'x', 't')
    jacobian, u = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    dy_x = jacobian['y']['x']
    dy_t = jacobian['y']['t']
    dy_xx = hessian['y']['x']['x']
    return dy_t + u['y'] * dy_x - 0.01 / np.pi * dy_xx


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 0.99)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc = pinnx.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = pinnx.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = pinnx.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2500, num_boundary=100, num_initial=160
)
net = pinnx.nn.FNN([2] + [20] * 3 + [1], "tanh", bst.init.KaimingUniform())
model = pinnx.Model(data, net)

model.compile(bst.optim.Adam(1e-3))
model.train(iterations=10000)
# model.compile("L-BFGS")
# model.train()

X = geomtime.random_points(100000)
err = 1
while err > 0.005:
    f = model.predict(X, operator=pde)
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    x_id = np.argmax(err_eq)
    print("Adding new point:", X[x_id], "\n")
    data.add_anchors(X[x_id])
    early_stopping = pinnx.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model.compile(bst.optim.Adam(1e-3))
    model.train(iterations=10000, disregard_previous_best=True, callbacks=[early_stopping])
    model.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))
    losshistory, train_state = model.train(1000, display_every=100)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
