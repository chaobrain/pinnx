import brainstate as bst
import numpy as np

import pinnx


def gen_traindata():
    data = np.load("../dataset/reaction.npz")
    t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    Ca = np.reshape(ca, (-1, 1))
    Cb = np.reshape(cb, (-1, 1))
    return np.hstack((X, T)), Ca, Cb


kf = bst.ParamState(0.05)
D = bst.ParamState(1.0)


def pde(neu, x):
    x = pinnx.array_to_dict(x, "x", "t")
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), "ca", "cb")
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)
    ca, cb = y['ca'], y['cb']
    dca_t = jacobian['ca']['t']
    dcb_t = jacobian['cb']['t']
    dca_xx = hessian['ca']['x']['x']
    dcb_xx = hessian['cb']['x']['x']
    eq_a = dca_t - 1e-3 * D.value * dca_xx + kf.value * ca * cb ** 2
    eq_b = dcb_t - 1e-3 * D.value * dcb_xx + 2 * kf.value * ca * cb ** 2
    return [eq_a, eq_b]


def fun_bc(x):
    return 1 - x[:, 0:1]


def fun_init(x):
    return np.exp(-20 * x[:, 0:1])


geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 10)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc_a = pinnx.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0
)
bc_b = pinnx.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
)
ic1 = pinnx.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
ic2 = pinnx.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

observe_x, Ca, Cb = gen_traindata()
observe_y1 = pinnx.icbc.PointSetBC(observe_x, Ca, component=0)
observe_y2 = pinnx.icbc.PointSetBC(observe_x, Cb, component=1)

data = pinnx.data.TimePDE(
    geomtime,
    pde,
    [bc_a, bc_b, ic1, ic2, observe_y1, observe_y2],
    num_domain=2000,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=50000,
)
net = pinnx.nn.FNN([2] + [20] * 3 + [2], "tanh")

model = pinnx.Model(data, net)
model.compile(
    bst.optim.Adam(0.001),
    external_trainable_variables=[kf, D]
)
variable = pinnx.callbacks.VariableValue([kf, D], period=1000, filename="variables.dat")
losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
