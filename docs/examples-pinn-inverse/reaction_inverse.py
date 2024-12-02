import brainstate as bst
import brainunit as u
import numpy as np

import pinnx


def gen_traindata():
    data = np.load("../dataset/reaction.npz")
    t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
    X, T = np.meshgrid(x, t)
    return {'x': X.flatten(), 't': T.flatten()}, {'ca': ca.flatten(), 'cb': cb.flatten()}


kf = bst.ParamState(0.05)
D = bst.ParamState(1.0)


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x)
    ca, cb = y['ca'], y['cb']
    dca_t = jacobian['ca']['t']
    dcb_t = jacobian['cb']['t']
    dca_xx = hessian['ca']['x']['x']
    dcb_xx = hessian['cb']['x']['x']
    eq_a = dca_t - 1e-3 * D.value * dca_xx + kf.value * ca * cb ** 2
    eq_b = dcb_t - 1e-3 * D.value * dcb_xx + 2 * kf.value * ca * cb ** 2
    return [eq_a, eq_b]


net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None, t=None),
    pinnx.nn.FNN([2] + [20] * 3 + [2], "tanh"),
    pinnx.nn.ArrayToDict(ca=None, cb=None),
)


def fun_bc(x):
    return {'ca': 1 - x['x'], 'cb': 1 - x['x']}


def fun_init(x):
    return {
        'ca': u.math.exp(-20 * x['x']),
        'cb': u.math.exp(-20 * x['x']),
    }


geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 10)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point('x', 't')

bc = pinnx.icbc.DirichletBC(fun_bc)
ic = pinnx.icbc.IC(fun_init)

observe_x, observe_y = gen_traindata()
observe_bc = pinnx.icbc.PointSetBC(observe_x, observe_y)

data = pinnx.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_bc],
    net,
    num_domain=2000,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=50000,
)

variable = pinnx.callbacks.VariableValue([kf, D], period=1000, filename="./variables.dat")
model = pinnx.Trainer(data, external_trainable_variables=[kf, D])
model.compile(bst.optim.Adam(0.001)).train(iterations=80000, callbacks=[variable])
model.saveplot(issave=True, isplot=True)
