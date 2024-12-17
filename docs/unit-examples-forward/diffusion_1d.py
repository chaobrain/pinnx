import brainstate as bst
import brainunit as u

import pinnx

unit_of_x = u.meter
unit_of_t = u.second
unit_of_f = 1 / u.second

c = 1. * u.meter ** 2 / u.second

geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)
geomtime = geomtime.to_dict_point(x=unit_of_x, t=unit_of_t)


def func(x):
    y = u.math.sin(u.math.pi * x['x'] / unit_of_x) * u.math.exp(-x['t'] / unit_of_t)
    return {'y': y}


bc = pinnx.icbc.DirichletBC(func)
ic = pinnx.icbc.IC(func)

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=unit_of_x, t=unit_of_t),
    pinnx.nn.FNN([2] + [32] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x, xi='x', xj='x')
    dy_t = jacobian["y"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    source = (
        u.math.exp(-x['t'] / unit_of_t) * (
        u.math.sin(u.math.pi * x['x'] / unit_of_x) -
        u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'] / unit_of_x)
    )
    )
    return dy_t - c * dy_xx + source * unit_of_f


problem = pinnx.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    net,
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    solution=func,
    num_test=10000,
)

trainer = pinnx.Trainer(problem)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=10000)
trainer.saveplot(issave=True, isplot=True)
