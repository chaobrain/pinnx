import brainstate as bst
import brainunit as u

import pinnx as pinnx

C = bst.ParamState(2.0)


def pde(x, y):
    jacobian = net.jacobian(x, x='t')
    hessian = net.hessian(x, xi='x', xj='x')

    dy_t = jacobian["y"]["t"]
    dy_xx = hessian["y"]["x"]["x"]
    return (
        dy_t
        - C.value * dy_xx
        + u.math.exp(-x['t'])
        * (u.math.sin(u.math.pi * x['x']) -
           u.math.pi ** 2 * u.math.sin(u.math.pi * x['x']))
    )


def func(x):
    y = u.math.sin(u.math.pi * x['x']) * u.math.exp(-x['t'])
    return {'y': y}


geom = pinnx.geometry.Interval(-1, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain).to_dict_point('x', 't')

bc = pinnx.icbc.DirichletBC(func)
ic = pinnx.icbc.IC(func)

x = {
    'x': u.math.linspace(-1, 1, num=10),
    't': u.math.full((10,), 1)
}
observe_y = pinnx.icbc.PointSetBC(x, func(x))

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None, t=None),
    pinnx.nn.FNN([2] + [32] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

problem = pinnx.problem.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    net,
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=x,
    solution=func,
    num_test=10000,
)

variable = pinnx.callbacks.VariableValue(C, period=1000)
trainer = pinnx.Trainer(problem, external_trainable_variables=C)
trainer.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"]).train(iterations=50000, callbacks=[variable])
trainer.saveplot(issave=True, isplot=True)
