import brainstate as bst
import brainunit as u
import numpy as np

import pinnx as dde


def pde(net, x):
    x = dde.array_to_dict(x, ["r", "theta"])
    approx = lambda x: dde.array_to_dict(net(dde.dict_to_array(x)), ["y"])
    jacobian = dde.grad.jacobian(approx, x)
    hessian = dde.grad.hessian(approx, x)

    dy_r = jacobian["y"]["r"]
    dy_rr = hessian["y"]["r"]["r"]
    dy_thetatheta = hessian["y"]["theta"]["theta"]
    return x['r'] * dy_r + x['r'] ** 2 * dy_rr + dy_thetatheta


def solution(x):
    r, theta = x[:, 0:1], x[:, 1:]
    return r * np.cos(theta)

# TODO: Rectangle compat with brainunit
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
bc = dde.icbc.DirichletBC(
    lambda x: np.cos(x[:, 1:2]),
    lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 1),
)


# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
def feature_transform(x):
    return u.math.concatenate([x[..., 0:1] * u.math.sin(x[..., 1:2]),
                               x[..., 0:1] * u.math.cos(x[..., 1:2])], axis=-1)


net = dde.nn.FNN([geom.dim] + [20] * 3 + [1], "tanh")
net.apply_feature_transform(feature_transform)

data = dde.problem.PDE(
    geom,
    pde,
    bc,
    num_domain=2540,
    num_boundary=80,
    solution=solution
)

model = dde.Trainer(data)
model.compile(bst.optim.Adam(1e-3), metrics=["l2 relative error"])
model.train(iterations=15000)
model.saveplot(issave=True, isplot=True)
