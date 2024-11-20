import brainstate as bst
import brainunit as u
import numpy as np

import pinnx as dde


def pde(net, x):
    x = dde.array_to_dict(x, "r", "theta")
    approx = lambda x: dde.array_to_dict(net(dde.dict_to_array(x)), "y")
    jacobian = dde.grad.jacobian(approx, x)
    hessian = dde.grad.hessian(approx, x)

    dy_r = jacobian["y"]["r"]
    dy_rr = hessian["y"]["r"]["r"]
    dy_thetatheta = hessian["y"]["theta"]["theta"]
    return x['r'] * dy_r + x['r'] ** 2 * dy_rr + dy_thetatheta


def solution(x):
    r, theta = x[:, 0:1], x[:, 1:]
    return r * np.cos(theta)


geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2 * np.pi])
bc_rad = dde.icbc.DirichletBC(
    geom,
    lambda x: np.cos(x[:, 1:2]),
    lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 1),
)
data = dde.data.PDE(
    geom, pde, bc_rad, num_domain=2540, num_boundary=80, solution=solution
)

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh")


# Use [r*sin(theta), r*cos(theta)] as features,
# so that the network is automatically periodic along the theta coordinate.
def feature_transform(x):
    return u.math.concatenate([x[..., 0:1] * u.math.sin(x[..., 1:2]),
                               x[..., 0:1] * u.math.cos(x[..., 1:2])], axis=-1)


net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)
model.compile(bst.optim.Adam(1e-3), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=15000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
