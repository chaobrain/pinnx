import matplotlib.pyplot as plt
import numpy as np

import pinnx
from ADR_solver import solve_ADR


# PDE
def pde(x, y, v):
    D = 0.01
    k = 0.01
    dy_t = pinnx.grad.jacobian(y, x, j=1)
    dy_xx = pinnx.grad.hessian(y, x, j=0)
    return dy_t - D * dy_xx + k * y ** 2 - v


geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)

bc = pinnx.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
ic = pinnx.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

pde = pinnx.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=200,
    num_boundary=40,
    num_initial=20,
    num_test=500,
)

# Function space
func_space = pinnx.data.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.data.PDEOperator(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=1000
)

# Net
net = pinnx.nn.DeepONet(
    [50, 128, 128, 128],
    [2, 128, 128, 128],
    "tanh",
    "Glorot normal",
)

model = pinnx.Trainer(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(iterations=50000)
pinnx.utils.plot_loss_history(losshistory)

func_feats = func_space.random(1)
xs = np.linspace(0, 1, num=100)[:, None]
v = func_space.eval_batch(func_feats, xs)[0]
x, t, u_true = solve_ADR(
    0,
    1,
    0,
    1,
    lambda x: 0.01 * np.ones_like(x),
    lambda x: np.zeros_like(x),
    lambda u: 0.01 * u ** 2,
    lambda u: 0.02 * u,
    lambda x, t: np.tile(v[:, None], (1, len(t))),
    lambda x: np.zeros_like(x),
    100,
    100,
)
u_true = u_true.T
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])[0]
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((np.tile(v_branch, (100 * 100, 1)), x_trunk))
u_pred = u_pred.reshape((100, 100))
print(pinnx.metrics.l2_relative_error(u_true, u_pred))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
