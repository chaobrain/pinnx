import matplotlib.pyplot as plt
import numpy as np

import pinnx as pinnx

dim_x = 5


# PDE
def pde(x, y, v):
    dy_x = pinnx.grad.jacobian(y, x, j=0)
    dy_t = pinnx.grad.jacobian(y, x, j=1)
    return dy_t + dy_x


# The same problem as advection_unaligned_pideeponet.py
# But consider time as the 2nd space coordinate
# to demonstrate the implementation of 2D problems
geom = pinnx.geometry.Rectangle([0, 0], [1, 1])


def func_ic(x, v):
    return v


def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)


ic = pinnx.icbc.DirichletBC(geom, func_ic, boundary)

pde = pinnx.data.PDE(geom, pde, ic, num_domain=200, num_boundary=200)

# Function space
func_space = pinnx.data.GRF(kernel="ExpSineSquared", length_scale=1)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.data.PDEOperator(pde, func_space, eval_pts, 1000, function_variables=[0])

# Net
net = pinnx.nn.DeepONet(
    [50, 128, 128, 128],
    [dim_x, 128, 128, 128],
    "tanh",
    "Glorot normal",
)


def periodic(x):
    x, t = x[:, :1], x[:, 1:]
    x = x * 2 * np.pi
    return concat([cos(x), sin(x), cos(2 * x), sin(2 * x), t], 1)


net.apply_feature_transform(periodic)

model = pinnx.Trainer(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(iterations=10000)
pinnx.utils.plot_loss_history(losshistory)

x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
u_true = np.sin(2 * np.pi * (x - t[:, None]))
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = np.sin(2 * np.pi * eval_pts)[:, 0]
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((np.tile(v_branch, (100 * 100, 1)), x_trunk))
u_pred = u_pred.reshape((100, 100))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
print(pinnx.metrics.l2_relative_error(u_true, u_pred))
