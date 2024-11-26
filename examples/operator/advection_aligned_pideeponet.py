import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
import brainstate as bst
import pinnx

dim_x = 5


# PDE
def pde(neu, x):
    approx = lambda x: neu(pinnx.dict_to_array(x))
    jacobian = pinnx.grad.jacobian(approx, pinnx.array_to_dict(x, ['x', 't']))
    dy_x = jacobian['x']
    dy_t = jacobian['t']
    return dy_t + dy_x


geom = pinnx.geometry.Interval(0, 1)
timedomain = pinnx.geometry.TimeDomain(0, 1)
geomtime = pinnx.geometry.GeometryXTime(geom, timedomain)


def func_ic(x, v):
    return v


ic = pinnx.icbc.IC(geomtime, func_ic, lambda _, on_initial: on_initial)
pde = pinnx.data.TimePDE(geomtime, pde, ic, num_domain=250, num_initial=50, num_test=500)

# Function space
func_space = pinnx.data.GRF(kernel="ExpSineSquared", length_scale=1)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.data.PDEOperatorCartesianProd(
    pde,
    func_space,
    eval_pts,
    1000,
    function_variables=[0],
    num_test=100,
    batch_size=32
)

# Net
net = pinnx.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [dim_x, 128, 128, 128],
    "tanh",
)


def periodic(x):
    x, t = x[:, :1], x[:, 1:]
    x = x * 2 * np.pi
    return u.math.concatenate(
        [u.math.cos(x),
         u.math.sin(x),
         u.math.cos(2 * x),
         u.math.sin(2 * x),
         t],
        axis=-1
    )


net.apply_feature_transform(periodic)

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.0005))
losshistory, train_state = model.train(iterations=50000)
pinnx.utils.plot_loss_history(losshistory)

x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
u_true = np.sin(2 * np.pi * (x - t[:, None]))
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = np.sin(2 * np.pi * eval_pts).T
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((v_branch, x_trunk))
u_pred = u_pred.reshape((100, 100))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()
print(pinnx.metrics.l2_relative_error(u_true, u_pred))
