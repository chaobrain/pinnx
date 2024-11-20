import matplotlib.pyplot as plt
import numpy as np

import pinnx as dde


# PDE equation
def pde(xy, uvp, aux):
    mu = 0.01
    # first order
    du_x = dde.grad.jacobian(uvp, xy, i=0, j=0)
    dv_y = dde.grad.jacobian(uvp, xy, i=1, j=1)
    dp_x = dde.grad.jacobian(uvp, xy, i=2, j=0)
    dp_y = dde.grad.jacobian(uvp, xy, i=2, j=1)
    # second order
    du_xx = dde.grad.hessian(uvp, xy, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(uvp, xy, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(uvp, xy, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(uvp, xy, component=1, i=1, j=1)
    motion_x = mu * (du_xx + du_yy) - dp_x
    motion_y = mu * (dv_xx + dv_yy) - dp_y
    mass = du_x + dv_y
    return motion_x, motion_y, mass


# Geometry
geom = dde.geometry.Rectangle([0, 0], [1, 1])


# Boundary condition
# other boundary conditions will be enforced by output transform
def bc_slip_top_func(x, aux_var):
    # using (perturbation / 10 + 1) * x * (1 - x)
    return (aux_var / 10 + 1.) * dde.backend.as_tensor(x[:, 0:1] * (1 - x[:, 0:1]))


bc_slip_top = dde.icbc.DirichletBC(
    geom=geom,
    func=bc_slip_top_func,
    on_boundary=lambda x, on_boundary: np.isclose(x[1], 1.),
    component=0
)

# PDE object
pde = dde.data.PDE(
    geom,
    pde,
    ic_bcs=[bc_slip_top],
    num_domain=5000,
    num_boundary=4000,  # sampling a bit more points on boundary (1000 on top bc)
    num_test=500,
)

# Function space
func_space = dde.data.GRF(length_scale=0.2)

# Data
n_pts_edge = 101  # using the size of true solution, but this is unnecessary
eval_pts = np.linspace(0, 1, num=n_pts_edge)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts,
    num_function=1000,
    function_variables=[0],
    num_test=100,
    batch_size=50
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [n_pts_edge, 128, 128, 128],
    [2, 128, 128, 128],
    "tanh",
    "Glorot normal",
    num_outputs=3,
    multi_output_strategy="independent"
)


# Output transform for zero boundary conditions
def out_transform(inputs, outputs):
    x, y = inputs[1][:, 0], inputs[1][:, 1]
    # horizontal velocity on left, right, bottom
    u = outputs[:, :, 0] * (x * (1 - x) * y)[None, :]
    # vertical velocity on all edges
    v = outputs[:, :, 1] * (x * (1 - x) * y * (1 - y))[None, :]
    # pressure on bottom
    p = outputs[:, :, 2] * y[None, :]
    return dde.backend.stack((u, v, p), axis=2)


net.apply_output_transform(out_transform)

# Model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, decay=("inverse time", 10000, 0.5))
losshistory, train_state = model.train(iterations=50000)
dde.utils.plot_loss_history(losshistory)
# save model if needed
# model.save('stokes_weights')

# Evaluation
func_feats = func_space.random(1)
v = func_space.eval_batch(func_feats, eval_pts)
v[:] = 0.  # true solution uses zero perturbation
xv, yv = np.meshgrid(eval_pts[:, 0], eval_pts[:, 0], indexing='ij')
xy = np.vstack((np.ravel(xv), np.ravel(yv))).T
sol_pred = model.predict((v, xy))[0]
sol_true = np.load('../dataset/stokes.npz')['arr_0']
print('Error on horizontal velocity:', dde.metrics.l2_relative_error(sol_true[:, 0], sol_pred[:, 0]))
print('Error on vertical velocity:', dde.metrics.l2_relative_error(sol_true[:, 1], sol_pred[:, 1]))
print('Error on pressure:', dde.metrics.l2_relative_error(sol_true[:, 2], sol_pred[:, 2]))


# Plot
def plot_sol(sol, ax, pressure_lim=0.03, vec_space=4, vec_scale=.5, label=""):
    ax.imshow(sol[:, :, 2].T,
              origin="lower",
              vmin=-pressure_lim,
              vmax=pressure_lim,
              cmap="turbo",
              alpha=.6)
    ax.quiver(xv[::vec_space, ::vec_space] * 100,
              yv[::vec_space, ::vec_space] * 100,
              sol[::vec_space, ::vec_space, 0],
              sol[::vec_space, ::vec_space, 1], color="k", scale=vec_scale)
    ax.axis('off')
    ax.set_title(label)


fig, ax = plt.subplots(1, 2, dpi=200)
plot_sol(sol_true.reshape(101, 101, 3), ax[0], label="True")
plot_sol(sol_pred.reshape(101, 101, 3), ax[1], label="Predicted")
# save plot if needed
# plt.savefig('stokes_plot.png')
plt.show()
