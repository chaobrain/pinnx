import brainstate
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

import pinnx


# PDE equation
def pde(xy, uvp, aux):
    mu = 0.01
    fix, xy = xy

    # Each output u[b, n] depends only on the point xy[n], so the pointwise derivatives
    # of all outputs are forward-mode directional derivatives along a one-hot tangent.
    def directional(fn, axis):
        tangent = jax.numpy.zeros_like(xy).at[:, axis].set(1.)
        return lambda pts: jax.jvp(fn, (pts,), (tangent,))[1]

    f = lambda pts: net((fix, pts))
    f_x, f_y = directional(f, 0), directional(f, 1)
    jacobian = {'x': f_x(xy), 'y': f_y(xy)}
    hessian = {'xx': directional(f_x, 0)(xy), 'yy': directional(f_y, 1)(xy)}

    # first order
    du_x = jacobian['x']['u']
    dv_y = jacobian['y']['v']
    dp_x = jacobian['x']['p']
    dp_y = jacobian['y']['p']
    # second order
    du_xx = hessian['xx']['u']
    du_yy = hessian['yy']['u']
    dv_xx = hessian['xx']['v']
    dv_yy = hessian['yy']['v']
    motion_x = mu * (du_xx + du_yy) - dp_x
    motion_y = mu * (dv_xx + dv_yy) - dp_y
    mass = du_x + dv_y
    return motion_x, motion_y, mass


# Net


# Output transform for zero boundary conditions
def out_transform(inputs, outputs):
    x, y = inputs[1][..., 0], inputs[1][..., 1]
    # horizontal velocity on left, right, bottom
    u_ = outputs[..., 0] * (x * (1 - x) * y)[None, :]
    # vertical velocity on all edges
    v = outputs[..., 1] * (x * (1 - x) * y * (1 - y))[None, :]
    # pressure on bottom
    p = outputs[..., 2] * y[None, :]
    return u.math.stack((u_, v, p), axis=2)


n_pts_edge = 101  # using the size of true solution, but this is unnecessary

net = brainstate.nn.Sequential(
    pinnx.nn.DeepONetCartesianProd(
        [n_pts_edge, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        num_outputs=3,
        multi_output_strategy="independent",
        output_transform=out_transform
    ),
    pinnx.nn.ArrayToDict(u=None, v=None, p=None)
)

# Geometry
geom = pinnx.geometry.Rectangle([0, 0], [1, 1]).to_dict_point('x', 'y')


# Boundary condition
# other boundary conditions will be enforced by output transform
def bc_slip_top_func(x, aux):
    # using (perturbation / 10 + 1) * x * (1 - x)
    x = x[1][..., 0]
    u_ = (aux / 10 + 1.) * u.math.asarray(x * (1 - x))
    return {'u': u_}


bc_slip_top = pinnx.icbc.DirichletBC(
    func=bc_slip_top_func,
    on_boundary=lambda x, on_boundary: u.math.isclose(x['y'], 1.),
)

# Function space
func_space = pinnx.fnspace.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=n_pts_edge)[:, None]
data = pinnx.problem.PDEOperatorCartesianProd(
    geom,
    pde,
    bc_slip_top,
    func_space,
    eval_pts,
    approximator=net,
    num_function=1000,
    function_variables=[0],
    num_fn_test=100,
    batch_size=50,
    num_domain=5000,
    num_boundary=4000,  # sampling a bit more points on boundary (1000 on top bc)
    num_test=500,
)

# Trainer
trainer = pinnx.Trainer(data)
trainer.compile(braintools.optim.SGD(1e-6)).train(iterations=50)
# trainer.compile(braintools.optim.Adam(braintools.optim.ExponentialDecayLR(1e-5, 10000, 1 / 1.5))).train(iterations=50000)
trainer.saveplot()

# Evaluation
func_feats = func_space.random(1)
v = func_space.eval_batch(func_feats, eval_pts)
v[:] = 0.  # true solution uses zero perturbation
xv, yv = np.meshgrid(eval_pts[:, 0], eval_pts[:, 0], indexing='ij')
xy = np.vstack((np.ravel(xv), np.ravel(yv))).T
sol_pred = trainer.predict((v, xy))
sol_pred = jax.tree.map(lambda x: x[0], sol_pred)
sol_true = np.load('../dataset/stokes.npz')['arr_0']
print('Error on horizontal velocity:', pinnx.metrics.l2_relative_error(sol_true[:, 0], sol_pred['u']))
print('Error on vertical velocity:', pinnx.metrics.l2_relative_error(sol_true[:, 1], sol_pred['v']))
print('Error on pressure:', pinnx.metrics.l2_relative_error(sol_true[:, 2], sol_pred['p']))


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
plot_sol(pinnx.utils.dict_to_array(sol_pred).reshape(101, 101, 3), ax[1], label="Predicted")
# save plot if needed
# plt.savefig('stokes_plot.png')
plt.show()
