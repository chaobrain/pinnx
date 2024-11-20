import brainstate as bst
import numpy as np
import optax

import pinnx

a = 1
d = 1
Re = 1


def pde(net, x):
    x = pinnx.array_to_dict(x, 'x', 'y', 'z', 't')
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), 'u_vel', 'v_vel', 'w_vel', 'p')

    jacobian, u = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    u_vel, v_vel, w_vel, p = u['u_vel'], u['v_vel'], u['w_vel'], u['p']

    du_vel_dx = jacobian['u_vel']['x']
    du_vel_dy = jacobian['u_vel']['y']
    du_vel_dz = jacobian['u_vel']['z']
    du_vel_dt = jacobian['u_vel']['t']
    du_vel_dx_dx = hessian['u_vel']['x']['x']
    du_vel_dy_dy = hessian['u_vel']['y']['y']
    du_vel_dz_dz = hessian['u_vel']['z']['z']

    dv_vel_dx = jacobian['v_vel']['x']
    dv_vel_dy = jacobian['v_vel']['y']
    dv_vel_dz = jacobian['v_vel']['z']
    dv_vel_dt = jacobian['v_vel']['t']
    dv_vel_dx_dx = hessian['v_vel']['x']['x']
    dv_vel_dy_dy = hessian['v_vel']['y']['y']
    dv_vel_dz_dz = hessian['v_vel']['z']['z']

    dw_vel_dx = jacobian['w_vel']['x']
    dw_vel_dy = jacobian['w_vel']['y']
    dw_vel_dz = jacobian['w_vel']['z']
    dw_vel_dt = jacobian['w_vel']['t']
    dw_vel_dx_dx = hessian['w_vel']['x']['x']
    dw_vel_dy_dy = hessian['w_vel']['y']['y']
    dw_vel_dz_dz = hessian['w_vel']['z']['z']

    dp_dx = jacobian['p']['x']
    dp_dy = jacobian['p']['y']
    dp_dz = jacobian['p']['z']

    momentum_x = (
        du_vel_dt
        + (u_vel * du_vel_dx + v_vel * du_vel_dy + w_vel * du_vel_dz)
        + dp_dx
        - 1 / Re * (du_vel_dx_dx + du_vel_dy_dy + du_vel_dz_dz)
    )
    momentum_y = (
        dv_vel_dt
        + (u_vel * dv_vel_dx + v_vel * dv_vel_dy + w_vel * dv_vel_dz)
        + dp_dy
        - 1 / Re * (dv_vel_dx_dx + dv_vel_dy_dy + dv_vel_dz_dz)
    )
    momentum_z = (
        dw_vel_dt
        + (u_vel * dw_vel_dx + v_vel * dw_vel_dy + w_vel * dw_vel_dz)
        + dp_dz
        - 1 / Re * (dw_vel_dx_dx + dw_vel_dy_dy + dw_vel_dz_dz)
    )
    continuity = du_vel_dx + dv_vel_dy + dw_vel_dz

    return [momentum_x, momentum_y, momentum_z, continuity]


def u_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def v_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def w_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def p_func(x):
    return (
        -0.5
        * a ** 2
        * (
            np.exp(2 * a * x[:, 0:1])
            + np.exp(2 * a * x[:, 1:2])
            + np.exp(2 * a * x[:, 2:3])
            + 2
            * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
            + 2
            * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
            + 2
            * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
        )
        * np.exp(-2 * d ** 2 * x[:, 3:4])
    )


spatial_domain = pinnx.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
temporal_domain = pinnx.geometry.TimeDomain(0, 1)
spatio_temporal_domain = pinnx.geometry.GeometryXTime(spatial_domain, temporal_domain)

boundary_condition_u = pinnx.icbc.DirichletBC(
    spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = pinnx.icbc.DirichletBC(
    spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_w = pinnx.icbc.DirichletBC(
    spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
)

initial_condition_u = pinnx.icbc.IC(
    spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
)
initial_condition_v = pinnx.icbc.IC(
    spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
)
initial_condition_w = pinnx.icbc.IC(
    spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
)

data = pinnx.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [
        boundary_condition_u,
        boundary_condition_v,
        boundary_condition_w,
        initial_condition_u,
        initial_condition_v,
        initial_condition_w,
    ],
    num_domain=50000,
    num_boundary=5000,
    num_initial=5000,
    num_test=10000,
)

net = pinnx.nn.FNN([4] + 4 * [50] + [4], "tanh", bst.init.KaimingUniform())

model = pinnx.Model(data, net)

model.compile(
    bst.optim.Adam(1e-3),
    loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100]
)
model.train(iterations=30000)
model.compile(
    bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)),
    loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100]
)
losshistory, train_state = model.train(5000, display_every=200)

x, y, z = np.meshgrid(
    np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
)

X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

t_0 = np.zeros(1000).reshape(1000, 1)
t_1 = np.ones(1000).reshape(1000, 1)

X_0 = np.hstack((X, t_0))
X_1 = np.hstack((X, t_1))

output_0 = model.predict(X_0)
output_1 = model.predict(X_1)

u_pred_0 = output_0[:, 0].reshape(-1)
v_pred_0 = output_0[:, 1].reshape(-1)
w_pred_0 = output_0[:, 2].reshape(-1)
p_pred_0 = output_0[:, 3].reshape(-1)

u_exact_0 = u_func(X_0).reshape(-1)
v_exact_0 = v_func(X_0).reshape(-1)
w_exact_0 = w_func(X_0).reshape(-1)
p_exact_0 = p_func(X_0).reshape(-1)

u_pred_1 = output_1[:, 0].reshape(-1)
v_pred_1 = output_1[:, 1].reshape(-1)
w_pred_1 = output_1[:, 2].reshape(-1)
p_pred_1 = output_1[:, 3].reshape(-1)

u_exact_1 = u_func(X_1).reshape(-1)
v_exact_1 = v_func(X_1).reshape(-1)
w_exact_1 = w_func(X_1).reshape(-1)
p_exact_1 = p_func(X_1).reshape(-1)

f_0 = model.predict(X_0, operator=pde)
f_1 = model.predict(X_1, operator=pde)

l2_difference_u_0 = pinnx.metrics.l2_relative_error(u_exact_0, u_pred_0)
l2_difference_v_0 = pinnx.metrics.l2_relative_error(v_exact_0, v_pred_0)
l2_difference_w_0 = pinnx.metrics.l2_relative_error(w_exact_0, w_pred_0)
l2_difference_p_0 = pinnx.metrics.l2_relative_error(p_exact_0, p_pred_0)
residual_0 = np.mean(np.absolute(f_0))

l2_difference_u_1 = pinnx.metrics.l2_relative_error(u_exact_1, u_pred_1)
l2_difference_v_1 = pinnx.metrics.l2_relative_error(v_exact_1, v_pred_1)
l2_difference_w_1 = pinnx.metrics.l2_relative_error(w_exact_1, w_pred_1)
l2_difference_p_1 = pinnx.metrics.l2_relative_error(p_exact_1, p_pred_1)
residual_1 = np.mean(np.absolute(f_1))

print("Accuracy at t = 0:")
print("Mean residual:", residual_0)
print("L2 relative error in u:", l2_difference_u_0)
print("L2 relative error in v:", l2_difference_v_0)
print("L2 relative error in w:", l2_difference_w_0)
print("\n")
print("Accuracy at t = 1:")
print("Mean residual:", residual_1)
print("L2 relative error in u:", l2_difference_u_1)
print("L2 relative error in v:", l2_difference_v_1)
print("L2 relative error in w:", l2_difference_w_1)
