import brainstate as bst
import numpy as np
import optax

import pinnx

Re = 20
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)


def pde(net, x):
    x = pinnx.array_to_dict(x, ["x", 'y'])
    approx = lambda x: pinnx.array_to_dict(net(pinnx.dict_to_array(x)), ['u_vel', 'v_vel', 'p'])

    jacobian, u = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    u_vel, v_vel, p = u['u_vel'], u['v_vel'], u['p']
    u_vel_x = jacobian['u_vel']['x']
    u_vel_y = jacobian['u_vel']['y']
    u_vel_xx = hessian['u_vel']['x']['x']
    u_vel_yy = hessian['u_vel']['y']['y']

    v_vel_x = jacobian['v_vel']['x']
    v_vel_y = jacobian['v_vel']['y']
    v_vel_xx = hessian['v_vel']['x']['x']
    v_vel_yy = hessian['v_vel']['y']['y']

    p_x = jacobian['p']['x']
    p_y = jacobian['p']['y']

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]


def u_func(x):
    return 1 - np.exp(l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])


def v_func(x):
    return l / (2 * np.pi) * np.exp(l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])


def p_func(x):
    return 1 / 2 * (1 - np.exp(2 * l * x[:, 0:1]))


def boundary_outflow(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], 1)


spatial_domain = pinnx.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

boundary_condition_u = pinnx.icbc.DirichletBC(
    spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = pinnx.icbc.DirichletBC(
    spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_right_p = pinnx.icbc.DirichletBC(
    spatial_domain, p_func, boundary_outflow, component=2
)

data = pinnx.data.PDE(
    spatial_domain,
    pde,
    [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
    num_domain=2601,
    num_boundary=400,
    num_test=100000,
)

net = pinnx.nn.FNN([2] + 4 * [50] + [3], "tanh")

model = pinnx.Trainer(data, net)

model.compile(bst.optim.Adam(1e-3))
model.train(iterations=30000)
model.compile(bst.optim.OptaxOptimizer(optax.lbfgs(1e-3, linesearch=None)))
losshistory, train_state = model.train(2000)

X = spatial_domain.random_points(100000)
output = model.predict(X)
u_pred = output[:, 0]
v_pred = output[:, 1]
p_pred = output[:, 2]

u_exact = u_func(X).reshape(-1)
v_exact = v_func(X).reshape(-1)
p_exact = p_func(X).reshape(-1)

f = model.predict(X, operator=pde)

l2_difference_u = pinnx.metrics.l2_relative_error(u_exact, u_pred)
l2_difference_v = pinnx.metrics.l2_relative_error(v_exact, v_pred)
l2_difference_p = pinnx.metrics.l2_relative_error(p_exact, p_pred)
residual = np.mean(np.absolute(f))

print("Mean residual:", residual)
print("L2 relative error in u:", l2_difference_u)
print("L2 relative error in v:", l2_difference_v)
print("L2 relative error in p:", l2_difference_p)
