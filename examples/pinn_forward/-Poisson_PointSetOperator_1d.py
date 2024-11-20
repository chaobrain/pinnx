import pinnx


def pde(x, y):
    dy_xx = pinnx.grad.hessian(y, x)
    return dy_xx - 2


def dy_x(x, y, X):
    dy_x = pinnx.grad.jacobian(y, x)
    return dy_x


def boundary_l(x, on_boundary):
    return on_boundary and pinnx.utils.isclose(x[0], -1)


def func(x):
    return (x + 1) ** 2


def d_func(x):
    return 2 * (x + 1)


geom = pinnx.geometry.Interval(-1, 1)
bc_l = pinnx.icbc.DirichletBC(geom, func, boundary_l)
boundary_pts = geom.random_boundary_points(2)
r_boundary_pts = boundary_pts[pinnx.utils.isclose(boundary_pts, 1)].reshape(-1, 1)
bc_r = pinnx.icbc.PointSetOperatorBC(r_boundary_pts, d_func(r_boundary_pts), dy_x)

data = pinnx.data.PDE(
    geom, pde, [bc_l, bc_r], num_domain=16, num_boundary=2, solution=func, num_test=100
)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh")

model = pinnx.Model(data, net)


def dy_x(x, y):
    dy_x = pinnx.grad.jacobian(y, x)
    return dy_x


def dy_xx(x, y):
    dy_xx = pinnx.grad.hessian(y, x)
    return dy_xx


# Print out first and second derivatives into a file during training on the boundary points
first_derivative = pinnx.callbacks.OperatorPredictor(
    geom.random_boundary_points(2), op=dy_x, period=200, filename="first_derivative.txt"
)
second_derivative = pinnx.callbacks.OperatorPredictor(
    geom.random_boundary_points(2),
    op=dy_xx,
    period=200,
    filename="second_derivative.txt",
)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(
    iterations=10000, callbacks=[first_derivative, second_derivative]
)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
