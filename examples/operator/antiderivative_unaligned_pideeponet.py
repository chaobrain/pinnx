import matplotlib.pyplot as plt
import numpy as np

import pinnx

# PDE
geom = pinnx.geometry.TimeDomain(0, 1)


def pde(x, u, v):
    return pinnx.grad.jacobian(u, x) - v


ic = pinnx.icbc.IC(geom, lambda _: 0, lambda _, on_initial: on_initial)
pde = pinnx.data.PDE(geom, pde, ic, num_domain=20, num_boundary=2, num_test=40)

# Function space
func_space = pinnx.data.GRF(length_scale=0.2)

# Problem
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = pinnx.data.PDEOperator(pde, func_space, eval_pts, 1000, num_test=1000)

# Net
net = pinnx.nn.DeepONet(
    [50, 128, 128, 128],
    [1, 128, 128, 128],
    "tanh",
)


# Hard constraint zero IC
def zero_ic(inputs, outputs):
    return outputs * inputs[1]


net.apply_output_transform(zero_ic)

model = pinnx.Trainer(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(iterations=40000)

pinnx.utils.plot_loss_history(losshistory)

x = np.linspace(0, 1, num=50)
v = np.sin(np.pi * x)
u = np.ravel(model.predict((np.tile(v, (50, 1)), x[:, None])))
u_true = 1 / np.pi - np.cos(np.pi * x) / np.pi
print(pinnx.metrics.l2_relative_error(u_true, u))
plt.figure()
plt.plot(x, u_true, "k")
plt.plot(x, u, "r")
plt.show()
