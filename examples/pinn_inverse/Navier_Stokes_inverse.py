"""
An inverse problem of the Navier-Stokes equation of incompressible flow around cylinder with Re=100

References: https://doi.org/10.1016/j.jcp.2018.10.045 Section 4.1.1
"""

import re

import brainstate as bst
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import pinnx

# true values
C1true = 1.0
C2true = 0.01


# Load training data
def load_training_data(num):
    data = loadmat("../../docs/dataset/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Problem
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]
    return [x_train, y_train, t_train, u_train, v_train, p_train]


# Parameters to be identified
C1 = bst.ParamState(0.0)
C2 = bst.ParamState(0.0)


# Define Navier Stokes Equations (Time-dependent PDEs)
def Navier_Stokes_Equation(neu, x):
    x = pinnx.array_to_dict(x, ["x", "y", "t"])
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), ["u", "v", "p"])
    jacobian, y = pinnx.grad.jacobian(approx, x, return_value=True)
    hessian = pinnx.grad.hessian(approx, x)

    u = y['u']
    v = y['v']
    p = y['p']
    du_x = jacobian['u']['x']
    du_y = jacobian['u']['y']
    du_t = jacobian['u']['t']
    dv_x = jacobian['v']['x']
    dv_y = jacobian['v']['y']
    dv_t = jacobian['v']['t']
    dp_x = jacobian['p']['x']
    dp_y = jacobian['p']['y']
    du_xx = hessian['u']['x']['x']
    du_yy = hessian['u']['y']['y']
    dv_xx = hessian['v']['x']['x']
    dv_yy = hessian['v']['y']['y']
    continuity = du_x + dv_y
    x_momentum = du_t + C1.value * (u * du_x + v * du_y) + dp_x - C2.value * (du_xx + du_yy)
    y_momentum = dv_t + C1.value * (u * dv_x + v * dv_y) + dp_y - C2.value * (dv_xx + dv_yy)
    return [continuity, x_momentum, y_momentum]


# Define Spatio-temporal domain
# Rectangular
Lx_min, Lx_max = 1.0, 8.0
Ly_min, Ly_max = -2.0, 2.0
# Spatial domain: X × Y = [1, 8] × [−2, 2]
space_domain = pinnx.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
# Time domain: T = [0, 7]
time_domain = pinnx.geometry.TimeDomain(0, 7)
# Spatio-temporal domain
geomtime = pinnx.geometry.GeometryXTime(space_domain, time_domain)

# Get the training data: num = 7000
[ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=7000)
ob_xyt = np.hstack((ob_x, ob_y, ob_t))
observe_u = pinnx.icbc.PointSetBC(ob_xyt, ob_u, component=0)
observe_v = pinnx.icbc.PointSetBC(ob_xyt, ob_v, component=1)

# Training datasets and Loss
data = pinnx.data.TimePDE(
    geomtime,
    Navier_Stokes_Equation,
    [observe_u, observe_v],
    num_domain=700,
    num_boundary=200,
    num_initial=100,
    anchors=ob_xyt,
)

# Neural Network setup
layer_size = [3] + [50] * 6 + [3]
net = pinnx.nn.FNN(layer_size, "tanh")
model = pinnx.Trainer(data, net)

# callbacks for storing results
fnamevar = "variables.dat"
variable = pinnx.callbacks.VariableValue([C1, C2], period=100, filename=fnamevar)

# Compile, train and save trainer
model.compile(bst.optim.Adam(1e-3), external_trainable_variables=[C1, C2])
loss_history, train_state = model.train(
    iterations=10000, callbacks=[variable], display_every=1000, disregard_previous_best=True
)
pinnx.saveplot(loss_history, train_state, issave=True, isplot=True)
model.compile(bst.optim.Adam(1e-4), external_trainable_variables=[C1, C2])
loss_history, train_state = model.train(
    iterations=10000, callbacks=[variable], display_every=1000, disregard_previous_best=True
)
pinnx.saveplot(loss_history, train_state, issave=True, isplot=True)
# trainer.save(save_path = "./NS_inverse_model/trainer")
f = model.predict(ob_xyt, operator=Navier_Stokes_Equation)
print("Mean residual:", np.mean(np.absolute(f)))

# Plot Variables:
# reopen saved data using callbacks in fnamevar
lines = open(fnamevar, "r").readlines()
# read output data in fnamevar
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
l, c = Chat.shape
plt.semilogy(range(0, l * 100, 100), Chat[:, 0], "r-")
plt.semilogy(range(0, l * 100, 100), Chat[:, 1], "k-")
plt.semilogy(range(0, l * 100, 100), np.ones(Chat[:, 0].shape) * C1true, "r--")
plt.semilogy(range(0, l * 100, 100), np.ones(Chat[:, 1].shape) * C2true, "k--")
plt.legend(["C1hat", "C2hat", "True C1", "True C2"], loc="right")
plt.xlabel("Epochs")
plt.title("Variables")
plt.show()

# Plot the velocity distribution of the flow field:
for t in range(0, 8):
    [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=140000)
    xyt_pred = np.hstack((ob_x, ob_y, t * np.ones((len(ob_x), 1))))
    uvp_pred = model.predict(xyt_pred)
    x_pred, y_pred, t_pred = xyt_pred[:, 0], xyt_pred[:, 1], xyt_pred[:, 2]
    u_pred, v_pred, p_pred = uvp_pred[:, 0], uvp_pred[:, 1], uvp_pred[:, 2]
    x_true = ob_x[ob_t == t]
    y_true = ob_y[ob_t == t]
    u_true = ob_u[ob_t == t]
    fig, ax = plt.subplots(2, 1)
    cntr0 = ax[0].tricontourf(x_pred, y_pred, u_pred, levels=80, cmap="rainbow")
    cb0 = plt.colorbar(cntr0, ax=ax[0])
    cntr1 = ax[1].tricontourf(x_true, y_true, u_true, levels=80, cmap="rainbow")
    cb1 = plt.colorbar(cntr1, ax=ax[1])
    ax[0].set_title("u-PINN " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[0].set_ylabel("Y", fontsize=7.5, family="Arial")
    ax[1].set_title("u-Reference solution " + "(t=" + str(t) + ")", fontsize=9.5)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=7.5, family="Arial")
    ax[1].set_ylabel("Y", fontsize=7.5, family="Arial")
    fig.tight_layout()
    plt.show()
