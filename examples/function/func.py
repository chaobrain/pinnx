import brainstate as bst
import numpy as np

import pinnx


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return x * np.sin(5 * x)


geom = pinnx.geometry.Interval(-1, 1)
num_train = 160
num_test = 100
data = pinnx.data.Function(geom, func, num_train, num_test)

net = pinnx.nn.FNN([1] + [20] * 3 + [1], "tanh", bst.init.LecunUniform())

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

pinnx.saveplot(losshistory, train_state, issave=False, isplot=True)
