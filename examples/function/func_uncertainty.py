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
num_train = 100
num_test = 1000
data = pinnx.data.Function(geom, func, num_train, num_test)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh", bst.init.KaimingUniform())

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
uncertainty = pinnx.callbacks.DropoutUncertainty(period=1000)
losshistory, train_state = model.train(iterations=30000, callbacks=uncertainty)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
