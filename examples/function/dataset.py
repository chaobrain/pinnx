import brainstate as bst
import numpy as np

import pinnx

train_data = np.loadtxt("../../docs/dataset/dataset.train")
test_data = np.loadtxt("../../docs/dataset/dataset.test")

data = pinnx.problem.DataSet(
    X_train=train_data[:, (0,)],
    y_train=train_data[:, (1,)],
    X_test=test_data[:, (0,)],
    y_test=test_data[:, (1,)],
    standardize=True,
)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh", bst.init.KaimingUniform())

model = pinnx.Trainer(data)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=50000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
