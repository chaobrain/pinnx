"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import brainstate as bst

import pinnx

fname_train = "../dataset/dataset.train"
fname_test = "../dataset/dataset.test"

data = pinnx.data.DataSet(
    fname_train=fname_train,
    fname_test=fname_test,
    col_x=(0,),
    col_y=(1,),
    standardize=True,
)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh", bst.init.KaimingUniform())

model = pinnx.Model(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=50000)

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)
