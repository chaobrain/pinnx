import brainstate as bst
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import pinnx


def pde(neu, x):
    x = pinnx.array_to_dict(x, ["x"])
    approx = lambda x: pinnx.array_to_dict(neu(pinnx.dict_to_array(x)), ["u"])
    hessian = pinnx.grad.hessian(approx, x)
    dy_xx = hessian["u"]["x"]["x"]
    return -dy_xx - np.pi ** 2 * u.math.sin(np.pi * x['x'])


def boundary(x, on_boundary):
    return on_boundary


def func(x):
    return np.sin(np.pi * x)


geom = pinnx.geometry.Interval(-1, 1)
bc = pinnx.icbc.DirichletBC(geom, func, boundary)
data = pinnx.data.PDE(geom, pde, bc, 16, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [1]
net = pinnx.nn.FNN(layer_size, "tanh")

model = pinnx.Trainer(data, net)
model.compile(bst.optim.Adam(0.001), metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=10000)
# Optional: Save the trainer during training.
# checkpointer = pinnx.callbacks.ModelCheckpoint(
#     "trainer/trainer", verbose=1, save_better_only=True
# )
# Optional: Save the movie of the network solution during training.
# ImageMagick (https://imagemagick.org/) is required to generate the movie.
# movie = pinnx.callbacks.MovieDumper(
#     "trainer/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
# )
# loss_history, train_state = trainer.train(iterations=10000, callbacks=[checkpointer, movie])

pinnx.saveplot(losshistory, train_state, issave=True, isplot=True)

# Optional: Restore the saved trainer with the smallest training loss
# trainer.restore(f"trainer/trainer-{train_state.best_step}.ckpt", verbose=1)
# Plot PDE residual
x = geom.uniform_points(1000, True)
y = model.predict(x, operator=pde)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()
