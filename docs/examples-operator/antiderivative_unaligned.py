import brainstate
import braintools
import numpy as np

import pinnx


def load_unaligned(split, num_samples, rng):
    """Draws (function, location, value) triples from the aligned antiderivative dataset.

    In the unaligned format each input function is paired with a single location, so
    a sample is ``(v, x) -> y`` instead of the aligned ``v -> y`` on a shared grid.
    """
    d = np.load(f"../dataset/antiderivative_aligned_{split}.npz", allow_pickle=True)
    v, x = d["X"][0].astype(np.float32), d["X"][1].astype(np.float32)
    y = d["y"].astype(np.float32)
    i = rng.integers(0, v.shape[0], num_samples)
    j = rng.integers(0, x.shape[0], num_samples)
    return (v[i], x[j]), y[i, j][:, None]


# Load dataset
rng = np.random.default_rng(1234)
X_train, y_train = load_unaligned("train", 10000, rng)
X_test, y_test = load_unaligned("test", 10000, rng)

# Choose a network
m = 100
dim_x = 1
net = pinnx.nn.DeepONet(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
)

# problem
problem = pinnx.problem.TripleDataset(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    approximator=net,
)

# Define a Trainer
trainer = pinnx.Trainer(problem)
# Compile and Train
trainer.compile(braintools.optim.Adam(0.001)).train(iterations=10000)
# Plot the loss trajectory
trainer.saveplot(issave=True, isplot=True)
