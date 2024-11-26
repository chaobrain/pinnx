from typing import Sequence
import brainstate as bst
import numpy as np

from pinnx import utils
from .base import Problem


class DataSet(Problem):
    """Fitting Problem set.

    Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training output data.
        X_test (np.ndarray): Testing input data.
        y_test (np.ndarray): Testing output data.
        standardize (bool, optional): Standardize input data. Defaults to False.
    """

    def __init__(
        self,
        X_train: bst.typing.ArrayLike,
        y_train: bst.typing.ArrayLike,
        X_test: bst.typing.ArrayLike,
        y_test: bst.typing.ArrayLike,
        standardize: bool = False,
        approximator: bst.nn.Module = None,
        loss_fn: str = 'MSE',
        loss_weights: Sequence[float] = None,
    ):

        assert X_train.ndim == 2, "X_train must be 2D."
        assert y_train.ndim == 2, "y_train must be 2D."
        assert X_test.ndim == 2, "X_test must be 2D."
        assert y_test.ndim == 2, "y_test must be 2D."

        self.train_x = X_train.astype(bst.environ.dftype())
        self.train_y = y_train.astype(bst.environ.dftype())
        self.test_x = X_test.astype(bst.environ.dftype())
        self.test_y = y_test.astype(bst.environ.dftype())

        self.scaler_x = None
        if standardize:
            self.scaler_x, self.train_x, self.test_x = utils.standardize(self.train_x, self.test_x)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y

    def test(self):
        return self.test_x, self.test_y

    def transform_inputs(self, x):
        if self.scaler_x is None:
            return x
        return self.scaler_x.transform(x)
