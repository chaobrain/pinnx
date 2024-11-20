import brainstate as bst
import numpy as np

from pinnx import utils
from .data import Data


class DataSet(Data):
    """Fitting Data set.

    Args:
        col_x: List of integers.
        col_y: List of integers.
    """

    def __init__(
        self,
        X_train: bst.typing.ArrayLike = None,
        y_train: bst.typing.ArrayLike = None,
        X_test: bst.typing.ArrayLike = None,
        y_test: bst.typing.ArrayLike = None,
        fname_train: str = None,
        fname_test: str = None,
        col_x=None,
        col_y=None,
        standardize: bool = False,
    ):
        if X_train is not None:

            assert y_train is not None, "y_train is None."
            assert X_test is not None, "X_test is None."
            assert y_test is not None, "y_test is None."

            self.train_x = X_train.astype(bst.environ.dftype())
            self.train_y = y_train.astype(bst.environ.dftype())
            self.test_x = X_test.astype(bst.environ.dftype())
            self.test_y = y_test.astype(bst.environ.dftype())

        elif fname_train is not None:

            assert fname_test is not None, "fname_test is None."

            train_data = np.loadtxt(fname_train)
            self.train_x = train_data[:, col_x].astype(bst.environ.dftype())
            self.train_y = train_data[:, col_y].astype(bst.environ.dftype())
            test_data = np.loadtxt(fname_test)
            self.test_x = test_data[:, col_x].astype(bst.environ.dftype())
            self.test_y = test_data[:, col_y].astype(bst.environ.dftype())

        else:
            raise ValueError("No training data.")

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
