# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================

from __future__ import annotations

import abc
from typing import Callable, Sequence, Any, Tuple

import brainstate
import jax

from pinnx.utils.losses import get_loss

Inputs = Any
Targets = Any
Auxiliary = Any
Outputs = Any
LOSS = jax.typing.ArrayLike

__all__ = [
    'Problem',
]


class Problem(abc.ABC):
    """
    Base Problem Class.

    A problem is defined by the approximator and the loss function.

    Attributes:
        approximator: The approximator.
        loss_fn: The loss function.
        loss_weights: A list specifying scalar coefficients (Python floats) to
            weight the loss contributions. The loss value that will be minimized by
            the trainer will then be the weighted sum of all individual losses,
            weighted by the `loss_weights` coefficients.
    """

    approximator: brainstate.nn.Module
    loss_fn: Callable | Sequence[Callable]

    def __init__(
        self,
        approximator: brainstate.nn.Module = None,
        loss_fn: str | Callable[[Inputs, Outputs], LOSS] = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        """
        Initialize the problem.

        Args:
            approximator: The approximator.
            loss_fn: The loss function. If the same loss is used for all errors,
                then `loss` is a String name of a loss function or a loss function.
                If different errors use different losses, then `loss` is a list
                whose size is equal to the number of errors.
            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the trainer will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients.
        """
        # approximator
        if approximator is not None:
            self.define_approximator(approximator)
        else:
            self.approximator = None

        # loss function
        self.loss_fn = get_loss(loss_fn)

        # loss weights
        if loss_weights is not None:
            assert isinstance(loss_weights, (list, tuple)), "loss_weights must be a list or tuple."
        self.loss_weights = loss_weights

    def define_approximator(
        self,
        approximator: brainstate.nn.Module,
    ) -> Problem:
        """
        Define the approximator.

        Args:
            approximator: The approximator.

        """
        assert isinstance(approximator, brainstate.nn.Module), "approximator must be an instance of brainstate.nn.Module."
        self.approximator = approximator
        return self

    def data_losses(self, targets, outputs):
        """
        Compare ``outputs`` with ``targets`` using ``loss_fn``, then apply ``loss_weights``.

        If ``loss_fn`` is a list, its entries are applied to the target components in order.
        ``loss_weights`` are matched to the target components in order (or to the single
        loss when the targets are an array).
        """
        if isinstance(self.loss_fn, (list, tuple)):
            if not isinstance(targets, dict) or len(self.loss_fn) != len(targets):
                raise ValueError(f"Got {len(self.loss_fn)} loss functions, which must match "
                                 f"the number of target components.")
            losses = {
                k: f({k: targets[k]}, {k: outputs[k]})[k]
                for f, k in zip(self.loss_fn, targets)
            }
        else:
            losses = self.loss_fn(targets, outputs)

        if self.loss_weights is not None:
            if isinstance(losses, dict):
                if len(self.loss_weights) != len(losses):
                    raise ValueError(f"Expected {len(losses)} loss weights, got {len(self.loss_weights)}.")
                losses = {k: w * v for (k, v), w in zip(losses.items(), self.loss_weights)}
            else:
                if len(self.loss_weights) != 1:
                    raise ValueError(f"Expected 1 loss weight, got {len(self.loss_weights)}.")
                losses = self.loss_weights[0] * losses
        return losses

    def losses(self, inputs, outputs, targets, **kwargs):
        """
        Return a list of losses, i.e., constraints.
        """
        raise NotImplementedError("Problem.losses is not implemented.")

    def losses_train(self, inputs, outputs, targets, **kwargs):
        """
        Return a list of losses for training dataset, i.e., constraints.
        """
        with brainstate.environ.context(fit=True):
            return self.losses(inputs, outputs, targets, **kwargs)

    def losses_test(self, inputs, outputs, targets, **kwargs):
        """
        Return a list of losses for test dataset, i.e., constraints.
        """
        with brainstate.environ.context(fit=False):
            return self.losses(inputs, outputs, targets, **kwargs)

    @abc.abstractmethod
    def train_next_batch(self, batch_size=None) -> Tuple[Inputs, Targets] | Tuple[Inputs, Targets, Auxiliary]:
        """
        Return a training dataset of the size `batch_size`.
        """

    @abc.abstractmethod
    def test(self) -> Tuple[Inputs, Targets] | Tuple[Inputs, Targets, Auxiliary]:
        """
        Return a test dataset.
        """
