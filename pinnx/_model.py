import functools
from typing import Union, Sequence, Callable, Optional

import brainstate as bst
import braintools
import brainunit as u
import numpy as np

from pinnx._display import training_display
from . import metrics as metrics_module
from . import utils
from .callbacks import CallbackList, Callback
from .data.data import Data

__all__ = [
    "LossHistory",
    "Model",
    "TrainState"
]


def mean_absolute_error(y_true, y_pred):
    return braintools.metric.absolute_error(y_true, y_pred).mean()


def mean_squared_error(y_true, y_pred):
    return braintools.metric.squared_error(y_true, y_pred).mean()


def mean_l2_relative_error(y_true, y_pred):
    return braintools.metric.l2_norm(y_pred, y_true).mean()


def softmax_cross_entropy(y_true, y_pred):
    return braintools.metric.softmax_cross_entropy(y_pred, y_true).mean()


LOSS_DICT = {
    # mean absolute error
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,

    # mean squared error
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,

    # mean l2 relative error
    "mean l2 relative error": mean_l2_relative_error,

    # softmax cross entropy
    "softmax cross entropy": softmax_cross_entropy,
}


def get_loss(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get_loss, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)


class Model:
    """
    A ``Model`` trains a ``NN`` on a ``Data``.

    Args:
        data: ``pinnx.data.Data`` instance.
        net: ``pinnx.nn.NN`` instance.
        external_trainable_variables: A trainable ``pinnx.Variable`` object or a list
                of trainable ``pinnx.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered.
    """
    __module__ = 'pinnx'
    optimizer: bst.optim.Optimizer  # optimizer
    net: bst.nn.Module  # neural network
    data: Data  # data
    params: bst.util.FlattedDict  # trainable variables

    def __init__(
        self,
        data: Data,
        net: bst.nn.Module,
        external_trainable_variables: Union[bst.ParamState, Sequence[bst.ParamState]] = None,
    ):
        self.net = net
        assert isinstance(self.net, bst.nn.Module), "net must be a Module instance."
        self.data = data
        assert isinstance(self.data, Data), "data must be a Data instance."

        self.metrics = None
        self.train_state = TrainState()
        self.loss_history = LossHistory()
        self.stop_training = False

        # external trainable variables
        params = self.net.states(bst.ParamState)
        if external_trainable_variables is None:
            external_trainable_variables = []
        else:
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
        for i, var in enumerate(external_trainable_variables):
            assert isinstance(var, bst.ParamState), ("external_trainable_variables must be a "
                                                     "list of ParamState instance.")
            params[('external_trainable_variable', i)] = var
        self.params = params

    @utils.timing
    def compile(
        self,
        optimizer: bst.optim.Optimizer,
        loss: str = "MSE",
        metrics: Union[str, Sequence[str]] = None,
        loss_weights: Sequence = None,
    ):
        """
        Configures the model for training.

        Args:
            optimizer: String name of an optimizer, or an optimizer class instance.
            loss: If the same loss is used for all errors, then `loss` is a String name
                of a loss function or a loss function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients.

        """
        print("Compiling model...")

        # loss function
        loss_fn = get_loss(loss)
        loss_weights = u.math.asarray(loss_weights) if loss_weights is not None else loss_weights

        # optimizer
        assert isinstance(optimizer, bst.optim.Optimizer), "optimizer must be an Optimizer instance."
        self.optimizer = optimizer
        self.optimizer.register_trainable_weights(self.params)

        # metrics may use model variables such as self.net,
        # and thus are instantiated after compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

        def fn_outputs(training: bool, inputs):
            with bst.environ.context(fit=training):
                if isinstance(inputs, tuple):
                    inputs = tuple(map(lambda x: u.math.asarray(x), inputs))
                else:
                    inputs = u.math.asarray(inputs)
                return self.net(inputs)

        def fn_outputs_losses(training, inputs, targets):
            with bst.environ.context(fit=training):
                if isinstance(inputs, tuple):
                    inputs = tuple(map(lambda x: u.math.asarray(x), inputs))
                else:
                    inputs = u.math.asarray(inputs)
                outputs_ = self.net(inputs)
                # Data losses
                if targets is not None:
                    targets = u.math.asarray(targets)
                if training:
                    losses = self.data.losses_train(targets, outputs_, loss_fn, inputs, self)
                else:
                    losses = self.data.losses_test(targets, outputs_, loss_fn, inputs, self)
                if not isinstance(losses, list):
                    losses = [losses]
                losses = u.math.stack(losses)
                # Weighted losses
                if loss_weights is not None:
                    losses *= loss_weights
                return outputs_, losses

        def fn_outputs_losses_train(inputs, targets):
            return fn_outputs_losses(True, inputs, targets)

        def fn_outputs_losses_test(inputs, targets):
            return fn_outputs_losses(False, inputs, targets)

        def fn_train_step(inputs, targets):
            grads = bst.augment.grad(
                lambda: u.math.sum(fn_outputs_losses_train(inputs, targets)[1]),
                grad_states=self.params
            )()
            self.optimizer.update(grads)

        # Callables
        self.fn_outputs = bst.compile.jit(fn_outputs, static_argnums=0)
        self.fn_outputs_losses_train = bst.compile.jit(fn_outputs_losses_train)
        self.fn_outputs_losses_test = bst.compile.jit(fn_outputs_losses_test)
        self.fn_train_step = bst.compile.jit(fn_train_step)

    def _outputs(self, training, inputs):
        outs = self.fn_outputs(training, inputs)
        return utils.to_numpy(outs)

    def _outputs_losses(self, training, inputs, targets):
        outs = (self.fn_outputs_losses_train(inputs, targets)
                if training else
                self.fn_outputs_losses_test(inputs, targets))
        return utils.to_numpy(outs[0]), utils.to_numpy(outs[1])

    @utils.timing
    def train(
        self,
        iterations: int,
        batch_size: int = None,
        display_every: int = 1000,
        disregard_previous_best: bool = False,
        callbacks: Union[Callback, Sequence[Callback]] = None,
        model_restore_path: str = None,
        model_save_path: str = None,
    ):
        """
        Trains the model.

        Args:
            iterations (Integer): Number of iterations to train the model, i.e., number
                of times the network weights are updated.
            batch_size: Integer, tuple, or ``None``.

                - If you solve PDEs via ``pinnx.data.PDE`` or ``pinnx.data.TimePDE``, do not use `batch_size`,
                  and instead use `pinnx.callbacks.PDEPointResampler
                  <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEPointResampler>`_,
                  see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_resample.py>`_.
                - For DeepONet in the format of Cartesian product, if `batch_size` is an Integer,
                  then it is the batch size for the branch input;
                  if you want to also use mini-batch for the trunk net input,
                  set `batch_size` as a tuple, where the fist number is the batch size for the branch net input
                  and the second number is the batch size for the trunk net input.
            display_every (Integer): Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``pinnx.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path (String): Path where parameters were previously saved.
            model_save_path (String): Prefix of filenames created for the checkpoint.
        """

        if self.metrics is None:
            raise ValueError("Compile the model before training.")

        # callbacks
        callbacks = CallbackList(callbacks=[callbacks] if isinstance(callbacks, Callback) else callbacks)
        callbacks.set_model(self)

        # disregard previous best
        if disregard_previous_best:
            self.train_state.disregard_best()

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")

        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test()
        callbacks.on_train_begin()
        self._train(iterations, display_every, batch_size, callbacks)
        callbacks.on_train_end()

        print("")
        training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.loss_history, self.train_state

    def _train(self, iterations, display_every, batch_size, callbacks):
        for i in range(iterations):
            callbacks.on_epoch_begin()
            callbacks.on_batch_begin()

            # get data
            self.train_state.set_data_train(*self.data.train_next_batch(batch_size))

            # train one batch
            self.fn_train_step(
                self.train_state.X_train,
                self.train_state.y_train,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            callbacks.on_batch_end()
            callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _test(self):
        # evaluate the training data
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
        )

        # evaluate the test data
        (
            self.train_state.y_pred_test,
            self.train_state.loss_test
        ) = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
        )

        # metrics
        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.loss_history.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True

        training_display(self.train_state)

    def predict(
        self,
        x,
        operator: Optional[Callable] = None,
        callbacks: Union[Callback, Sequence[Callback]] = None,
    ):
        """Generates predictions for the input samples. If `operator` is ``None``,
        returns the network output, otherwise returns the output of the `operator`.

        Args:
            x: The network inputs. A Numpy array or a tuple of Numpy arrays.
            operator: A function takes arguments (`neural_net`, `inputs`) and outputs a tensor. `inputs` and
                `outputs` are the network input and output tensors, respectively. `operator` is typically
                chosen as the PDE (used to define `pinnx.data.PDE`) to predict the PDE residual.
            callbacks: List of ``pinnx.callbacks.Callback`` instances. List of callbacks
                to apply during prediction.
        """
        if isinstance(x, tuple):
            x = tuple(u.math.asarray(xi, dtype=bst.environ.dftype()) for xi in x)
        else:
            x = u.math.asarray(x, dtype=bst.environ.dftype())

        callbacks = CallbackList(callbacks=[callbacks] if isinstance(callbacks, Callback) else callbacks)
        callbacks.set_model(self)
        callbacks.on_predict_begin()
        if operator is not None:
            y = operator(functools.partial(self.fn_outputs, False), x)
        else:
            y = self._outputs(False, x)
        callbacks.on_predict_end()
        return y

    def save(self, save_path, verbose: int = 0):
        """Saves all variables to a disk file.

        Args:
            save_path (string): Prefix of filenames to save the model file.
            verbose (int): Verbosity mode, 0 or 1.

        Returns:
            string: Path where model is saved.
        """
        import braintools

        # save path
        save_path = f"{save_path}-{self.train_state.epoch}.msgpack"

        # avoid the duplicate ParamState save
        data = bst.graph.Dict(model=self.net, optimizer=self.optimizer)
        checkpoint = bst.graph.states(data).to_nest()

        braintools.file.msgpack_save(save_path, checkpoint)

        if verbose > 0:
            print(
                "Epoch {}: saving model to {} ...\n".format(
                    self.train_state.epoch, save_path
                )
            )
        return save_path

    def restore(self, save_path, verbose: int = 0):
        """Restore all variables from a disk file.

        Args:
            save_path (string): Path where model was previously saved.
            verbose (int): Verbosity mode, 0 or 1.
        """
        import braintools
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))

        data = bst.graph.Dict(model=self.net, optimizer=self.optimizer)
        checkpoint = bst.graph.states(data).to_nest()
        braintools.file.msgpack_load(save_path, target=checkpoint)


class TrainState:
    __module__ = 'pinnx'

    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Test results
        self.loss_test = None
        self.y_pred_test = None
        self.y_std_test = None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train = np.inf
        self.best_loss_test = np.inf
        self.best_y = None
        self.best_ystd = None
        self.best_metrics = None

    def set_data_train(self, X_train, y_train, *args):
        self.X_train = X_train
        self.y_train = y_train

    def set_data_test(self, X_test, y_test, *args):
        self.X_test = X_test
        self.y_test = y_test

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf


class LossHistory:
    __module__ = 'pinnx'

    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)


class ZCSModel(Model):
    """Derived `Model` class for ZCS support."""

    def __init__(self, data, net):
        super().__init__(data, net)
        # store ZCS parameters, sent to user for PDE calculation
        self.zcs_parameters = None

    def _compile_pytorch(self, lr, loss_fn, decay):
        """pytorch"""
        super()._compile_pytorch(lr, loss_fn, decay)

        def process_inputs_zcs(inputs):
            # get inputs
            branch_inputs, trunk_inputs = inputs

            # convert to tensors with grad disabled
            branch_inputs = u.math.asarray(branch_inputs)
            trunk_inputs = u.math.asarray(trunk_inputs)

            # create ZCS scalars
            n_dim_crds = trunk_inputs.shape[1]
            zcs_scalars = [
                u.math.asarray(0.0).requires_grad_() for _ in range(n_dim_crds)
            ]

            # add ZCS to truck inputs
            zcs_vector = u.math.stack(zcs_scalars)
            trunk_inputs = trunk_inputs + zcs_vector[None, :]

            # return inputs and ZCS scalars
            return (branch_inputs, trunk_inputs), {"leaves": zcs_scalars}

        def outputs_losses_zcs(training, inputs, targets, auxiliary_vars, losses_fn):
            # aux
            self.net.auxiliary_vars = None
            if auxiliary_vars is not None:
                self.net.auxiliary_vars = u.math.asarray(auxiliary_vars)

            # inputs
            inputs, self.zcs_parameters = process_inputs_zcs(inputs)

            # forward
            self.net.train(mode=training)
            outputs_ = self.net(inputs)

            # losses
            if targets is not None:
                targets = u.math.asarray(targets)
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            losses = u.math.stack(losses)

            # TODO: regularization

            # weighted
            if self.loss_weights is not None:
                losses *= u.math.asarray(self.loss_weights)

            return outputs_, losses

        def outputs_losses_train_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        def train_step_zcs(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
                total_loss = u.math.sum(losses)
                return total_loss

            self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # overwrite callables
        self.outputs_losses_train = outputs_losses_train_zcs
        self.outputs_losses_test = outputs_losses_test_zcs
        self.train_step = train_step_zcs
