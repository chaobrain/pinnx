# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import braintools


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
