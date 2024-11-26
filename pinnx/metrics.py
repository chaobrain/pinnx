import brainstate as bst
import brainunit as u
import numpy as np


def accuracy(y_true, y_pred):
    return u.math.mean(u.math.equal(u.math.argmax(y_pred, axis=-1),
                                    u.math.argmax(y_true, axis=-1)))


def l2_relative_error(y_true, y_pred):
    return u.linalg.norm(y_true - y_pred) / u.linalg.norm(y_true)


def nanl2_relative_error(y_true, y_pred):
    """Return the L2 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = u.math.nan_to_num(err)
    y_true = u.math.nan_to_num(y_true)
    return u.linalg.norm(err) / u.linalg.norm(y_true)


def mean_l2_relative_error(y_true, y_pred):
    """Compute the average of L2 relative error along the first axis."""
    return u.math.mean(
        u.linalg.norm(y_true - y_pred, axis=1) /
        u.linalg.norm(y_true, axis=1)
    )


def _absolute_percentage_error(y_true, y_pred):
    return 100 * u.math.abs(
        (y_true - y_pred) /
        np.clip(np.abs(y_true), np.finfo(bst.environ.dftype()).eps, None)
    )


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(_absolute_percentage_error(y_true, y_pred))


def max_absolute_percentage_error(y_true, y_pred):
    return np.amax(_absolute_percentage_error(y_true, y_pred))


def absolute_percentage_error_std(y_true, y_pred):
    return np.std(_absolute_percentage_error(y_true, y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "nanl2 relative error": nanl2_relative_error,
        "mean l2 relative error": mean_l2_relative_error,
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret metric function identifier:", identifier)
