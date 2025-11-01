``pinnx.metrics`` module
========================

.. currentmodule:: pinnx.metrics
.. automodule:: pinnx.metrics

This module provides metrics for evaluating model performance and solution accuracy. These metrics can be used
to assess how well the neural network approximates the true solution.

Metrics
-------

**accuracy**: Classification accuracy.

**l2_relative_error**: Relative L2 error between predicted and true values.

**nanl2_relative_error**: L2 relative error ignoring NaN values.

**mean_l2_relative_error**: Mean relative L2 error across multiple outputs.

**mean_squared_error**: Mean squared error (MSE).

**mean_absolute_percentage_error**: Mean absolute percentage error (MAPE).

**max_absolute_percentage_error**: Maximum absolute percentage error.

**absolute_percentage_error_std**: Standard deviation of absolute percentage error.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   accuracy
   l2_relative_error
   nanl2_relative_error
   mean_l2_relative_error
   mean_squared_error
   mean_absolute_percentage_error
   max_absolute_percentage_error
   absolute_percentage_error_std
