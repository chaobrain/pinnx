``pinnx.utils`` module
======================

.. currentmodule:: pinnx.utils
.. automodule:: pinnx.utils

This module provides utility functions for array operations, data conversion, visualization, sampling,
and other helper functions used throughout PINNx.

Conversion Helpers
------------------

Functions for converting between dictionary and array representations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   array_to_dict
   dict_to_array

.. currentmodule:: pinnx.utils.array_ops

Array Operations
----------------

Utilities for tensor and array manipulation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   is_tensor
   istensorlist
   convert_to_array
   hstack
   zero_padding

.. currentmodule:: pinnx.utils

Training Utilities
------------------

Functions for training, visualization, and result management.

**apply**: Apply a function element-wise.

**standardize**: Standardize data to zero mean and unit variance.

**saveplot**: Save and plot training results.

**plot_loss_history**: Plot loss history over iterations.

**save_loss_history**: Save loss history to file.

**plot_best_state**: Plot best model state.

**save_best_state**: Save best model state.

**dat_to_csv**: Convert .dat files to .csv format.

**isclose**: Check if values are close within tolerance.

**smart_numpy**: Convert to NumPy array intelligently.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   apply
   standardize
   saveplot
   plot_loss_history
   save_loss_history
   plot_best_state
   save_best_state
   dat_to_csv
   isclose
   smart_numpy

.. currentmodule:: pinnx.utils.internal

Internal Helpers
----------------

Internal utility functions for PINNx operations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   timing
   merge_dict
   subdict
   check_not_none
   run_if_all_none
   run_if_any_none
   vectorize
   return_tensor
   to_numpy
   make_dict
   save_animation
   list_to_str

.. currentmodule:: pinnx.utils.losses

Loss Functions
--------------

Loss functions for training neural networks.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   mean_absolute_error
   mean_squared_error
   mean_l2_relative_error
   softmax_cross_entropy
   get_loss

.. currentmodule:: pinnx.utils.sampler

Sampling Helpers
----------------

Batch sampling utilities.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchSampler

.. currentmodule:: pinnx.utils.sampling

Sampling Strategies
-------------------

Point sampling strategies for training data generation.

**sample**: Generic sampling function.

**pseudorandom**: Pseudorandom sampling.

**quasirandom**: Quasi-random (low-discrepancy) sampling.

**InitialPointGenerator**: Generates initial training points.

**check_random_state**: Validates random state.

**check_dimension**: Validates dimension parameter.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   sample
   pseudorandom
   quasirandom
   InitialPointGenerator
   check_random_state
   check_dimension

.. currentmodule:: pinnx.utils.transformers

Transformers
------------

Data transformation and preprocessing utilities.

**Transformer**: Base transformer class.

**Identity**: Identity transformation.

**StringEncoder**: Encode strings to numerical values.

**LogN**: Logarithmic transformation.

**CategoricalEncoder**: Encode categorical variables.

**LabelEncoder**: Encode labels.

**Normalize**: Normalize data.

**Pipeline**: Chain multiple transformers.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Transformer
   Identity
   StringEncoder
   LogN
   CategoricalEncoder
   LabelEncoder
   Normalize
   Pipeline

.. currentmodule:: pinnx.utils
