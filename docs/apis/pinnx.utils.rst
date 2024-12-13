``pinnx.utils`` module
======================

.. currentmodule:: pinnx.utils 
.. automodule:: pinnx.utils 

Dict and Array Converters
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   array_to_dict
   dict_to_array


Display training progress.
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   pformat
   tree_repr
   TrainingDisplay
   training_display


Array Operations
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   is_tensor
   istensorlist
   convert_to_array
   hstack
   zero_padding
   Sequence


External Functions
------------------

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
   Axes3D
   Pool


Internal Functions
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   wraps
   apply
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
   tree_repr
   get_num_args
   get_activation
   Callable
   Union


Loss Functions
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   mean_absolute_error
   mean_squared_error
   mean_l2_relative_error
   softmax_cross_entropy
   get_loss
   LOSS_DICT


Sampler
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchSampler


Sampling
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   sample


Transformer
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LabelBinarizer
   Transformer
   Identity
   StringEncoder
   LogN
   CategoricalEncoder
   LabelEncoder
   Normalize
   Pipeline


