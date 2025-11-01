``pinnx.fnspace`` module
========================

.. currentmodule:: pinnx.fnspace
.. automodule:: pinnx.fnspace

This module provides function spaces for operator learning and uncertainty quantification. Function spaces
represent infinite-dimensional spaces of functions, useful for generating training data and input functions.

Function Spaces
---------------

**wasserstein2**: Computes the 2-Wasserstein distance between distributions.

**FunctionSpace**: Base class for function spaces.

**PowerSeries**: Function space of power series expansions.

**Chebyshev**: Chebyshev polynomial function space.

**GRF**: Gaussian Random Field for 1D functions.

**GRF_KL**: Gaussian Random Field with Karhunen-Loève expansion.

**GRF2D**: Gaussian Random Field for 2D functions.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   wasserstein2
   FunctionSpace
   PowerSeries
   Chebyshev
   GRF
   GRF_KL
   GRF2D
