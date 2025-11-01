``pinnx.nn`` module
===================

.. currentmodule:: pinnx.nn
.. automodule:: pinnx.nn

This module provides neural network architectures for physics-informed learning, including feedforward networks,
DeepONet for operator learning, and utilities for converting between array and dictionary representations.

Model Interfaces
----------------

Core classes for building and managing neural network models.

**NN**: Base neural network interface.

**Model**: Wrapper for combining multiple network components.

**DictToArray**: Converts dictionary inputs (with units) to arrays.

**ArrayToDict**: Converts array outputs back to dictionaries (with units).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   NN
   Model
   DictToArray
   ArrayToDict

Neural Network Architectures
----------------------------

Network architectures for function approximation and operator learning.

**FNN**: Fully-connected feedforward neural network.

**PFNN**: Parallel feedforward neural network with multiple outputs.

**DeepONet**: Deep Operator Network for learning nonlinear operators.

**DeepONetCartesianProd**: DeepONet with Cartesian product structure.

**PODDeepONet**: Proper Orthogonal Decomposition-enhanced DeepONet.

**MIONetCartesianProd**: Multiple-input operator network.

**PODMIONet**: POD-enhanced multiple-input operator network.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FNN
   PFNN
   DeepONet
   DeepONetCartesianProd
   PODDeepONet
   MIONetCartesianProd
   PODMIONet
