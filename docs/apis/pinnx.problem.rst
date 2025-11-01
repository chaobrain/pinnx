``pinnx.problem`` module
========================

.. currentmodule:: pinnx.problem
.. automodule:: pinnx.problem

This module defines problem types for physics-informed neural networks, including PDEs, integro-differential
equations, and operator learning problems. Problems encapsulate the geometry, differential equations,
boundary/initial conditions, and neural network approximators.

Problem Interface
-----------------

Base class for all problem types.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Problem

Dataset Utilities
-----------------

Classes for managing training data and function spaces.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DataSet
   Function
   MfDataSet
   MfFunc
   TripleDataset
   TripleCartesianProd
   QuadrupleDataset
   QuadrupleCartesianProd

Differential Equation Problems
------------------------------

Problem classes for various types of differential equations.

**PDE & TimePDE**: For solving partial differential equations (steady-state and time-dependent).

**IDE**: For integro-differential equations.

**FPDE & TimeFPDE**: For fractional PDEs using fractional derivatives.

**PDEOperator**: For learning PDE operators (e.g., DeepONet).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   IDE
   PDE
   TimePDE
   FPDE
   TimeFPDE
   PDEOperator
   PDEOperatorCartesianProd
