``pinnx.grad`` module
=====================

.. currentmodule:: pinnx.grad
.. automodule:: pinnx.grad

This module provides automatic differentiation utilities for computing gradients, Jacobians, and Hessians
of neural network outputs with respect to inputs. These are essential for evaluating PDE residuals.

Differentiation Functions
--------------------------

**jacobian**: Computes the Jacobian matrix (first derivatives) of network outputs with respect to inputs.

**hessian**: Computes the Hessian matrix (second derivatives) of network outputs with respect to inputs.

**gradient**: Computes gradients using JAX's automatic differentiation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   jacobian
   hessian
   gradient

