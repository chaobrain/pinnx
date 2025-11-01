``pinnx.icbc`` module
=====================

.. currentmodule:: pinnx.icbc
.. automodule:: pinnx.icbc

This module provides initial conditions (ICs) and boundary conditions (BCs) for constraining PDE solutions.
Boundary conditions can be specified as Dirichlet, Neumann, Robin, periodic, or operator-based constraints.

Interfaces
----------

Base classes for initial and boundary conditions.

**ICBC**: Base class for all initial and boundary conditions.

**BC**: Base class specifically for boundary conditions.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ICBC
   BC

Boundary Conditions
-------------------

Various types of boundary conditions for PDEs.

**DirichletBC**: Specifies the value of the solution on the boundary (u = g).

**NeumannBC**: Specifies the normal derivative on the boundary (∂u/∂n = g).

**RobinBC**: Linear combination of value and derivative (a·u + b·∂u/∂n = g).

**PeriodicBC**: Periodic boundary conditions (u(x_left) = u(x_right)).

**OperatorBC**: General operator-based boundary condition.

**PointSetBC**: Boundary condition at specific points.

**PointSetOperatorBC**: Operator boundary condition at specific points.

**Interface2DBC**: Boundary condition at 2D interfaces.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DirichletBC
   NeumannBC
   RobinBC
   PeriodicBC
   OperatorBC
   PointSetBC
   PointSetOperatorBC
   Interface2DBC

Initial Conditions
------------------

Initial conditions for time-dependent problems.

**IC**: Specifies the initial state of the solution (u(x, t=0) = g(x)).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   IC
