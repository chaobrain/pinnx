``pinnx.callbacks`` module
==========================

.. currentmodule:: pinnx.callbacks
.. automodule:: pinnx.callbacks

This module provides callback functions for monitoring and controlling the training process. Callbacks can be used
for model checkpointing, early stopping, logging, visualization, and adaptive training strategies.

Callbacks
---------

Classes for training monitoring and control.

**Callback**: Base class for all callbacks.

**CallbackList**: Container for managing multiple callbacks.

**ModelCheckpoint**: Saves model checkpoints during training.

**EarlyStopping**: Stops training when monitored metric stops improving.

**Timer**: Tracks and reports training time.

**DropoutUncertainty**: Estimates prediction uncertainty using dropout.

**VariableValue**: Monitors and logs variable values during training.

**OperatorPredictor**: Makes predictions with trained operator networks.

**MovieDumper**: Creates animations of solution evolution.

**PDEPointResampler**: Adaptively resamples collocation points based on residuals.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Callback
   CallbackList
   ModelCheckpoint
   EarlyStopping
   Timer
   DropoutUncertainty
   VariableValue
   OperatorPredictor
   MovieDumper
   PDEPointResampler

