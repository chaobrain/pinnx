``pinnx`` documentation
========================

`PINNx <https://github.com/chaobrain/pinnx>`_ is a library for scientific machine learning and physics-informed learning.
It is rewritten according to `DeepXDE <https://github.com/lululxvi/deepxde>`_ but is enhanced by our
`Brain Dynamics Programming (BDP) ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.
For example, it leverages

- `brainstate <https://brainstate.readthedocs.io/>`_ for just-in-time compilation,
- `brainunit <https://brainunit.readthedocs.io/>`_ for dimensional analysis,
- `braintools <https://braintools.readthedocs.io/>`_ for checkpointing, loss functions, and other utilities.


----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

        .. code-block:: bash

            pip install -U pinnx[cpu]

    .. tab-item:: GPU (CUDA 12.0)

        .. code-block:: bash

            pip install -U pinnx[cuda12]

    .. tab-item:: TPU

        .. code-block:: bash

            pip install -U pinnx[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

Quick Start
^^^^^^^^^^^
To be added.



User guide
^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    examples-unit.rst
    examples-unitless.rst

About PINNx
^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    about.rst


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^

We are building the `brain dynamics programming ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.

