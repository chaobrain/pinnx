PINNx: Physics-Informed Neural Networks for Scientific Machine Learning in JAX
================================================================================

.. image:: https://github.com/chaobrain/pinnx/actions/workflows/build.yml/badge.svg
   :target: https://github.com/chaobrain/pinnx/actions/workflows/build.yml
   :alt: Build Status

.. image:: https://readthedocs.org/projects/pinnx/badge/?version=latest
   :target: https://pinnx.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://badge.fury.io/py/pinnx.svg
   :target: https://badge.fury.io/py/pinnx
   :alt: PyPI Version

.. image:: https://img.shields.io/github/license/chaobrain/pinnx
   :target: https://github.com/chaobrain/pinnx/blob/master/LICENSE
   :alt: License


``PINNx`` is a library for scientific machine learning and physics-informed learning in JAX.
It is a rewrite of `DeepXDE <https://github.com/lululxvi/deepxde>`_ but is enhanced by our
`brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.

For example, it leverages

- `brainstate <https://brainstate.readthedocs.io/>`_ for just-in-time compilation,
- `brainunit <https://brainunit.readthedocs.io/>`_ for dimensional analysis,
- `braintools <https://braintools.readthedocs.io/>`_ for checkpointing, loss functions, and other utilities.


----


Installation
^^^^^^^^^^^^

Install the stable version with ``pip``:

.. code-block:: bash

   pip install pinnx --upgrade


Install ``pinnx`` on CPU or GPU with JAX:

.. tab-set::

    .. tab-item:: CPU

        .. code-block:: bash

            pip install pinnx[cpu]

    .. tab-item:: CUDA 12

        .. code-block:: bash

            pip install pinnx[cuda12]

    .. tab-item:: CUDA 13

        .. code-block:: bash

            pip install pinnx[cuda13]

    .. tab-item:: TPU

        .. code-block:: bash

            pip install pinnx[tpu]


Quick Start
^^^^^^^^^^^

Define a PINN with explicit variables and physical units.

.. code-block:: python

    import braintools
    import brainunit as u
    import pinnx

    # geometry
    geometry = pinnx.geometry.GeometryXTime(
        geometry=pinnx.geometry.Interval(-1, 1.),
        timedomain=pinnx.geometry.TimeDomain(0, 0.99)
    ).to_dict_point(x=u.meter, t=u.second)

    uy = u.meter / u.second
    v = 0.01 / u.math.pi * u.meter ** 2 / u.second

    # boundary conditions
    bc = pinnx.icbc.DirichletBC(lambda x: {'y': 0. * uy})
    ic = pinnx.icbc.IC(lambda x: {'y': -u.math.sin(u.math.pi * x['x'] / u.meter) * uy})

    # PDE equation
    def pde(x, y):
        jacobian = approximator.jacobian(x)
        hessian = approximator.hessian(x)
        dy_x = jacobian['y']['x']
        dy_t = jacobian['y']['t']
        dy_xx = hessian['y']['x']['x']
        residual = dy_t + y['y'] * dy_x - v * dy_xx
        return residual

    # neural network
    approximator = pinnx.nn.Model(
        pinnx.nn.DictToArray(x=u.meter, t=u.second),
        pinnx.nn.FNN(
            [geometry.dim] + [20] * 3 + [1],
            "tanh",
            braintools.init.KaimingUniform()
        ),
        pinnx.nn.ArrayToDict(y=uy)
    )

    # problem
    problem = pinnx.problem.TimePDE(
        geometry,
        pde,
        [bc, ic],
        approximator,
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
    )

    # training
    trainer = pinnx.Trainer(problem)
    trainer.compile(braintools.optim.Adam(1e-3)).train(iterations=15000)
    trainer.compile(braintools.optim.LBFGS(1e-3)).train(2000, display_every=500)
    trainer.saveplot(issave=True, isplot=True)


----



.. toctree::
    :maxdepth: 1

    forward_examples.md
    foward_unitless_examples.md
    inverse_examples.md
    More Examples <https://github.com/chaobrain/pinnx/tree/main/docs>
    about.rst




See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``pinnx`` is one part of our `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   changelog.md
   apis/pinnx.rst
   apis/pinnx.callbacks.rst
   apis/pinnx.fnspace.rst
   apis/pinnx.grad.rst
   apis/pinnx.geometry.rst
   apis/pinnx.icbc.rst
   apis/pinnx.metrics.rst
   apis/pinnx.nn.rst
   apis/pinnx.problem.rst
   apis/pinnx.utils.rst
