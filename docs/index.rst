``pinnx`` documentation
========================

`PINNx <https://github.com/chaobrain/pinnx>`_ is a library for scientific machine learning and physics-informed learning
in JAX. It enables to define PINN problem with explicit variables (e.g. ``x``, ``y``, ``z``) and physical units
(e.g. ``meter``, ``second``, ``kelvin``) and to solve the problem with neural networks.

`PINNx <https://github.com/chaobrain/pinnx>`_ is built on top of our `Brain Dynamics Programming (BDP) ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.
For example, it leverages `brainstate <https://brainstate.readthedocs.io/>`_ for just-in-time compilation,
`brainunit <https://brainunit.readthedocs.io/>`_ for dimensional analysis,
`braintools <https://braintools.readthedocs.io/>`_ for checkpointing, loss functions, and other utilities.


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


.. code-block:: python

    import brainstate as bst
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
            bst.init.KaimingUniform()
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
    trainer.compile(bst.optim.Adam(1e-3)).train(iterations=15000)
    trainer.compile(bst.optim.LBFGS(1e-3)).train(2000, display_every=500)
    trainer.saveplot(issave=True, isplot=True)



User guide
^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    about.rst


.. toctree::
    :maxdepth: 2

    unit-examples-forward.rst
    unit-examples-inverse.rst
    More Examples <https://github.com/chaobrain/pinnx/tree/main/docs>


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^

We are building the `brain dynamics programming ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

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