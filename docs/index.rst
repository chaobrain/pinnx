``pinnx`` documentation
========================

`PINNx <https://github.com/chaobrain/pinnx>`_ is a library for scientific machine learning and physics-informed learning.

It is rewritten according to `DeepXDE <https://github.com/lululxvi/deepxde>`_ but is enhanced by our
`Brain Dynamics Programming (BDP) ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.
For example, it leverages
`brainstate <https://brainstate.readthedocs.io/>`_ for just-in-time compilation,
`brainunit <https://brainunit.readthedocs.io/>`_ for dimensional analysis,
`braintools <https://braintools.readthedocs.io/>`_ for checkpointing, loss functions, and other utilities.


- Solving different problems using PINN
    - solving forward/inverse ordinary/partial differential equations (ODEs/PDEs) [`SIAM Rev. <https://doi.org/10.1137/19M1274067>`_]
    - solving forward/inverse integro-differential equations (IDEs) [`SIAM Rev. <https://doi.org/10.1137/19M1274067>`_]
    - fPINN: solving forward/inverse fractional PDEs (fPDEs) [`SIAM J. Sci. Comput. <https://doi.org/10.1137/18M1229845>`_]
    - NN-arbitrary polynomial chaos (NN-aPC): solving forward/inverse stochastic PDEs (sPDEs) [`J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2019.07.048>`_]
    - PINN with hard constraints (hPINN): solving inverse design/topology optimization [`SIAM J. Sci. Comput. <https://doi.org/10.1137/21M1397908>`_]
- Improving PINN accuracy
    - residual-based adaptive sampling [`SIAM Rev. <https://doi.org/10.1137/19M1274067>`_, `Comput. Methods Appl. Mech. Eng. <https://doi.org/10.1016/j.cma.2022.115671>`_]
    - gradient-enhanced PINN (gPINN) [`Comput. Methods Appl. Mech. Eng. <https://doi.org/10.1016/j.cma.2022.114823>`_]
    - PINN with multi-scale Fourier features [`Comput. Methods Appl. Mech. Eng. <https://doi.org/10.1016/j.cma.2021.113938>`_]
- (physics-informed) deep operator network (DeepONet)
    - DeepONet: learning operators [`Nat. Mach. Intell. <https://doi.org/10.1038/s42256-021-00302-5>`_]
    - DeepONet extensions, e.g., POD-DeepONet [`Comput. Methods Appl. Mech. Eng. <https://doi.org/10.1016/j.cma.2022.114778>`_]
    - MIONet: learning multiple-input operators [`SIAM J. Sci. Comput. <https://doi.org/10.1137/22M1477751>`_]
    - Fourier-DeepONet [`Comput. Methods Appl. Mech. Eng. <https://doi.org/10.1016/j.cma.2023.116300>`_], Fourier-MIONet [`arXiv <https://arxiv.org/abs/2303.04778>`_]
    - physics-informed DeepONet [`Sci. Adv. <https://doi.org/10.1126/sciadv.abi8605>`_]
    - multifidelity DeepONet [`Phys. Rev. Research <https://doi.org/10.1103/PhysRevResearch.4.023210>`_]
    - DeepM&Mnet: solving multiphysics and multiscale problems [`J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2021.110296>`_, `J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2021.110698>`_]
    - Reliable extrapolation [`Comput. Methods Appl. Mech. Eng. <https://doi.org/10.1016/j.cma.2023.116064>`_]
- multifidelity neural network (MFNN)
    - learning from multifidelity data [`J. Comput. Phys. <https://doi.org/10.1016/j.jcp.2019.109020>`_, `PNAS <https://doi.org/10.1073/pnas.1922210117>`_]


Features
--------

PINNx has implemented many algorithms as shown above and supports many features:

- enables the user code to be compact, resembling closely the mathematical formulation.
- **complex domain geometries** without tyranny mesh generation. The primitive geometries are interval, triangle, rectangle, polygon, disk, ellipse, star-shaped, cuboid, sphere, hypercube, and hypersphere. Other geometries can be constructed as constructive solid geometry (CSG) using three boolean operations: union, difference, and intersection. PINNx also supports a geometry represented by a point cloud.
- 5 types of **boundary conditions** (BCs): Dirichlet, Neumann, Robin, periodic, and a general BC, which can be defined on an arbitrary domain or on a point set; and approximate distance functions for **hard constraints**.
- 3 **automatic differentiation** (AD) methods to compute derivatives: reverse mode (i.e., backpropagation), forward mode, and zero coordinate shift (ZCS).
- different **neural networks**: fully connected neural network (FNN), stacked FNN, residual neural network, (spatio-temporal) multi-scale Fourier feature networks, etc.
- many **sampling methods**: uniform, pseudorandom, Latin hypercube sampling, Halton sequence, Hammersley sequence, and Sobol sequence. The training points can keep the same during training or be resampled (adaptively) every certain iterations.
- 4 **function spaces**: power series, Chebyshev polynomial, Gaussian random field (1D/2D).
- **data-parallel training** on multiple GPUs.
- different **optimizers**: Adam, L-BFGS, etc.
- conveniently **save** the model during training, and **load** a trained model.
- **callbacks** to monitor the internal states and statistics of the model during training: early stopping, etc.
- **uncertainty quantification** using dropout.
- **float16**, **float32**, and **float64**.
- many other useful features: different (weighted) losses, learning rate schedules, metrics, etc.


User guide
----------

.. toctree::
  :maxdepth: 1

  examples-function.rst
  examples-pinn-forward.rst
  examples-pinn-inverse.rst
  examples-operator.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
