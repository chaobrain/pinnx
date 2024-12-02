# PINNx: Physics-Informed Neural Networks for Scientific Machine Learning in JAX


<p align="center">
  	<img alt="Header image of pinnx." src="https://github.com/chaobrain/pinnx/blob/main/docs/_static/pinnx.png" width=50%>
</p> 


[![Build Status](https://github.com/chaobrain/pinnx/actions/workflows/build.yml/badge.svg)](https://github.com/chaobrain/pinnx/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/pinnx/badge/?version=latest)](https://pinnx.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/pinnx.svg)](https://badge.fury.io/py/pinnx)
[![License](https://img.shields.io/github/license/chaobrain/pinnx)](https://github.com/chaobrain/pinnx/blob/master/LICENSE)

``PINNx`` is a library for scientific machine learning and physics-informed learning. 
It is rewritten according to [DeepXDE](https://github.com/lululxvi/deepxde) but is enhanced by our 
[Brain Dynamics Programming (BDP) ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/). 
For example, it leverages 
[brainstate](https://brainstate.readthedocs.io/) for just-in-time compilation,
[brainunit](https://brainunit.readthedocs.io/) for dimensional analysis, 
[braintools](https://braintools.readthedocs.io/) for checkpointing, loss functions, and other utilities.


``PINNx`` implements the following algorithms, but with the flexibility and efficiency of JAX:

- solving different PINN problems
    - solving forward/inverse ordinary/partial differential equations (ODEs/PDEs) [[SIAM Rev.](https://doi.org/10.1137/19M1274067)]
    - solving forward/inverse integro-differential equations (IDEs) [[SIAM Rev.](https://doi.org/10.1137/19M1274067)]
    - fPINN: solving forward/inverse fractional PDEs (fPDEs) [[SIAM J. Sci. Comput.](https://doi.org/10.1137/18M1229845)]
    - NN-arbitrary polynomial chaos (NN-aPC): solving forward/inverse stochastic PDEs (sPDEs) [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2019.07.048)]
    - PINN with hard constraints (hPINN): solving inverse design/topology optimization [[SIAM J. Sci. Comput.](https://doi.org/10.1137/21M1397908)]
- improving PINN accuracy
    - residual-based adaptive
      sampling [[SIAM Rev.](https://doi.org/10.1137/19M1274067), [Comput. Methods Appl. Mech. Eng.](https://doi.org/10.1016/j.cma.2022.115671)]
    - gradient-enhanced PINN (gPINN) [[Comput. Methods Appl. Mech. Eng.](https://doi.org/10.1016/j.cma.2022.114823)]
    - PINN with multi-scale Fourier features [[Comput. Methods Appl. Mech. Eng.](https://doi.org/10.1016/j.cma.2021.113938)]
- (physics-informed) deep operator network (DeepONet)
    - DeepONet: learning operators [[Nat. Mach. Intell.](https://doi.org/10.1038/s42256-021-00302-5)]
    - DeepONet extensions, e.g., POD-DeepONet [[Comput. Methods Appl. Mech. Eng.](https://doi.org/10.1016/j.cma.2022.114778)]
    - MIONet: learning multiple-input operators [[SIAM J. Sci. Comput.](https://doi.org/10.1137/22M1477751)]
    - Fourier-DeepONet [[Comput. Methods Appl. Mech. Eng.](https://doi.org/10.1016/j.cma.2023.116300)],
      Fourier-MIONet [[arXiv](https://arxiv.org/abs/2303.04778)]
    - physics-informed DeepONet [[Sci. Adv.](https://doi.org/10.1126/sciadv.abi8605)]
    - multifidelity DeepONet [[Phys. Rev. Research](https://doi.org/10.1103/PhysRevResearch.4.023210)]
    - DeepM&Mnet: solving multiphysics and multiscale problems [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110296), [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2021.110698)]
    - Reliable extrapolation [[Comput. Methods Appl. Mech. Eng.](https://doi.org/10.1016/j.cma.2023.116064)]
- multifidelity neural network (MFNN)
    - learning from multifidelity data [[J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2019.109020), [PNAS](https://doi.org/10.1073/pnas.1922210117)]

## Installation

- Install the stable version with `pip`:

``` sh
pip install pinnx --upgrade
```


## Documentation

The official documentation is hosted on Read the Docs: [https://pinnx.readthedocs.io/](https://pinnx.readthedocs.io/)


## See also the BDP ecosystem

We are building the Brain Dynamics Programming ecosystem: https://ecosystem-for-brain-dynamics.readthedocs.io/

