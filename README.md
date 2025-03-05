# PINNx: Physics-Informed Neural Networks for Scientific Machine Learning in JAX


<p align="center">
  	<img alt="Header image of pinnx." src="https://github.com/chaobrain/pinnx/blob/main/docs/_static/pinnx.png" width=40%>
</p> 


[![Build Status](https://github.com/chaobrain/pinnx/actions/workflows/build.yml/badge.svg)](https://github.com/chaobrain/pinnx/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/pinnx/badge/?version=latest)](https://pinnx.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/pinnx.svg)](https://badge.fury.io/py/pinnx)
[![License](https://img.shields.io/github/license/chaobrain/pinnx)](https://github.com/chaobrain/pinnx/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/890937840.svg)](https://doi.org/10.5281/zenodo.14970037)

``PINNx`` is a library for scientific machine learning and physics-informed learning in JAX. 
It is a rewrite of [DeepXDE](https://github.com/lululxvi/deepxde) but is enhanced by our 
[Brain Dynamics Programming (BDP) ecosystem](https://ecosystem-for-brain-dynamics.readthedocs.io/). 
For example, it leverages 

- [brainstate](https://brainstate.readthedocs.io/) for just-in-time compilation,
- [brainunit](https://brainunit.readthedocs.io/) for dimensional analysis, 
- [braintools](https://braintools.readthedocs.io/) for checkpointing, loss functions, and other utilities.


## Quickstart


Define a PINN with explicit variables and physical units.

```python
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

```



## Installation

- Install the stable version with `pip`:

``` sh
pip install pinnx --upgrade
```


## Documentation

The official documentation is hosted on Read the Docs: [https://pinnx.readthedocs.io/](https://pinnx.readthedocs.io/)


## See also the BDP ecosystem

We are building the Brain Dynamics Programming ecosystem: https://ecosystem-for-brain-dynamics.readthedocs.io/

