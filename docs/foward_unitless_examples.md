# PINN Forward Unitless Examples

This section contains examples demonstrating how to solve forward problems using Physics-Informed Neural Networks (PINNs) **without explicit physical units**. These examples work with dimensionless or normalized equations, following the traditional PINN approach.

## What are Unitless Examples?

Unitless (or dimensionless) examples use normalized variables and equations where:

- All quantities are scaled to be of order unity (O(1))
- Physical dimensions are removed through non-dimensionalization
- Equations are expressed in terms of dimensionless parameters (e.g., Reynolds number, Mach number)
- Results require post-processing to convert back to physical units

This approach is widely used in traditional computational physics and can be beneficial when:
- Working with multi-scale problems
- Comparing solutions across different parameter regimes
- Simplifying complex equations
- Following established non-dimensional formulations

## Comparison with Unit-Aware Examples

While PINNx supports **both** unit-aware and unitless formulations, each has its advantages:

| **Unitless (this section)** | **Unit-Aware** (see PINN Forward Examples) |
|------------------------------|---------------------------------------------|
| Traditional approach | PINNx's innovative feature |
| Requires manual non-dimensionalization | Automatic dimensional handling |
| Results need scaling back | Results in physical units directly |
| Better for classical benchmarks | Better for real-world applications |
| More examples available | Growing collection |

## Comprehensive Example Collection

This section features an extensive collection of over 30 examples covering:

### Elliptic PDEs
- **Poisson Equation**: Various boundary conditions (Dirichlet, Neumann, Robin, Periodic)
- **Helmholtz Equation**: Multiple geometries including domains with holes
- **Laplace Equation**: Complex domain shapes

### Parabolic PDEs
- **Diffusion Equation**: Standard, with exact BC, with resampling
- **Heat Equation**: Time-dependent problems
- **Diffusion-Reaction**: Coupled equations

### Hyperbolic PDEs
- **Klein-Gordon Equation**: Wave propagation
- **Schrodinger Equation**: Quantum mechanics

### Nonlinear PDEs
- **Allen-Cahn Equation**: Phase field modeling
- **Burgers Equation**: Nonlinear advection-diffusion
- **Navier-Stokes**: Fluid dynamics (Kovasznay flow, Beltrami flow)

### Fractional PDEs
- **Fractional Poisson**: 1D, 2D, and 3D formulations
- **Fractional Diffusion**: Time-fractional derivatives

### Structural Mechanics
- **Euler Beam**: Classical beam theory
- **Linear Elasticity**: 2D plate problems

### Ordinary Differential Equations (ODEs)
- **Second-order ODEs**: Various boundary conditions
- **ODE Systems**: Coupled equations
- **Lotka-Volterra**: Population dynamics
- **Volterra Integro-Differential Equations (IDE)**: Memory effects

### Advanced Features Demonstrated
- **Hyperparameter Optimization (HPO)**: Automated tuning
- **Hard Constraints**: Exact boundary condition enforcement
- **Residual-based Adaptive Refinement (RAR)**: Dynamic point resampling
- **Point Set Operators**: Custom boundary operators

Each example includes detailed implementations showing:
- Problem formulation and non-dimensionalization
- Neural network architecture selection
- Training configuration and loss functions
- Result visualization and validation
- Comparison with analytical/numerical solutions (when available)

```{toctree}
:maxdepth: 1

examples-pinn-forward/Allen_Cahn.ipynb
Helmholtz equation over a 2D square domain <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Helmholtz_Dirichlet_2d.py>
Helmholtz equation over a 2D square domain: Hyper-parameter optimization <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Helmholtz_Dirichlet_2d_HPO.py>
Helmholtz equation over a 2D square domain with a hole <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Helmholtz_Dirichlet_2d_hole.py>
Helmholtz sound-hard scattering problem with absorbing boundary conditions <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Helmholtz_Sound_hard_ABC_2d.py>
Diffusion equation <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/diffusion_1d.py>
Diffusion equation with hard initial and boundary conditions <https://github.com/chatobrain/pinnx/blob/main/docs/examples-pinn-forward/diffusion_1d_exactBC.py>
Diffusion equation with training points resampling <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/diffusion_1d_resample.py>
Diffusion-reaction equation <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/diffusion_reaction.py>
Linear elastostatic equation over a 2D square domain <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/elasticity_plate.py>
Euler beam <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Euler_beam.py>
Klein-Gordon equation <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Klein_Gordon.py>
Kovasznay flow <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Kovasznay_flow.py>
Lotka-Volterra equation <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Lotka_Volterra.py>
Second-order ODE system <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/ode_2nd.py>
A simple ODE system <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/ode_ide.py>
Second order ODE <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/ode_system.py>
Poisson equation in 1D with Dirichlet boundary conditions <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_Dirichlet.py>
Poisson equation in 1D with hard boundary conditions <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_Dirichlet_1d_exactBC.py>
Poisson equation over L-shaped domain <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_Lshape.py>
Poisson equation in 1D with Dirichlet/Neumann boundary conditions <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_Neumann_1d.py>
Poisson equation in 1D with Dirichlet/Periodic boundary conditions <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_periodic_1d.py>
Poisson equation in 1D with Dirichlet/PointSetOperator boundary conditions<https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_PointSetOperator_1d.py>
Poisson equation in 1D with Dirichlet/Robin boundary conditions <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Poisson_Robin_1d.py>
Lotka-Volterra ODE equation<https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/ode_Lotka_Volterra.py>
Beltrami flow <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Beltrami_flow.py>
Fractional diffusion equation <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/fractional_diffusion_1d.py>
Fractional Poisson equation in 1D<https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/fractional_Poisson_1d.py>
Fractional Poisson equation in 2D<https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/fractional_Poisson_2d.py>
Fractional Poisson equation in 3D<https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/fractional_Poisson_3d.py>
Schrodinger equation <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/Schrodinger.py>
Volterra IDE <https://github.com/chaobrain/pinnx/blob/main/docs/examples-pinn-forward/ode_Volterra_IDE.py>
```