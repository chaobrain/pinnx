# PINN Forward Examples

This section contains examples demonstrating how to solve forward problems using Physics-Informed Neural Networks (PINNs) **with physical units**. These examples showcase PINNx's unique capability to handle dimensional analysis and physical units directly in the neural network training process.

## What are Forward Problems?

Forward problems involve solving partial differential equations (PDEs) where the governing equations, domain geometry, and boundary/initial conditions are fully known. The goal is to find the solution field (e.g., temperature, velocity, displacement) throughout the domain.

## Physical Units in PINNx

Unlike traditional PINN implementations, these examples use PINNx's **unit-aware framework**, which:

- Accepts inputs and outputs with explicit physical units (e.g., meters, seconds, Pascals)
- Automatically handles dimensional analysis during training
- Ensures physical consistency across all computations
- Improves training stability and convergence
- Makes results directly interpretable in real-world units

## Featured Examples

The examples below cover a diverse range of physics problems:

- **Fluid Dynamics**: Beltrami flow, Burgers equation with/without adaptive refinement
- **Heat Transfer**: Heat equation, diffusion equation with various configurations
- **Structural Mechanics**: Euler beam under different loading conditions
- **Wave Propagation**: Helmholtz equation in various geometries
- **Elliptic PDEs**: Laplace equation on complex domains

Each example demonstrates best practices for:
- Setting up problems with physical units
- Defining boundary and initial conditions
- Configuring neural network architectures
- Training with dimensional awareness
- Visualizing and validating results

```{toctree}
:maxdepth: 1

unit-examples-forward/Beltrami_flow.ipynb
unit-examples-forward/diffusion_1d.ipynb
unit-examples-forward/Euler_beam.ipynb
unit-examples-forward/Helmholtz_Dirichlet_2d.ipynb
unit-examples-forward/burgers.ipynb
unit-examples-forward/Burgers_RAR.ipynb
unit-examples-forward/heat.ipynb
unit-examples-forward/heat_resample.ipynb
unit-examples-forward/Laplace_disk.ipynb
```
