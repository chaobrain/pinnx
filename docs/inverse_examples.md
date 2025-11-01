# PINN Inverse Examples

This section contains examples demonstrating how to solve inverse problems using Physics-Informed Neural Networks (PINNs) **with physical units**. Inverse problems are among the most powerful applications of PINNs, allowing you to discover unknown parameters, fields, or even physical laws from observational data.

## What are Inverse Problems?

Inverse problems involve determining unknown parameters or functions within a governing equation using observed data. Unlike forward problems where everything is known except the solution, inverse problems require simultaneously:

1. **Solving the PDE** to find the solution field
2. **Inferring unknown parameters** such as:
   - Material properties (diffusion coefficients, permeability, viscosity)
   - Source terms or reaction rates
   - Initial conditions or boundary values
   - Spatially-varying fields (heterogeneous properties)

This is particularly valuable in real-world scenarios where:
- Direct measurement of parameters is difficult or impossible
- Only sparse observations of the solution are available
- Physical properties vary in space or time
- Multiple parameters need to be identified simultaneously

## Why PINNs Excel at Inverse Problems

Traditional inverse problem solvers often struggle with:
- **Ill-posedness**: Multiple parameter sets may fit the data
- **Computational cost**: Iterative optimization can be prohibitively expensive
- **Noise sensitivity**: Measurement errors can lead to unstable solutions
- **High dimensionality**: Many unknown parameters to identify

PINNs address these challenges by:
- Embedding physical laws directly in the loss function (regularization)
- Providing smooth, continuous representations of fields
- Handling sparse and noisy data naturally
- Enabling parameter discovery with minimal observations
- Scaling to high-dimensional parameter spaces

## Physical Units in Inverse Problems

These examples use PINNx's **unit-aware framework**, which is especially valuable for inverse problems:

- **Dimensional consistency**: Ensures inferred parameters have correct physical units
- **Better conditioning**: Unit normalization improves optimization stability
- **Physical constraints**: Unit analysis helps constrain parameter search spaces
- **Interpretability**: Results are directly usable without unit conversion
- **Validation**: Easier to verify if discovered parameters are physically reasonable

## Featured Examples

The examples demonstrate parameter inference across multiple physics domains:

### Diffusion and Transport
- **1D Diffusion Inverse**: Inferring diffusion coefficient from temperature measurements
- **Reaction-Diffusion**: Discovering reaction rates in coupled systems

### Fluid Dynamics
- **Navier-Stokes Inverse**: Identifying viscosity and pressure fields from velocity data
- **Brinkman-Forchheimer**: Inferring permeability and form drag in porous media flow

### Field Reconstruction
- **Elliptic Inverse Field**: Reconstructing spatially-varying coefficient fields
- **Heterogeneous Parameter Identification**: Finding space-dependent material properties

### Chemical Kinetics
- **Reaction Inverse**: Determining reaction rate constants from concentration data

## Key Techniques Demonstrated

Each example showcases important inverse problem methodologies:

- **Data assimilation**: Incorporating sparse observations into the physics-informed loss
- **Multi-objective optimization**: Balancing data fitting with PDE residuals
- **Uncertainty quantification**: Assessing confidence in inferred parameters (where applicable)
- **Regularization strategies**: Using physics to constrain ill-posed problems
- **Sparse data handling**: Learning from limited measurements
- **Noise robustness**: Dealing with measurement uncertainties
- **Parameter initialization**: Effective starting points for optimization
- **Convergence monitoring**: Tracking both solution and parameter convergence

## Example Structure

Each inverse problem example includes:

1. **Problem formulation**: Defining knowns and unknowns
2. **Synthetic data generation**: Creating observations (or using real data)
3. **Network architecture**: Designing networks for both solution and parameters
4. **Loss function**: Combining data misfit and PDE residuals
5. **Training strategy**: Optimization approaches for inverse problems
6. **Results validation**: Comparing inferred vs. true parameters
7. **Sensitivity analysis**: Understanding parameter identifiability

These examples serve as templates for solving your own inverse problems in physics and engineering.

```{toctree}
:maxdepth: 1

unit-examples-inverse/elliptic_inverse_filed.ipynb
unit-examples-inverse/brinkman_forchheimer.ipynb
unit-examples-inverse/diffusion_reaction_rate.ipynb
unit-examples-inverse/reaction_inverse.ipynb
unit-examples-inverse/diffusion_1d_inverse.ipynb
unit-examples-inverse/Navier_Stokes_inverse.ipynb

```
