# Changelog

## Version 0.0.4 (2026-10-21)

This release makes PINNx work out of the box on current JAX and the BrainX stack. On a fresh install of 0.0.3, most PDE examples crashed: the default `Hammersley` sampler imported the undeclared `scikit-optimize` package, and many code paths still used APIs that newer JAX and `brainstate` releases have removed. 0.0.4 fixes these crashes along with a number of numerically incorrect results, adds a test suite of 222 tests that runs in CI on Linux, macOS and Windows, and verifies that every documentation example runs.

### Breaking changes
- **Python 3.12+ and JAX 0.9+ are now required.** Support for Python 3.10 and 3.11 is dropped.
- **Units now come from `saiunit`.** PINNx depends on `saiunit>=0.5.2` instead of `brainunit`, which is now a thin re-export of `saiunit`. Use `import saiunit as u`. Code that imports `brainunit` keeps working, because both packages share the same `Quantity` and `Unit` classes.
- **The vendored scikit-optimize code has been removed:** `pinnx.utils.transformers`, plus `InitialPointGenerator`, `Space`, `Real`, `Integer`, `Categorical`, `check_random_state` and `check_dimension` in `pinnx.utils.sampling`. Use `pinnx.utils.sampling.sample` / `quasirandom`, or `scipy.stats.qmc` directly.
- `pinnx.utils.standardize` now returns `(scaler, X_train, X_test)`, as documented. The scaler exposes `transform` and `inverse_transform`.
- `pinnx.problem.TripleCartesianProd` adds a trailing output axis to 2-D targets so that they match the `(N1, N2, n_outputs)` output of `pinnx.nn.DeepONetCartesianProd`.
- `pinnx.utils.get_activation` raises `ValueError` for unknown activation names instead of failing later.

### Packaging
- Declare `numpy`, `scipy`, `matplotlib`, `brainstate>=0.5.0` and `braintools>=0.3.0` as runtime dependencies. They were already imported at package import time.
- Fix the accelerator extras: `pinnx[cpu]`, `pinnx[cuda12]`, `pinnx[cuda13]` and `pinnx[tpu]` now install the matching `jax[...]` extra. `jaxlib` has no such extras.
- Ship a PEP 561 `py.typed` marker, so type checkers use PINNx's inline annotations.

### Bug fixes

#### Sampling and utilities
- Reimplement the `Hammersley`, `Halton`, `Sobol` and `LHS` samplers on `scipy.stats.qmc`. The default training-point distribution no longer needs an optional package.
- `utils.isclose` and `utils.smart_numpy` accept NumPy arrays, Python scalars and lists.
- The `"mean l2 relative error"` loss is now actually relative, and `softmax_cross_entropy` receives its arguments in the correct order.
- `utils.hstack` and `utils.zero_padding` preserve JAX arrays inside `jit`.
- Replace `jax.tree_map` and other APIs that JAX has removed.

#### Geometry
- Fix crashes in `Polygon`, `Triangle`, `Rectangle`, `Ellipse`, `StarShaped`, `Cuboid` and `Hypercube` caused by in-place JAX array assignment, lists passed to `jnp` functions, and 2-D `jnp.cross`.
- Fix incorrect results:
  - `Ellipse` bounding box and diameter.
  - `StarShaped` bounding box and `on_boundary`.
  - `Polygon` boundary sampling and normals.
  - `PointCloud.boundary_normal` ordering.
  - The `Cinf` boundary-constraint factor.
  - `Interval.boundary_constraint_factor(where="right")`.
- Boundary normals from dict-point geometries are now dimensionless instead of carrying coordinate units.
- `boundary_normal` and `periodic_point` work under `jit`.

#### Initial and boundary conditions
- `PeriodicBC` now differentiates with respect to the inputs, not the outputs.
- `DirichletBC` accepts a constant value.
- `Interface2DBC` is ported to the dict-based pipeline.

#### Problems
- PDE problems with `pde=None` can be trained.
- `loss_weights` is now applied by data-driven problems (`Function`, `DataSet`, `TripleDataset`, `TripleCartesianProd`, `QuadrupleDataset`, `QuadrupleCartesianProd`, `MfFunc`, `MfDataSet`).
- Fix multi-fidelity and quadruple datasets: missing base-class initialization and a wrong batch structure.
- Fix `TripleCartesianProd` mini-batching.
- `FPDE`, `TimeFPDE` and `IDE`:
  - Test losses no longer drop rows of the test set.
  - Resampling resets the cached integral matrix.
  - A learnable fractional order `alpha` is used in the loss.
- `PDEOperator` and `PDEOperatorCartesianProd` return the correct targets and auxiliary variables, and reset their caches when training points are resampled.

#### Training, networks and callbacks
- `Trainer.save` and `Trainer.restore` work, including optimizer state.
- `Trainer.train` uses the batch size given to `compile`.
- `compile(metrics="...")` accepts a single metric name.
- Metrics are skipped when the problem has no reference solution.
- `ModelCheckpoint` and `MovieDumper` work with dict-valued losses and models.
- Fix `PODDeepONet`, the `MIONetCartesianProd` and `PODMIONet` merger paths, and `NN.num_trainable_parameters`.
- `ArrayToDict` accepts unit-valued quantities such as `1 / u.second`.
- `fnspace`: unknown GRF kernels raise `ValueError`, and `np.trapz` is replaced by `np.trapezoid` for NumPy 2.

### Examples and documentation
- All 61 examples in `docs/` run with only the declared dependencies:
  - The hyper-parameter optimization example uses a Sobol search instead of scikit-optimize.
  - `antiderivative_unaligned` builds its data from the bundled dataset.
  - The Stokes operator example computes derivatives in forward mode and no longer exhausts memory.
- Fix broken and outdated links in the README and the documentation. The README logo is now served from `raw.githubusercontent.com`, so it renders on PyPI.
- The documentation builds again: notebooks are no longer executed during the build, and the API reference no longer lists removed symbols.

### CI
- `pytest` runs on Python 3.12 and 3.13 on Linux, macOS and Windows, and test failures now fail the build.
- The flake8 syntax/undefined-name check is now blocking.
- The release workflow verifies that `py.typed` is included in the wheel.
- Bump `actions/checkout`, `actions/setup-python` and `codecov/codecov-action` to v7.

## Version 0.0.3

### Major Changes

#### Brainstate to Braintools Migration
- Replaced all `brainstate` imports with `braintools` across the entire codebase
- Updated import statements in all core modules, examples, and documentation

#### Build System Refactoring
- Removed `setup.py` in favor of modern `pyproject.toml` configuration
- Reorganized project requirements into separate files:
  - `requirements-dev.txt` for development dependencies
  - `requirements-doc.txt` for documentation dependencies
- Updated copyright information across the project

### CI/CD Enhancements

#### GitHub Actions Improvements
- Added Python 3.13 support to CI pipeline
- Integrated `mypy` for static type checking
- Added `ruff` linter for code quality checks
- Enhanced build workflow with additional testing configurations
- Improved release workflow with better automation

#### New GitHub Templates and Policies
- Added issue templates:
  - Bug report template (`.github/ISSUE_TEMPLATE/bug_report.yml`)
  - Feature request template (`.github/ISSUE_TEMPLATE/feature_request.yml`)
  - Documentation request template (`.github/ISSUE_TEMPLATE/documentation.yml`)
- Added pull request template (`.github/pull_request_template.md`)
- Added security policy (`.github/SECURITY.md`)
- Added Dependabot configuration (`.github/dependabot.yml`)
- Added funding information (`.github/FUNDING.yml`)

### Documentation Improvements

#### API Documentation
- Added comprehensive API documentation for new modules:
  - `pinnx.geometry` module with all geometry classes
  - `pinnx.icbc` module for initial and boundary conditions
  - `pinnx.nn` module for neural network components
  - `pinnx.problem` module for problem definitions
  - `pinnx.utils` module with utility functions
- Enhanced existing API documentation for:
  - `pinnx.callbacks`
  - `pinnx.fnspace`
  - `pinnx.grad`
  - `pinnx.metrics`

#### Documentation Structure
- Added `forward_examples.md` for forward problem examples
- Added `forward_unitless_examples.md` for unitless examples
- Added `inverse_examples.md` for inverse problem examples
- Removed deprecated documentation files:
  - `docs/examples-unitless.rst`
  - `docs/unit-examples-forward.rst`
  - `docs/unit-examples-inverse.rst`
- Removed `docs/auto_generater.py` (no longer needed)
- Reorganized `docs/index.rst` for better navigation and clarity
- Corrected project name in `docs/Makefile`

#### Configuration Updates
- Upgraded Python version in ReadTheDocs configuration (`.readthedocs.yaml`)
- Updated OS version requirements in ReadTheDocs
- Updated contact email in Code of Conduct
- Significantly expanded `CONTRIBUTING.md` with detailed guidelines

### Code Quality and Testing

#### New Test Files
- Added `pinnx/grad_test.py` for gradient computation testing

#### Code Improvements
- Removed unused imports across multiple files
- Updated imports in all example files (function, operator, PINN forward/inverse)
- Enhanced code formatting and consistency
- Updated references from `DeepXDE` to `PINNx` throughout the codebase

### Examples Updates
- Updated all forward problem examples with improved imports
- Updated all inverse problem examples with corrected imports
- Updated all operator learning examples
- Updated all function approximation examples
- Refreshed all Jupyter notebooks with updated code

### Version Management
- Bumped version to 0.0.3 (`pinnx.__version__`)
- Added version information to `pinnx/__init__.py`
- Created this changelog file and integrated it into documentation

### Miscellaneous
- Added DOI badge to README.md
- Updated `.gitignore` with additional patterns
- Enhanced `README.md` with improved project information


