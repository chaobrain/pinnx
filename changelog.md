# Changelog

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
  - Removed standalone `requirements.txt`
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
- Bumped version to 0.0.3 in `pyproject.toml`
- Added version information to `pinnx/__init__.py`
- Created this changelog file and integrated it into documentation

### Miscellaneous
- Added DOI badge to README.md
- Updated `.gitignore` with additional patterns
- Enhanced `README.md` with improved project information


