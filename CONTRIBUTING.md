# Contributing to PINNx

Thank you for your interest in contributing to PINNx! We welcome contributions from the community.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code of Conduct](#code-of-conduct)

## Getting Started

PINNx is a library for scientific machine learning and physics-informed learning in JAX. Before contributing, please:

1. Read the [documentation](https://pinnx.readthedocs.io/)
2. Check existing [issues](https://github.com/chaobrain/pinnx/issues) and [pull requests](https://github.com/chaobrain/pinnx/pulls)
3. Review our [Code of Conduct](CODE_OF_CONDUCT.md)

## Development Setup

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/pinnx.git
cd pinnx
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/chaobrain/pinnx.git
```

### Install Development Dependencies

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install PINNx in development mode with all dependencies:

```bash
pip install -e ".[cpu]"  # For CPU development
# or
pip install -e ".[cuda12]"  # For CUDA 12
```

3. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, JAX version)
- Minimal code example that demonstrates the issue

### Suggesting Enhancements

For feature requests or enhancements:

- Open an issue with a clear description
- Explain the use case and why it would be useful
- Provide examples if possible

### Contributing Code

1. **Find or create an issue**: Check if there's an existing issue for your contribution. If not, create one to discuss your proposed changes.

2. **Create a branch**: 

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

3. **Make your changes**: Write your code following our [coding guidelines](#coding-guidelines).

4. **Write tests**: Add tests for your changes.

5. **Update documentation**: Update relevant documentation if needed.

6. **Commit your changes**:

```bash
git add .
git commit -m "Description of your changes"
```

Follow conventional commit format when possible:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring

## Coding Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Code Quality

- Write clear, self-documenting code
- Add comments for complex logic
- Ensure your code is compatible with Python 3.9+
- Use JAX best practices (pure functions, avoid side effects)

### Example

```python
import jax.numpy as jnp
import brainunit as u


def compute_residual(
    x: dict,
    y: dict,
    v: u.Quantity
) -> jnp.ndarray:
    """Compute PDE residual.
    
    Args:
        x: Dictionary of input coordinates
        y: Dictionary of solution values
        v: Diffusion coefficient with units
        
    Returns:
        Residual values
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
pytest tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test edge cases and error conditions
- Aim for good test coverage

Example:

```python
def test_geometry_interval():
    """Test Interval geometry creation and sampling."""
    geom = pinnx.geometry.Interval(-1, 1)
    assert geom.dim == 1
    
    points = geom.random_points(100)
    assert points.shape == (100, 1)
    assert jnp.all(points >= -1) and jnp.all(points <= 1)
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When condition occurs
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

### Building Documentation

```bash
cd docs
make html
```

View the documentation at `docs/_build/html/index.html`

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest upstream changes:

```bash
git fetch upstream
git rebase upstream/main
```

2. **Push to your fork**:

```bash
git push origin your-branch-name
```

3. **Create a Pull Request**:
   - Go to the PINNx repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template with:
     - Description of changes
     - Related issue numbers
     - Testing performed
     - Documentation updates

4. **Review process**:
   - Maintainers will review your PR
   - Address any feedback or requested changes
   - Once approved, your PR will be merged

### PR Guidelines

- Keep PRs focused on a single feature/fix
- Write clear commit messages
- Ensure all tests pass
- Update documentation as needed
- Add yourself to contributors if you'd like

## Code of Conduct

Please note that this project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

If you have questions about contributing:

- Open a [discussion](https://github.com/chaobrain/pinnx/discussions)
- Check the [documentation](https://pinnx.readthedocs.io/)
- Contact the maintainers at chao.brain@qq.com

## License

By contributing to PINNx, you agree that your contributions will be licensed under the Apache-2.0 license.

---

Thank you for contributing to PINNx! Your efforts help make scientific machine learning more accessible to everyone.
