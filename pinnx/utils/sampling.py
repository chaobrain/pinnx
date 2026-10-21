# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


import warnings

import brainstate
import numpy as np
from scipy.stats import qmc

__all__ = ["sample"]


def sample(n_samples, dimension, sampler="pseudo"):
    """Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError(f"{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    return np.random.random(size=(n_samples, dimension)).astype(brainstate.environ.dftype())


def _halton(n_samples, dimension):
    # 1st point: [0, 0, ...]
    return qmc.Halton(d=dimension, scramble=False).random(n_samples + 1)[1:]


def quasirandom(n_samples, dimension, sampler):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    if sampler == "LHS":
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        x = qmc.LatinHypercube(d=dimension, rng=rng).random(n_samples)
    elif sampler == "Halton":
        x = _halton(n_samples, dimension)
    elif sampler == "Hammersley":
        # 1st point: [0, 0, ...]
        grid = np.arange(1, n_samples + 1)[:, None] / (n_samples + 1)
        if dimension == 1:
            x = grid
        else:
            x = np.hstack((grid, _halton(n_samples, dimension - 1)))
    elif sampler == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        skip = 1 if dimension < 3 else 2
        with warnings.catch_warnings():
            # Sobol' balance properties need 2^m points; DeepXDE does not require it.
            warnings.simplefilter("ignore", UserWarning)
            x = qmc.Sobol(d=dimension, scramble=False).random(n_samples + skip)[skip:]
    else:
        raise ValueError(f"{sampler} sampling is not available.")
    return np.asarray(x, dtype=brainstate.environ.dftype())
