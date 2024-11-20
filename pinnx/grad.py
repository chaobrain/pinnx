# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Callable

import brainstate as bst
import jax

__all__ = [
    'jacobian',
    'hessian',
]


def jacobian(
    approx_fn: Callable,
    xs: jax.Array,
    return_value: bool = False,
    mode: str = 'backward',
):
    """
    Compute `Jacobian matrix <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
    J as J[i, j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and j = 0, ..., dim_x - 1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int or None): `i`th row. If `i` is ``None``, returns the `j`th column
            J[:, `j`].
        j (int or None): `j`th column. If `j` is ``None``, returns the `i`th row
            J[`i`, :], i.e., the gradient of y_i. `i` and `j` cannot be both ``None``,
            unless J has only one element, which is returned.

    Returns:
        (`i`, `j`)th entry J[`i`, `j`], `i`th row J[`i`, :], or `j`th column J[:, `j`].
    """
    if mode == 'backward':
        grad_model = bst.augment.jacrev(approx_fn, argnums=0, return_value=return_value)
    elif mode == 'forward':
        grad_model = bst.augment.jacfwd(approx_fn, argnums=0, return_value=return_value)
    else:
        raise ValueError('Invalid mode. Choose between backward and forward.')
    return jax.vmap(lambda x: grad_model(x))(xs)


def hessian(
    approx_fn: Callable,
    xs: jax.Array,
    return_value: bool = False,
):
    """
    Compute `Hessian matrix <https://en.wikipedia.org/wiki/Hessian_matrix>`_ H as
    H[i, j] = d^2y / dx_i dx_j, where i,j = 0, ..., dim_x - 1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i, j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        approx_fn: Function to compute the gradient.
        xs: Input Tensor of shape (batch_size, dim_x).
        return_value: If True, return the value of the function.

    Returns:
        H[`i`, `j`].
    """
    return jax.vmap(bst.augment.hessian(approx_fn, argnums=0, return_value=return_value))(xs)
