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

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable

from pinnx import utils


class ICBC(ABC):
    @abstractmethod
    def filter(self, X):
        pass

    @abstractmethod
    def collocation_points(self, X):
        pass

    @abstractmethod
    def error(self, X, inputs, outputs, beg, end, approx: Callable, *args):
        """
        Returns the loss.
        """
        # aux_var is used in PI-DeepONet, where aux_var is the input function evaluated
        # at x.


def npfunc_range_autocache(func):
    """Call a NumPy function on a range of the input ndarray.

    If the backend is pytorch, the results are cached based on the id of X.
    """
    # For some BCs, we need to call self.func(X[beg:end]) in BC.error(). For backend
    # tensorflow.compat.v1/tensorflow, self.func() is only called once in graph mode,
    # but for backend pytorch, it will be recomputed in each iteration. To reduce the
    # computation, one solution is that we cache the results by using @functools.cache
    # (https://docs.python.org/3/library/functools.html). However, numpy.ndarray is
    # unhashable, so we need to implement a hash function and a cache function for
    # numpy.ndarray. Here are some possible implementations of the hash function for
    # numpy.ndarray:
    # - xxhash.xxh64(ndarray).digest(): Fast
    # - hash(ndarray.tobytes()): Slow
    # - hash(pickle.dumps(ndarray)): Slower
    # - hashlib.md5(ndarray).digest(): Slowest
    # References:
    # - https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/16592241#16592241
    # - https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary/47922199
    # Then we can implement a cache function or use memoization
    # (https://github.com/lonelyenvoy/python-memoization), which supports custom cache
    # key. However, IC/BC is only for pinnx.data.PDE, where the ndarray is fixed. So we
    # can simply use id of X as the key, as what we do for gradients.

    cache = {}

    @wraps(func)
    def wrapper_nocache(X, beg, end, _):
        return func(X[beg:end])

    @wraps(func)
    def wrapper_nocache_auxiliary(X, beg, end, aux_var):
        return func(X[beg:end], aux_var[beg:end])

    @wraps(func)
    def wrapper_cache(X, beg, end, _):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]

    @wraps(func)
    def wrapper_cache_auxiliary(X, beg, end, aux_var):
        # Even if X is the same one, aux_var could be different
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end], aux_var[beg:end])
        return cache[key]

    if utils.get_num_args(func) == 1:
        return wrapper_nocache
    if utils.get_num_args(func) == 2:
        return wrapper_nocache_auxiliary
    raise ValueError("The function should have 1 or 2 arguments.")
