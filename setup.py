# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import io
import os
import re
import sys
import time

from setuptools import find_packages
from setuptools import setup

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'pinnx/', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]
if len(sys.argv) > 2 and sys.argv[2] == '--python-tag=py3':
    version = version
else:
    # version += '.post{}'.format(time.strftime("%Y%m%d", time.localtime()))
    version += 'post20250106'

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    README = f.read()

# installation packages
packages = find_packages(
    exclude=[
        "docs*", "tests*", "examples*", "experiments*", "build*",
        "dist*", "pinnx.egg-info*", "pinnx/__pycache__*",
    ]
)

# setup
setup(
    name='pinnx',
    version=version,
    description='Physics-Informed Neural Networks for Scientific Machine Learning in JAX.',
    long_description=README,
    long_description_content_type="text/markdown",
    author='PINNx Developers',
    author_email='chao.brain@qq.com',
    packages=packages,
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'jax',
        'brainunit>=0.0.3',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'brainstate',
        'braintools',
        'optax',
    ],
    url='https://github.com/chaobrain/pinnx',
    project_urls={
        "Bug Tracker": "https://github.com/chaobrain/pinnx/issues",
        "Documentation": "https://pinnx.readthedocs.io/",
        "Source Code": "https://github.com/chaobrain/pinnx",
    },
    extras_require={
        'cpu': ['jaxlib'],
        'cuda12': ['jaxlib[cuda12]'],
        'tpu': ['jaxlib[tpu]'],
    },
    keywords=(
        'computational neuroscience, '
        'brain-inspired computation, '
        'brain dynamics programming'
    ),
    classifiers=[
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache-2.0 license',
)
