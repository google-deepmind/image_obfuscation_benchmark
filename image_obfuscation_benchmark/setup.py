#!/usr/bin/python
#
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py', 'mock', 'numpy', 'scipy', 'tensorflow', 'tensorflow_datasets',
    'tensorflow_hub', 'tqdm']

setup(
    name='image_obfuscation_benchmark',
    version='1.0',
    description='Package to use the image obfuscation benchmark.',
    url='https://github.com/deepmind/image_obfuscation_benchmark',
    author='DeepMind',
    author_email='stimberg@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)
