# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


setup(
    name='sound-spaces',
    version='0.1.1',
    packages=[
        'soundspaces',
        'ss_baselines'
    ],
    install_requires=[
        'habitat',
        'habitat-sim',
        'torch',
        'gym==0.10.9',
        'numpy>=1.16.1',
        'yacs>=0.1.5',
        'numpy-quaternion>=2019.3.18.14.33.20',
        'attrs>=19.1.0',
        'opencv-python>=3.3.0',
        'imageio>=2.2.0',
        'imageio-ffmpeg>=0.2.0',
        'scipy>=1.0.0',
        'tqdm>=4.0.0',
        'numba',
        'Pillow',
        'pydub',
        'getch',
        'matplotlib',
        'librosa',
        'torchsummary',
        'gitpython',
        'networkx',
        'tqdm',
        'notebook',
        'moviepy',
        'tensorflow',
        'astropy',
        'scikit-image'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
