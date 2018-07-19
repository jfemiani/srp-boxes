#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='srp-boxes',
    version='0.1',
    packages=find_packages(),
    install_requires=['torch', 'torchvision', 'numpy', 'pandas', 'matplotlib'],

    # Which files should be installed alongside the sources
    package_data={
        # 'package':['*.ext', ...]
    },

    # Matadata for use e.g. on PyPI
    description='3D Boxes From IMagery and Point Clouds',
    author='Xian Liu',
    author_email='liux13@miamioh.edu',
    url='https://github.com/liux13/srp-boxes',
    licens='MIT',
    keywords='lidar box detection',
)
