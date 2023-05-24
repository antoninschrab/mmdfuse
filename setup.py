#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

dist = setup(
    name="mmdfuse",
    version="1.0.0",
    description="MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting",
    author="Antonin Schrab",
    author_email="a.lastname@ucl.ac.uk",
    license="MIT License",
    packages=["mmdfuse", ],
    install_requires=["jax", "jaxlib"],
    python_requires=">=3.9",
)
