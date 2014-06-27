#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='rdfsim',
    version='0.3',
    description='Modelling RDF data as a vector space',
    author='Yves Raimond',
    author_email='yves.raimond@bbc.co.uk',
    packages=['rdfsim'],
    install_requires=[
        'numpy',
        'RDF',
        'scipy',
    ],
)
