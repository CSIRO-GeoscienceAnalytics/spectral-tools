
# spectraltools

A collection of tools for working with and using spectral data. Documentation for the various modules can be
found in the docs folder.

## Pre Installation Requirements

1) setuptools >= 18.0
2) Cython

    In regard to cython you will probably need to ensure you have a compatible C
    compiler for your OS. Probably best to take a look at the [Cython documentation](https://cython.readthedocs.io/en/latest/index.html)
    to see what is required there. Who knows you may already have one installed.

## Installation

$ python setup.py install

## What is this?

This package is a small collection of tools that I have developed or had a need for over the years. The primary focus is to work with spectral reflectance data. The overall package could probably by structured a lot better but you get that.

It does have a precompiled cython file in the spectraltools/ext folder for performing upper convex hull corections. If you need to compile it yourself then you will need Cython installed and a C compiler. The compiled version is for python 3.10 and a 64 bit system.

I have tried various python only implementations of hull correction routines but haven't found them to be fast enough. I probably need to look closer at `qhull`.

## Whats in the spectraltools package?

It is comprised of a number of sub-packages.

1) ext (package): A Cython implementation for convex hulls.

    from `spectarltools.ext import chulls', then to use it you would call chulls.get_absorption()

2) extraction (package): A module for extracting spectral feature information from spectra

    from `spectraltools.extraction import extraction

3) io (package): An module with a bunch of spectral data parsers.

    from `spectraltools.io import parse_tsg, parse_envi, parse_csv, parse_numpy

4) python_hulls (package): A pure python implementation of convex hulls. This is way slower than the cython implementation but you get that.

    from `spectraltools.python_hulls import phull

5) speclib (package): A work in progress for doing spectral unmixing using NMF. Its still in Dev. I may delete it.