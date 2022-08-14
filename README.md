
# spex

A collection of tools for working with and using spectral reflectance data. Documentation for the various modules can be 
found in the docs/spex folder. 

## Pre Installation Requirements

1) setuptools >= 18.0
2) Cython 
    
    In regard to cython you will probably need to ensure you have a compatible C 
    compiler for your OS. Probably best to take a look at the [Cython documentation](https://cython.readthedocs.io/en/latest/index.html) 
    to see what is required there. Who knows you may already have one installed.

## Installation

$ python setup.py install

## What is this?

This package is a collection of tools that I have developed or had a need for over the years. The primary focus is to 
work with spectral reflectance data. The overall package could probably by structured a lot better but you get that.

It does require you to have `setuptools` installed first as well as `cython`. The later is required so that hull 
removal/correction routines can actually work with large amounts of spectral data in a realistic timeframe. I have tried 
various python only implementations of hull correction routines but haven't found them to be fast enough. I probably 
need to look closer at `qhull`.

## Whats in the spex package?

It is comprised of a number of sub-packages.

1) ext (package): A Cython implementation for convex hulls.

    from `spex.ext.chulls` import get_absorption

2) extraction (package): A class for extracting spectral feature information from spectra

    from `spex.extraction.specex` import SpectralExtraction

3) io (package): An abstract class to get at spectral reflecatnce data in different formats. It is also a required input 
into the SpectralExtraction class.

    from `spex.io.instruments` import Tsg, ImageData, NumpySpectra, CsvSpectra

4) python_hulls (package): A pure python implementation of convex hulls. This is way slower than the `spex.ext.chulls`
implementation but you get that.

    from `spex.python_hulls.phull` import get_absorption

5) speclib (package): A work in progress for doing spectral unmixing using NMF. Its still in Dev. I may delete it.

6) utilities (package): A collection of small utilities for doing some odds and ends. I may deprecate it
