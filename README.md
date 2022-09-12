
# spectraltools

A very small collection of tools for working with spectral data. Documentation for the various modules can be
found in the docs folder. I used to have a class for parsing TSG data but after Ben Chi (https://github.com/FractalGeoAnalytics/pytsg) wrote a much nicer and cleaner version I have adopted his instead. Nice work Ben.

I used to have spectral unmixing routines in here as well but have removed them as they were really a bit (read a LOT) adhoc. They may come back in the future.

## Pre Installation Requirements (only if you need to build the extension file for doing convex hulls)

1) setuptools >= 18.0
2) Cython

    In regard to cython you will probably need to ensure you have a compatible C
    compiler for your OS. Probably best to take a look at the [Cython documentation](https://cython.readthedocs.io/en/latest/index.html)
    to see what is required there. Who knows you may already have one installed.

## Installation (assumimg you are not building the Cython extension and are simply using the prebuilt .c and .pyx files)

$ python setup.py install

## What is this?

This package is a small collection of tools that I have developed or had a need for over the years. The primary focus is to work with spectral reflectance data. 
I have recently gone through it and cleaned it up and dropped a bunch of half finished stuff.

It does have a precompiled cython file in the spectraltools/ext folder for performing upper convex hull corections. If you need to compile it yourself then you will need Cython installed and a C compiler. The compiled version is for python 3.10 and a 64 bit system.

I have tried various python only implementations of hull correction routines but haven't found them to be fast enough. I probably need to look closer at `qhull`. With that said their is python only hulls routine in the package.

Additionaly, if you want to run the feature extraction method then you need to run your routine in a main guard. If you dont then it will not allow you to use the 
multiprocessor component. This initself is not a bad thing and for small datasets (<20000 spectra I guess) it probably isnt needed anyway. If you are extracting features from spectral imagery however then not using the main guard and the main_guard=True keyword in spectraltools.extraction.extract_spectral_features will run noticeably slower due to the large number of samples.

## Whats in the spectraltools package?

It is comprised of a number of sub-packages.

1) ext (package): A Cython implementation for convex hulls.

    The main file is the chulls.pyx file from which the chulls.c and chulls*.pyd is generated. A setup file to regenerate the c and pyd files is also in there.
    You can directly call the chulls.get_absorption method but its better to use the spectraltools.hulls.convexhulls.uc_hulls method instead. This is a wrapper for the     chulls one and as such has type hinting etc.

2) extraction (package): A module for extracting spectral feature information from spectra

    from spectraltools.extraction.extraction import extract_spectral_features

3) io (package): An module with a bunch of convenience spectral data parsers. The parse_numpy one is probably a bit dumb but whatever.

    from spectraltools.io import parse_tsg, parse_envi, parse_csv, parse_numpy

4) python_hulls (package): A pure python implementation of convex hulls (phulls) and a warpper for the cython extension when you really need speed. 

    from spectraltools.hulls.phull import get_absorption 
    or,
    from spectraltools.hulls.convexhulls import get_absorption
