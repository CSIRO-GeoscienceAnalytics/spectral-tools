"""
Documentation was created with pdoc3: pdoc --html spex --config latex_math=True  --output-dir docs\

This package has a collection of tools that I have had to develop or had a need for over the years.

It is comprised of a number of sub-packages.

1) ext (package): A Cython implementation of for convex hulls.

from `spex.ext.chulls` import get_absorption

2) extraction (package): A class for extracting spectral feature information from spectra

from `spex.extraction.specex` import SpectralExtraction

3) io (package): An abstract class that allow you to get at the spectral data of a given type of instrument. It is also a
required input into the SpectralExtraction class

from `spex.io.instruments` import Tsg, ImageData, NumpySpectra, CsvSpectra

4) python_hulls (package): A pure python implementation of convex hulls. This is way slower than the spex.ext.chulls
implementation but you get that

from `spex.python_hulls.phull` import get_absorption

5) speclib (package): A work in progress for doing spectral unmixing using NMF. Its still in Dev. I may delete it.

6) utilities (package): A collection of small utilities for doing some odds and ends. I may deprecate it
"""