"""
This package is a Cython implementation of a convex hull routine. It was originally put together by Jess Robertson
and then modified by me later (minimal modification - a couple of bugs fixes on edge cases). Its main purpose is to
provide a fast implementation as a pure python version (at least as best as I could do or find) was just to slow over the
tens of thousands of spectra we can encounter in a typical dataset.

"""
