"""
A pure python implementation of convex hulls. It uses scipy.spatial.convexhull
Its not super fast but its an alternative if you dont have cython and cant compile the pyx files
It takes about 7 mins to run 240K spectra with 2100 spectral bands. So not super bad but still slow.
For comparison that is the equivalent of 2.4km of HyLogger data
"""