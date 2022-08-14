import cython
from cython.parallel import prange, parallel
from numpy import empty, double
from multiprocessing import cpu_count
cdef int NUM_THREADS = cpu_count()

def get_absorption(wavelengths, spectra, hull_type):
    """
    Modified: Andrew Rodger, CSIRO Mineral Resources (15/02/2020)

    Original author: Jess Robertson, CSIRO Mineral Resources

    date: March 2016

    (sk)Hulls for the skull god!

    Apply a hull correction to incoming spectral data, or alternatively return the hull of the spectra

    Args:
        spectra (ndarray): the spectral data for processing. Can be a single spectrum, or N samples (NxB), or an image (NxMxB)
        wavelengths (ndarray): the wavelengths corresponding to the spectral data
        hull_type (int): 0 is hull quotient (1.0 - spectrum/hull), 1 is hull removed (hull - spectrum), and 2 is the actual sk(Hull)

    Returns:
        ndarray: an array (the same shape as spectral data array) of hull corrected spectra

    """
    # make the arrays doubles
    spectra = double(spectra)
    wavelengths = double(wavelengths)

    # Allocate some memory for absorptions
    absorption = empty(shape=spectra.shape)

    # see if its the hull quotient or hull removed
    if hull_type == 0:
        hull_type = 0
    elif hull_type == 1:
        hull_type = 1
    else:
        hull_type = 2

    # Do lookup for dimension-specific call
    if (spectra.ndim == 1):
        extract_hull_0d(spectra, absorption, wavelengths, hull_type)

    elif (spectra.ndim == 2):
        extract_hull_1d(spectra, absorption, wavelengths, hull_type)

    elif (spectra.ndim == 3):
        extract_hull_2d(spectra, absorption, wavelengths, hull_type)
    else:
        raise NotImplementedError(
            'Extraction not available for raster dimension >= 3')
    return absorption

# CYTHON CODE
@cython.cdivision(True)
@cython.boundscheck(False)
cdef void extract_subhull(double [:] spectrum, double [:] absorption,
                          double[:] wavelengths, int start, int end, int hull_type) nogil:
    """ Extracts a subhull between the given start and end locations for
        the given index in a spectra instance.

        Parameters:
            spectrum - the spectral reflectance values
            absorption - an array (same size as spectrum) to store the
                absorption features in.
            wavelengths - the wavelengths for the spectral channels
            start, end - the start and end points to extract the hull from.
    """
    # Calculate trial hull given end points
    cdef int iidx, minidx = 0
    cdef double minabsorb = 0, hull = 0
    cdef double m = (spectrum[end] - spectrum[start]) \
                    / (wavelengths[end] - wavelengths[start])
    for iidx in range(start, end+1):
        hull = m * (wavelengths[iidx] - wavelengths[start]) + spectrum[start]

        if (hull_type == 0):
            absorption[iidx] = 0 if (hull == 0) else (1 - spectrum[iidx] / hull)
        elif (hull_type ==1):
            absorption[iidx] = 0 if (hull == 0) else (hull - spectrum[iidx])
        else:
            absorption[iidx] = 0 if (hull == 0) else (hull - spectrum[iidx])

        # Make sure we keep the minimum absorption around
        if absorption[iidx] < minabsorb:
            minabsorb = absorption[iidx]
            minidx = iidx
        if hull_type == 2:
            if hull >= absorption[iidx]:
                absorption[iidx] = hull

    # Determine whether we need to subdivide the hull
    if (minabsorb < 0):
        # Check upper half
        if minidx - start > 0 and minidx != end:
            extract_subhull(spectrum, absorption, wavelengths, start, minidx, hull_type)
        else:
            if (hull_type == 2):
                absorption[minidx] = hull
            else:
                absorption[minidx] = 0

        # Check lower half
        if end - minidx > 0 and minidx != start:
            extract_subhull(spectrum, absorption, wavelengths, minidx, end, hull_type)
        else:
            if (hull_type ==2):
                absorption[minidx] = hull
            else:
                absorption[minidx] = 0

# Specializations for different array shapes
cdef void extract_hull_0d(double [:] spectrum, double [:] absorption,
                          double [:] wavelengths, int hull_type) nogil:
    """ Extract hull for a single spectrum
    """
    cdef int nwvl = wavelengths.shape[0]-1
    extract_subhull(spectrum, absorption, wavelengths, 0, nwvl, hull_type)

@cython.boundscheck(False)
cdef void extract_hull_1d(double [:, :] spectrum, double [:, :] absorption,
                          double [:] wavelengths, int hull_type) nogil:
    """ Extract hull for a 1D line of spectra
    """
    cdef int iidx, nx = spectrum.shape[0], nwvl = spectrum.shape[1]-1
    with nogil, parallel(num_threads=NUM_THREADS):
        for iidx in prange(nx, schedule='dynamic'):
            extract_subhull(spectrum[iidx], absorption[iidx], wavelengths, 0, nwvl, hull_type)

@cython.boundscheck(False)
cdef void extract_hull_2d(double [:, :, :] spectrum, double[:, :, :] absorption,
                          double [:] wavelengths, int hull_type) nogil:
    """ Extract hull for a 2D raster of spectra
    """
    cdef int iidx, jidx
    cdef int nx = spectrum.shape[0], ny = spectrum.shape[1], nwvl = spectrum.shape[2]-1
    with nogil, parallel(num_threads=NUM_THREADS):
        for iidx in prange(nx, schedule='dynamic'):
            for jidx in range(ny):
                extract_subhull(spectrum[iidx, jidx], absorption[iidx, jidx], wavelengths, 0, nwvl, hull_type)


