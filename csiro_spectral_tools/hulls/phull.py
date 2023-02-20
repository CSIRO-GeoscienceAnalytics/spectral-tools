
import numpy as np
from numpy.typing import NDArray
from scipy import spatial


def get_absorption(wavelengths: NDArray, spectral_data: NDArray, hull_type: int=0) -> NDArray:
    """get_absorption 
    Calculate either a hull quotient, hull removed or get the hull of the incoming spectral data

    Args:
        wavelengths (NDArray): the ordinates [#bands] corresponding to the spectral_data
        spectral_data (NDArray): 1D [#bands], 2D [N, #bands], 3D [N,M, #bands]
        hull_type (int, optional): 0 = hull quotient, 1 = hull removed, 2 = hull. Defaults to 0.

    Returns:
        NDArray: The hull solution according to the selected hull_type
    """

    def _snail_hull(wavelengths, spectra, hull_type=0):
        
        hq = []
        x_shape = wavelengths.shape[0]
        for val in spectra:
            points = np.concatenate((wavelengths[..., None], val[..., None]), axis=1)
            hull = spatial.ConvexHull(points)

            start = np.where(hull.vertices == x_shape - 1)[0][0]
            stop = np.where(hull.vertices == 0)[0][0]
            # okay need to work from the right hand side to the left and ditch stuff in between e.g. lower convex hull

            if stop < start:
                locations = [int(location) for location in hull.vertices[start:]]
                locations.append(int(stop))
            else:
                locations = [int(location) for location in hull.vertices[start:stop + 1]]
            locations = np.unique(locations)

            # get the hull value at each ordinate value
            y_out = np.interp(wavelengths, wavelengths[locations], val[locations])

            if hull_type == 0:
                hq.append(1.0 - val / y_out)
            elif hull_type == 1:
                hq.append(y_out - val)
            else:
                hq.append(y_out)
        return hq

    # get the size of the input spectral data
    ndims = spectral_data.ndim
    if ndims == 1:
        return np.asarray(_snail_hull(wavelengths, spectral_data[None, ...], hull_type=hull_type))
    elif ndims == 2:
        hull_array = _snail_hull(wavelengths, spectral_data, hull_type=hull_type)
        hull_array = np.asarray(hull_array)
        return hull_array
    elif ndims == 3:
        hull_array = [_snail_hull(wavelengths, spectra, hull_type=hull_type) for spectra in spectral_data]
        return np.resize(np.asarray(hull_array), spectral_data.shape)
    else:
        print('Yeah thats not gonna happen! Your array has ', ndims, ' dimensions.')
        return 0


