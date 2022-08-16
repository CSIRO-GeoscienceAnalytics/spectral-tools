import numpy as np
from scipy.spatial import ConvexHull


def get_absorption(wavelengths, spectral_data, hull_type=0):
    """
    Get either the hull quotient, hull removed, or the hull of the input spectral_data

    Args:
        wavelengths (ndarray): A numpy array of ordinate wavelengths [M]

        spectral_data (ndarray): A numpy array of spectral data 1D: [M], 2D: [N,M], 3D: [N,P,M]

        hull_type (int): 0: Hull Quotient (default), 1:Hull Removed, 2: Hull Only

    Returns:
        ndarray: hull corrected spectra corresponding to the hull_type selected

    """
    def do_one(wavelengths_in, spectrum, hull_type_in='upper'):
        """
        Calculates the upper convex hull of the spectrum and interpolates the result to the input wavelengths

        Args:
            wavelengths_in (ndarray): numpy array of wavelengths [M]

            spectrum (ndarray): numpy array of the spectrum [M]

            hull_type_in (str): 'upper' or 'lower'. Its fixed for this though.

        Returns:
            ndarray: numpy array of the hull [M]

        """
        points = zip(wavelengths_in, spectrum)
        temp = convex_hull(points, hull_type_in=hull_type_in)
        temp_2 = np.asarray(sorted(temp))
        return np.interp(wavelengths, temp_2[:, 0], temp_2[:, 1])

    def which_type_of_hull(spectral_data_in, hull_array_in, hull_type_in=0):
        """
        Performs the hull correction to the incoming spectral data.
        Args:
            spectral_data_in (ndarray): the input spectral data

            hull_array_in (ndarray): the hull data (same length as spectral_data_in

            hull_type_in (int): 0 is hull quotient, 1 is hull removed. Default is 0

        Returns:
            ndarray: the hull corrected spectra

        """
        if hull_type_in == 0:
            return 1.0 - spectral_data_in/hull_array_in
        elif hull_type_in == 1:
            return hull_array_in - spectral_data_in
        else:
            return hull_array_in

    def convex_hull(points, hull_type_in="upper"):
        """
        Computes the convex hull of a set of 2D points. Implements Andrew's monotone chain algorithm. O(n log n) complexity.

        Args:
            points (tuple): (x,y) pairs representing the points

            hull_type_in (str): "upper" or "lower" Default: "upper"

        Returns:
            a list of vertices of the convex hull in counter-clockwise order starting from the vertex with the
            lexicographically smallest coordinates.

        """

        def cross(o, a, b):
            """
            calculate the cross product

            """
            # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
            # Returns a positive value, if OAB makes a counter-clockwise turn,
            # negative for clockwise turn, and zero if the points are collinear.
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def upper_hull(points_in):
            """
            builds the upper hull and returns it

            Args:
                points_in (points): the points

            Returns:
                the upper hull

            """
            # build and return the upper hull
            upper = []
            for p in reversed(points_in):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            return upper

        def lower_hull(points_in):
            """
            builds the lower hull

            Args:
                points_in (points): the points

            Returns:
                the lower hull

            """
            # Build and return the lower hull
            lower = []
            for p in points_in:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            return lower

        # Sort the points lexicographically (tuples are compared lexicographically).
        # Remove duplicates to detect the case we have just one unique point.
        points = sorted(set(points))

        # Boring case: no points or a single point, possibly repeated multiple times.
        if len(points) <= 1:
            return points

        if hull_type_in == "upper":
            # Build upper hull
            return upper_hull(points)
        elif hull_type_in == "lower":
            return lower_hull(points)
        else:
            low = lower_hull(points)
            up = upper_hull(points)
            # Concatenation of the lower and upper hulls gives the convex hull.
            # Last point of each list is omitted because it is repeated at the beginning of the other list.
            return low[:-1] + up[:-1]

    def snail_hull(wavelengths, spectra, hull_type=0):
        """
    Get either the hull quotient, hull removed, or the hull of the input spectral_data

    Args:
        wavelengths (ndarray): A numpy array of ordinate wavelengths [M]

        spectra (ndarray): A numpy array of spectral data 1D: [M], 2D: [N,M], 3D: [N,P,M]

        hull_type (int): 0: Hull Quotient (default), 1:Hull Removed, 2: Hull Only

    Returns:
        ndarray: hull corrected spectra corresponding to the hull_type selected        """
        hq = []
        x_shape = wavelengths.shape[0]
        for val in spectra:
            points = np.concatenate((wavelengths[..., None], val[..., None]), axis=1)
            hull = ConvexHull(points)

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
        return np.asarray(snail_hull(wavelengths, spectral_data[None, ...], hull_type=hull_type))
    elif ndims == 2:
        hull_array = snail_hull(wavelengths, spectral_data, hull_type=hull_type)
        hull_array = np.asarray(hull_array)
        return hull_array
    elif ndims == 3:
        hull_array = [snail_hull(wavelengths, spectra, hull_type=hull_type) for spectra in spectral_data]
        return np.resize(np.asarray(hull_array), spectral_data.shape)
    else:
        print('Yeah thats not gonna happen! Your array has ', ndims, ' dimensions.')
        return 0


