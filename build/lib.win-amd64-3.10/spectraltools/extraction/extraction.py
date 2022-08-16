"""
A module for extracting spectral feature information from a supplied spectral dataset and its ordinates.
from spectraltools.extraction import extract_spectral_features
"""

import multiprocessing as mp
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev as cheb
from numpy.typing import NDArray
from scipy.signal import find_peaks
from spectraltools.ext import chulls

warnings.simplefilter('ignore', np.RankWarning)  # stop polyfit rankwarnings

@dataclass
class Features:
    extracted_features: NDArray
    max_features: int
    do_hull: bool
    hull_type: int
    invert: bool
    fit_type: str
    resolution: float
    ordinates_inspection_range:  list[float, float]
    prominence: float
    height: float
    threshold: float
    distance: int
    width: float
    wlen: float


def _mp_leadin(spectral_array: NDArray, ordinates: NDArray, distance: int, max_features: int, prominence: float, height: float, threshold: float, width: int, wlen: float, fit_type: str, resolution: float) -> list:

    chunks = mp.cpu_count()
    sub_arrays = np.array_split(spectral_array, chunks)
    inputs = [(subarray, ordinates, distance, max_features, prominence, height, threshold, width, wlen, fit_type, resolution) for subarray in
            sub_arrays]
    pool = mp.Pool(chunks)
    pool_output = pool.starmap(_mp_process_data, inputs)  # Returns a list of lists
    pool.close()
    pool.join()
    return [entry for val in pool_output for entry in val]


def _mp_process_data(spectral_array, ordinates, distance, max_features, prominence, height, threshold, width, wlen, fit_type, resolution)-> list:
    """
    Used for multiprocessing of image files (since they are usually large). Allows for large speed ups in processing

    Args:
        spectral_array (iterable): the spectral data
        ordinates (numpy): the spectral ordinates associated with the spectral data
        distance (int): how many bands apart at a minimum should features be
        max_features (int): the maximum number of features to return
        prominence (float): a cutoff value for the feature depth (below which its assumed its not a feature)

    Returns: the calculated feature information

    """
    feature_info = []
    if spectral_array.ndim == 2:
        for spectrum in spectral_array:
            # return the peaks_ordinates, prominences and widths
            feature_info.append(_process_signal(spectrum, ordinates, distance=distance,
                                                    max_features=max_features, prominence=prominence,
                                                    height=height, threshold=threshold,
                                                    width=width, wlen=wlen, fit_type=fit_type, resolution=resolution))
    if spectral_array.ndim == 3:
        for row in spectral_array:
            for spectrum in row:
                # return the peaks_ordinates, prominences and widths
                feature_info.append(
                    _process_signal(spectrum, ordinates, distance=distance, max_features=max_features,
                                        prominence=prominence,
                                        height=height, threshold=threshold,
                                        width=width, wlen=wlen, fit_type=fit_type, resolution=resolution))
    return feature_info


def _process_signal(signal: NDArray, ordinates: NDArray, max_features: int = 4, height: float = None, threshold: float = None,
                        distance: int = None, prominence: float = None, width: float = None, wlen: float = None,
                        fit_type: str = 'cheb', resolution: float = None):
    """
    Return the peaks for a given spectrum

    Args:
        signal (ndarray): the signal to be processed (usually a numpy array)

        ordinates (ndarray): the ordinates corresponding to the signal (usually a numpy array)

        distance (int, optional): Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
            Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.

        prominence (number or ndarray or sequence, optional): Required prominence of peaks. Either a number, ``None``,
            an array matching `x` or a 2-element sequence of the former. The first element is always interpreted as the
            minimal and the second, if supplied, as the maximal required prominence.

        max_features (int): maximum number of features to report back

    Returns:
        ndarray: A numpy array of size (9 x number of requested features) values representing the feature parameters for each of
            the found features

    Comments:
        The 9 parameters associated with a single feature are as follows,

        0: feature wavelength

        1: feature depth (given as prominence in the find_peaks routine). These can be considered as relative depths.
        See the explanation of prominence in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

        2: feature width (FWHM)

        3: feature asymmetry. A number between -1 and 1. -1 is heavily left symmetrical, 0 is symmetrical and 1 is
        heavily right symmetrical

        4: feature peak heights. These are different than the feature depths. The feature peak heights are the height
        from the base line to the top of the peak

        5: Wavelength location of the left shoulder of a feature (from a prominence point of view)

        6: Wavelength location of the right shoulder of a feature (from a prominence point of view)

        7: Wavelength location of the left hand side of the FWHM

        8: Wavelength location of the right hand side of the FWHM
    """
    def _remove_excess_features(max_features, peaks, peaks_properties, indices, ordinates):
        if len(indices) >= max_features:
            indices = indices[:max_features]
        peaks = peaks[indices]

        #reduce and reorder the properties
        for key in peaks_properties:
            peaks_properties[key] = peaks_properties[key][indices]

        peak_ordinates = ordinates[peaks]

        return peak_ordinates, peaks_properties


    def _padout_features(max_features, peaks_properties, peak_ordinates):
        ordinates_size = peak_ordinates.size
        if ordinates_size < max_features:
            peak_ordinates = np.pad(peak_ordinates, (0, max_features - ordinates_size), 'constant')
            for key in peaks_properties:
                peaks_properties[key] = np.append(peaks_properties[key], np.zeros(max_features - ordinates_size))
        return peak_ordinates, peaks_properties


    def _create_final_return_values(ordinates, peaks_properties, peak_ordinates):
        left_ips = np.interp(peaks_properties['left_ips'], np.arange(ordinates.shape[0]), ordinates)
        right_ips = np.interp(peaks_properties['right_ips'], np.arange(ordinates.shape[0]), ordinates)
        widths = right_ips - left_ips
        left_bases = np.interp(peaks_properties['left_bases'], np.arange(ordinates.shape[0]), ordinates)
        right_bases = np.interp(peaks_properties['right_bases'], np.arange(ordinates.shape[0]), ordinates)
        asymmetry = np.zeros(left_ips.shape)
        indx = (right_ips - left_ips) > 0
        asymmetry[indx] = 2 * (
                (right_ips[indx] - np.array(peak_ordinates)[indx]) / (right_ips[indx] - left_ips[indx]) - 0.5)

        return np.asarray(peak_ordinates), peaks_properties['prominences'], widths, asymmetry, \
                        peaks_properties['peak_heights'], left_bases, right_bases, left_ips, right_ips


    def _chebyshev_fit(signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen, resolution):

        # see if a user resolution has been set
        if resolution is None:
            resolution = 1.0
        resolution_multiplier = int((ordinates[-1] - ordinates[0]) // resolution)

        # set the default cheb degree
        cheb_deg = 75
        if signal.size // 2 < 75:
            cheb_deg = signal.size // 2

        # fit and interpolate
        chubby = cheb.fit(ordinates, signal, cheb_deg)
        o, s = chubby.linspace(resolution_multiplier)
        end_buffer = distance

        # create an end-buffer where the fit is regarded as untrustworthy
        s = s[(o >= ordinates[end_buffer]) & (o <= ordinates[-end_buffer])]
        o = o[(o >= ordinates[end_buffer]) & (o <= ordinates[-end_buffer])]

        # get the peaks and associated data
        return _raw_fit(s, o, max_features, height, threshold, distance, prominence, width, wlen)


    def _raw_fit(signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen):

        # get the indicies for the peaks. If the width isn't set to 0 (or some other value) it wont return peaks_properties
        peaks, peaks_properties = find_peaks(signal, height=height, threshold=threshold, distance=distance,
                                            prominence=prominence, width=width, wlen=wlen, rel_height=0.5)

        # sort from deepest to smallest
        indices = np.flip(np.argsort(peaks_properties['prominences']))

        # reorder and get rid of additional unwanted features
        peak_ordinates, peaks_properties = _remove_excess_features(max_features, peaks, peaks_properties, indices, ordinates)

        # pad out the results if needed to suit the users number of features request
        peak_ordinates, peaks_properties = _padout_features(max_features, peaks_properties, peak_ordinates)

        # Interpolate the left and right locations of the FWHM to wavelength space
        return _create_final_return_values(ordinates, peaks_properties, peak_ordinates)

    # set the width and height if it hasnt been already
    if width is None:
        width = 0.0
    if height is None:
        height = 0.0

    # see how the user wants to get the features. Will add more as time goes on
    match fit_type:
        case 'cheb':
            return_values = _chebyshev_fit(signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen, resolution)
        case 'raw':
            return_values = _raw_fit(signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen)
        case _:
            return_values = _chebyshev_fit(signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen, resolution)

    return return_values


def extract_spectral_features(instrument_data: NDArray, ordinates: NDArray, max_features: int = 4,
    do_hull: bool = False, hull_type: int = 0, invert: bool = False, main_guard: bool = False,
    fit_type: str = 'cheb', resolution: float = 1.0, ordinates_inspection_range:  Optional[list[float, float]] = None,
    prominence: Optional[float] = None, height: Optional[float] = None, threshold: Optional[float] = None,
    distance: Optional[int] = None, width: Optional[float] = None, wlen: Optional[float] = None) -> Features:
    """
    This is an internal method so we can work with spectral image data.

    Returns:
        feature_info[N, M, 9] where N is the number of rows and M is the number of columns

    Comments:
        The 9 parameters associated with a single feature are as follows:

        0: feature wavelength

        1: feature depth (given as prominence in the find_peaks routine). These can be considered as relative depths.
        See the explanation of prominence in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

        2: feature width (FWHM)

        3:Feature asymmetry. A number between -1 and 1. -1 is heavily left symmetrical, 0 is symmetrical and 1 is
        heavily right symmetrical

        4: feature peak heights. These are different than the feature depths. The feature peak heights are the height
        from the base line to the top of the peak

        5: Wavelength location of the left shoulder of a feature (from a prominence point of view)

        6: Wavelength location of the right shoulder of a feature (from a prominence point of view)

        7: Wavelength location of the left hand side of the FWHM

        8: Wavelength location of the right hand side of the FWHM

    """


    def _hulls_if_required(ordinates_in: NDArray, spectral_array: NDArray, do_hull: bool = False, hull_type: int = 0) -> NDArray:
        # run a hull process if required
        if do_hull:
            spectral_array = chulls.get_absorption(ordinates_in, spectral_array, hull_type=hull_type)
        if hull_type == 3:
            # do a baseline correction. Should really only do this for spectral data
            spectral_array = chulls.get_absorption(ordinates_in, 1.0 - spectral_array, hull_type=1)
        return spectral_array


    def _extract_subset(spectral_data_in: NDArray, ordinates: NDArray, ordinates_inspection_range: list[float, float]) -> tuple[NDArray, NDArray]:
        # subset the data on wavelength if its called for
        range_index = np.searchsorted(ordinates, ordinates_inspection_range)
        ordinates_out = ordinates[range_index[0]:range_index[1] + 1]
        if spectral_data_in.ndim == 1:
            spectral_array = spectral_data_in[range_index[0]:range_index[1] + 1]
        elif spectral_data_in.ndim == 2:
            spectral_array = spectral_data_in[:, range_index[0]:range_index[1] + 1]
        elif spectral_data_in.ndim == 3:
            spectral_array = spectral_data_in[:, :, range_index[0]:range_index[1] + 1]
        else:
            return -99
        spectral_array[spectral_array < 0] = 0
        return ordinates_out,spectral_array


    def _generator(spectral_array: NDArray) -> NDArray:
        """
        a generator for a numpy data array
        The assumption is that the last axis is the actual data while the other leading axis are spatial or sample
        e.g. data_array of shape [N, M] has N sample with M spectral bands, a data array of shape [N, P, M] has N rows by
        P columns with M spectral bands

        If the incoming array is one dimensional it will simply return that spectrum
        :return: an iterator over the spectra
        """
        ndims = spectral_array.ndim
        if ndims == 1:
            yield spectral_array
        elif ndims == 2:
            for spectrum in spectral_array:
                yield spectrum
        elif ndims == 3:
            for row in spectral_array:
                for spectrum in row:
                    yield spectrum
        else:
            return np.zeros([1])


    feature_info = []
    # subset the spectral data if called for
    ordinates_in, spectral_array = _extract_subset(instrument_data, ordinates, ordinates_inspection_range)
    # not really sure if thats help with memory usage
    instrument_data = None
    ordinates = None

    # invert the data if called for
    if invert:
        spectral_array = 1.0 - spectral_array

    # do a hull correction if called for
    spectral_array = _hulls_if_required(ordinates_in, spectral_array, do_hull=do_hull, hull_type=hull_type)

    if spectral_array.ndim == 1:
        # TODO This could be run in multiprocessing
        for signal in _generator(spectral_array):
            # return the peaks_ordinates, prominences and widths
            feature_info.append(
                _process_signal(signal, ordinates_in, distance=distance, max_features=max_features,
                                        prominence=prominence, height=height, threshold=threshold,
                                        width=width, wlen=wlen, fit_type=fit_type, resolution=resolution))
    else:  # multiprocessing
        if main_guard:
            feature_info = _mp_leadin(spectral_array, ordinates_in, distance, max_features, prominence,
                                        height, threshold, width, wlen, fit_type, resolution)
        else:
            if spectral_array.ndim == 3:
                for row in spectral_array:
                    for signal in _generator(row):
                        # return the peaks_ordinates, prominences and widths
                        feature_info.append(
                            _process_signal(signal, ordinates_in, distance=distance,
                                                    max_features=max_features, prominence=prominence,
                                                    height=height, threshold=threshold,
                                                    width=width, wlen=wlen, resolution=resolution))
            if spectral_array.ndim == 2:
                for signal in spectral_array:
                    feature_info.append(
                        _process_signal(signal, ordinates_in, distance=distance,
                                                max_features=max_features, prominence=prominence,
                                                height=height, threshold=threshold,
                                                width=width, wlen=wlen, resolution=resolution))

    match spectral_array.ndim:
        case 1:
            feature_info = np.asarray(feature_info)
        case 2:
            feature_info = np.asarray(feature_info)
        case 3:
            feature_info = np.reshape(np.asarray(feature_info),
                                    (spectral_array.shape[0], spectral_array.shape[1], 9, max_features))

    package = Features(feature_info, max_features, do_hull, hull_type, invert,
    fit_type, resolution, ordinates_inspection_range, prominence, height, threshold, distance, width, wlen)

    return package


@dataclass
class FeatureExtraction:
    """
    A class for extracting spectral feature information from a spectral dataset. The dataset is represented by an
    instance of the Instrument class e.g. from `spex.io.instruments`

    The features that are returned (either peaks or absorptions depending on how you set it up) are not just
    randomly selected between your ordinate_inspection_range they are ordered. Meaning if you said give me 4
    features then it is the 4 deepest features in descending order.

    The actual calculation of the location of the peaks is done via the find_peaks routine in scikit-learn. To get
    actual peak locations in a wavelength format they are further inferred from the fitting of a 3 point
    quadratic (since the band where the deepest point occurs might not technically be the actual location of the
    deepest feature) and solving ``dy/dx = 0 = -b^2/2a``

    find_peaks: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    I would recommend having a read through the various routines and returns from fit_peaks so you can get to grips
    with what it actually being returned from this class.

    It is up to the user to select the appropriate processing required in terms of hull corrections via the
    ``do_hull`` keyword and/or invert keyword.

    Warnings:
        YOU CAN DO SOME REALLY WEIRD STUFF IF YOU DON'T PAY ATTENTION

        If you are chasing the spectral absorption features.If the incoming data is spectral reflectance then set
        ``do_hull=True`` and ``hull_type`` to your preferred. If the data is already in the appropriate format i.e. hull removed
        etc, then this isn't needed and do_hull=False should be set.

        If the incoming spectra are TIR reflectance spectra then you set ``do_hull=False`` and ``hull_type=3``
        A bit weird but it tells the class that what you actually want is a baseline correction. If you don't want a
        baseline correction then just leave hull_type as the default. As an aside, the hull type actually applied with
        those settings is a hull removed type so it does not alter the heights of the peaks during the baseline
        correction.

        When you do the above with TIR spectra you are actually finding the locations of the peaks!!!
        Not the reflectance absorptions. If we assume ``1=R+E+T`` (reflectance + emissivity + transmittance) then this is
        finding the locations of the emissivity absorptions (assuming T is zero).

    If you want to use a different hull type for TIR then do the following, set ``invert=True`` (does a 1-spectra),
    set ``do_hull=True`` (this acts as a baseline correction) and set ``hull_type`` to your preference.

    If you want the reflectance absorptions (emissivity peaks) of your TIR spectra then set ``do_hull=True``,
    ``invert=False`` and ``hull_type=0`` (hull quotient) or ``1`` (hull removed). There is no baseline correction here though.

    So lots of weirdness can ensue if you don't pay attention to the keywords and your spectra.

    Attributes:
        dimension_shape (tuple): The shape of the spectral data set

        dimensions (int): how many dimensions are in the spectral dataset

        do_hull (bool): using a hull correction prior to feature extraction

        feature_info (ndarray): the extracted feature information

        distance (int): closest allowable feature distance

        feature_search_indices (ndarray): A boolean array of the same sample size as the spectral data where the
            feature_search_space criteria was met.

        feature_search_parameters (ndarray): The result of running a feature search over the extracted spectral features

        feature_search_space (list): A list of tuples designating the search criteria of the feature_info

        hull_type: The hull type used in the feature extraction

        instrument (obj): An instance of the `spex.io.instruments` class

        invert (bool): Should the spectral data be inverted prior to feature extraction

        max_features (int): How many spectral features to find

        ordinates (ndarray): the ordinates of the spectral data

        ordinates_inspection_range (list): The spectral range over which to perform feature extraction

        prominence_depth (float): minimum acceptable prominence depth

        range_index (list): the indices of the ordinates array that correspond to the ordinates_inspection_range

    """
    instrument_data: NDArray
    ordinates: NDArray
    max_features: int = 4
    do_hull: bool = False
    hull_type: int = 0
    invert: bool = False
    main_guard: bool = False
    fit_type: str = 'cheb'
    resolution: float = 1.0
    ordinates_inspection_range:  Optional[list[float, float]] = None
    prominence: Optional[float] = None
    height: Optional[float] = None
    threshold: Optional[float] = None
    distance: Optional[int] = None
    width: Optional[float] = None
    wlen: Optional[float] = None

    def __post_init__(self):
        if self.ordinates_inspection_range is None:
            self.ordinates_inspection_range = [self.ordinates[0], self.ordinates[-1]]
            self.spectral_features = None

    # # clustering parameters
    # self.scalar = None
    # self.clusterer = None
    # self.features_to_include = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    # self.number_of_features = 2
    # self.feature = 0
    # self.min_cluster_size = 5
    # self.min_samples = None
    # self.cluster_selection_epsilon = 0.0
    # self.cluster_selection_method = 'eom'
    # self.allow_single_cluster = False
    # self.prediction_data = False

    def extract_features(self):
        """
        Process the spectral data according to the class instantiation variables and return the feature results

        Returns:
            An array of values for each found feature (all zeros if no feature is found). The array has dimensions
            of (N x 9 x number of features requested), where is N are the number of spectral samples. The features are given
            from the largest to the smallest so entry features[0,:,0] is for the first feature, feature[0,:,1] is for the
            second and so on.

        Comments:
            The 9 parameters associated with a single feature are as follows:

            0: feature wavelength

            1: feature depth (given as prominence in the find_peaks routine). These can be considered as relative depths.
            See the explanation of prominence in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

            2: feature width (FWHM)

            3: feature asymmetry. A number between -1 and 1. -1 is heavily left symmetrical, 0 is symmetrical and 1 is
            heavily right symmetrical

            4: feature peak heights. These are different than the feature depths. The feature peak heights are the height
            from the base line to the top of the peak

            5: Wavelength location of the left shoulder of a feature (from a prominence point of view)

            6: Wavelength location of the right shoulder of a feature (from a prominence point of view)

            7: Wavelength location of the left hand side of the FWHM

            8: Wavelength location of the right hand side of the FWHM

        """

        self._process_spectral_array()
        return self.spectral_features

        # todo get rid of the range check. if folks want to search outside of their range then let them.


    def _generator(self, spectral_array: NDArray) -> NDArray:
        """
        a generator for a numpy data array
        The assumption is that the last axis is the actual data while the other leading axis are spatial or sample
        e.g. data_array of shape [N, M] has N sample with M spectral bands, a data array of shape [N, P, M] has N rows by
        P columns with M spectral bands

        If the incoming array is one dimensional it will simply return that spectrum
        :return: an iterator over the spectra
        """
        ndims = spectral_array.ndim
        if ndims == 1:
            yield spectral_array
        elif ndims == 2:
            for spectrum in spectral_array:
                yield spectrum
        elif ndims == 3:
            for row in spectral_array:
                for spectrum in row:
                    yield spectrum
        else:
            return np.zeros([1])


    def _process_spectral_array(self) -> NDArray:
        """
        This is an internal method so we can work with spectral image data.

        Returns:
            feature_info[N, M, 9] where N is the number of rows and M is the number of columns

        Comments:
            The 9 parameters associated with a single feature are as follows:

            0: feature wavelength

            1: feature depth (given as prominence in the find_peaks routine). These can be considered as relative depths.
            See the explanation of prominence in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

            2: feature width (FWHM)

            3:Feature asymmetry. A number between -1 and 1. -1 is heavily left symmetrical, 0 is symmetrical and 1 is
            heavily right symmetrical

            4: feature peak heights. These are different than the feature depths. The feature peak heights are the height
            from the base line to the top of the peak

            5: Wavelength location of the left shoulder of a feature (from a prominence point of view)

            6: Wavelength location of the right shoulder of a feature (from a prominence point of view)

            7: Wavelength location of the left hand side of the FWHM

            8: Wavelength location of the right hand side of the FWHM

        """
        feature_info = []
        # subset the spectral data if called for
        ordinates_in, spectral_array = self._extract_subset()

        # invert the data if called for
        if self.invert:
            spectral_array = 1.0 - spectral_array

        # do a hull correction if called for
        spectral_array = self._hulls_if_required(ordinates_in, spectral_array)

        if self.instrument_data.ndim == 1:
            # TODO This could be run in multiprocessing
            for signal in self._generator(spectral_array):
                # return the peaks_ordinates, prominences and widths
                feature_info.append(
                    _process_signal(signal, ordinates_in, distance=self.distance, max_features=self.max_features,
                                            prominence=self.prominence, height=self.height, threshold=self.threshold,
                                            width=self.width, wlen=self.wlen, fit_type=self.fit_type, resolution=self.resolution))
        else:  # multiprocessing
            if self.main_guard:
                feature_info = _mp_leadin(spectral_array, ordinates_in, self.distance, self.max_features, self.prominence,
                                            self.height, self.threshold, self.width, self.wlen, self.fit_type, self.resolution)
            else:
                if self.instrument_data.ndim == 3:
                    for row in spectral_array:
                        for signal in self._generator(row):
                            # return the peaks_ordinates, prominences and widths
                            feature_info.append(
                                _process_signal(signal, ordinates_in, distance=self.distance,
                                                        max_features=self.max_features, prominence=self.prominence,
                                                        height=self.height, threshold=self.threshold,
                                                        width=self.width, wlen=self.wlen, resolution=self.resolution))
                if self.instrument_data.ndim == 2:
                    for signal in spectral_array:
                        feature_info.append(
                            _process_signal(signal, ordinates_in, distance=self.distance,
                                                    max_features=self.max_features, prominence=self.prominence,
                                                    height=self.height, threshold=self.threshold,
                                                    width=self.width, wlen=self.wlen, resolution=self.resolution))

        match self.instrument_data.ndim:
            case 1:
                feature_info = np.asarray(feature_info)
            case 2:
                feature_info = np.asarray(feature_info)
            case 3:
                feature_info = np.reshape(np.asarray(feature_info),
                                        (spectral_array.shape[0], spectral_array.shape[1], 9, self.max_features))

        self.spectral_features = feature_info


    def _hulls_if_required(self, ordinates_in, spectral_array):
        # run a hull process if required
        if self.do_hull:
            spectral_array = chulls.get_absorption(ordinates_in, spectral_array, hull_type=self.hull_type)
        if self.hull_type == 3:
            # do a baseline correction. Should really only do this for spectral data
            spectral_array = chulls.get_absorption(ordinates_in, 1.0 - spectral_array, hull_type=1)
        return spectral_array


    def _extract_subset(self):
        # subset the data on wavelength if its called for
        range_index = np.searchsorted(self.ordinates, self.ordinates_inspection_range)
        ordinates_in = self.ordinates[range_index[0]:range_index[1] + 1]
        if self.instrument_data.ndim == 1:
            spectral_array = self.instrument_data[range_index[0]:range_index[1] + 1]
        elif self.instrument_data.ndim == 2:
            spectral_array = self.instrument_data[:, range_index[0]:range_index[1] + 1]
        elif self.instrument_data.ndim == 3:
            spectral_array = self.instrument_data[:, :, range_index[0]:range_index[1] + 1]
        else:
            return -99
        spectral_array[spectral_array < 0] = 0
        return ordinates_in,spectral_array
