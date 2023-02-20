"""
A collection of routines for feature extraction
"""
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev as cheb
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from csiro_spectral_tools.hulls.convexhulls import uc_hulls

warnings.simplefilter("ignore", np.RankWarning)  # stop polyfit rankwarnings


@dataclass
class Features:
    """
    The Features class is the output from running spectraltools.extraction.extract_spectral_features
    Besides the actual features found it contains all the options that were used to run the extraction.
    """

    extracted_features: NDArray
    max_features: int
    do_hull: bool
    hull_type: int
    invert: bool
    fit_type: str
    resolution: float
    ordinates_inspection_range: Tuple[float, float]
    prominence: float
    height: float
    threshold: float
    distance: int
    width: float
    wlen: float


def _mp_leadin(
    spectral_array: NDArray,
    ordinates: NDArray,
    distance: int,
    max_features: int,
    prominence: float,
    height: float,
    threshold: float,
    width: int,
    wlen: float,
    fit_type: str,
    resolution: float,
) -> list:
    """_mp_leadin
    This method preps the incoming data so it can be run with your laptop/pc multiprocessing.
    To do this requires the main code is run in a main guard. If its not and you say it is then expect havoc to ensue.

    Args:
        spectral_array (NDArray): 2D [N, #bands] or 3D [N, M, #bands]
        ordinates (NDArray): ordinates of the incoming spectral_array [#bands]
        distance (int): The closest distance in bands that one feature can be to another
        max_features (int): How many features you want returned. If less are found they are returned as zeros
        prominence (float): minimum prominence to consider. prominence is like a contour height
        height (float): minimum base to peak height to consider
        threshold (float): unused
        width (int): minimum width in bands to consider
        wlen (float): buffer around peaks for defining a feature. This can do weird stuff. I leave it as None
        fit_type (str): 'cheb' or 'raw': default is 'cheb'. Produces smoothing in noisy spectra and lets you define the output resolution
        resolution (float): desired output resolution

    Returns:
        list: The calculated spectral features. The incoming spectral arrays dimension are preserved.
    """

    chunks = mp.cpu_count()
    sub_arrays = np.array_split(spectral_array, chunks)
    inputs = [
        (
            subarray,
            ordinates,
            distance,
            max_features,
            prominence,
            height,
            threshold,
            width,
            wlen,
            fit_type,
            resolution,
        )
        for subarray in sub_arrays
    ]
    pool = mp.Pool(chunks)
    pool_output = pool.starmap(_mp_process_data, inputs)  # Returns a list of lists
    pool.close()
    pool.join()
    return [entry for val in pool_output for entry in val]


def _mp_process_data(
    spectral_array,
    ordinates,
    distance,
    max_features,
    prominence,
    height,
    threshold,
    width,
    wlen,
    fit_type,
    resolution,
) -> list:
    """
    Used for multiprocessing of image files (since they are usually large).
                Allows for large speed ups in processing

    Args:
        spectral_array (NDArray): 2D [N, #bands] or 3D [N, M, #bands]
        ordinates (NDArray): ordinates of the incoming spectral_array [#bands]
        distance (int): The closest distance in bands that one feature can be to another
        max_features (int): How many features you want returned. If less are found they are returned as zeros
        prominence (float): minimum prominence to consider. prominence is like a contour height
        height (float): minimum base to peak height to consider
        threshold (float): unused
        width (int): minimum width in bands to consider
        wlen (float): buffer around peaks for defining a feature. This can do weird stuff. I leave it as None
        fit_type (str): 'cheb' or 'raw': default is 'cheb'. Produces smoothing in noisy spectra and lets you
                define the output resolution
        resolution (float): desired output resolution

    Returns: the calculated feature information

    """
    # feature_info = []
    if spectral_array.ndim == 2:
        feature_info = [
            _process_signal(
                spectrum,
                ordinates,
                distance=distance,
                max_features=max_features,
                prominence=prominence,
                height=height,
                threshold=threshold,
                width=width,
                wlen=wlen,
                fit_type=fit_type,
                resolution=resolution,
            )
            for spectrum in spectral_array
        ]
    if spectral_array.ndim == 3:
        feature_info = [
            _process_signal(
                spectrum,
                ordinates,
                distance=distance,
                max_features=max_features,
                prominence=prominence,
                height=height,
                threshold=threshold,
                width=width,
                wlen=wlen,
                fit_type=fit_type,
                resolution=resolution,
            )
            for row in spectral_array
            for spectrum in row
        ]
    return feature_info


def _process_signal(
    signal: NDArray,
    ordinates: NDArray,
    max_features: int = 4,
    height: float = None,
    threshold: float = None,
    distance: int = None,
    prominence: float = None,
    width: float = None,
    wlen: float = None,
    fit_type: str = "cheb",
    resolution: float = None,
):
    """_process_signal _summary_

    Args:
        spectral_array (NDArray): 2D [N, #bands] or 3D [N, M, #bands]
        ordinates (NDArray): ordinates of the incoming spectral_array [#bands]
        distance (int): The closest distance in bands that one feature can be to another
        max_features (int): How many features you want returned. If less are found they are returned as zeros
        prominence (float): minimum prominence to consider. prominence is like a contour height
        height (float): minimum base to peak height to consider
        threshold (float): unused
        width (int): minimum width in bands to consider
        wlen (float): buffer around peaks for defining a feature. This can do weird stuff. I leave it as None
        fit_type (str): 'cheb' or 'raw': default is 'cheb'. Produces smoothing in noisy spectra and lets you
                      define the output resolution
        resolution (float): desired output resolution

    """

    def _remove_excess_features(max_features, peaks, peaks_properties, indices, ordinates):
        """_remove_excess_features"""
        if len(indices) >= max_features:
            indices = indices[:max_features]
        peaks = peaks[indices]

        # reduce and reorder the properties
        for key in peaks_properties:
            peaks_properties[key] = peaks_properties[key][indices]

        peak_ordinates = ordinates[peaks]

        return peak_ordinates, peaks_properties

    def _padout_features(max_features, peaks_properties, peak_ordinates):
        ordinates_size = peak_ordinates.size
        if ordinates_size < max_features:
            peak_ordinates = np.pad(peak_ordinates, (0, max_features - ordinates_size), "constant")
            for key in peaks_properties:
                peaks_properties[key] = np.append(
                    peaks_properties[key], np.zeros(max_features - ordinates_size)
                )
        return peak_ordinates, peaks_properties

    def _create_final_return_values(ordinates, peaks_properties, peak_ordinates):
        xp = np.arange(ordinates.shape[0])
        values_in = [
            peaks_properties["left_ips"],
            peaks_properties["right_ips"],
            peaks_properties["left_bases"],
            peaks_properties["right_bases"],
        ]

        values_out = np.interp(values_in, xp, ordinates)
        widths = values_out[1] - values_out[0]

        indx = widths > 0
        asymmetry = np.zeros(values_out[0].shape)
        asymmetry[indx] = 2 * (
            (values_out[1][indx] - np.array(peak_ordinates)[indx])
            / (values_out[1][indx] - values_out[0][indx])
            - 0.5
        )

        return (
            np.asarray(peak_ordinates),
            peaks_properties["prominences"],
            widths,
            asymmetry,
            peaks_properties["peak_heights"],
            values_out[2],
            values_out[3],
            values_out[0],
            values_out[1],
        )

    def _chebyshev_fit(
        signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen, resolution
    ):
        # set a distance if it hasnt already
        if distance is None:
            distance = 2

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
        interp_function = interp1d(ordinates, chubby(ordinates), kind="cubic", fill_value="extrapolate")
        o = np.linspace(ordinates[0], ordinates[-1], resolution_multiplier)
        s = interp_function(o)
        end_buffer = distance

        # create an end-buffer where the fit is regarded as untrustworthy
        s = s[(o >= ordinates[end_buffer]) & (o <= ordinates[-end_buffer])]
        o = o[(o >= ordinates[end_buffer]) & (o <= ordinates[-end_buffer])]

        # get the peaks and associated data
        return _raw_fit(s, o, max_features, height, threshold, distance, prominence, width, wlen)

    def _raw_fit(signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen):
        # get the indicies for the peaks. If the width isn't set to 0 (or some other value) it wont return peaks_properties
        peaks, peaks_properties = find_peaks(
            signal,
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            width=width,
            wlen=wlen,
            rel_height=0.5,
        )

        # sort from deepest to smallest
        indices = np.flip(np.argsort(peaks_properties["prominences"]))

        # reorder and get rid of additional unwanted features
        peak_ordinates, peaks_properties = _remove_excess_features(
            max_features, peaks, peaks_properties, indices, ordinates
        )

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
    if fit_type == "cheb":
        return_values = _chebyshev_fit(
            signal,
            ordinates,
            max_features,
            height,
            threshold,
            distance,
            prominence,
            width,
            wlen,
            resolution,
        )
    elif fit_type == "raw":
        return_values = _raw_fit(
            signal, ordinates, max_features, height, threshold, distance, prominence, width, wlen
        )
    else:
        return_values = _chebyshev_fit(
            signal,
            ordinates,
            max_features,
            height,
            threshold,
            distance,
            prominence,
            width,
            wlen,
            resolution,
        )

    return return_values


def extract_spectral_features(
    instrument_data: NDArray,
    ordinates: NDArray,
    max_features: int = 4,
    do_hull: bool = False,
    hull_type: int = 0,
    invert: bool = False,
    main_guard: bool = False,
    fit_type: str = "cheb",
    resolution: float = 1.0,
    ordinates_inspection_range: Optional[Tuple[float, float]] = None,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    threshold: Optional[float] = None,
    distance: Optional[int] = None,
    width: Optional[float] = None,
    wlen: Optional[float] = None,
) -> Features:
    """extract_spectral_features
    This routine will extract spectral features from the input spectral data.

    Args:
        instrument_data (NDArray): incoming spectral data array 1D, 2D [N, Bands], 3D [Row, Col, Bands]
        ordinates (NDArray): ordinates corresponding to the spectral array [Bands]
        max_features (int, optional): Maximum number of features to look for. Defaults to 4.
        do_hull (bool, optional): Apply a hull correction prior to looking for features. Need to do this with
                                        reflectance. Defaults to False.
        hull_type (int, optional): 0 - hull quotient, 1 - hull removal, 2 - return hull. Defaults to 0.
        invert (bool, optional): Invert the incoming spectra. Defaults to False.
        main_guard (bool, optional): If run in a main guard makes use of mutliproccessing. Defaults to False.
        fit_type (str, optional): 'cheb', 'raw', 'crude'. Defaults to 'cheb'.
        resolution (float, optional): if fit_type = 'cheb' this defines the desired resolution of the result.
                                        Defaults to 1.0.
        ordinates_inspection_range (Optional[list[float, float]], optional): start and stop wavelengths over
                                        which to look. Defaults to None.
        prominence (Optional[float], optional):minimum prominence value to consider. Defaults to None.
        height (Optional[float], optional): minimum base to peak height to consider. Defaults to None.
        threshold (Optional[float], optional): nothing at the moment. Not implemented. Defaults to None.
        distance (Optional[int], optional): minimum distance in bands that features are allowed to exist
                                        together. Defaults to None.
        width (Optional[float], optional): minimum width to consider. Defaults to None.
        wlen (Optional[float], optional): minimum distance around a feature to use
                                        (setting this can cause issues). Defaults to None.

    Returns:
        Features: a class containing various input selections and of course the features found

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

    def _crude_fit(signal_array, ordinates):
        depths = signal_array[..., :].max(axis=(signal_array.ndim - 1))
        waves = ordinates[signal_array[..., :].argmax(axis=(signal_array.ndim - 1))]
        return depths, waves

    def _hulls_if_required(
        ordinates_in: NDArray, spectral_array: NDArray, do_hull: bool = False, hull_type: int = 0
    ) -> NDArray:
        # run a hull process if required
        if do_hull:
            spectral_array = uc_hulls(ordinates_in, spectral_array, hull_type=hull_type)
        if hull_type == 3:
            # do a baseline correction. Should really only do this for spectral data
            spectral_array = uc_hulls(ordinates_in, 1.0 - spectral_array, hull_type=1)
        return spectral_array

    def _extract_subset(
        spectral_data_in: NDArray, ordinates: NDArray, ordinates_inspection_range: Tuple[float, float]
    ) -> tuple[NDArray, NDArray]:
        # subset the data on wavelength if its called for
        range_index = np.searchsorted(ordinates, ordinates_inspection_range)
        ordinates_out = ordinates[range_index[0] : range_index[1] + 1]
        if spectral_data_in.ndim == 1:
            spectral_array = spectral_data_in[range_index[0] : range_index[1] + 1]
        elif spectral_data_in.ndim == 2:
            spectral_array = spectral_data_in[:, range_index[0] : range_index[1] + 1]
        elif spectral_data_in.ndim == 3:
            spectral_array = spectral_data_in[:, :, range_index[0] : range_index[1] + 1]
        else:
            return -99
        spectral_array[spectral_array < 0] = 0
        return ordinates_out, spectral_array

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

    if fit_type != "crude":
        if spectral_array.ndim == 1:
            # This could be run in multiprocessing. I(N FACT THIS IS PROBS WRONG IF ITS A SINGLE SPECTRUM!!)
            for signal in _generator(spectral_array):
                # return the peaks_ordinates, prominences and widths
                feature_info.append(
                    _process_signal(
                        signal,
                        ordinates_in,
                        distance=distance,
                        max_features=max_features,
                        prominence=prominence,
                        height=height,
                        threshold=threshold,
                        width=width,
                        wlen=wlen,
                        fit_type=fit_type,
                        resolution=resolution,
                    )
                )
        else:  # multiprocessing
            if main_guard:
                feature_info = _mp_leadin(
                    spectral_array,
                    ordinates_in,
                    distance,
                    max_features,
                    prominence,
                    height,
                    threshold,
                    width,
                    wlen,
                    fit_type,
                    resolution,
                )
            else:
                if spectral_array.ndim == 3:
                    feature_info = [
                        _process_signal(
                            signal,
                            ordinates_in,
                            distance=distance,
                            max_features=max_features,
                            prominence=prominence,
                            height=height,
                            threshold=threshold,
                            width=width,
                            wlen=wlen,
                            resolution=resolution,
                        )
                        for row in spectral_array
                        for signal in row
                    ]

                if spectral_array.ndim == 2:
                    feature_info = [
                        _process_signal(
                            signal,
                            ordinates_in,
                            distance=distance,
                            max_features=max_features,
                            prominence=prominence,
                            height=height,
                            threshold=threshold,
                            width=width,
                            wlen=wlen,
                            resolution=resolution,
                        )
                        for signal in spectral_array
                    ]

        if spectral_array.ndim == 1:
            feature_info = np.asarray(feature_info)
        elif spectral_array.ndim == 2:
            feature_info = np.asarray(feature_info)
        elif spectral_array.ndim == 3:
            feature_info = np.reshape(
                np.asarray(feature_info),
                (spectral_array.shape[0], spectral_array.shape[1], 9, max_features),
            )
        package = Features(
            feature_info,
            max_features,
            do_hull,
            hull_type,
            invert,
            fit_type,
            resolution,
            ordinates_inspection_range,
            prominence,
            height,
            threshold,
            distance,
            width,
            wlen,
        )
    else:
        depths, waves = _crude_fit(spectral_array, ordinates_in)
        feature_info = np.stack([waves, depths], axis=-1)
        package = Features(
            feature_info,
            1,
            do_hull,
            hull_type,
            invert,
            fit_type,
            resolution,
            ordinates_inspection_range,
            prominence,
            height,
            threshold,
            distance,
            width,
            wlen,
        )

    return package
