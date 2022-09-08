"""
THIS IS A MESS AT THIS STAGE AND SHOULD BE IGNORED. I NEED TO CLEAN IT UP

SO the documentation is this: DON'T USE ME

The references internally refer to data sources that I am almost certain I have set up incorrectly in the package. By this
I mean its okat to call them how I do when its on my machine but when someone installs the package on their machine via
`python setup.py install` I think its meant to use resource packages.

So this I will fix a bit later
"""
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
from plotly import graph_objects as go
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr
from sklearn.decomposition import non_negative_factorization
from sklearn.preprocessing import normalize
from spectraltools.ext.convexhulls import uc_hulls


class SpectralMix(object):
    def __init__(self, instrument_wavelengths, spectral_input, spectral_range=None, hull=True, tir=False, library=0):
        """
        Initialisation routine for the SpectralMix class object. This class attempts to perform spectral unmixing
        according to the users selected spectral library (3 available - 1 VNIR/SWIR, 2 TIR). An NMF model is used to
        perform linear spectral unmixing.

        Args:
            instrument_wavelengths (ndarry):
            spectral_input ()ndarray:
            spectral_range (list):
            hull (bool):
            tir (bool):
            library (int):
        """
        libraries = ['merged_vnswir_culled', 'tsg_tir_ms9', 'tsg_jhu_merge_tir']
        stream = pkg_resources.resource_stream(__name__, 'spectral_libraries/colors_tsg.csv')
        self.colors = pd.read_csv(stream)

        self.spectral_input = spectral_input
        self.instrument_wavelengths = instrument_wavelengths

        self.library_df = None
        self.library_spectra = None
        self.library = library
        self.library_wavelengths = None

        self.instrument_library_spectra = None
        self.instrument_library_df = None

        self.average_instrument_library_spectra = None
        self.average_instrument_library_df = None

        # if any cuts are made to the spectral library e.g. drop spectra for whatever reason then the result is saved
        # to the culled_df and culled_spectra values so as not to destroy the original datasets
        self.culled_df = None
        self.culled_spectra = None
        self.which_culled = None
        self.cull_type = None

        self.hull = hull
        self.hull_type = 1
        self.spectral_range = spectral_range
        self.tir = tir
        if self.tir:
            self.hull = False

        self.nmf_results = None
        # this can take on one of three values: full, average or culled
        self.which_nmf = None
        self.rms_fit_error = None
        self.r2_fit_error = None

        #library = os.path.join(library_directory, libraries[library])
        # set some other variables up
        # ensure that the library doesnt require extrapolation
        library = libraries[library]
        self.set_library(library)
        self.convert_library_to_instrument()
        self.make_average_instrument_library()
        self.range_indices = np.searchsorted(self.instrument_wavelengths, spectral_range) 

    def set_external_library(self, wavelengths, spectra, names):
        """
        Use external user supplied spectra as the spectral library
        Args:
            wavelengths (ndarry): An array of wavelengths (B) in nanometers corresponding to the number of entries in an
                individual spectrum contained in spectra.
            spectra (ndarray): An array of spectra (NxB) representing a user defined spectral endmember
            names (list): A list of names (N) for each spectrum in spectra

        Returns:
            Nothing: Sets internal variables

        """
        self.library_df = pd.DataFrame(names, columns=['mineral'])
        self.library_spectra = spectra
        self.library_wavelengths = wavelengths

        self.convert_library_to_instrument()
        self.make_average_instrument_library()
        self.range_indices = np.searchsorted(self.instrument_wavelengths, self.spectral_range) #find_indices(self.spectral_range, self.instrument_wavelengths)

    def set_library(self, library=None):
        """
        Set which spectral library to use (from the 3 inbuilt available)
        Args:
            library (str): The library to use ofr calculations

        Returns:
            Nothing: Sets internal variables
        """
        if library is None:
            print("Nope. You didn't select a library")
        else:
            library_df = library + '_df.csv'
            library_spectra = library + '_spectra.csv'
            library_df_read = pkg_resources.resource_stream(__name__, 'spectral_libraries/' + library_df)
            self.library_df = pd.read_csv(library_df_read, index_col=0)
            library_spectra_read = pkg_resources.resource_stream(__name__, 'spectral_libraries/' + library_spectra)
            spectra_df = pd.read_csv(library_spectra_read, index_col=0)
            self.library_spectra = spectra_df.values
            self.library_wavelengths = 1000.0 * spectra_df.columns.values.astype('float')

    def get_library(self):
        """
        Return a pandas dataframe of entries in the spectral library and a numpy array of the spectra.
        Returns:
            Dataframe: The ancillary data associated with the spectral library
            ndarray: A numpy array of spectral data corresponding to the library

        """
        return self.library_df, self.library_spectra

    def convert_library_to_instrument(self):
        """
        Converts the internal spectral library to the same spectral domain as the spectral data that is to be analysed.

        Returns:
            Nothing: Sets internal variables

        """
        cs = CubicSpline(self.library_wavelengths, self.library_spectra, extrapolate=False, axis=1)
        # want to exclude any wavelengths that required extrapolation
        in_range = np.unique(np.where(np.isfinite(cs(self.instrument_wavelengths)))[1])
        self.instrument_library_spectra = cs(self.instrument_wavelengths)[:, in_range]
        self.instrument_library_df = self.library_df.copy()
        # reassign the instrument data so the range matches the library
        self.instrument_wavelengths = self.instrument_wavelengths[in_range]
        self.spectral_input = self.spectral_input[:, in_range]

    def get_instrument_library(self):
        """
        Return a pandas dataframe of entries in the spectral library and a numpy array of the spectra at the same
            spectral space as the incoming spectral data.

        Returns:
            Dataframe: The ancillary data associated with the spectral library
            ndarray: A numpy array of spectral data corresponding to the library

        """
        return self.instrument_library_df, self.instrument_library_spectra

    def make_average_instrument_library(self):
        """
        This assumes that the spectral libraries contain mineral spectra. It groupsby the mineral names and produces an
            average spectrum for each mineral.

        Returns:
            Nothing: Sets the `internal average_instrument_library_spectra` and `average_instrument_library_df` variables

        """
        grouped_dict = self.instrument_library_df.groupby('mineral').indices
        mean_spectra = []
        temp_df = pd.DataFrame()
        for val in grouped_dict:
            indices = grouped_dict[val]
            mean_spectra.append(np.mean(self.instrument_library_spectra[indices, :], axis=0))
            temp_df = temp_df.append(self.instrument_library_df.iloc[indices[0], :])
        temp_df = temp_df.reset_index(drop=True)

        self.average_instrument_library_spectra = np.asarray(mean_spectra)
        self.average_instrument_library_df = temp_df

    def get_average_instrument_library(self):
        """
        Returns the instrument spectral library averages (grouped and averaged based on mineral name)

        Returns:
            Dataframe: The ancillary data associated with the spectral library
            ndarray: A numpy array of spectral data corresponding to the library

        """
        if self.average_instrument_library_df is None:
            self.make_average_instrument_library()
        return self.average_instrument_library_df, self.average_instrument_library_spectra

    def tag_spectra_below_a_maximum_threshold(self, threshold=0.01, fit_to='full'):
        """

        Args:
            threshold ():
            fit_to ():

        Returns:

        """
        # set the user spectral input
        if self.spectral_input is None:
            print('You need to enter input data to run the analysis ')
            return 0

        if fit_to == 'culled':
            if self.culled_df is None:
                print('You need to cull something first')
                return 0

        # get the library & instrument data
        _, lib_spec, inst_spec, wavelengths = self.get_library_and_instrument_data(fit_to)

        # do a hull correction if asked for
        if self.hull or self.tir:
            lib_spec, inst_spec, _ = self._hull_corrections(wavelengths, inst_spec, lib_spec, self.tir)

        return np.max(inst_spec, axis=1) < threshold

    def r2_error(self, fit_to='full'):
        """

        Args:
            fit_to ():

        Returns:

        """
        # set the user spectral input
        if self.spectral_input is None:
            print('You need to enter input data to run the analysis ')
            return 0

        if fit_to == 'culled':
            if self.culled_df is None:
                print('You need to cull something first')
                return 0

        # get the library & instrument data
        _, lib_spec, inst_spec, wavelengths = self.get_library_and_instrument_data(fit_to)

        # do a hull correction if asked for
        if self.hull or self.tir:
            lib_spec, inst_spec, _ = self._hull_corrections(wavelengths, inst_spec, lib_spec, self.tir)

        synth = np.dot(self.nmf_results[0], self.nmf_results[1])

        # normalise the spectra
        synthetics, inst_spec = self._normalise_the_spectra(inst_spec, synth)

        # store the result and return them
        r2 = [pearsonr(val1, val2)[0] for val1, val2 in zip(inst_spec, synthetics)]

        return np.asarray(r2)

    def rms_error(self, fit_to='full'):
        """

        Args:
            fit_to ():

        Returns:

        """
        # set the user spectral input
        if self.spectral_input is None:
            print('You need to enter input data to run the analysis ')
            return 0

        if fit_to == 'culled':
            if self.culled_df is None:
                print('You need to cull something first')
                return 0

        # get the library & instrument data
        _, lib_spec, inst_spec, wavelengths = self.get_library_and_instrument_data(fit_to)

        # do a hull correction if asked for
        if self.hull or self.tir:
            lib_spec, inst_spec, _ = self._hull_corrections(wavelengths, inst_spec, lib_spec, self.tir)

        synth = np.dot(self.nmf_results[0], self.nmf_results[1])

        # normalise the spectra
        synthetics, inst_spec = self._normalise_the_spectra(inst_spec, synth)

        # store the result and return them
        rms_error = np.sqrt(
            np.sum(np.square(inst_spec - synthetics), axis=1) / wavelengths.shape[0])
        return rms_error

    def fit(self, fit_to='full', solver='mu', threshold=1.e-4):
        """

        Args:
            fit_to ():
            solver ():
            threshold ():

        Returns:

        """
        # set the user spectral input
        if self.spectral_input is None:
            print('You need to enter input data to run the analysis ')
            return 0

        if fit_to == 'culled':
            if self.culled_df is None:
                print('You need to cull something first')
                return 0

        # get the library & instrument data
        _, lib_spec, inst_spec, wavelengths = self.get_library_and_instrument_data(fit_to)

        # do a hull correction if asked for
        if self.hull or self.tir:
            lib_spec, inst_spec, _ = self._hull_corrections(wavelengths, inst_spec, lib_spec, self.tir)

        # store what we performed the NMF on
        self.which_nmf = fit_to

        # normalise the spectra
        lib_spec, inst_spec = self._normalise_the_spectra(inst_spec, lib_spec)

        # do the NMF calculation
        if 'mu' in solver:
            nmf_results = non_negative_factorization(inst_spec, H=lib_spec, update_H=False, init=None,
                                                     n_components=lib_spec.shape[0], max_iter=600, solver='mu',
                                                    beta_loss=1, tol=1.e-5, random_state=42)
        else:
            nmf_results = non_negative_factorization(inst_spec, H=lib_spec, update_H=False, init=None,
                                                 n_components=lib_spec.shape[0], max_iter=600, solver='cd',
                                                 beta_loss=2, random_state=42, tol=1.e-4)


        # apply a threshold to zero out ridiculously small values
        indices = np.where(nmf_results[0] < threshold)
        nmf_results[0][indices] = 0.0

        # normalize the abundance values between 0 and 1
        part_one = nmf_results[0] / np.expand_dims(np.sum(nmf_results[0], axis=1), axis=1)

        # store the result and return them
        self.nmf_results = part_one, nmf_results[1]
        rms_error = self.rms_error(fit_to=fit_to)
        r2_error = self.r2_error(fit_to=fit_to)
        self.rms_fit_error = rms_error
        self.r2_fit_error = r2_error
        return part_one, nmf_results[1], rms_error, r2_error

    @staticmethod
    def _normalise_the_spectra(inst_spec, lib_spec):
        """

        Args:
            inst_spec ():
            lib_spec ():

        Returns:

        """
        lib_spec = normalize(lib_spec, norm='max', axis=1)
        inst_spec = normalize(inst_spec, norm='max', axis=1)
        return lib_spec, inst_spec

    def _hull_corrections(self, wavelengths, inst_spec, lib_spec, tir=False):
        """

        Args:
            wavelengths ():
            inst_spec ():
            lib_spec ():
            tir ():

        Returns:

        """
        inst_hull = uc_hulls(wavelengths, inst_spec, 2)
        if tir:
            lib_spec = uc_hulls(wavelengths, 1.0 - lib_spec, self.hull_type)
            inst_spec = uc_hulls(wavelengths, 1.0 - inst_spec, self.hull_type)
        else:
            lib_spec = uc_hulls(wavelengths, lib_spec, self.hull_type)
            inst_spec = uc_hulls(wavelengths, inst_spec, self.hull_type)

        lib_spec = np.nan_to_num(lib_spec)
        inst_spec = np.nan_to_num(inst_spec)
        lib_spec[lib_spec < 0] = 0
        inst_spec[inst_spec < 0] = 0

        return lib_spec, inst_spec, inst_hull

    def get_library_and_instrument_data(self, which_library):
        """

        Args:
            which_library ():

        Returns:

        """
        lib_df = None
        lib_spec = None
        range_index = self.range_indices

        inst_spec = self.spectral_input[:, range_index[0]:range_index[1]]
        if which_library == 'full':
            lib_df = self.instrument_library_df
            lib_spec = self.instrument_library_spectra[:, range_index[0]:range_index[1]]
        elif which_library == 'average':
            lib_df = self.average_instrument_library_df
            lib_spec = self.average_instrument_library_spectra[:, range_index[0]:range_index[1]]
        elif which_library == 'culled':
            lib_df = self.culled_df
            lib_spec = self.culled_spectra

        wavelengths = self.instrument_wavelengths[range_index[0]:range_index[1]]
        lib_spec[lib_spec < 0] = 0
        inst_spec[inst_spec < 0] = 0
        return lib_df, lib_spec, inst_spec, wavelengths

    def get_nmf_results(self):
        """

        Returns:

        """
        return self.nmf_results, self.which_nmf

    def plot_spectral_fit_at_ordinate(self, value, ordinates, plot_type='ref', what='mineral', top=3, total_contribution=False, ax=None,
                          fill_between=True, mask=0, title=None, color=None, additional_label='', legend=True, return_contributions=False):
        """

        Args:
            value ():
            ordinates ():
            plot_type ():
            what ():
            top ():
            total_contribution ():
            ax ():
            fill_between ():
            mask ():
            title ():
            color ():
            additional_label ():
            legend ():
            return_contributions ():

        Returns:

        """
        ax = ax or plt.gca()
        # get the fit
        nmf = self.nmf_results
        df, _, instrument_spectra, wavelengths = self.get_library_and_instrument_data(self.which_nmf)

        if not np.isscalar(mask):
            index = np.square(ordinates - ordinates[mask][np.square(ordinates[mask] - value).argmin()]).argmin()
            rms = self.rms_fit_error[mask][index]
            r2 = self.r2_fit_error[mask][index]
        else:
            index = np.square(ordinates - value).argmin()
            rms = self.rms_fit_error[index]
            r2 = self.r2_fit_error[index]

        args = np.argsort(nmf[0][index, :])[-top:]
        top_minerals = np.flip(df['mineral'].loc[args].values)
        contributions = np.round(np.flip(nmf[0][index, args]), 2)

        # get total contribution
        if total_contribution:
            grp = df.groupby(what).indices
            name = []
            amount = []
            for val in grp:
                name.append(val)
                amount.append(np.sum(nmf[0][index, grp[val]]))

            args = np.argsort(amount)[-top:]
            contributions = np.round(np.asarray(amount)[args], 2)
            top_minerals = np.asarray(name)[args]
        return_minerals = (top_minerals, contributions)

        # get the nmf interpretation of the spectrum and the actual spectrum
        synthetic_spectrum = nmf[0][index, :].dot(nmf[1])
        actual_spectrum = instrument_spectra[index, :]

        # 1: TIR = True : Only need a baseline correction to the actual spectra
        # 2: Hull = True : Only need to get the actual spectra hull removed
        hull = None
        sf = None
        temp_spectrum = None
        if self.tir:
            actual_spectrum = uc_hulls(wavelengths, 1.0 - actual_spectrum, 1)
            sf = np.max(actual_spectrum) / np.max(synthetic_spectrum)
        if self.hull:
            hull = uc_hulls(wavelengths, actual_spectrum, 2)
            temp_spectrum = uc_hulls(wavelengths, actual_spectrum, 1)
            sf = np.max(temp_spectrum) / np.max(synthetic_spectrum)

        if sf is None:
            scale_factor = np.max(actual_spectrum) / np.max(synthetic_spectrum)
        else:
            scale_factor = sf

        # now its only about the display
        # can either be reflectance or hull
        # tir can only be reflectance
        if self.tir:
            synthetic_spectrum = synthetic_spectrum * scale_factor
        else:
            if plot_type == 'ref':
                synthetic_spectrum = hull - synthetic_spectrum * scale_factor
            else:
                synthetic_spectrum = synthetic_spectrum * scale_factor
                actual_spectrum = temp_spectrum

        if top != 0:
            minerals = [val + '(' + str(contributions[index]) + ')' for index, val in enumerate(top_minerals)]
            seperator = ', '
            label = seperator.join(minerals) + ', RMS:' + str(np.round(rms, 3)) + ', R2:' + str(np.round(r2, 3)) + ':' + additional_label
        else:
            label = 'RMS:' + str(np.round(rms, 3)) + ', R2:' + str(np.round(r2, 3)) + ':' + additional_label

        # actual_spectrum += offset
        # synthetic_spectrum += offset
        if fill_between:
            line = ax.plot(wavelengths, actual_spectrum, color='k')
            if color:
                line2 = ax.plot(wavelengths, synthetic_spectrum, color=color, label=label)
            else:
                line2 = ax.plot(wavelengths, synthetic_spectrum, color='firebrick', label=label)
            line3 = ax.fill_between(wavelengths, actual_spectrum, synthetic_spectrum,
                                    where=synthetic_spectrum > actual_spectrum, facecolor='blue', alpha=0.3)
            line4 = ax.fill_between(wavelengths, actual_spectrum, synthetic_spectrum,
                                    where=synthetic_spectrum < actual_spectrum, facecolor='green', alpha=0.3)
        else:
            line = ax.plot(wavelengths, actual_spectrum, color='k')
            if color:
                line2 = ax.plot(wavelengths, synthetic_spectrum, color=color, label=label)
            else:
                line2 = ax.plot(wavelengths, synthetic_spectrum, color='firebrick', label=label)
        ax.set_xlabel('Wavelength (nm)', fontsize=16)
        ax.set_ylabel('Reflectance', fontsize=16)
        if title:
            ax.set_title(title, fontsize=20)
        if legend:
            ax.legend()
        return wavelengths, actual_spectrum, synthetic_spectrum, return_minerals

    def plot_spectral_fit(self, index, plot_type='ref', what='mineral', top=3, total_contribution=False, ax=None,
                          fill_between=True):
        """

        Args:
            index ():
            plot_type ():
            what ():
            top ():
            total_contribution ():
            ax ():
            fill_between ():

        Returns:

        """
        ax = ax or plt.gca()
        # get the fit
        nmf = self.nmf_results
        df, _, instrument_spectra, wavelengths = self.get_library_and_instrument_data(self.which_nmf)

        args = np.argsort(nmf[0][index, :])[-top:]
        top_minerals = np.flip(df['mineral'].loc[args].values)
        contributions = np.round(np.flip(nmf[0][index, args]), 2)

        # get total contribution
        if total_contribution:
            grp = df.groupby(what).indices
            name = []
            amount = []
            for val in grp:
                name.append(val)
                amount.append(np.sum(nmf[0][index, grp[val]]))

            args = np.argsort(amount)[-top:]
            contributions = np.round(np.asarray(amount)[args], 2)
            top_minerals = np.asarray(name)[args]

        # get the nmf interpretation of the spectrum and the actual spectrum
        synthetic_spectrum = np.dot(nmf[0][index, :], nmf[1])
        actual_spectrum = instrument_spectra[index, :]

        # 1: TIR = True : Only need a baseline correction to the actual spectra
        # 2: Hull = True : Only need to get the actual spectra hull removed
        hull = None
        sf = None
        temp_spectrum = None
        if self.tir:
            actual_spectrum = uc_hulls(wavelengths, 1.0 - actual_spectrum, 1)
            sf = np.max(actual_spectrum) / np.max(synthetic_spectrum)
        if self.hull:
            hull = uc_hulls(wavelengths, actual_spectrum, 2)
            temp_spectrum = uc_hulls(wavelengths, actual_spectrum, 1)
            sf = np.max(temp_spectrum) / np.max(synthetic_spectrum)

        if sf is None:
            scale_factor = np.max(actual_spectrum) / np.max(synthetic_spectrum)
        else:
            scale_factor = sf

        # now its only about the display
        # can either be reflectance or hull
        # tir can only be reflectance
        if self.tir:
            synthetic_spectrum = synthetic_spectrum * scale_factor
        else:
            if plot_type == 'ref':
                synthetic_spectrum = hull - synthetic_spectrum * scale_factor
            else:
                synthetic_spectrum = synthetic_spectrum * scale_factor
                actual_spectrum = temp_spectrum

        minerals = [val + '(' + str(contributions[index]) + ')' for index, val in enumerate(top_minerals)]
        seperator = ' + '
        label = seperator.join(minerals)

        if fill_between:
            line = ax.plot(wavelengths, actual_spectrum, color='k')
            line2 = ax.plot(wavelengths, synthetic_spectrum, color='firebrick', label=label)
            line3 = ax.fill_between(wavelengths, actual_spectrum, synthetic_spectrum,
                                    where=synthetic_spectrum > actual_spectrum, facecolor='blue', alpha=0.5)
            line4 = ax.fill_between(wavelengths, actual_spectrum, synthetic_spectrum,
                                    where=synthetic_spectrum < actual_spectrum, facecolor='green', alpha=0.5)
        else:
            line = ax.plot(wavelengths, actual_spectrum, color='k')
            line2 = ax.plot(wavelengths, synthetic_spectrum, color='firebrick', label=label)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.legend()
        return wavelengths, actual_spectrum, synthetic_spectrum

    def cull_by_rank(self, rank=10, direction='gt'):
        """

        Args:
            rank ():
            direction ():

        Returns:

        """
        # cut out the last rank nmf results above or below e.g. cut the top 10 out top_or_bottom_cut(rank=10, direction='above)
        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        proportions = 100. * np.sum(self.nmf_results[0], axis=0) / np.sum(np.sum(self.nmf_results[0], axis=0))
        orders = np.argsort(proportions)
        which_nmf = self.which_nmf

        temp_df, temp_spectra, _, _ = self.get_library_and_instrument_data(which_nmf)
        temp_df = temp_df.iloc[orders, :]
        temp_spectra = temp_spectra[orders, :]
        self.which_culled = which_nmf

        if direction.lower() == 'gt':
            temp_df = temp_df.iloc[-rank:, :].reset_index(drop=True)
            temp_spectra = temp_spectra[-rank:, :]
        else:
            temp_df = temp_df.iloc[:rank, :].reset_index(drop=True)
            temp_spectra = temp_spectra[:rank, :]

        self.cull_type = 'top_or_bottom_' + direction
        self.culled_df = temp_df
        self.culled_spectra = temp_spectra
        return self.fit(fit_to='culled')

    def keep_library_above(self, threshold=0.0):
        """

        Args:
            threshold ():

        Returns:

        """
        # cut the last NMF result at some % proportion as either gt or lt
        if self.nmf_results is None:
            self.which_nmf = 'full'
            # print('You need to fit some data first')
            # return ()

        # see what the last nmf results were generated from
        which_nmf = self.which_nmf
        temp_df, temp_spectra, _, wavelengths = self.get_library_and_instrument_data(which_nmf)

        # do a hull correction if asked for
        if self.hull or self.tir:
            lib_spec, _, _ = self._hull_corrections(wavelengths, temp_spectra, temp_spectra, self.tir)

        indices = np.where(np.max(lib_spec, axis=1) > threshold)[0]

        return self.keep_specific(keep_these=indices, indices_supplied=True)

    def keep_specific(self, keep_these=None, indices_supplied=False, keep_type='mineral'):
        """

        Args:
            keep_these ():
            indices_supplied ():
            keep_type ():

        Returns:

        """
        # cut the last NMF result at some % proportion as either gt or lt
        if self.nmf_results is None:
            self.which_nmf = 'full'
            # print('You need to fit some data first')
            # return ()

        # see what the last nmf results were generated from
        which_nmf = self.which_nmf
        temp_df, temp_spectra, _, _ = self.get_library_and_instrument_data(which_nmf)

        self.which_culled = which_nmf
        self.cull_type = 'grouped_' + keep_type
        # first find out where the stuff is in the df so we can flag which spectra to drop as well
        if not indices_supplied:
            indices = temp_df[temp_df[keep_type].str.contains('|'.join(keep_these))].index.values
        else:
            indices = keep_these

        if len(indices) > 0:
            temp_df = temp_df.iloc[indices, :].reset_index(drop=True)
            temp_spectra = temp_spectra[indices, :]
            self.culled_df = temp_df
            self.culled_spectra = temp_spectra
            return self.fit(fit_to='culled')

    def drop_specific(self, drop_these=None, indices_supplied=False, drop_type='mineral', solver='mu'):
        """

        Args:
            drop_these ():
            indices_supplied ():
            drop_type ():
            solver ():

        Returns:

        """
        # cut the last NMF result at some % proportion as either gt or lt
        if self.nmf_results is None:
            which_nmf = 'full'
        else:
            # see what the last nmf results were generated from
            which_nmf = self.which_nmf

        temp_df, temp_spectra, _, _ = self.get_library_and_instrument_data(which_nmf)

        self.which_culled = which_nmf
        self.cull_type = 'grouped_' + drop_type
        # first find out where the stuff is in the df so we can flag which spectra to drop as well
        if not indices_supplied:
            indices = temp_df[temp_df[drop_type].str.contains('|'.join(drop_these))].index.values
        else:
            indices = drop_these

        if len(indices) > 0:
            temp_df = temp_df.drop(indices, axis=0).reset_index(drop=True)
            temp_spectra = np.delete(temp_spectra, indices, axis=0)
            self.culled_df = temp_df
            self.culled_spectra = temp_spectra
            return self.fit(fit_to='culled', solver=solver)

    def cull_by_individual_and_total_sample_contribution(self, cull_value1=None, cull_value2=None):
        """

        Args:
            cull_value1 ():
            cull_value2 ():

        Returns:

        """
        # cut the last NMF result at some % proportion as either gt or lt
        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        temp_nmf = self.nmf_results[0]
        if cull_value1:
            cull_value1 = cull_value1 / 100.0
        else:
            cull_value1 = 1.0 / temp_nmf.shape[1]

        if cull_value2:
            cull_value2 = cull_value2 / 100.0
        else:
            cull_value2 = 1.0 / temp_nmf.shape[1]

        args = np.where(temp_nmf < cull_value1)
        temp_nmf[args[0], args[1]] = 0
        # now we add them up as a function of the number of samples
        sample_sum = np.sum(temp_nmf, axis=0) / temp_nmf.shape[0]
        args = np.where(sample_sum > cull_value2)[0]

        temp_df, temp_spectra, _, _ = self.get_library_and_instrument_data(self.which_nmf.lower())
        temp_df = temp_df.iloc[args, :].reset_index(drop=True)
        temp_spectra = temp_spectra[args, :]

        # store the culled data frame and spectral data and rerun the unmixing
        self.culled_df = temp_df
        self.culled_spectra = temp_spectra
        self.which_culled = self.which_nmf

        self.cull_type = 'double_cull_'  # + str(cull_value)
        return self.fit(fit_to='culled')

    def cull_by_cumulative_proportion(self, cull_value=10, direction='gt', cumulative=True):
        """

        Args:
            cull_value ():
            direction ():
            cumulative ():

        Returns:

        """
        # cut the last NMF result at some % proportion as either gt or lt
        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        temp_df, temp_spectra, temp_nmf = self.cull_entries(cumulative, direction, cull_value)

        # store the culled data frame and spectral data and rerun the unmixing
        self.culled_df = temp_df
        self.culled_spectra = temp_spectra
        self.which_culled = self.which_nmf

        self.cull_type = 'proportion_' + direction + str(cull_value)
        return self.fit(fit_to='culled')

    def cull_entries(self, cumulative, direction, proportion):
        """

        Args:
            cumulative ():
            direction ():
            proportion ():

        Returns:

        """
        # these are the individual spectral sample proportions in the library
        proportions = 100. * np.sum(self.nmf_results[0], axis=0) / np.sum(self.nmf_results[0])
        # set them from low to high
        orders = np.argsort(proportions)
        props = proportions[orders]

        temp_nmf = self.nmf_results
        t1 = temp_nmf[0][:, orders]
        t2 = temp_nmf[1][orders, :]
        temp_nmf = [t1, t2]

        temp_df, temp_spectra, _, _ = self.get_library_and_instrument_data(self.which_nmf.lower())
        temp_df = temp_df.iloc[orders, :].reset_index(drop=True)
        temp_spectra = temp_spectra[orders, :]

        cutoff_index = self._proportion_cutoff_indices(direction, proportion, props, cumulative=cumulative)
        temp_df = temp_df.iloc[cutoff_index, :].reset_index(drop=True)
        temp_spectra = temp_spectra[cutoff_index, :]
        t1 = temp_nmf[0][:, cutoff_index]
        t2 = temp_nmf[1][cutoff_index, :]
        temp_nmf = [t1, t2]
        return temp_df, temp_spectra, temp_nmf

    @staticmethod
    def _proportion_cutoff_indices(direction, proportion, props, cumulative=True):
        """

        Args:
            direction ():
            proportion ():
            props ():
            cumulative ():

        Returns:

        """
        if cumulative:
            if direction.lower() == 'gt':
                cutoff_index = np.where(np.cumsum(props) >= proportion)[0]
            else:
                cutoff_index = np.where(np.cumsum(props) <= proportion)[0]
            if len(cutoff_index) == 0:
                cutoff_index = range(props.shape[0])
        else:
            if direction.lower() == 'gt':
                cutoff_index = np.where(props >= proportion)[0]
            else:
                cutoff_index = np.where(props <= proportion)[0]
            if len(cutoff_index) == 0:
                cutoff_index = range(props.shape[0])

        return cutoff_index

    def plot_proportions(self, plot_what='mineral', stacked=True):
        """

        Args:
            plot_what ():
            stacked ():

        Returns:

        """
        # plot the proportions of the last NMF run
        # todo return the axes and the figure so people can put it where they want
        # what_thing really can be anything from the data frame. Will it make sense though? Maybe not but that is
        # your call

        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        proportions = 100. * np.sum(self.nmf_results[0], axis=0) / np.sum(self.nmf_results[0])
        orders = np.argsort(proportions)
        props = proportions[orders]
        names = None

        which_nmf = self.which_nmf
        if which_nmf.lower() == 'full':
            names = self.instrument_library_df[plot_what].iloc[orders]
        elif which_nmf.lower() == 'average':
            names = self.average_instrument_library_df[plot_what].iloc[orders]
        elif which_nmf == 'culled':
            names = self.culled_df[plot_what].iloc[orders]

        # make a temporary data frame
        temp_df = pd.DataFrame(np.expand_dims(props, axis=0), columns=names)
        # todo allow people to change the color map
        colors = plt.cm.tab20b(np.linspace(0, 1, len(names)))
        if stacked:
            temp_df.plot.bar(stacked=True, legend=False, color=colors)
            # todo work out how many cols are needed based on len(names) and adjust accordingly
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)
            plt.subplots_adjust(right=0.55)
        else:
            plt.bar(np.arange(props.shape[0]), props, tick_label=names, color=colors)
            plt.xticks(rotation=90)
            plt.tight_layout()
        plt.show()

    def cull_by_grouped(self, cull_value=0.0, grouping='mineral'):
        """

        Args:
            cull_value ():
            grouping ():

        Returns:

        """
        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        _, names, props = self._group_and_sort_by_proportion(grouping)

        # see what values are below this & drop them
        cull_this = np.where(props <= cull_value)[0]
        if len(cull_this) > 0:
            names = names[cull_this]
            return self.drop_specific(drop_these=names, drop_type=grouping)
        else:
            return 0

    def plot_grouped_proportions(self, group=None, stacked=True, ax=None):
        """

        Args:
            group ():
            stacked ():
            ax ():

        Returns:

        """
        if group is None:
            group = ['mineral']
        ax = ax or plt.gca()
        # plot the proportions of the last NMF run
        # what_thing really can be anything from the data frame. Will it make sense though? Maybe not but that is
        # your call

        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        cum_sum, names, props = self._group_and_sort_by_proportion(group)

        temp = []
        for index, name in enumerate(names):
            temp.append(name + '|' + str(np.round(props[index], 2)) + '|' + str(np.round(cum_sum[index], 2)))

        names = list(np.asarray(temp))

        # make a temporary data frame
        temp_df = pd.DataFrame(np.expand_dims(props, axis=0), columns=names)
        # todo allow people to change the color map
        colors = plt.cm.tab20b(np.linspace(0, 1, len(names)))

        if len(group) > 1:
            stacked = True
        if stacked:
            temp_df.plot.bar(stacked=True, legend=False, color=colors)
            # todo work out how many cols are needed based on len(names) and adjust accordingly
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)
            plt.subplots_adjust(right=0.4)
        else:
            plt.bar(names, props, color=colors)
            plt.xticks(rotation=90)
            plt.tight_layout()

        plt.show()

    def _group_and_sort_by_proportion(self, group):
        """

        Args:
            group ():

        Returns:

        """
        temp_df, _, _, _ = self.get_library_and_instrument_data(self.which_nmf)
        proportions = 100. * np.sum(self.nmf_results[0], axis=0) / np.sum(self.nmf_results[0])
        minerals_grouped = temp_df.groupby(group).indices
        names = []
        proportion = []
        for val in minerals_grouped:
            names.append(val)
            proportion.append(np.sum(proportions[minerals_grouped[val]]))
        orders = np.argsort(proportion)
        props = np.asarray(proportion)[orders]
        names = np.asarray(names)[orders]
        cumsum = np.cumsum(props)
        return cumsum, names, props

    def plot_stacked_weight(self, x_ordinate, what_thing='mineral', partitions=1, ax=None, legend_ax=None,
                            do_plotly=True, save_name='stacked_samples', hide_value=0, use_tsg_colors=False,
                            mask=0, title="Relative Mineral Proportions", xtitle='Depth (m)'):
        """

        Args:
            x_ordinate ():
            what_thing ():
            partitions ():
            ax ():
            legend_ax ():
            do_plotly ():
            save_name ():
            hide_value ():
            use_tsg_colors ():
            mask ():
            title ():
            xtitle ():

        Returns:

        """
        if do_plotly is False:
            ax = ax or plt.gca()
            legend_ax = legend_ax or plt.gca()

        # plot the stacked weights from the last NMF run
        # which_nmf determines where the last result came from
        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        temp_df, _, _, _ = self.get_library_and_instrument_data(self.which_nmf)
        args = np.where(np.sum(self.nmf_results[0], axis=0) / self.nmf_results[0].shape[0] > hide_value)[0]
        temp_weights = self.nmf_results[0].copy()
        temp_weights = temp_weights[:, args]
        names = np.unique(temp_df[what_thing].iloc[args])
        temp_df = temp_df.iloc[args, :].reset_index(drop=True)

        # todo add in the nmf_results[1] into the mix so it can be passed back
        new_weight = []
        # sum the results as a function of 'what_thing'. If its 'mineral' then this is effectively grouping by
        # mineral name todo change this a pandas groupby and indices call instead
        for name in names:
            args = np.where(temp_df[what_thing] == name)[0]
            new_weight.append(np.sum(temp_weights[:, args], axis=1))
        new_weight = np.transpose(np.asarray(new_weight))
        new_weight /= np.expand_dims(np.nansum(new_weight, axis=1), axis=1)

        # find where the indices are that will let us split the data according to the user partition size
        split_locations = np.searchsorted(x_ordinate, np.arange(partitions, x_ordinate[-1], partitions))
        if np.isscalar(mask):
            mask = np.ones(x_ordinate.shape).astype(bool)

        x_ordinate_split = np.split(x_ordinate, split_locations)
        new_weight_split = np.split(new_weight, split_locations)
        mask_split = np.split(mask, split_locations)

        depth = []
        width = []
        weight = []

        count = 0
        if np.min(np.diff(x_ordinate)) < partitions:
            for xval, weight_val, mask_val in zip(x_ordinate_split, new_weight_split, mask_split):
                if xval.shape[0] > 0:
                    depth.append(xval[0])
                    width.append(xval[-1]-xval[0])
                    any_masked = float(np.where(mask_val)[0].shape[0])
                    # if nothing is masked then dont average
                    if any_masked > 0:
                        weight.append(np.sum(weight_val[mask_val, :], axis=0)/any_masked)
                    else:
                        weight.append(np.sum(weight_val[mask_val, :], axis=0))
        else:
            depth = x_ordinate
            width = np.zeros((depth.shape[0])) + partitions
            weight = new_weight[mask]

        weight = np.asarray(weight)
        depth = np.asarray(depth)
        width = np.asarray(width)

        tdepth = [str(int(val)) for val in depth]
        new_dataframe = pd.DataFrame(weight, columns=names, index=tdepth)
        colors = plt.cm.tab20b(np.linspace(0, 1, len(names)))
        if not do_plotly:
            new_dataframe.plot(ax=ax, kind='bar', stacked=True, width=np.asarray(width), legend=False, color=colors)
        else:
            if hide_value > 0:
                args = np.where(weight < hide_value)
                weight[args[0], args[1]] = np.nan
                # normalize
                weight /= np.expand_dims(np.nansum(weight, axis=1), axis=1)

            data = []
            for var in np.arange(weight.shape[1]):
                if what_thing == 'mineral':
                    thing = 'Mineral'
                elif what_thing == 'group':
                    thing = 'Group'
                else:
                    use_tsg_colors = False

            color_data = pd.read_csv("./working_up_colours.csv")
            for val in np.unique(temp_df['group']):
                unique_minerals = np.unique(temp_df[temp_df['group'] == val][what_thing])
                number_of_minerals = unique_minerals.shape[0]
                group_color_name = color_data[color_data['Group'] == val]['name'].iloc[0]
                # lets get alpha values
                rgb = color_data[color_data['Group'] == val][['Red', 'Blue', 'Green']].iloc[0]
                alphas = np.linspace(1.0, 0.5, number_of_minerals)
                # colors = sb.dark_palette(group_color_name, n_colors=3 + number_of_minerals)
                # rgb = np.asarray(colors[-number_of_minerals:]) * int(255)
                # rgb = rgb.astype(int)
                for index, mineral in enumerate(unique_minerals):
                    #color = "rgb(" + str(rgb[index, 0]) + "," + str(rgb[index, 1]) + "," + str(rgb[index, 2]) + ")"
                    color = "rgba(" + str(rgb['Red']) + "," + str(rgb['Green']) + "," + str(rgb['Blue']) + "," + str(alphas[index]) + ")"
                    location = np.where(names == mineral)[0][0]
                    data.append(go.Bar(name=mineral, x=depth + width / 2, y=weight[:, location], width=width,
                                         marker_color=color))

            fig = go.Figure(data=data)
            # Change the bar mode
            # todo put in keywords for titles and labels
            title = title + ' , Bin Size: ' + str(partitions) + ': Minimum Spatial Threshold = ' + str(hide_value)
            fig.update_layout(barmode='stack', title=title, xaxis_title=xtitle,
                              yaxis_title="Relative Proportion", legend_orientation="h", font_size=20)
            fig.write_image(save_name+".png", width=1920, height=1080, scale=2)
            fig.write_image(save_name+".pdf", width=1920, height=1080, scale=2)
            fig.write_html(save_name+".html")

        return names, depth, np.nan_to_num(weight, 0)

    def plot_stacked_weight_hack(self, x_ordinate, what_thing='mineral', partitions=100, ax=None, legend_ax=None,
                            do_plotly=True, save_name='stacked_samples', hide_value=0, use_tsg_colors=False,
                            mask=0, title="Relative Mineral Proportions"):
        """

        Args:
            x_ordinate ():
            what_thing ():
            partitions ():
            ax ():
            legend_ax ():
            do_plotly ():
            save_name ():
            hide_value ():
            use_tsg_colors ():
            mask ():
            title ():

        Returns:

        """

        if do_plotly is False:
            ax = ax or plt.gca()
            legend_ax = legend_ax or plt.gca()

        if self.nmf_results is None:
            print('You need to fit some data first')
            return ()

        temp_df, _, _, _ = self.get_library_and_instrument_data(self.which_nmf)
        args = np.where(np.sum(self.nmf_results[0], axis=0) / self.nmf_results[0].shape[0] > hide_value)[0]
        temp_weights = self.nmf_results[0].copy()[:, args]
        names = np.unique(temp_df[what_thing].iloc[args])
        temp_df = temp_df.iloc[args, :].reset_index(drop=True)

        new_weight = []
        # sum the results as a function of 'what_thing'. If its 'mineral' then this is effectively grouping by
        # mineral name todo change this a pandas groupby and indices call instead
        for name in names:
            args = np.where(temp_df[what_thing] == name)[0]
            new_weight.append(np.sum(temp_weights[:, args], axis=1))
        new_weight = np.transpose(np.asarray(new_weight))
        new_weight /= np.expand_dims(np.nansum(new_weight, axis=1), axis=1)

        # find where the indices are that will let us split the data according to the user partition size
        split_locations = np.searchsorted(x_ordinate, np.arange(partitions, x_ordinate[-1], partitions))
        if np.isscalar(mask):
            mask = np.ones(x_ordinate.shape).astype(bool)

        x_ordinate_split = np.split(x_ordinate, split_locations)
        new_weight_split = np.split(new_weight, split_locations)
        mask_split = np.split(mask, split_locations)

        depth = []
        width = []
        weight = []

        count=0
        if np.min(np.diff(x_ordinate)) < partitions:
            for xval, weight_val, mask_val in zip(x_ordinate_split, new_weight_split, mask_split):
                if xval.shape[0] > 0:
                    depth.append(xval[0])
                    width.append(xval[-1]-xval[0])
                    any_masked = float(np.where(mask_val)[0].shape[0])
                    # if nothing is masked then dont average
                    if any_masked > 0:
                        weight.append(np.sum(weight_val[mask_val, :], axis=0)/any_masked)
                    else:
                        weight.append(np.sum(weight_val[mask_val, :], axis=0))
        else:
            depth = x_ordinate
            width = np.zeros((depth.shape[0])) + partitions
            weight = new_weight[mask]

        weight = np.asarray(weight)
        depth = np.asarray(depth)
        width = np.asarray(width)

        tdepth = [str(int(val)) for val in depth]
        new_dataframe = pd.DataFrame(weight, columns=names, index=tdepth)
        colors = plt.cm.tab20b(np.linspace(0, 1, len(names)))
        if not do_plotly:
            new_dataframe.plot(ax=ax, kind='bar', stacked=True, width=np.asarray(width), legend=False, color=colors)
        else:
            if hide_value > 0:
                args = np.where(weight < hide_value)
                weight[args[0], args[1]] = np.nan
                # normalize
                weight /= np.expand_dims(np.nansum(weight, axis=1), axis=1)

            from plotly.subplots import make_subplots
            fig = make_subplots(rows=3, cols=1, #shared_xaxes=True,
                                specs=[[{}], [{"rowspan":2}], [None]])

            # todo add in a bit for group plots not just mineral
            data = []
            color_data = pd.read_csv("./spectral_libraries/working_up_colours_mkII.csv")
            for val in np.unique(temp_df['group']):
                unique_minerals = np.unique(temp_df[temp_df['group'] == val][what_thing])
                number_of_minerals = unique_minerals.shape[0]
                group_color_name = color_data[color_data['Group'] == val]['name'].iloc[0]
                # lets get alpha values
                rgb = color_data[color_data['Group'] == val][['Red', 'Blue', 'Green']].iloc[0]
                alphas = np.linspace(1.0, 0.5, number_of_minerals)
                # colors = sb.dark_palette(group_color_name, n_colors=3 + number_of_minerals)
                # rgb = np.asarray(colors[-number_of_minerals:]) * int(255)
                # rgb = rgb.astype(int)
                for index, mineral in enumerate(unique_minerals):
                    #color = "rgb(" + str(rgb[index, 0]) + "," + str(rgb[index, 1]) + "," + str(rgb[index, 2]) + ")"
                    color = "rgba(" + str(rgb['Red']) + "," + str(rgb['Green']) + "," + str(rgb['Blue']) + "," + str(alphas[index]) + ")"
                    location = np.where(names == mineral)[0][0]
                    fig.add_trace(go.Bar(name=mineral, x=depth + width / 2, y=weight[:, location], width=width,
                                         marker_color=color), row=2, col=1)

            # MSDP11 HACK
            msdp11_df = pd.read_csv('msdp11_log.csv')
            lithology = msdp11_df.columns.values[2:]
            x = msdp11_df['X'].values.astype(int)
            w = msdp11_df['Width'].values.astype(int)
            colors=["brown", "saddlebrown", "lightyellow", "pink", "red", "skyblue", "olivedrab", "darkgray", "gray", "indianred", "purple"]

            for index, var in enumerate(lithology):
                y = msdp11_df[var].values
                fig.add_trace(go.Bar(name=var, x=x+w/2, y=y, width=w, marker_color=colors[index]), row=1, col=1)

            # Change the bar mode
            # todo put in keywords for titles and labels
            title = title + ' , Bin Size: ' + str(partitions) + ': Minimum Spatial Threshold = ' + str(hide_value)
            fig.update_layout(barmode='stack', title=title, legend_orientation="h", font_size=20)
            fig.update_yaxes(showticklabels=False, row=1, col=1)
            fig.update_xaxes(title_text='Depth (m)', row=2, col=1)
            fig.update_yaxes(title_text='Relative Proportion', row=2, col=1)
            fig.write_image(save_name+".png", width=1920, height=1080, scale=2)
            fig.write_image(save_name+".pdf", width=1920, height=1080, scale=2)
            fig.write_html(save_name+".html")

        return names, depth, np.nan_to_num(weight, 0)

    def plot_library_spectra(self, search_item='mineral', names=None, plot_hull=False, tir=False):
        """

        Args:
            search_item ():
            names ():
            plot_hull ():
            tir ():

        Returns:

        """
        if names is None:
            return 0
        df = self.instrument_library_df
        wavelengths = self.instrument_wavelengths[self.range_indices[0]:self.range_indices[1]]
        spectra = self.instrument_library_spectra[:, self.range_indices[0]:self.range_indices[1]]
        if plot_hull:
            if tir:
                spectra = uc_hulls(wavelengths, 1.0 - spectra, 1)
            else:
                spectra = uc_hulls(wavelengths, spectra, 1)

        indices = df.loc[df[search_item].isin(names)].index.values
        if len(indices) > 0:
            for val in indices:
                plt.plot(wavelengths, spectra[val, :], label=df[search_item].iloc[val])
            plt.legend()
            plt.show()

    def compare_instrument_spectra_to_specific(self, index, search_type='mineral', search_item='kaolinite', hull=False,
                                               normalise=False):
        """

        Args:
            index ():
            search_type ():
            search_item ():
            hull ():
            normalise ():

        Returns:

        """
        df = self.instrument_library_df
        wavelengths = self.instrument_wavelengths[self.range_indices[0]:self.range_indices[1]]
        spectra = self.instrument_library_spectra[:, self.range_indices[0]:self.range_indices[1]]

        instrument_data = self.spectral_input[index, self.range_indices[0]:self.range_indices[1]]
        if hull:
            instrument_data = uc_hulls(wavelengths, instrument_data, 1)
        if normalise:
            instrument_data = instrument_data / np.max(instrument_data)

        indices = np.where(df[search_type].str.contains(search_item))[0]
        if len(indices) > 0:
            plt.plot(wavelengths, instrument_data, label='Instrument', color='k')
            for val in indices:
                spectrum = spectra[val, :]
                if hull:
                    spectrum = uc_hulls(wavelengths, spectrum, 1)
                if normalise:
                    spectrum = spectrum / np.max(spectrum)
                plt.plot(wavelengths, spectrum, label=df[search_type].iloc[val])
            plt.legend()
            plt.show()

    def mixtures(self, which_library='full', tir=False):
        """
        
        Args:
            which_library ():
            tir ():

        Returns:

        """
        # singleton
        lib_df, lib_spec, inst_spec, wavelengths = self.get_library_and_instrument_data(which_library)
        lib_spec, inst_spec, inst_hull = self._hull_corrections(wavelengths, inst_spec, lib_spec, tir=tir)

        # get all of the possible combinations of the end members up to a max mixture level e.g 1 through to 4
        lib_elements = list(np.arange(lib_spec.shape[0]))
        mixtures = []
        max_mixture_level = 4 # hard coded for trial
        for val in range(1, max_mixture_level+1):
            mixtures.append(list(combinations(lib_elements, val)))

        # run the NMF to calculate the weights for the potential mixture levels
        rms_error = []
        for val in mixtures[0]:
            nmf_results = non_negative_factorization(inst_spec, H=np.reshape(lib_spec[val[1], :], [1, -1]),
                                                     update_H=False, init=None,
                                                     n_components=2, max_iter=600, solver='mu',
                                                     beta_loss=1, tol=1.e-4, random_state=42)[:2]
            rms_error.append(np.sqrt(np.sum(np.square(np.dot(nmf_results[0][0], nmf_results[0][1]) - inst_spec), axis=1)))

        # calculate the rms error between the various mixture level fits
        # pick the lowest rms error as the winner for a given spectrum (not sure if this will just default to the one with the most mixtures)
