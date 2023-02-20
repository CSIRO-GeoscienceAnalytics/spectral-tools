import os
from pathlib import Path

import hdbscan
import numpy as np
import plotly.express as px
import pytest
import spectral.io.envi as envi
from scipy.interpolate import interp1d

from spectraltools.extraction.extraction import extract_spectral_features
from spectraltools.hulls.convexhulls import uc_hulls

dir_path = os.path.dirname(os.path.realpath(__file__))
test_data = Path(dir_path) / "../spectraltools/data"
ouput_path = Path(dir_path) / "output"

# lets get a HyMap image
path = test_data / "Corescan"
# define the files.
files = list(path.glob(r"**/*.bil"))
file_pairs = [(val, Path(str(val).replace(".bil", ".hdr"))) for val in files]


@pytest.fixture
def clean_output():
    products_dir = ouput_path / "depth_products"
    if products_dir.exists():
        import shutil

        shutil.rmtree(str(products_dir))


def test_hymap(clean_output):
    bands = []
    products = [(550, 750), (700, 1300), (2050, 2150), (2100, 2250), (2210, 2300)]
    for files in file_pairs:
        # create an output directory
        products_dir = ouput_path / "depth_products"
        products_dir.mkdir(parents=True, exist_ok=True)

        # get the image data
        img_data = envi.open(str(files[1]), str(files[0]))
        bands = img_data.bands.centers
        for product in products:
            # extract feature/s. Here you really want to set the ordinates_inspection_range
            # to the thing you are after
            features = extract_spectral_features(
                img_data[:, :, :],
                1000.0 * np.array(img_data.bands.centers),
                do_hull=True,
                main_guard=False,
                ordinates_inspection_range=[product[0], product[1]],
                distance=1,
                fit_type="crude",
            )

            # Simple example 1: now write the product to file
            md = {
                "description": f"Simple maximum hull quotient depth between {str(product[0])} and {str(product[1])}",
                "lines": img_data.metadata["lines"],
                "samples": img_data.metadata["samples"],
                "bands": "1",
                "data type": "4",
                "interleave": "bip",
            }
            band_name = f"{str(product[0])}_{str(product[1])}"
            md["band names"] = [band_name]
            fileout_name = f"{band_name}.hdr"
            fileout_name = str(products_dir / fileout_name)
            img_out = envi.create_image(fileout_name, md, force=True)
            img_out.open_memmap(writeable=True)
            features.extracted_features[:, :, 1]

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Example 2. Single feature but keep the wavelength, depth and fwhm
            # same as above but lets say you are wanting to keep the wavelength, depth and FWHM of a feature in each wavelength region defined in products.

            # This would be the equivilent call if you wanted a more sophisticated feature extraction. Much slower though.
            # features = extract_spectral_features(img_data[:,:,:], 1000.0*np.array(img_data.bands.centers), do_hull=True, main_guard=True,
            # max_features=1, ordinates_inspection_range=[product[0], product[1]], distance=1, fit_type='cheb', resolution=1)

            # md = {}
            # md['description'] = 'Simple maximum hull quotient depth between ' + str(product[0]) + ' and ' + str(product[1])
            # md['lines'] = img_data.metadata['lines']
            # md['samples'] = img_data.metadata['samples']
            # md['bands'] = '3'
            # md['data type'] = '4' #float32 see https://www.l3harrisgeospatial.com/docs/idl_data_types.html
            # md['interleave'] = 'bip'
            # band_name = str(product[0]) + '_' + str(product[1])
            # md['band names'] = ['Wavelength', 'Depth', 'FWHM']
            # fileout_name = band_name+'.hdr'
            # fileout_name = str(products_dir / fileout_name)
            # img_out = envi.create_image(fileout_name, md, force=True)
            # img_out_memmap = img_out.open_memmap(writeable=True)
            # img_out_memmap = features.extracted_features[:,:,[0,1,2],0]

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Example 3. Multiple features (lets say 2) keeping the wavelength, depth and fwhm
            # This would be the equivilent call if you wanted a more sophisticated feature extraction. Much slower though.
            # features = extract_spectral_features(img_data[:,:,:], 1000.0*np.array(img_data.bands.centers), do_hull=True, main_guard=True,
            # max_features=2, ordinates_inspection_range=[product[0], product[1]], distance=1, fit_type='cheb', resolution=1)

            # md = {}
            # md['description'] = 'Simple maximum hull quotient depth between ' + str(product[0]) + ' and ' + str(product[1])
            # md['lines'] = img_data.metadata['lines']
            # md['samples'] = img_data.metadata['samples']
            # md['bands'] = '6'
            # md['data type'] = '4' #float32 see https://www.l3harrisgeospatial.com/docs/idl_data_types.html
            # md['interleave'] = 'bip'
            # band_name = str(product[0]) + '_' + str(product[1])
            # md['band names'] = ['Wavelength1', 'Depth1', 'FWHM1', 'Wavelength2', 'Depth2', 'FWHM2']
            # fileout_name = band_name+'.hdr'
            # fileout_name = str(products_dir / fileout_name)
            # img_out = envi.create_image(fileout_name, md, force=True)
            # img_out_memmap = img_out.open_memmap(writeable=True)
            # features_shape = features.extracted_features.shape
            # img_out_memmap = features.extracted_features[:,:,[0,1,2],[0,1]].reshape([features_shape[0], features_shape[1], -1])


@pytest.mark.skip
def test_hymap_extended():
    """Everything from this point forward is not setup to run through the entire list of HyMap files.
    However, the approach would be exactly the same as the cell above. You would create a loop that runs over
    the file_pairs and does various things like the processing you want and then writing outputs to files
    (dependent on what you want of course).
    I have left them as small snippets so as not to confuse the whole notebook.
        What if we want to do some hull quotient stuff ourself. This is how you can do it for a single file.
        Again loop and output as required like the above stuff. However, this is really what the
        extract_spectral_features with fit_type = 'crude' is doing.
        Just figured you might want a bit more control.
    """

    # lets just do the first BH file
    single_image = envi.open(str(file_pairs[0][1]), str(file_pairs[0][0]))
    bands = 1000.0 * np.array(single_image.bands.centers)
    # let define where we want to look
    indices = np.searchsorted(bands, [2100, 2250])
    # lets set the data into reflectance between 0-1. You need to directely reference the images like this as the spectral library uses numpy memmaps to reference the data when it is an image file
    temp_img = single_image[:, :, :] / 10000.0  # type: ignore
    # get the hull corrected image
    hull_corrected_image = uc_hulls(
        bands[indices[0] : indices[1] + 1], temp_img[:, :, indices[0] : indices[1] + 1], 0
    )
    # can now find out what the maximum depth etc is. From a memory perspective I would normally just tack the .max(axis=2) onto the previous line
    hull_corrected_image.max(axis=2)


@pytest.mark.skip
def test_hymap_extended2():
    """What if you want to extract some features and do some clustering?
    A HUGE HUGE note here!!!
    The images in terms of spectral processing really are quite large and are equivalent to tens of km's of
                drillcore so they take a long time. This probably needs to be done on bowen and it should really be
                done in a main guard. In the e.g. bit below the process_my_hymap_stuff() would allow one to use the
                main_guard = True keyword in extract_spectral_features
    I am not sure how one implements that on bowen though (Fang should know how). Ultimately some of this
                stuff is better run not in a notebook but in a script/s
    Because this example in this file is run in a main guard I have set the main_guard keyword to true
    """

    # first lets open our image
    # lets just do the first BH file as previously. I am only reopening them for this cell as a
    # demonstration. No need to continuously reopen these
    single_image = envi.open(str(file_pairs[0][1]), str(file_pairs[0][0]))
    bands = 1000.0 * np.array(single_image.bands.centers)
    # now lets do a feature extraction.
    features = extract_spectral_features(
        single_image[:, :, :],  # type: ignore
        bands,
        do_hull=True,
        max_features=3,
        ordinates_inspection_range=[2100, 2390],
        distance=2,
        fit_type="cheb",
        resolution=1,
        main_guard=False,
    )
    # can cluster features if wanted. Need to be warned though this might not scale very well with really
    # large datasets an aletrnative might be needed here
    clst = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=3)
    # lets cluster on the 2 deepest wavelengths. The reshape is to flatten the data into a Nxfeatures
    # array since it was originally from the entire image
    clst.fit(features.extracted_features[:, :, 0, [0, 1]].reshape(-1, 2))
    # show me the cluster labels
    labels = clst.labels_

    fig = px.scatter(
        x=features.extracted_features[:, :, 0, 0], y=features.extracted_features[:, :, 0, 1], color=labels
    )
    fig.write_image(ouput_path / "plot.png")


@pytest.mark.skip
def test_hymap_extended3():
    """What about if you want the GFZ spectral libraries. Well here is how you would grab one of them.
    Again just stick stuff in a loop if you want to collect them all up at once
    """
    single_image = envi.open(str(file_pairs[0][1]), str(file_pairs[0][0]))
    bands = 1000.0 * np.array(single_image.bands.centers)

    files = [test_data / "HyLogger/UDD10701.ini", test_data / "HyLogger/UDD10701.hdr"]
    gfz_data = envi.open(str(files[1]), str(files[0]))
    # lets setup an interpolating function so we can interpolate the spectra to the HyMap wavelengths
    f = interp1d(gfz_data.bands.centers, gfz_data.spectra / 10000.0, fill_value="extrapolate", kind="cubic")
    # convert our spectral library to the HyMap wavelengths (actually just a 1D interpolation).
    # Remember they are reflectance*10000
    f(bands)


# bands was defined somewhere earlier. If not its the HyMap bands. Both the GFZ and HyMap bands
# need to be in the same units

if __name__ == "__main__":
    test_hymap()
