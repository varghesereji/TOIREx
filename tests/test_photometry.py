from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
from astropy.table import Table
from astropy.io import fits
from pathlib import Path

from toirex.photometry import calculate_snr
from toirex.photometry import calculate_magnitude
from toirex.photometry import save_photometry
from toirex.photometry import _make_daostarfinder
from toirex.photometry import targetfind_auto
from toirex.photometry import targetfind_manual
from toirex.photometry import aperture_photometry_subrot
from toirex.photometry import make_epsf


@patch("toirex.photometry.get_logger")
def test_calculate_snr_aperture(mock_get_logger):
    """Test SNR calculation for aperture photometry"""

    phot_table = Table({
        "flux_net": [100.0, 200.0, 300.0],
        "var_net": [25.0, 100.0, 225.0],
        })

    result = calculate_snr(phot_table)

    expected_snr = [20.0, 20.0, 20.0]
    assert "snr" in result.colnames
    np.testing.assert_allclose(result['snr'], expected_snr)


@patch("toirex.photometry.get_logger")
def test_calculate_snr_psf(mock_get_logger):
    """Test SNR calculation for psf photometry"""

    phot_table = Table({
        "flux_fit": [100.0, 200.0, 300.0],
        "flux_err": [5.0, 10.0, 15.0],
        })

    result = calculate_snr(phot_table)

    expected_snr = [20.0, 20.0, 20.0]
    assert "snr" in result.colnames
    np.testing.assert_allclose(result['snr'], expected_snr)


@patch("toirex.photometry.get_logger")
def test_calculate_magnitude_aperture(mock_get_logger):
    """Test SNR calculation for psf photometry"""

    phot_table = Table({
        "flux_net": [100.0, 200.0, 300.0],
        "var_net": [25.0, 100.0, 225.0],
        })

    result = calculate_magnitude(phot_table)

    expected_mags = [-5.0, -5.75257499, -6.1928034]
    expected_magerrs = [0.05428681023790647,
                        0.05428681023790647,
                        0.05428681023790647]
    assert "mag" in result.colnames
    assert "mag_err" in result.colnames

    np.testing.assert_allclose(result['mag'], expected_mags)
    np.testing.assert_allclose(result['mag_err'], expected_magerrs)


@patch("toirex.photometry.get_logger")
def test_calculate_magnitude_psf(mock_get_logger):
    """Test SNR calculation for psf photometry"""

    phot_table = Table({
        "flux_fit": [100.0, 200.0, 300.0],
        "flux_err": [5.0, 10.0, 15.0],
        })

    result = calculate_magnitude(phot_table)

    expected_mags = [-5.0, -5.75257499, -6.1928034]
    expected_magerrs = [0.05428681023790647,
                        0.05428681023790647,
                        0.05428681023790647]
    assert "mag" in result.colnames
    assert "mag_err" in result.colnames

    np.testing.assert_allclose(result['mag'], expected_mags)
    np.testing.assert_allclose(result['mag_err'], expected_magerrs)


@patch("toirex.photometry.get_logger")
def test_save_aperturephotometry(mock_get_logger, tmp_path):
    """Test saving photometry without calculating magnitude."""

    input_file = tmp_path / "test.fits"
    fits.PrimaryHDU().writeto(input_file)

    phot_table = Table({
        "id": [1, 2, 3],
        "flux_net": [100.0, 200.0, 300.0],
        "var_net": [25.0, 100.0, 225.0],
    })

    output_file = save_photometry(
        input_file,
        phot_table,
        save_magnitude=True,
    )

    expected_snr = [20.0, 20.0, 20.0]
    expected_mags = [-5.0, -5.75257499, -6.1928034]
    expected_magerrs = [0.05428681023790647,
                        0.05428681023790647,
                        0.05428681023790647]

    with fits.open(output_file) as hdul:
        phot = Table(hdul["PHOTOMETRY"].data)

        assert "snr" in phot.colnames
        assert "mag" in phot.colnames
        assert "mag_err" in phot.colnames

        history = hdul[0].header.get("HISTORY", [])
        assert "Calculated signal-to-noise ratio of each source" in history
        assert "Calculated instrument magnitude" in history

        np.testing.assert_allclose(phot['snr'], expected_snr)
        np.testing.assert_allclose(phot['mag'], expected_mags)
        np.testing.assert_allclose(phot['mag_err'], expected_magerrs)


@patch("toirex.photometry.get_logger")
def test_save_psfphotometry(mock_get_logger, tmp_path):
    """Test saving photometry without calculating magnitude."""

    input_file = tmp_path / "test.fits"
    fits.PrimaryHDU().writeto(input_file)

    phot_table = Table({
        "id": [1, 2, 3],
        "flux_fit": [100.0, 200.0, 300.0],
        "flux_err": [5.0, 10.0, 15.0],
    })

    output_file = save_photometry(
        input_file,
        phot_table,
        save_magnitude=True,
    )

    expected_snr = [20.0, 20.0, 20.0]
    expected_mags = [-5.0, -5.75257499, -6.1928034]
    expected_magerrs = [0.05428681023790647,
                        0.05428681023790647,
                        0.05428681023790647]

    with fits.open(output_file) as hdul:
        phot = Table(hdul["PHOTOMETRY"].data)

        assert "snr" in phot.colnames
        assert "mag" in phot.colnames
        assert "mag_err" in phot.colnames

        history = hdul[0].header.get("HISTORY", [])
        assert "Calculated signal-to-noise ratio of each source" in history
        assert "Calculated instrument magnitude" in history

        np.testing.assert_allclose(phot['snr'], expected_snr)
        np.testing.assert_allclose(phot['mag'], expected_mags)
        np.testing.assert_allclose(phot['mag_err'], expected_magerrs)


#############################
# Tests on DAOFinder
#############################
@patch("toirex.photometry.DAOStarFinder")
def test_make_daostarfinder_n_brightest(mock_daofinder):
    """Test DAOStarFinder initialization with n_brightest."""

    _make_daostarfinder(
        fwhm=7,
        threshold=50,
        n_brightest=10,
    )

    mock_daofinder.assert_called_once_with(
        fwhm=7,
        threshold=50,
        n_brightest=10,
    )


@patch("toirex.photometry.DAOStarFinder")
def test_make_daostarfinder_brightest(mock_daofinder):
    """Test DAOStarFinder initialization with brightest."""

    with patch("toirex.photometry._DAOSTARFINDER_SUPPORTS_N_BRIGHTEST", False):
        _make_daostarfinder(
            fwhm=7,
            threshold=50,
            n_brightest=10,
        )

    mock_daofinder.assert_called_once_with(
        fwhm=7,
        threshold=50,
        brightest=10,
    )


# Test on automatic source detection
@patch("toirex.photometry.imageplot")
@patch("toirex.photometry.table_to_centroids")
@patch("toirex.photometry._make_daostarfinder")
@patch("toirex.photometry.sigma_clipped_stats")
@patch("toirex.photometry.fits.getdata")
@patch("toirex.photometry.get_logger")
def test_targetfind_auto(mock_get_logger,
                         mock_getdata,
                         mock_stats,
                         mock_make_daofind,
                         mock_table_to_centroids,
                         mock_imageplot):
    """Test automatic source detection."""

    data = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    mock_getdata.return_value = data
    mock_stats.return_value = (0.0, 10.0, 1.0)

    sources = Table({
        "x_centroid": [10.0, 20.0],
        "y_centroid": [15.0, 25.0],
    })

    mock_daofind = Mock(return_value=sources)
    mock_make_daofind.return_value = mock_daofind

    mock_table_to_centroids.return_value = [(15.0, 10.0), (25.0, 20.0)]

    fname = Path("test.fits")

    result = targetfind_auto(
        fname,
        fwhm=7.0,
        threshold=50,
        n_brightest=10,
        show_plot=False,
        aperture_radii=(10, 15, 20),
        plot_dirs="plots",
    )

    mock_getdata.assert_called_once_with(fname, ext=0)

    mock_stats.assert_called_once_with(data)

    mock_make_daofind.assert_called_once_with(
        fwhm=7.0,
        threshold=50,
        n_brightest=10,
        exclude_border=True,
        xycoords=None,
    )

    mock_daofind.assert_called_once()
    detected_data = mock_daofind.call_args.args[0]

    np.testing.assert_array_equal(
        detected_data,
        data - 10.0,
    )

    mock_table_to_centroids.assert_called_once_with(
        sources,
        keys=("y_centroid", "x_centroid"),
    )

    mock_imageplot.assert_called_once()

    assert "x_0" in result.colnames
    assert "y_0" in result.colnames

    np.testing.assert_array_equal(
        result["x_0"],
        [10.0, 20.0],
    )

    np.testing.assert_array_equal(
        result["y_0"],
        [15.0, 25.0],
    )


@patch("toirex.photometry.imageplot")
@patch("toirex.photometry.get_logger")
def test_targetfind_manual(mock_get_logger, mock_imageplot, tmp_path):
    """Test manual source selection."""

    fname = tmp_path / "test.fits"

    centroids_0 = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
        ])

    mock_imageplot.return_value = np.array([
        [11.0, 21.0],
        [31.0, 41.0],
        ])

    result = targetfind_manual(
        fname,
        centroids_0,
        aperture_radii=(10, 15, 20),
        plot_dirs=tmp_path,
        )

    mock_get_logger.return_value.info.assert_called_once_with(
        f"Manual finding sources in {str(fname)}"
        )

    plot_name = tmp_path / "test_selectedsources.pdf"

    mock_imageplot.assert_called_with(
        fname,
        ext=0,
        title="Select sources",
        line_profile="aperture",
        get_target=False,
        centroid_list=centroids_0,
        save_plot=plot_name,
        aperture_radii=(10, 15, 20),
        )

    assert "x_0" in result.colnames
    assert "y_0" in result.colnames
    np.testing.assert_array_equal(
        result["x_0"],
        [21.0, 41.0],
        )


@patch("toirex.photometry.save_photometry")
@patch("toirex.photometry.aperture_photometry")
@patch("toirex.photometry.fits.getdata")
@patch("toirex.photometry.CircularAnnulus")
@patch("toirex.photometry.CircularAperture")
@patch("toirex.photometry.get_logger")
def test_aperture_photometry_subrot(mock_get_logger,
                                    mock_circular_aperture,
                                    mock_circular_annulus,
                                    mock_getdata,
                                    mock_aperture_photometry,
                                    mock_save_photometry):
    """Test aperture photometry."""

    config = {
        "photometry": {
            "APERTURE": "CircularAperture",
            "ANNULUS": "CircularAnnulus",
            "RADIUS": "10",
            "BKGWINDOWS": "(15, 20)",
            "SAVE_MAGNITUDE": "Y",
        },
        "inputs": {
            "FLUXEXT": "0",
            "VAREXT": "1",
        },
    }

    positions = Table({
        "x_0": [10.0, 20.0],
        "y_0": [15.0, 25.0],
    })

    fname = Path("test.fits")

    data = np.ones((100, 100))
    var = np.ones((100, 100))

    mock_getdata.side_effect = [data, var]

    # Mock aperture
    aperture = Mock()
    aperture.area = 100.0
    mock_circular_aperture.return_value = aperture

    # Mock annulus
    annulus_aperture = Mock()

    mask1 = Mock()
    mask1.data = np.ones((2, 2), dtype=bool)
    mask1.multiply.side_effect = [
        np.array([
            [10.0, 10.0],
            [10.0, 10.0],
        ]),
        np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
    ]

    mask2 = Mock()
    mask2.data = np.ones((2, 2), dtype=bool)
    mask2.multiply.side_effect = [
        np.array([
            [20.0, 20.0],
            [20.0, 20.0],
        ]),
        np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
    ]

    annulus_aperture.to_mask.return_value = [mask1, mask2]
    mock_circular_annulus.return_value = annulus_aperture

    # Mock aperture photometry result
    phot = Table({
        "xcenter": [10.0, 20.0],
        "ycenter": [15.0, 25.0],
        "aperture_sum": [1000.0, 2000.0],
        "aperture_sum_err": [10.0, 20.0],
    })

    mock_aperture_photometry.return_value = phot

    mock_save_photometry.return_value = Path("test.phot.fits")

    result = aperture_photometry_subrot(
        config,
        fname,
        positions,
    )

    # Check aperture positions and radius
    expected_positions = np.array([
        [10.0, 15.0],
        [20.0, 25.0],
    ])

    assert mock_circular_aperture.call_count == 1

    aperture_args = mock_circular_aperture.call_args

    np.testing.assert_array_equal(
        aperture_args.args[0],
        expected_positions,
        )

    assert aperture_args.kwargs["r"] == 10.0

    # Check annulus positions and radii

    assert mock_circular_annulus.call_count == 1

    annulus_args = mock_circular_annulus.call_args

    np.testing.assert_array_equal(
        annulus_args.args[0],
        expected_positions,
        )

    assert annulus_args.kwargs["r_in"] == 15.0
    assert annulus_args.kwargs["r_out"] == 20.0

    # Check input data
    assert mock_getdata.call_count == 2

    mock_getdata.assert_any_call(
        fname,
        ext=0,
    )

    mock_getdata.assert_any_call(
        fname,
        ext=1,
    )

    # Check aperture photometry
    mock_aperture_photometry.assert_called_once()

    call_args = mock_aperture_photometry.call_args

    assert np.array_equal(
        call_args.args[0],
        data,
    )

    assert call_args.args[1] is aperture

    np.testing.assert_array_equal(
        call_args.kwargs["error"],
        np.sqrt(var),
    )

    # Check the photometry table before saving
    saved_phot = mock_save_photometry.call_args.args[1]

    assert "bkg" in saved_phot.colnames
    assert "bkg_var" in saved_phot.colnames
    assert "bkg_sum" in saved_phot.colnames
    assert "bkg_var_sum" in saved_phot.colnames
    assert "flux_net" in saved_phot.colnames
    assert "var_net" in saved_phot.colnames
    assert "x_fit" in saved_phot.colnames
    assert "y_fit" in saved_phot.colnames

    # Check background calculation
    np.testing.assert_array_equal(
        saved_phot["bkg"],
        [10.0, 20.0],
    )

    np.testing.assert_array_equal(
        saved_phot["bkg_var"],
        [0.25, 0.25],
    )

    # bkg_sum = bkg * aperture area
    np.testing.assert_array_equal(
        saved_phot["bkg_sum"],
        [1000.0, 2000.0],
    )

    # bkg_var_sum = bkg_var * aperture area
    np.testing.assert_array_equal(
        saved_phot["bkg_var_sum"],
        [25.0, 25.0],
    )

    # flux_net = aperture_sum - bkg_sum
    np.testing.assert_array_equal(
        saved_phot["flux_net"],
        [0.0, 0.0],
    )

    # var_net = aperture_sum_err² + bkg_var_sum
    np.testing.assert_array_equal(
        saved_phot["var_net"],
        [125.0, 425.0],
    )

    # Check column renaming
    assert "xcenter" not in saved_phot.colnames
    assert "ycenter" not in saved_phot.colnames

    np.testing.assert_array_equal(
        saved_phot["x_fit"],
        [10.0, 20.0],
    )

    np.testing.assert_array_equal(
        saved_phot["y_fit"],
        [15.0, 25.0],
    )

    # Check save_magnitude and save_photometry arguments
    mock_save_photometry.assert_called_once_with(
        fname,
        saved_phot,
        history="Aperture photometry table added on file update.",
        save_magnitude=True,
        flext=0,
    )

    # Check return value
    assert result == Path("test.phot.fits")


@patch("toirex.photometry.plot_epsf")
@patch("toirex.photometry.EPSFBuilder")
@patch("toirex.photometry.extract_stars")
@patch("toirex.photometry.PSFPhotometry")
@patch("toirex.photometry.CircularGaussianSigmaPRF")
@patch("toirex.photometry.get_logger")
def test_make_epsf(mock_get_logger,
                   mock_psf_model,
                   mock_psf_photometry,
                   mock_extract_stars,
                   mock_epsf_builder,
                   mock_plot_epsf):
    """Test ePSF construction from supplied star positions."""

    frame = np.ones((100, 100))
    err = np.ones((100, 100))

    star_positions = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
        [70.0, 80.0],
        [20.0, 70.0],
        [40.0, 30.0],
        [60.0, 50.0],
        [80.0, 20.0],
        [25.0, 25.0],
        [75.0, 75.0],
    ])

    # Mock PSF model
    psf_model = Mock()
    mock_psf_model.return_value = psf_model

    # Mock PSF photometry
    psfphot = Mock()

    phot = Table({
        "flags": [0] * 10,
        "flux_init": [
            10.0, 20.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 100.0,
        ],
        "x_fit": [
            10.0, 20.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 100.0,
        ],
        "y_fit": [
            11.0, 21.0, 31.0, 41.0, 51.0,
            61.0, 71.0, 81.0, 91.0, 101.0,
        ],
    })

    psfphot.return_value = phot
    mock_psf_photometry.return_value = psfphot

    # Mock extracted stars
    epsf_stars = [Mock()] * 5
    mock_extract_stars.return_value = epsf_stars

    # Mock EPSFBuilder
    epsf = Mock()
    fitted_stars = Mock()

    epsf_builder = Mock(return_value=(epsf, fitted_stars))
    mock_epsf_builder.return_value = epsf_builder

    plot_fname = "epsf_plot.pdf"

    result = make_epsf(
        frame,
        err=err,
        star_positions=star_positions,
        aperture_radius=4,
        fwhm=7.0,
        threshold=50,
        cutout_size=25,
        fit_shape=(15, 15),
        oversample=4,
        plot_fname=plot_fname,
    )

    # Check PSF model
    mock_psf_model.assert_called_once_with(
        flux=1,
        sigma=7.0 / 2.355,
    )

    # DAOStarFinder should not be created when positions are supplied
    # (there is no _make_daostarfinder mock in this test).

    # Check PSFPhotometry
    mock_psf_photometry.assert_called_once_with(
        psf_model,
        (15, 15),
        finder=None,
        aperture_radius=4,
    )

    psfphot.assert_called_once_with(
        frame,
        error=err,
        init_params=star_positions,
    )

    # 90th percentile of the fluxes is 91.
    # Therefore only the source with flux_init=100 is selected.
    #
    # Check that extract_stars receives the selected position.
    extract_table = mock_extract_stars.call_args.args[1]

    np.testing.assert_array_equal(
        extract_table["x"],
        [100.0],
    )

    np.testing.assert_array_equal(
        extract_table["y"],
        [101.0],
    )

    assert mock_extract_stars.call_args.kwargs["size"] == 25

    # Check EPSFBuilder
    mock_epsf_builder.assert_called_once_with(
        oversampling=4,
        smoothing_kernel="quadratic",
        recentering_maxiters=10,
        maxiters=10,
        progress_bar=True,
    )

    epsf_builder.assert_called_once_with(epsf_stars)

    # Check plot
    mock_plot_epsf.assert_called_once_with(
        epsf,
        fitted_stars,
        plot_fname=plot_fname,
    )

    # Check return value
    assert result is epsf
# End
