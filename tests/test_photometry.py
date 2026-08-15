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



# End
