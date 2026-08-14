from unittest.mock import patch

import numpy as np
from astropy.table import Table
from astropy.io import fits

from toirex.photometry import calculate_snr
from toirex.photometry import calculate_magnitude
from toirex.photometry import save_photometry


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

# End
