from unittest.mock import patch

import numpy as np
from astropy.table import Table

from toirex.photometry import calculate_snr
from toirex.photometry import calculate_magnitude


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

# End
