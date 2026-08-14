from unittest.mock import patch

import numpy as np
from astropy.table import Table

from toirex.photometry import calculate_snr


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

# End
