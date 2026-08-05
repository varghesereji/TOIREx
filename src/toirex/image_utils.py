#!/usr/bin/env python3

from photutils.centroids import centroid_2dg
from photutils.profiles import RadialProfile
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astropy.time import Time
import astropy.units as u

from .obscatalog import read_catalog


def get_radial_profile(image,
                       xypos,
                       edge_radii):
    """
    Compute the radial profile of a source.

    Parameters
    ----------
    image : ndarray
        Input image.
    xypos : tuple of float
        Source centre as ``(x, y)``.
    edge_radii : array_like
        Radii defining the annular bin edges.

    Returns
    -------
    photutils.profiles.RadialProfile
        Radial profile object.
    """

    return RadialProfile(
        image,
        xypos,
        radii=edge_radii
        )


def select_source(data, error=None, mask=None):
    centroid = centroid_2dg(data)
    # print("centroid", centroid)
    return centroid


def wcs_correction(frame_name, cat_name, config):
    catalog = read_catalog(cat_name)
    xy = (catalog['X'],
          catalog['Y'])
    ra_hms = list(catalog['RA'])
    dec_dms = list(catalog['Dec'])
    pmra = list(catalog['pmRA'])
    pmdec = list(catalog['pmDec'])

    ref_epoch = Time(float(config['wcs']['REF_EPOCH']), format='jyear')
    obs_epoch = Time(float(config['wcs']['OBS_EPOCH']), format='jyear')

    # Convert to SkyCoord and apply proper motion
    sky = SkyCoord(
        ra=ra_hms,
        dec=dec_dms,
        unit=(u.hourangle, u.deg),
        pm_ra_cosdec=pmra * u.mas/u.yr,
        pm_dec=pmdec * u.mas/u.yr,
        frame='icrs',
        obstime=f'J{ref_epoch}'
    )

    # Apply proper motion correction to the observation epoch
    sky_obs = sky.apply_space_motion(new_obstime=obs_epoch)

    # --- Fit WCS from matched points
    #  2nd-order SIP distortion; you can set None for TAN only
    fitted_wcs = fit_wcs_from_points(xy,
                                     sky_obs,
                                     sip_degree=2
                                     )
    with fits.open(frame_name, mode='update') as hdul:
        hdr = hdul[0].header

        # Convert WCS objects to header card
        wcs_header = fitted_wcs.to_header()
        # Update header in place
        hdr.update(wcs_header)

        # Commit changes to disk
        hdul.flush()
    print(f"WCS successfully added to {frame_name}")

# End
