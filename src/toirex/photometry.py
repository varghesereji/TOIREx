#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import ast
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianPSF
from photutils.psf import GaussianPSF
from photutils.psf import PSFPhotometry

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry

import matplotlib.pyplot as plt
from .utils import read_txt_file
from .plottings import imageplot
from .io_utils import convert_radec


def targetfind_auto(fname):
    data = fits.getdata(fname, ext=0)
    mean, median, std = sigma_clipped_stats(data)
    daofind = DAOStarFinder(fwhm=6.0, threshold=50,
                            brightest=None, exclude_border=True)
    cl_data = data - median
    plt.figure()
    plt.imshow(cl_data, origin='lower', vmin=0, vmax=mean+std)

    sources = daofind(cl_data)
    print(sources)
    id_no = sources['id']
    x_pos = sources['xcentroid']
    y_pos = sources['ycentroid']
    for i, index in enumerate(id_no):
        plt.text(x_pos[i], y_pos[i], index)
    plt.show()
    positions = Table()
    positions['x_0'] = x_pos
    positions['y_0'] = y_pos
    return positions


def targetfind_manual(fname):
    centroids = imageplot(fname, ext=0, title="Select sources",
                          line_profile="aperture", get_target=False)
    positions = Table()
    positions['x_0'] = centroids[:, 1]
    positions['y_0'] = centroids[:, 0]
    return positions

# Aperture photometry


def aperture_photometry_subrot(config, fname, positions):
    positions = np.array([positions['x_0'],
                          positions['y_0']]).T
    apertures = CircularAperture(positions, r=4)
    flext = int(config['inputs']['FLUXEXT'])
    varext = config['inputs']['VAREXT']
    try:
        varext = int(varext)
    except ValueError:
        print("Using flux array as variance")
        varext = flext

    data = fits.getdata(fname, ext=flext)
    var = fits.getdata(fname, ext=varext)
    phot = aperture_photometry(data, apertures,
                               error=np.sqrt(var))
    print(phot)
    opfname = save_photometry(
        fname, phot,
        history="Aperture photometry table added on file update.",
        flext=flext
        )

    return opfname

# PSF photometry


def psf_photometry_subrot(config, fname, positions):
    if config['photometry']['MODEL'] == 'CircularGaussianPSF':
        fwhm = config['photometry']['FWHM']
        psf_model = CircularGaussianPSF(flux=1, fwhm=fwhm)
    elif config['photometry']['MODEL'] == 'GaussianPSF':
        psf_fwhm = config['photometry']['PSF_FWHM']
        psf_fwhm = list(float(x) for x in ast.literal_eval(psf_fwhm))
        psf_angle = float(config['photometry']['PSF_ANGLE'])
        psf_model = GaussianPSF(flux=1,
                                x_fwhm=psf_fwhm[0],
                                y_fwhm=psf_fwhm[1],
                                theta=psf_angle)

    fit_shape = (15, 15)
    psfphot = PSFPhotometry(psf_model, fit_shape,
                            aperture_radius=4)
    flext = int(config['inputs']['FLUXEXT'])
    varext = config['inputs']['VAREXT']
    try:
        varext = int(varext)
    except ValueError:
        print("Using flux array as variance")
        varext = flext

    data = fits.getdata(fname, ext=flext)
    var = fits.getdata(fname, ext=varext)
    error = np.sqrt(var)
    phot = psfphot(data, error=error, init_params=positions)
    opfname = save_photometry(
        fname, phot,
        history='PSF photometry table added on file update.',
        flext=flext
    )
    return opfname


# WCS conversion


def save_to_wcs(final_fname):
    opdir = Path(final_fname.parent)
    print(opdir)
    with fits.open(final_fname) as hdul:
        primary_header = hdul[0].header
        w = WCS(primary_header)
        phot_table = Table(hdul['PHOTOMETRY'].data)
        # print(phot_table)
        x = phot_table['x_fit']
        y = phot_table['y_fit']
        # print(x)
        ra, dec = w.wcs_pix2world(x, y, 0)
        ra, dec = convert_radec(ra, dec)
        # print(ra, dec)
        colnames = phot_table.colnames
        reordered = Table()
        reordered[colnames[0]] = phot_table[colnames[0]]
        reordered['RA'] = ra
        reordered['Dec'] = dec
        for name in colnames[1:]:
            reordered[name] = phot_table[name]
        # print(reordered)
        out_table_name = final_fname.stem + ".wcs.fits"
        # out_table_path = final_fname.parent
        out_table_name = opdir / out_table_name
        hdu = fits.BinTableHDU(data=reordered, header=primary_header,
                               name='PHOTOMETRY')
        hdul_out = fits.HDUList([fits.PrimaryHDU(header=primary_header), hdu])
        hdul_out.writeto(out_table_name, overwrite=True)
        print("{} saved with WCS coordinates".format(out_table_name))


# File Saving

def save_photometry(fname, phot_table, history="Photometry table added",
                    flext=0):
    table_hdu = fits.BinTableHDU(phot_table, name="PHOTOMETRY")
    primary_hdu = fits.PrimaryHDU(
        header=fits.getheader(fname, ext=flext)
        )
    hdul = fits.HDUList([primary_hdu, table_hdu])
    opdir = Path(fname.parent)
    opfname = fname.stem + ".phot.fits"
    opfname = opdir / opfname
    hdul.writeto(opfname, overwrite=True)
    return opfname

# Extraction


def photometry_extraction(config, dirname):
    # dictkw = config['inits']['DICTKW']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "Readytoextract_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)
    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        for txtline in txtfile_full:
            frametoextract = txtline[0]
            frametoextract = opdir / frametoextract
            if config['photometry']['FINDSOURCE'] == 'AUTO':
                centroids = targetfind_auto(frametoextract)
            elif config['photometry']['FINDSOURCE'] == 'MANUAL':
                centroids = targetfind_manual(frametoextract)
            # Doing photometry
            if config['photometry']['METHOD'] == 'PSF':
                withphot = psf_photometry_subrot(config, frametoextract,
                                                 positions=centroids)
            elif config['photometry']['METHOD'] == 'Aperture':
                withphot = aperture_photometry_subrot(config, frametoextract,
                                                      positions=centroids)
            print("Photometry data saved to {}".format(withphot))
            save_to_wcs(withphot)
