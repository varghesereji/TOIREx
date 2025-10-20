#!/usr/bin/env python3

from pathlib import Path

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import matplotlib.pyplot as plt
from .utils import read_txt_file
from .plottings import imageplot


def targetfind_auto(fname):
    data = fits.getdata(fname, ext=0)
    mean, median, std = sigma_clipped_stats(data)
    daofind = DAOStarFinder(fwhm=6.0, threshold=50,
                            brightest=None, exclude_border=True)
    cl_data = data # - median
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


def targetfind_manual(fname):
    centroids = imageplot(fname, ext=0, title="Select sources",
                          line_profile="aperture", get_target=False)
    print(centroids)


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
                targetfind_auto(frametoextract)
            elif config['photometry']['FINDSOURCE'] == 'MANUAL':
                targetfind_manual(frametoextract)
