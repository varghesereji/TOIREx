#!/usr/bin/env python3
import re
import numpy as np
import subprocess
from scipy import signal
from astropy.stats import mad_std
from astropy.io import fits


def extract_number_from_fname(fname):
    numbers = re.findall(r"\d+", fname)
    return numbers


def read_fits_header(filename, ext=0):
    '''
    Function to read the header of the fits file.
    Input
    -----
    filename: Name of the fits file to read.
    ext: Extension of the fits file. Default 0
    '''
    header = fits.getheader(filename, ext=ext)
    return header


def read_fits_data(filename, ext=0):
    data = fits.getdata(filename, ext=ext)
    return data


def open_in_editor(path, config):
    """
    Open a text file in the desired text editor.
    """
    editor = config['inits']['EDITOR']
    subprocess.run([editor, str(path)])


def DitherDetection(ObjectFile, ContWindowSelection,
                    startLoc=None, avgHWindow=21, TraceHWidth=5):

    """identify the center of a spectrum window """
    if isinstance(ObjectFile, str):
        ObjectFile = read_fits_data(ObjectFile)

    if startLoc is None:
        startLoc = ObjectFile.shape[1]//2
    # Starting labelling Reference XD cut data;
    WindowStart = ContWindowSelection[0]
    WindowEnd = ContWindowSelection[1]
    RefXD = np.nanmedian(ObjectFile[WindowStart:WindowEnd,
                                    startLoc-avgHWindow:startLoc+avgHWindow],
                         axis=1)
    Refpixels = np.arange(len(RefXD))+WindowStart
    Bkg = signal.order_filter(
        RefXD, domain=[True]*TraceHWidth*5, rank=int(TraceHWidth*5/10)
    )
    Flux = np.abs(RefXD - Bkg)
    ThreshMask = RefXD > (Bkg + np.abs(mad_std(Flux))*6)
    centerpix = np.sum(
        Flux[ThreshMask]*Refpixels[ThreshMask]
    ) / np.sum(Flux[ThreshMask])

    return centerpix


def read_txt_file(filename):
    txtline = []
    with open(filename, 'r') as txtfile:
        for line in txtfile:
            txtline.append(line.strip().split(" "))
    return txtline

# End
