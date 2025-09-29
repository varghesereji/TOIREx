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


def read_txt_file(filename):
    txtline = []
    with open(filename, 'r') as txtfile:
        for line in txtfile:
            txtline.append(line.strip().split(" "))
    return txtline

# End
