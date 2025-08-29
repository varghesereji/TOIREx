from astropy.io import fits


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

# End

