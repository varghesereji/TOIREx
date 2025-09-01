import re
from pathlib import Path

from .utils import read_fits_header


'''
Functions for SpecTANSPEC instrument
'''


def sort_filename_key_function_SpecTANSPEC(fname):
    '''
    Function which return the key to sort SpecTANSPEC frames.
    '''
    basename = Path(fname).name
    number = re.search(r'(\d+)\.Z\.fits', basename)
    return int(number.group(1)) if number else -1


def standardiseheader_spectanspec(fname):
    header = read_fits_header(fname)
    header['FNUM'] = int(header['FNUM'])
    return header


def frameselect_decision_spectanspec(fname):
    '''
    This function will decide weather to select a frame or not.
    Since TANSPEC have both image and spectroscopy mode, need to decied weather
    to select certain files or not.
    '''
    header = read_fits_header(fname)
    if (header['NAXIS1'] == 2048) & (header['NAXIS2'] == 2048):
        return True
    else:
        return False


'''
Functions for TIRSPEC instrument
'''


def sort_filename_key_function_TIRSPEC(fname):
    '''
    Function which return the key to sort SpecTANSPEC frames.
    '''
    basename = Path(fname).name
    number = re.search(r'(\d+)\.Z\.fits', basename)
    return int(number.group(1)) if number else -1


def standardiseheader_tirspec(fname):
    header = read_fits_header(fname)
    fnum = sort_filename_key_function_TIRSPEC(fname)
    header['FNUM'] = fnum
    return header


def frameselect_decision_tirspec(fname):
    '''
    This function will decide weather to select a frame or not.
    Since TANSPEC have both image and spectroscopy mode, need to decied weather
    to select certain files or not.
    '''
    return True


functions_dict = {
    'SpecTANSPEC': {
        'filename_sort_func': sort_filename_key_function_SpecTANSPEC,
        'StandardiseHeader': standardiseheader_spectanspec,
        'frame_select_function': frameselect_decision_spectanspec,
        'catalog_headers': [
            'FNUM', 'A_UTC', 'DATE_OBS', 'CONTGAIN',
            'ITIMEREQ', 'FILTER', 'GRATING', 'SLIT',
            'CALMIR', 'OBJECT',
            'ARGONL', 'NEONL', 'CONT1L', 'CONT2L',
            'A_TRGTRA', 'A_TRGTDE'
                            ]
    },
    'TIRSPEC': {
        'filename_sort_func': sort_filename_key_function_TIRSPEC,
        'StandardiseHeader': standardiseheader_tirspec,
        'frame_select_function': frameselect_decision_tirspec,
        'catalog_headers': [
            'FNUM', 'TIME', 'DATE', 'UPPER',
            'LOWER', 'SLIT', 'CALMIR', 'TARGET',
            'TCSRA', 'TCSDEC'
                            ]
    }
    }
# End
