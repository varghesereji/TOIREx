#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict
from astropy.io import ascii

from .instrument import instrument_class


def extract_catalog_entries(fname, dictkw):
    '''
    This function will take the required kewords from
    the header and enter it into a list.
    This list will be used to enter into the catalog.
    '''
    # Standardising header with the function appropriate to the instrument.
    # std_header_func = functions_dict[dictkw]['StandardiseHeader']
    instrument = instrument_class[dictkw]
    std_header = instrument.standardise_header(fname)

    # Using the keywords required for specific instrument.
    required_kws = instrument.catalog_headers
    entries = [Path(fname).name]
    for kws in required_kws:
        entries.append(str(std_header[kws]).replace(" ", "_"))
    return entries


def create_catalog(dirname, config):
    '''
    Function to create the catalogue of
    files available in the list.
    Inputs
    ------
    dirname: Name of the directory.
    config: The main config file.
    '''
    catalog_name = config['outputs']['CATALOGUE_NAME']
    op_path = Path(config['outputs']['OP_DIR']) / dirname
    file_path = op_path / catalog_name

    # If the catalog already exists, skip this step.
    # Else, create a new one.
    print("Catalog name: {}".format(file_path))
    if file_path.exists():
        print("Catalog for this directory already exists. Skipping...")
        return

    # Importing all fits files in the directory.
    dirname = Path(dirname)
    fitsfiles = list(dirname.glob("*.fits"))
    # Regular expression to remove from the catalog
    remove_regs = config['outputs']['EXCLUDE_REG'].strip().split(",")
    for regs in remove_regs:
        sel_names = dirname.glob(regs)
        for fname in sel_names:
            if fname in fitsfiles:
                fitsfiles.remove(fname)
    dictkw = config['inits']['DICTKW']  # Calling the dictionary keyword.
    instrument = instrument_class[dictkw]
    # Sorting the filenames based on FNUM
    # fnamesortfunc = functions_dict[dictkw]['filename_sort_func']
    fnamesortfunc = instrument.sort_filename_key
    sorted_files = sorted(fitsfiles, key=fnamesortfunc)

    filename_infos = []
    catalog_headers = instrument.catalog_headers
    catalog_headers.insert(0, 'FNAME')
    catalog_headers.append('FLAG')
    for filename in sorted_files:
        # To avoid selecting wrong frames,
        # making a decision weather a file to select or not.
        frame_decision_function = instrument.frame_select
        if not frame_decision_function(filename):
            continue
        entries_list = extract_catalog_entries(filename, dictkw)
        flagged_list = instrument.catalog_flag(entries_list)
        if config['inputs']['SKY'] == 'Y':
            flagged_list = flag_sky(flagged_list)
        filename_infos.append(flagged_list)
    ascii.write(list(zip(*filename_infos)),
                file_path, names=catalog_headers,
                format="fixed_width", overwrite=True)


def read_catalog(dirname, config):
    catalog_name = config['outputs']['CATALOGUE_NAME']
    catalog_path = Path(config['outputs']['OP_DIR']) / dirname
    file_path = catalog_path / catalog_name
    dictkw = config['inits']['DICTKW']
    instrument = instrument_class[dictkw]
    header_keys = instrument.catalog_headers
    header_keys.append("FLAG")
    header_keys.insert(0, "FNAME")
    catalogue_dict = defaultdict(list)
    with open(file_path, 'r') as catalogs:
        for entries in catalogs:
            entry_list = entries.strip().split(' ')
            for n, element in enumerate(entry_list):
                catalogue_dict[header_keys[n]].append(element)
    return catalogue_dict


def flag_sky(catalogue_entries):
    """
    The function to identify sky frames
    and flag them.
    """
    fname = catalogue_entries[0]
    if re.search("sky", fname, re.IGNORECASE):
        catalogue_entries[-1] = 'SKY'
    return catalogue_entries


# End
