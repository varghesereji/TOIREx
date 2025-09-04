from pathlib import Path

from .instrument import functions_dict
from .utils import read_fits_header


def extract_catalog_entries(fname, dictkw):
    '''
    This function will take the required kewords from
    the header and enter it into a list.
    This list will be used to enter into the catalog.
    '''
    # Standardising header with the function appropriate to the instrument.
    std_header_func = functions_dict[dictkw]['StandardiseHeader']
    std_header = std_header_func(fname)

    # Using the keywords required for specific instrument.
    required_kws = functions_dict[dictkw]['catalog_headers']
    entries = [Path(fname).name]
    for kws in required_kws:
        entries.append(str(std_header[kws]))
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
    else:
        # Creating an empty txt file.
        with open(file_path, 'w') as _:
            pass

    # Importing all fits files in the directory.
    dirname = Path(dirname)
    fitsfiles = dirname.glob("*.fits")

    dictkw = config['inits']['DICTKW']  # Calling the dictionary keyword.

    # Sorting the filenames based on FNUM
    fnamesortfunc = functions_dict[dictkw]['filename_sort_func']
    sorted_files = sorted(fitsfiles, key=fnamesortfunc)

    for filename in sorted_files:
        # To avoid selecting wrong frames,
        # making a decision weather a file to select or not.
        frame_decision_function = functions_dict[dictkw][
            'frame_select_function']
        if not frame_decision_function(filename):
            continue
        entries_list = extract_catalog_entries(filename, dictkw)
        flagged_list = functions_dict[dictkw][
            'catalog_flag'](entries_list)
        # print(entries_list)
        with open(file_path, 'a') as catalog:
            catalog.write(' '.join(flagged_list) + '\n')

# End
