#!/usr/bin/env python3
from pathlib import Path

from .utils import open_in_editor
from .instrument import instruments


def feed_to_txt_file(grouped_files, config, dirname, group):
    """
    Write grouped flat and lamp files to separate text files and open them
    in an editor.

    This function takes the grouped files dictionary (produced during
    grouping), identifies the flat-field and lamp calibration frames
    associated with each science object, and writes them into two text
    files:

        - Objects_flats_group{group}.txt
        - Objects_lamps_group{group}.txt

    Each line in the output file contains one science object filename, followed
    by the list of associated flat or lamp calibration files. After writing,
    each file is opened in the editor specified in the config.

    Parameters
    ----------
    grouped_files : dict
        Dictionary containing grouped science object and calibration file
    names.
        Must include at least the keys 'OBJECT', instrument.flat_kw, and
    instrument.lamp_kw.
    config : dict
        Configuration dictionary. Expected to contain:
          - config['inits']['DICTKW']: key for selecting the instrument class
          - config['outputs']['OP_DIR']: output directory path
          - config['editor']: preferred text editor (used by open_in_editor)
    dirname : str or Path
        Subdirectory under the output directory where the text files will be
    written.
    group : int
        Group number identifier used in the output file names.

    Outputs
    -------
    Creates two text files in the directory:
        <OP_DIR>/<dirname>/Objects_flats_group{group}.txt
        <OP_DIR>/<dirname>/Objects_lamps_group{group}.txt

    Each file is opened in the configured text editor after being written.

    Notes
    -----
    - If no calibration files exist for a given object, only the object
    filename
      will be written on that line.
    - The mapping between calibration type (flat or lamp) and grouped_files
    keys
      is provided by the instrument class (instrument.flat_kw,
    instrument.lamp_kw).
    """

    dictkw = config['inits']['DICTKW']
    instrument = instruments[dictkw]

    lamp_kw = instrument['lamp_kw']  # Keywords for flats
    flat_kw = instrument['flat_kw']  # Keywords for lamps
    lsets = [flat_kw, lamp_kw]    # List of both set of keywords
    objects = grouped_files['OBJECT']  # List of target filenames

    # Name of text files. Frist one will be for flats, second one is for lamps.
    txt_fnames = [
        'Objects_flats_group{}.txt'.format(group),
        'Objects_lamps_group{}.txt'.format(group)
    ]

    # No lamps if the reduction is for photometry.
    # Therefore, removing that element from the txt filenames.
    if config['inits']['TODO'] == 'P':
        txt_fnames.pop()

    for lset, txtfile in enumerate(txt_fnames):
        txtfile_path = Path(config['outputs']['OP_DIR']) / \
                            dirname / txtfile
        lampset = lsets[lset]
        txtfile = open(txtfile_path, 'w')
        for obj_fname in objects:
            line = obj_fname + " "
            for lamp in lampset:
                lamp_fnames = grouped_files[lamp]
                if len(lamp_fnames) == 0:
                    pass
                else:
                    line += " ".join(lamp_fnames)
                    line += " "
            line += "\n"
            txtfile.write(line)
        txtfile.close()
        open_in_editor(txtfile_path, config)

# End
