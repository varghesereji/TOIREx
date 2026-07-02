#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from collections import defaultdict


from .setups import get_logger
from .utils import extract_number_from_fname
from .obscatalog import read_catalog
from .utils import open_in_editor
from .utils import read_txt_file
from .utils import combine_frames

from .plottings import imageplot
from .instrument import instruments
from .dithering import find_shift


def making_title_for_frame(fname, dirname, config):
    """
    Function to make title for the plot
    """
    dictkw = config['inits']['DICTKW']
    header_keys = instruments[dictkw]["grouping_keys"].copy()
    header_keys.append("FLAG")
    catalog_dict = read_catalog(dirname, config, showcatname=False)
    fnames_array = np.array(catalog_dict["FNAME"])
    fname_mask = fnames_array == fname
    title_str = [r"FNAME : $\bf{" + str(fname) + "}$\n"]
    for keys in header_keys:
        col = np.array(catalog_dict[keys])
        value = col[fname_mask][0]

        element = keys + r" : $\mathbf{" + str(value) + "}$\n"
        title_str.append(element)
    title_str = " ".join(title_str)
    # Escape underscores for mathtext
    title_str = title_str.replace("_", r"\_")
    return title_str


def manual_inspection_obj(config, dirname):
    """
    Interactively inspect science frames, select frames for combination,
    and generate grouping files for subsequent spectral extraction.

    This function reads object lists (`Objects_flats_group*.txt`) from the
    specified output directory and creates corresponding
    `ObjectsToCombine_group*.txt` files. Depending on the pipeline
    configuration, the user may manually accept or reject each science
    frame, or all frames may be accepted automatically (e.g., in AUTO mode
    or for time-series observations).

    When dithering is enabled, image shifts are computed relative to a
    reference frame using ``find_shift()``. The calculated integer pixel
    shifts are written alongside each filename. If the measured shift
    exceeds three times the estimated uncertainty, a blank line is inserted
    into the output file to indicate the start of a new combination group,
    and the current frame becomes the new reference frame.

    After the initial grouping file is created, it is opened in the user's
    preferred editor for manual modification. Blank lines separate groups of
    frames that should **not** be combined.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary containing initialization,
        visualization, dithering, and output settings.
    dirname : str or pathlib.Path
        Name of the directory containing the grouped object lists relative
        to ``config['outputs']['OP_DIR']``.

    Notes
    -----
    - Input files must follow the naming convention
      ``Objects_flats_group*.txt``.
    - Output files are written as
      ``ObjectsToCombine_group<group_number>.txt``.
    - In time-series mode (``config['inits']['TIMESERIES'] == 'Y'``),
      all frames are accepted automatically and separated by blank lines.
    - In AUTO mode, all frames are accepted automatically without user
      interaction.
    - When dithering is enabled, each output line contains::

          filename x_shift y_shift

      where the shifts are rounded to the nearest integer pixel.
    """
    logger = get_logger("manual_inspect")
    txtfile_re = "Objects_flats_group*.txt"
    txt_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(txt_path.glob(txtfile_re))
    Obj2CombFile = "ObjectsToCombine_group{}.txt"
    for txtf in files_list:
        number = extract_number_from_fname(txtf.name)
        Obj2Comb_fname = txt_path / Obj2CombFile.format(number[0])
        Obj2Comb_txt = open(Obj2Comb_fname, 'w')
        print("Group number running:", int(number[0]), "\n")
        logger.info("Calling file " + txtf.name)
        txt_read_list = read_txt_file(txtf)
        # Selecting only the science frames.
        targets_name = [i[0] for i in txt_read_list]
        acceptall = False
        add_space = False
        reference_frame = None
        if (config['inits']['TIMESERIES'] == 'Y'):
            acceptall = True
            add_space = True
            config['dither']['DITHERING'] == 'N'
            print("Making dithering = N since timeseries data")
        elif config['inits']['MODE'] == 'AUTO':
            acceptall = True
            add_space = True
        for target in targets_name:
            target_fname = Path(dirname) / target
            if not acceptall and config['visual']['SCIENCE'] == 'Y':
                print("Displaying ", target)
                title = making_title_for_frame(target,
                                               dirname,
                                               config)
                imageplot(target_fname, title=title)
            if acceptall:
                UserInput = 'aa'
            else:
                UserInput = input(
                    'Enter "r" to reject and "aa" to accept:'
                )
            if UserInput == 'r':
                print("Removing", target)
                targets_name.remove(target)
            elif UserInput == 'aa':
                print("Accepting", target)
                line_to_txt = target
                if config['dither']['DITHERING'] == 'Y':
                    if reference_frame is None:
                        reference_frame = target_fname
                        line_to_txt += " 0 0"
                    else:
                        img_shift = find_shift(reference_frame, target_fname,
                                               config)
                        shift = np.array(img_shift[0], dtype=np.float64)
                        shift = np.rint(shift).astype(int)
                        shift_err = img_shift[1]
                        distance = np.sqrt(np.sum(shift**2))
                        line_to_txt += " "
                        line_to_txt += " ".join(map(str, shift))
                        if np.isnan(shift_err):
                            shift_err = 2
                        if distance > 3 * shift_err:
                            reference_frame = target_fname
                            Obj2Comb_txt.write("\n")
                Obj2Comb_txt.write(line_to_txt+"\n")
            elif UserInput == 'acceptall':
                acceptall = True
                print("Accepting every single remaining images of this night")
            if add_space:
                Obj2Comb_txt.write("\n")
        Obj2Comb_txt.close()
        print("Selected filenames are entered into", Obj2Comb_fname)
        print("Add space between the lines which you do not want to group")
        print("Remove the space if you want to combine")
        open_in_editor(Obj2Comb_fname, config)


def manual_inspection_flats(config, dirname, framecat="FLATS"):
    logger = get_logger("manual_inspect")
    dictkw = config['inits']['DICTKW']
    if framecat == "FLATS":
        txtfile_re = "Objects_flats_group*.txt"
    elif framecat == "SKY":
        txtfile_re = "Objects_sky_group*.txt"
    op_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(op_path.glob(txtfile_re))
    print("Use the following instructions to select the frames")
    print("'r': Reject the frame for currect SCIENCE frame")
    print("'ra': Reject the frame from the analysis")
    print("'a': Accept the frame for current SCIENCE frame")
    print("'aa': Accept the frame for the analysis")
    print("Press Enter without anything: Accept all frames without inspection")
    for f in files_list:
        number = extract_number_from_fname(f.name)

        print("Group number running:", int(number[0]), "\n")
        logger.info("Calling file" + f.name)
        read_file = read_txt_file(f)
        always_accept_list = []
        always_reject_list = []
        acceptall = False
        if framecat == "FLATS":
            finalframe_txtfname = "Objects_finalflats_group{}.txt".format(
                number[0]
            )
        elif framecat == "SKY":
            finalframe_txtfname = "Objects_finalsky_group{}.txt".format(
                number[0]
            )
        finalframe_txt = open(op_path / finalframe_txtfname, "w")
        for line in read_file:
            object_name = line[0]
            flats_list = line[1:]
            if not acceptall:
                print("*"*30)
                print("Inspecting {} for {}".format(framecat.lower(),
                                                    object_name))
            for target in flats_list:
                if target in always_reject_list:
                    flats_list.remove(target)
                    continue
                if target in always_accept_list:
                    # print(target, "Is always accepted")
                    continue
                if not acceptall and config['visual']['FLAT'] == 'Y':
                    target_fname = Path(dirname) / target
                    print("Displaying ", target)
                    title = making_title_for_frame(target,
                                                   dirname,
                                                   config)
                    print(target)
                    imageplot(target_fname, title=title)
                if acceptall:
                    UserInput = 'aa'
                else:
                    UserInput = input(
                        'Enter according to above instruction:'
                    ) or 'acceptall'
                if UserInput == 'ra':
                    print("Completely Removing", target)
                    flats_list.remove(target)
                    always_reject_list.append(target)
                elif UserInput == 'aa':
                    print("Always Accepting", target)
                    always_accept_list.append(target)
                elif UserInput == 'r':
                    flats_list.remove(target)
                elif UserInput == 'acceptall':
                    acceptall = True
                    print(
                        "Accepting every single remaining images of this night"
                    )
                else:
                    print("Accepting", target)
            fluxexts = list(config['inputs']['FLUXEXT'])
            varexts = list(config['inputs']['VAREXT'])
            logger.info("Flux extensions: {}".format(fluxexts))
            logger.info("Variance extensions: {}".format(varexts))
            # logger.info("Combining {} by biweight".format(targets_path))
            op_fname = framecat.lower()
            comb_framename = combine_frames(
                flats_list, op_path,
                instruments[dictkw]['sort_filename_key'],
                method='biweight',
                op_prefix="Comb_{}_".format(op_fname),
                fluxext=fluxexts,
                varext=varexts)
            object_frame_list = object_name + " " + comb_framename + "\n"
            finalframe_txt.write(object_frame_list)
        finalframe_txt.close()


def manual_inspection_cals(config, dirname):
    logger = get_logger("manual_inspect")
    dictkw = config['inits']['DICTKW']
    txtfile_re = "Objects_lamps_group*.txt"
    op_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(op_path.glob(txtfile_re))
    print("Use the following instructions to select the frames")
    print("'r': Reject the frame for currect SCIENCE frame")
    print("'ra': Reject the frame from the analysis")
    print("'a': Accept the frame for current SCIENCE frame")
    print("'aa': Accept the frame for the analysis")
    print("Press Enter without anything: Accept all frames without inspection")

    for f in files_list:
        number = extract_number_from_fname(f.name)

        print("Group number running:", int(number[0]), "\n")
        logger.info("Calling file" + f.name)
        read_file = read_txt_file(f)
        always_accept_list = []
        always_reject_list = []
        acceptall = False
        finalcals_txtfname = "Objects_finalcals_group{}.txt".format(number[0])
        finalcals_txt = open(op_path / finalcals_txtfname, "w")
        for line in read_file:
            object_name = line[0]
            txtfile_line = [object_name]
            cals_list = line[1:]
            if not acceptall:
                print("*" * 30)
                print("Inspecting cals for {}".format(object_name))
            lamps_dict = separate_lamps(config, dirname, cals_list)
            for lamp, filenames in lamps_dict.items():
                for target in filenames:
                    if target in always_reject_list:
                        filenames.remove(target)
                        continue
                    if target in always_accept_list:
                        # print(target, "Is always accepted")
                        continue
                    if not acceptall and config['visual']['LAMP'] == 'Y':
                        target_fname = Path(dirname) / target
                        print("Displaying", target)
                        title = making_title_for_frame(target,
                                                       dirname,
                                                       config)
                        imageplot(target_fname, title=title)

                    if acceptall:
                        UserInput = 'aa'
                    else:
                        UserInput = input(
                            'Enter "r" to reject and "aa" to accept:'
                        ) or 'acceptall'
                    if UserInput == 'ra':
                        print("Completely Removing", target)
                        filenames.remove(target)
                        always_reject_list.append(target)
                    elif UserInput == 'aa':
                        print("Always Accepting", target)
                        always_accept_list.append(target)
                    elif UserInput == 'r':
                        filenames.remove(target)
                    elif UserInput == 'acceptall':
                        acceptall = True
                        print(
                            "Accepting every remaining images of this night"
                        )
                    else:
                        print("Accepting", target)
                fluxexts = list(config['inputs']['FLUXEXT'])
                varexts = list(config['inputs']['VAREXT'])
                logger.info("Flux extensions: {}".format(fluxexts))
                logger.info("Variance extensions: {}".format(varexts))

                comb_filename = combine_frames(
                    filenames, op_path,
                    instruments[dictkw]['sort_filename_key'],
                    method='biweight',
                    op_prefix="Comb_lamp_"+lamp.lower() + "_",
                    fluxext=fluxexts,
                    varext=varexts)

                txtfile_line.append(comb_filename)
            object_cal_list = " ".join(txtfile_line) + "\n"
            finalcals_txt.write(object_cal_list)
        finalcals_txt.close()


def separate_lamps(config, dir_path, lamps_list):
    """
    config: config file.
    dir_path: path object.
    lamps_list: list of lamps.
    """

    # Opening the catalogue
    # catalogue = dir_path / config['outputs']['CATALOGUE_NAME']
    catalogue_entries = read_catalog(dir_path, config, showcatname=False)
    catalogue_names = catalogue_entries['FNAME']  # Filenames form catalogue
    catalogue_flags = catalogue_entries['FLAG']  # Flags from catalogue

    # Mask to find position of file in the catalogue
    # Grouping the files based on flag.
    lamps_dict = defaultdict(list)
    for index, filename in enumerate(catalogue_names):
        if filename in lamps_list:
            flag = catalogue_flags[index]
            lamps_dict[flag].append(filename)
    return lamps_dict

# End
