from pathlib import Path
from collections import defaultdict

from ariastro import combine_process

from .setups import get_logger
from .utils import extract_number_from_fname
from .obscatalog import read_catalog
from .utils import open_in_editor
from .utils import read_txt_file
from .plottings import imageplot
from .instrument import instruments


def manual_inspection_obj(config, dirname):
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
        if (config['inits']['TIMESERIES'] == 'Y'):
            acceptall = True
            add_space = True
        elif config['inits']['MODE'] == 'AUTO':
            acceptall = True
            add_space = True
        for target in targets_name:
            if not acceptall and config['visual']['SCIENCE'] == 'Y':
                target_fname = Path(dirname) / target
                title = target
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
                Obj2Comb_txt.write(target+"\n")
            elif UserInput == 'acceptall':
                acceptall = True
                print("Accepting every single remaining images of this night")
            if add_space:
                Obj2Comb_txt.write("\n")
        Obj2Comb_txt.close()
        print("Filenames are entered into", Obj2Comb_fname)
        open_in_editor(Obj2Comb_fname, config)


def manual_inspection_flats(config, dirname):
    logger = get_logger("manual_inspect")
    dictkw = config['inits']['DICTKW']
    txtfile_re = "Objects_flats_group*.txt"
    op_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(op_path.glob(txtfile_re))
    print("Use the following instructions to select the frames")
    print("'r': Reject the frame for currect SCIENCE frame")
    print("'ra': Reject the frame from the analysis")
    print("'a': Accept the frame for current SCIENCE frame")
    print("'aa': Accept the frame for the analysis")
    print("'acceptall': Accept all frames without inspection")
    for f in files_list:
        number = extract_number_from_fname(f.name)

        print("Group number running:", int(number[0]), "\n")
        logger.info("Calling file" + f.name)
        read_file = read_txt_file(f)
        always_accept_list = []
        always_reject_list = []
        acceptall = False
        finalflat_txtfname = "Objects_finalflats_group{}.txt".format(number[0])
        finalflat_txt = open(op_path / finalflat_txtfname, "w")
        for line in read_file:
            object_name = line[0]
            flats_list = line[1:]
            if not acceptall:
                print("*"*30)
                print("Inspecting flats for {}".format(object_name))
            for target in flats_list:
                if target in always_reject_list:
                    flats_list.remove(target)
                    continue
                if target in always_accept_list:
                    # print(target, "Is always accepted")
                    continue
                if not acceptall and config['visual']['FLAT'] == 'Y':
                    target_fname = Path(dirname) / target
                    title = target_fname
                    print(target)
                    imageplot(target_fname, title=title)
                if acceptall:
                    UserInput = 'aa'
                else:
                    UserInput = input(
                        'Enter according to above instruction:'
                    )
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
            fnums = []
            for flat in flats_list:
                fnum = instruments[dictkw]['sort_filename_key'](flat)
                fnums.append(fnum)
            comb_fnums = "_".join([str(n) for n in fnums])
            targets_path = [Path(dirname) / frame for frame in flats_list]
            comb_flatname = "Comb_flats_{}.fits".format(comb_fnums)
            object_flat_list = object_name + " " + comb_flatname + "\n"
            finalflat_txt.write(object_flat_list)
            combine_fname = op_path / comb_flatname
            if combine_fname.exists():
                continue

            fluxexts = list(config['inputs']['FLUXEXT'])
            varexts = list(config['inputs']['VAREXT'])
            logger.info("Flux extensions: {}".format(fluxexts))
            logger.info("Variance extensions: {}".format(varexts))
            logger.info("Combining {} by biweight".format(targets_path))
            combine_process(targets_path,
                            combine_fname,
                            method='biweight',
                            fluxext=fluxexts,
                            varext=varexts)
            # print("Saved the flat frame", combine_fname)
        finalflat_txt.close()


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
    print("'acceptall': Accept all frames without inspection")

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
                        title = target
                        print(target)
                        imageplot(target_fname, title=title)

                    if acceptall:
                        UserInput = 'aa'
                    else:
                        UserInput = input(
                            'Enter "r" to reject and "aa" to accept:'
                        )
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
                fnums = []
                for cals in filenames:
                    fnum = instruments[dictkw]['sort_filename_key'](cals)
                    fnums.append(fnum)
                comb_fnums = "_".join([str(n) for n in fnums])
                targets_path = [Path(dirname) / frame for frame in filenames]
                comb_filename = lamp.lower() + "_comb_{}.fits".format(
                    comb_fnums)
                txtfile_line.append(comb_filename)
                combine_fname = op_path / comb_filename
                if not combine_fname.exists():
                    fluxexts = list(config['inputs']['FLUXEXT'])
                    varexts = list(config['inputs']['VAREXT'])
                    logger.info("Flux extensions: {}".format(fluxexts))
                    logger.info("Variance extensions: {}".format(varexts))
                    logger.info("Combining {} by biweight".format(targets_path))
                    combine_process(targets_path,
                                    combine_fname,
                                    method='biweight',
                                    fluxext=fluxexts,
                                    varext=varexts)
                    # print("Saved the lamp frame", combine_fname)
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
