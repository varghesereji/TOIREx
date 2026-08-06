#!/usr/bin/env python3

from pathlib import Path

from .setups import read_config, create_config
from .setups import read_args
from .setups import print_banner
from .setups import create_dir
from .setups import read_dirs
from .setups import add_dict_keywords
from .setups import setup_logger_from_config

from .obscatalog import create_catalog
from .grouping_frames import grouping_items, grouping_with_re
from .selecting_frames import feed_to_txt_file
from .manual_inspection import manual_inspection_obj
from .manual_inspection import manual_inspection_flats
from .manual_inspection import manual_inspection_cals
from .clean_frame import frame_correction
from .utils import get_instrument_dir
from .dithering import subtract_dithers
from .dithering import combine_dithers
from .spectral_reduction import spectral_reduction
from .photometry import photometry_extraction


def get_directories(config, required='list'):
    if required == 'string':
        datadirs = config['inits']['DATA']
    else:
        datadirs = config['inits']['DATA'].strip().split(",")
    opdir = config['outputs']['OP_DIR']
    return opdir, datadirs


def create_fileslog(config):
    '''
    Task 0.
    Creation of log of data files.
    Input
    -------
    Main config.
    '''
    opdir, all_datadirs = get_directories(config)
    print("Running Task 0")
    for datadir in all_datadirs:
        print("Running for the directory {}".format(datadir))
        create_catalog(datadir, config)


def select_files(config):
    '''
    Task 1.
    Selection of files for reduction.
    Input
    -------
    Main config.
    '''
    opdir, all_datadirs = get_directories(config)
    print("Running Task 1")
    for datadir in all_datadirs:
        # Grouping files
        print("Running for the directory {}".format(datadir))
        groups_dict = grouping_items(config, datadir)
        print("Use a space if you have more than one group.")
        print("Press 'n' if you want to enter the filename regular expression")
        selected_groups = input("Enter the group number you want to reduce:")
        if selected_groups == 'n':
            groups_dict = grouping_with_re(config, datadir)
            selected_groups = list(groups_dict.keys())
        else:
            selected_groups = selected_groups.strip().split(" ")
            print("You selected the group(s)", " ".join(selected_groups))
        for group in selected_groups:
            selected_group_fnames = groups_dict[int(group)]
            if len(selected_group_fnames['OBJECT']) > 0:
                feed_to_txt_file(selected_group_fnames, config, datadir, group)


def manual_inspect_obj(config):
    """
    Task 2.
    Visually inspect the object files.
    """
    opdir, all_datadirs = get_directories(config)
    print("Running Task 2")
    for datadir in all_datadirs:
        print("Working on ", datadir)
        manual_inspection_obj(config, datadir)
        if config['inputs']['SKY'] == 'Y':
            manual_inspection_flats(config, datadir,
                                    framecat="SKY")


def manual_inspect_cal(config):
    """
    Task 3
    Visually inspect the flat/cal files.
    """
    opdir, all_datadirs = get_directories(config)
    print("Running Task 3")
    for datadir in all_datadirs:
        print("Working on ", datadir)
        manual_inspection_flats(config, datadir)
        if config['inits']['TODO'] == 'S':
            manual_inspection_cals(config, datadir)


def combframe_flatcorr(config):
    """
    Task 4
    Flat correction and dither combination
    If specified in config file,
    sky subtrction will be done before
    flat correction.
    """
    opdir, all_datadirs = get_directories(config)
    print("Running Task 4")
    for datadir in all_datadirs:
        print("Working on ", datadir)
        frame_correction(config, datadir)


def frame_dithercombine(config):
    """
    Task 5
    Dither combination and nameing
    the output files
    """
    opdir, all_datadirs = get_directories(config)
    print("Running Task 5")
    for datadir in all_datadirs:
        if config['inits']['TODO'] == "S":
            subtract_dithers(config, datadir)
        elif config['inits']['TODO'] == "P":
            # print("Function to combine dither frames")
            combine_dithers(config, datadir)


def data_extraction(config):
    """
    Task 6
    Data extraction
    """
    opdir, all_datadirs = get_directories(config)
    print("Running Task 6")
    # data_dir = resources.files("toirex").joinpath("data")
    # print(data_dir / "TANSPEC")
    for datadir in all_datadirs:
        if config['inits']['TODO'] == "S":
            spectral_reduction(config, datadir)
        elif config['inits']['TODO'] == "P":
            photometry_extraction(config, datadir)


def main():
    parser = read_args()
    args = parser.parse_args()
    configfilename = args.config
    config = read_config(configfilename)
    config = add_dict_keywords(config)
    instrument = config['inits']['INSTRUMENT']

    logger = setup_logger_from_config(config)
    logger.info("Pipline started")

    print_banner()
    print("\n You are reducting data observed with {}".format(instrument))
    print("="*50)
    get_instrument_dir(instrument)
    print("\n *** Very Very Important: Backup your RAW data first.", end="")
    print("Don't proceed without backup *** \n")

    # Reading the name of output directory from config file.
    opdir, config_data = get_directories(config, required='string')
    create_dir(opdir)
    try:
        with open(Path(opdir) / "StepsFinished", 'r') as stepsover:
            StepsOver = stepsover.read()
    except IOError:
        StepsOver = "Nothing..."
    print("\nSteps you have already finished: " + StepsOver + "\n")
    # Reading the directory config file.
    # This config does not exist if the pipeline is running for the first time.
    # Then this config file will be created, and write the name of data
    # directory into it. So the user do not have to enter it multiple times.
    # Data directories are selecting in the following order.
    # 1. If the dirconfig exists, read from that, and enter it to the main
    # config.
    # 2. If not exist in dirconfig, read from main config.
    # If exists in dirconfig, overwrite the main config.
    # 3. If both does not exist, take manual entry.

    dirconfig = Path(opdir) / "dirconfig.config"
    if dirconfig.exists():
        dconfig = read_config(dirconfig)
        nights = dconfig['inits']['DATA']
        # Updating the main config dict with the entried from
        # data config
        config['inits']['DATA'] = nights
        print("Nights selected in previous step: {}".format(nights))
    else:
        # If the data config does not exist, taking the entry from
        # main config.
        nights = config_data

    nights = config['inits']['DATA']
    if len(nights) == 0:
        # Since no entries in main config and data config,
        # taking manual entry.

        dirs_avl = read_dirs()  # Listing the available directories
        dirs_avl.remove(opdir)  # Removing the output directory.
        print("Available nights are:", ", ".join(dirs_avl))
        print("Use comma to separate multiple entries")
        ip_dirs = input(
            "Enter the names of data directories from the above list:")
        dirs = ip_dirs.strip().split(",")
        # Adding the input directories to config.
        dirs_str = ""
        for j in dirs:
            dirs_str += j+","
        # Enter the directories to main config file.
        # This entry will stay in the memory to use later.
        config['inits']['DATA'] = dirs_str[:-1]
    else:
        if isinstance(nights, list):
            dirs = nights
        else:
            dirs = nights.strip().split(",")
    print("The data are in:", ", ".join(dirs))

    # If the directory config file does not exist, it will be created here.
    # This will be read if the pipeline is running multiple times.
    # So the user do not have to enter the name of directory always.
    if not dirconfig.exists():
        dirs_str = ""
        dconfig = {}
        for j in dirs:
            dirs_str += j+","
        dconfig['inits'] = {'DATA': dirs_str[:-1]}
        config['inits']['DATA'] = dirs_str[:-1]
        create_config(dirconfig, dconfig)

    # Creating the directories to save the output.
    for subdir in dirs:
        create_dir(opdir+"/"+subdir)

    tasks_list = tasks_dict.keys()
    print('-'*50)
    print('\n The following are the tasks')

    for tasks in tasks_list:
        print('\nTask {}: {}'.format(tasks, tasks_dict[int(tasks)]['menu']))
    print('-'*50)
    # Calling tasks.
    print('\n')
    if config['inits']['MODE'] == 'MANUAL':
        print("Enter the serial numbers", end="")
        print("(Space separated if more than one task in succession).")
        task = input("Enter the tasks you want to run:")
        task_list = task.strip().split(' ')
    elif config['inits']['MODE'] == 'AUTO':
        print("The pipeline running in automatic mode")
        task_list = list(tasks_dict.keys())
    logger.info("Entered task(s):{}".format(" ".join(task_list)))
    for onetask in task_list:
        print('\n')
        tasks_dict[int(onetask)]['function'](config)
        print('\nTask {} over'.format(onetask))
        print('*'*50)
        with open(Path(opdir) / "StepsFinished", 'a') as stepsover:
            stepsover.write(str(onetask) + " ")
        logger.info(f"Finished task {onetask}")
    logger.info("Finished all given tasks")


tasks_dict = {
    0: {'function': create_fileslog,
        'menu': "Generate the catalog of fits files in each directory"
        },
    1: {'function': select_files,
        'menu': "Selection of object frames, flats, lamps etc to reduce"
        },
    2: {'function': manual_inspect_obj,
        'menu': "Visually inspect and/or reject object images one by one"
        },
    3: {'function': manual_inspect_cal,
        'menu': "Visually inspect and/or reject flat/cal images one by one"
        },
    4: {'function': combframe_flatcorr,
        'menu': "Apply Flat Correction and/or CR removal"
        },
    5: {'function': frame_dithercombine,
        'menu': "Combine/Subtracting dither frames"
        },
    6: {'function': data_extraction,
        'menu': "Data extraction"
        }
    }

if __name__ == "__main__":
    main()
