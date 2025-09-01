
from pathlib import Path

from .setups import read_config, create_config
from .setups import read_args
from .setups import print_banner
from .setups import create_dir
from .setups import read_dirs
from .setups import add_dict_keywords

from .obscatalog import create_catalog


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
    print("This function will be added soon")


def main():
    parser = read_args()
    args = parser.parse_args()
    configfilename = args.config
    config = read_config(configfilename)
    config = add_dict_keywords(config)
    instrument = config['inits']['INSTRUMENT']

    print_banner()
    print("\n You are reducting data observed with {}".format(instrument))
    print("="*50)

    # Reading the name of output directory from config file.
    opdir, config_data = get_directories(config, required='string')
    create_dir(opdir)

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
    print("Enter the serial numbers", end="")
    print("(Space separated if more than one task in succession).")
    task = input("Enter the tasks you want to run:")
    task_list = task.strip().split(' ')
    for onetask in task_list:
        print('\n')
        tasks_dict[int(onetask)]['function'](config)
        print('\nTask {} over'.format(onetask))
        print('*'*50)

tasks_dict = {
    0: {'function': create_fileslog,
        'menu': "Generate the catalog of fits files in each directory"
        },
    1: {'function': select_files,
        'menu': "Select files"
        }

    }

if __name__ == "__main__":
    main()
