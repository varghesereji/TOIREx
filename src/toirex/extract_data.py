
from .setups import read_config
from .setups import read_args
from .setups import print_banner
from .setups import create_dir
from .setups import read_dirs


def main():
    parser = read_args()
    args = parser.parse_args()
    configfilename = args.config
    config = read_config(configfilename)
    instrument = config['inits']['INSTRUMENT']

    print_banner()
    print("\n You are reducting data observed with {}".format(instrument))
    print("="*50)
    opdir = config['inits']['OP_DIR']
    create_dir(opdir)
    nights = config['inits']['DATA']
    if len(nights) == 0:

        dirs_avl = read_dirs()  # Listing the available directories
        dirs_avl.remove(opdir)  # Removing the output directory.
        print("Available nights are:", ", ".join(dirs_avl))
        ip_dirs = input(
            "Enter the names of data directories from the above list:")
        dirs = ip_dirs.strip().split(",")
        # Adding the input directories to config.
        config['inits']['DATA'] = dirs
    else:
        dirs = nights.strip().split(",")
    print("The data are in:", ", ".join(dirs))

    # Creating the directories to save the output
    for subdir in dirs:
        create_dir(opdir+"/"+subdir)


if __name__ == "__main__":
    main()
