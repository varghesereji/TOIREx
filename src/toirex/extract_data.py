from .utils import read_config
from .utils import read_args
from .utils import print_banner




def main():
    parser = read_args()
    args = parser.parse_args()
    configfilename = args.config
    configs = read_config(configfilename)
    instrument = configs['inits']['INSTRUMENT']

    # print(banner)
    print_banner()
    print("\n You are reducting data observed with {}".format(instrument))    


if __name__ == "__main":
    main()
