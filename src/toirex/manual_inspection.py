import numpy as np
from pathlib import Path

from .utils import extract_number_from_fname
from .plottings import imageplot


def manual_inspection_obj(config, dirname):
    txtfile_re = "Objects_lamps_group*.txt"
    txt_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(txt_path.glob(txtfile_re))
    for f in files_list:
        number = extract_number_from_fname(f.name)
        print("Group number running:", int(number[0]), "\n")
        read_file = np.genfromtxt(f, dtype=str)
        targets_name = list(read_file[:, 0])
        acceptall = False
        if (config['inits']['TIMESERIES'] == 'Y'):
            acceptall = True
        elif config['inits']['MODE'] == 'AUTO':
            acceptall = True
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
            elif UserInput == 'acceptall':
                acceptall = True
                print("Accepting every single remaining images of this night")

        if config['inits']['TODO'] == 'P':
            print("Photometry will be added soon")
        elif config['inits']['TODO'] == 'S':
            print("Finding dithers")
# End
