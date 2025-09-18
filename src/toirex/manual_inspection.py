import numpy as np
from pathlib import Path


from ariastro import combine_process

from .setups import get_logger
from .utils import extract_number_from_fname
from .plottings import imageplot



def manual_inspection_obj(config, dirname):
    logger = get_logger("manual_inspect")
    txtfile_re = "Objects_lamps_group*.txt"
    txt_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(txt_path.glob(txtfile_re))
    Obj2CombFile = "ObjectsToCombine_group{}.txt"
    for f in files_list:
        number = extract_number_from_fname(f.name)
        Obj2Comb_txt = open(txt_path / Obj2CombFile.format(number[0]), 'w')
        print("Group number running:", int(number[0]), "\n")
        logger.info("Calling file " + f.name)
        read_file = np.genfromtxt(f, dtype=str)
        targets_name = list(read_file[:, 0])
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


def manual_inspection_flats(config, dirname):
    logger = get_logger("manual_inspect")
    txtfile_re = "Objects_flats_group*.txt"
    op_path = Path(config['outputs']['OP_DIR']) / \
        dirname
    files_list = list(op_path.glob(txtfile_re))
    for f in files_list:
        number = extract_number_from_fname(f.name)

        print("Group number running:", int(number[0]), "\n")
        logger.info("Calling file" + f.name)
        read_file = np.genfromtxt(f, dtype=str)
        targets_name = list(read_file[0, 1:])
        acceptall = False
        for target in targets_name:
            if not acceptall and config['visual']['FLAT'] == 'Y':
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
        targets_path = [Path(dirname) / frame for frame in targets_name]
        comb_flatname = "Comb_flats_{}.fits".format(number[0])
        combine_fname = op_path / comb_flatname
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
        print("Saved the flat frame", combine_fname)


# End
