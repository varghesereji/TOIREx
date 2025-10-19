#!/usr/bin/env python3

from pathlib import Path

from .utils import read_txt_file


def photometry_extraction(config, dirname):
    dictkw = config['inits']['DICTKW']
    opdir = Path(config['outputs']['OP_DIR']) / dirname
    reduce_txtfname = "Readytoextract_group*.txt"
    txtfiles_groups = opdir.glob(reduce_txtfname)
    for groupfile in txtfiles_groups:
        txtfile_full = read_txt_file(groupfile)
        for txtline in txtfile_full:
            frametoextract = txtline[0]
            frametoextract = opdir / frametoextract
            print(frametoextract)
