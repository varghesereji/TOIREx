#!/usr/bin/env python3

import argparse
import configparser
import pyfiglet
from colorama import init, Fore, Style

from pathlib import Path

'''
===== Initial setups =====
'''


def create_dir(dirname):
    '''
    This function will create the directory.
    Input
    -----
    dirname: Name of the directory that need to be
    created.
    If the directory already exists, it will be skipped.
    '''
    path = Path(dirname)
    if not path.exists():
        path.mkdir(parents=True)
    else:
        print("{} directory already exists".format(dirname))


def read_dirs(dirname='.'):
    dirs = [d.name for d in Path(dirname).iterdir() if d.is_dir()]
    return dirs


def read_config(configfile):
    '''
    Function to read the config file.

    Input
    -----
    configfile: Name of the config file.

    Output
    -----
    config: Read config file. dict.
    '''
    config = configparser.ConfigParser()
    config.read(configfile)
    return config


def create_config(configfilename, entries):
    '''
    This function is to write a new config file.
    Inputs
    ------
    configfilename: Name of the config file with path.
    entries: Keywords and values as a dictionary.
    '''
    config = configparser.ConfigParser()
    for section, options in entries.items():
        config[section] = options
    with open(configfilename, "w") as configs:
        config.write(configs)


def read_args():
    '''
    Read the argument while execution.
    This function will take the config file.
    '''
    parser = argparse.ArgumentParser(description="Run data extraction pipeline")
    parser.add_argument('config', type=str, help="Config file name")
    return parser


def print_banner():
    '''
    This function is to print the banner of the pipeline.
    '''
    init(autoreset=True)  # Automatically reset colors after print

    text = "TOIRex"

    ascii_art = pyfiglet.figlet_format(text, font='slant')

    print('====== Welcome to ==========')
    print(Fore.CYAN + ascii_art + Style.RESET_ALL)
    print('\t Data reduction pipeline')


def add_dict_keywords(config):
    instrument = config['inits']['INSTRUMENT']
    todo = config['inits']['TODO']
    if (instrument == 'TANSPEC') & (todo == 'S'):
        dictkw = 'SpecTANSPEC'
    elif instrument == 'TIRSPEC':
        dictkw = 'TIRSPEC'
    config['inits']['DICTKW'] = dictkw
    return config




# End
