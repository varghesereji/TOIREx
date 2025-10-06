#!/usr/bin/env python3

import argparse
import configparser
import pyfiglet
from colorama import init, Fore, Style
import logging
from pathlib import Path


'''
===== Logging =====
'''
_logger = None  # module-level variable to store the configured logger


def setup_logger_from_config(config: dict,
                             log_file="TOIREx_logging.log") -> logging.Logger:
    """
    Configure logger using entries from the config.
    The logger name is taken from config['INSTRUMENT'].
    """
    global _logger
    if _logger is not None:
        return _logger  # already configured

    logger_name = config['inits']["INSTRUMENT"]
    level_str = config['logging']['LEVEL']
    level = getattr(logging, level_str, logging.INFO)
    log_dir = config["outputs"]["OP_DIR"]

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:  # avoid duplicate handlers
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # console handler
        # ch = logging.StreamHandler(sys.stdout)
        # ch.setFormatter(formatter)
        # logger.addHandler(ch)

        # file handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    _logger = logger
    return logger


def get_logger(name=None) -> logging.Logger:
    """
    Return the configured logger.
    If `name` is given, returns a child logger.
    """
    if _logger is None:
        raise RuntimeError(
            "Logger not set up. Call setup_logger_from_config first.")
    if name:
        return logging.getLogger(f"{_logger.name}.{name}")
    return _logger


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
    config.optionxform = str  # <-- preserve case
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
    config.optionxform = str  # preserve case
    # for section, options in entries.items():
    #     config[section] = options
    if "DEFAULT" in entries:
        for key, value in entries["DEFAULT"].items():
            config["DEFAULT"][key] = str(value)
    for section, options in entries.items():
        if section == 'DEFAULT':
            continue
        config.add_section(section)  # explicitly add section
        for key, value in options.items():
            config.set(section, key, str(value))  # explicitly set each key
    with open(configfilename, "w") as configs:
        config.write(configs)


def read_args():
    '''
    Read the argument while execution.
    This function will take the config file.
    '''
    parser = argparse.ArgumentParser(
        description="Run data extraction pipeline")
    parser.add_argument('config', type=str, help="Config file name")
    return parser


def print_banner():
    '''
    This function is to print the banner of the pipeline.
    '''
    init(autoreset=True)  # Automatically reset colors after print

    text = "TOIREx"

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
    if len(config['inputs']['VAREXT']) == 0:
        config['inputs']['VAREXT'] = 'None'
    return config


# End
