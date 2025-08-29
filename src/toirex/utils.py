import argparse
import configparser
import pyfiglet
from colorama import init, Fore, Style


def read_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    return config


def read_args():
    parser = argparse.ArgumentParser(description="Run data extraction pipeline")
    parser.add_argument('config', type=str, help="Config file name")
    return parser


def print_banner():
    init(autoreset=True)  # Automatically reset colors after print

    text = "TOIRex"
   
    ascii_art = pyfiglet.figlet_format(text, font='slant')  # Try different fonts!
   
    print('====== Welcome to ==========')
    print(Fore.CYAN + ascii_art + Style.RESET_ALL)
    print('\t Data reduction pipeline')
