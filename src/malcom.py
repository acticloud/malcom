#!/usr/bin/python3
import sys
import json
import random
import logging
import argparse
import experiments
from utils    import Utils
from pprint   import pprint
from utils    import Prediction
from stats    import ColumnStats
from mal_dict import MalDictionary

def init_logger(log_level_str):
    if log_level_str   == 'INFO':
        log_level = logging.INFO
    elif log_level_str == 'DEBUG':
        log_level = logging.DEBUG
    elif log_level_str == 'WARN':
        log_level = logging.WARN
    elif log_level_str == 'ERROR':
        log_level = logging.ERROR

    logger = logging.getLogger('')
    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    f = '~%(levelname)s~ %(filename)s:%(funcName)s:%(lineno)s --> %(message)s'
    formatter = logging.Formatter(f)
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

def init_parser():
    parser = argparse.ArgumentParser(
        description    = 'Malcom: Predicting things',
        epilog         = '''Satisfied ?''',
        formatter_class=argparse.MetavarTypeHelpFormatter
    )
    parser.add_argument('--log_level', type=str, default='INFO', required=False)
    parser.add_argument('--db', type = str, help = 'db name', required=False)
    return parser

if __name__ == '__main__':
    parser = init_parser()
    args   = parser.parse_args()
    init_logger(args.log_level)
    # experiments.predict_max_mem_tpch10()
    # experiments.plot_mem_error_air('airtraffic',10)
    experiments.plot_select_error_air('airtraffic',9)
    # experiments.analyze_select_error_air('airtraffic',9)
