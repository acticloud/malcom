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

#TODO rewrite
def topMemInstructions(q):
    blacklist = Utils.init_blacklist("mal_blacklist.txt")
    # stats = Utils.loadStatistics('tpch10_stats.txt')

    d = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, None)

    df = d.filter(lambda ins: ins.fname not in ['bind','tid','bind_idxbat'])
    ins_list = df.getInsList()

    ins_list.sort(key=lambda ins: -ins.mem_fprint)

    N = len(ins_list)
    y = []
    total_mem = sum([ins.mem_fprint for ins in ins_list])
    # for i in ins_list:
        # print(i.mem_fprint, i.short)
    mem = 0
    with open("results/topN/{}.txt".format(q),'w') as f:
        for i in range(0,N):
            if mem / total_mem >= 0.9:
                break
            mem += ins_list[i].mem_fprint
            f.write("{:12d} {}\n".format(ins_list[i].mem_fprint,ins_list[i].short))
    # for i in range(1,N):
    #     topN = ins_list[0:i]
    #     mem  = sum([ins.mem_fprint for ins in topN])
    #     y.append(int(100*mem/total_mem))
    #     # print(mem)
    # Utils.plotLine(range(1,N),y,"graphs/{}_mem.pdf".format(q),'perc of total','topN instructions')



    # return ret

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
        formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument('--log_level', type = str, default='INFO', required=False)
    parser.add_argument('--db', type = str, help = 'db name', required=False)

    # parser.add_argument('--runs', type = int, help = 'Number of variants', default= 3)


    return parser

if __name__ == '__main__':
    parser = init_parser()
    args   = parser.parse_args()
    init_logger(args.log_level)
    # experiments.plot_select_error([3])
    # experiments.test_airtraffic()
    # experiments.plot_select_error_airtraffic()
    # experiments.plot_mem_error_airtraffic()
    experiments.predict_max_mem_tpch10()
    # experiments.analyze_max_mem_airtraffic()
    # experiments.analyze_max_mem()
    # print(args.log_level)
    # test_test()
    # sanity_test()
    # experiments.examine_select(6)
    # experiments.plot_max_mem_error([1,3,6])
