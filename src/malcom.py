#!/usr/bin/python3
import argparse
import logging
import random
import json
import sys
from pprint   import pprint
from mal_dict import MalDictionary
from utils    import Prediction
from utils    import Utils
from stats    import ColumnStats

def print_usage():
    print("Usage: ./parser.py <trainset> <testset>")

def test_sampling(train_set, test_set):
    print("ntrain: {} ntest: {}".format(len(train_set.getInsList()),len(test_set.getInsList())))

    for ins in test_set.getInsList():
        l = train_set.findInstr(ins,True)
        if len(l) >= 1 and ins.mem_fprint > 1000000:
            print("{:10d} {:10d} {:10.1f}".format(l[0].mem_fprint,int(ins.mem_fprint/10),l[0].mem_fprint/ins.mem_fprint))

    # llist = [ins for ins in test_set.getInsList() if ins.mem_fprint > 10000000]
    #
    # nhits = [ins for ins in llist if train_set.findInstr(ins,True)[0].mem_fprint*10-ins.mem_fprint < 100000]
    #
    # print("{:10d} {:10d}".format(len(nhits),len(llist)))

def hold_out(data_set):
    sel_i = data_set.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'])
    print(len(sel_i.getInsList()))

    (train_set,test_set)  = sel_i.splitRandom(0.9,0.1)
    print(train_set.query_tags)
    print(len(train_set.getInsList()),len(test_set.getInsList()))

    # print(train_set.avgAcc(test_set))
    # train_set.printPredictions(test_set)
    print(train_set.avgCountAcc(test_set,0.01))

def hold_out2(train_set, test_set):
    sel_train = train_set.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'])
    sel_test  = test_set.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'])

    for i in sel_test.getInsList():
        print(i.short)
    l = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for p in l:
        (train_p,_) = sel_train.splitRandom(p)
        # for ins in train_p.getInsList():
        #     print(ins.short)
        print("Train set perc: ", p," length: ",len(train_p.getInsList()))

        for ins in sel_test.getInsList():
            # print("testing ins : ", ins.short)
            print(ins.lo, ins.hi)
            knn = train_p.predictCount(ins)
            p2 = train_p.predictCount2(ins)
            # p3 = train_p.pred
            print(ins.col, knn.col)
            print("Test    ins: ", ins.short, " Count: ", ins.cnt)
            print("closest ins: ", knn.short, " Count: ", knn.cnt, "Extrapolate: ", knn.extrapolate(ins))
            print("ArgDiv  ins: ", p2.short, "Count: ", p2.cnt, "ExtraExtrapolate: ", p2.extrapolate(ins)*p2.argDiv(ins))
            # print("knn5: ",knn.extrapolate(ins), 100* abs(knn.extrapolate(ins)-ins.cnt) / ins.cnt)
            # print("knn3: ",p2, 100* abs(p2-ins.cnt) / ins.cnt)

def hold_out3(train_set, test_set):
    sel_train = train_set.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'])
    sel_test  = test_set.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'])

    l = [0.1, 0.2, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    e = []
    for p in [1.0]:
        error = 0
        for ins in sel_test.getInsList():
            for i in [1,2,3,4,5,6,7,8,9,10]:
                (train_p,_) = sel_train.splitRandom(p)
                error       += train_p.errorCount(ins)
            print(error/10)
        e.append(error/(10*len(sel_test.getInsList())) )
    Utils.plotBar(l,e,"results/q3.pdf","Error %","perc of training set")
                # print("Error: ", train_p.errorCount(i), "%")
    print(sel_train.avgCountAcc(sel_test,0.1))


def test_pickle():
    d = MalDictionary.loadFromFile("test.pickle")
    for i in d.getInsList():
        print(i.short)

def test_approx():
    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    stats = Utils.loadStatistics('tpch10_stats.txt')

    d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran1_200_sf10.json", blacklist, stats)
    d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/01.json", blacklist, stats)

    approx = d2.approxGraph(d1)

def test_tpch6():
    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    stats = Utils.loadStatistics('tpch10_stats.txt')

    d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran6_200_disc_quan_sf10.json", blacklist, stats)
    d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/06.json", blacklist, stats)

    G = d2.approxGraph(d1)
    # print(G)
    train_tids = d1.filter(lambda ins: ins.fname in ['tid'])
    sel_train  = d1.filter(lambda ins: ins.fname in ['select', 'thetaselect'])# and ins.ctype not in ['bat[:bit]','bat[:hge]'])
    sel_test   = d2.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'])

    y = []
    nins = [3,15,30,60,90,120,150,180,210,240,270,300,330,360,390,450,480,540,600]
    for sel in nins:
        print("First {} instructions".format(int(sel/3)))
        sub_sel_train = sel_train.getFirst('clk',sel)
        sub_graph = d2.approxGraph(sub_sel_train.union(train_tids))
        # print(sub_graph)
        error = 0
        for ins in sel_test.getInsList():
            p = sub_sel_train.predictCountG(ins,sub_graph,True)
            print("Test    ins: ", ins.short, " Count: ", ins.cnt)
            print("closest ins: ", p.ins.short, " Count: ", p.ins.cnt, "Extrapolate: ", p.cnt)
            print("Error", 100* abs(p.cnt - ins.cnt) / ins.cnt)
            print("Knn5 Avg Error", 100* abs(p.avg - ins.cnt) / ins.cnt)
            error += 100* abs(p.avg - ins.cnt) / ins.cnt
            # if ins.col == 'l_quantity':
        y.append(error / len(sel_test.getInsList()))
            # print(i.short)
            # print(p.ins.short)
            # print(p.ins.cnt)
            # print(i.cnt, p.cnt)

    print(y)
    ind = [int(i/3) for i in nins]
    Utils.plotBar(ind, y, 'q6_disc_quan.pdf', 'Error perc', 'Number of instructions')
    # for i in sub_sel_train.getInsList():
        # print(i.short)

def test_tpch6_discount():
    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    stats = Utils.loadStatistics('tpch10_stats.txt')

    d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran6_200_discount_sf10.json", blacklist, stats)
    d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/06.json", blacklist, stats)

    G = d2.approxGraph(d1)
    # print(G)
    train_tids = d1.filter(lambda ins: ins.fname in ['tid'])
    sel_train  = d1.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.col == 'l_discount')# and ins.ctype not in ['bat[:bit]','bat[:hge]'])
    sel_test   = d2.filter(lambda ins: ins.fname in ['select', 'thetaselect'] and ins.ctype not in ['bat[:bit]','bat[:hge]'] and ins.col == 'l_discount')

    y = []
    nins = [3,15,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,600]
    nins2 = range(1,20)
    for sel in nins2:
        print("First {} instructions".format(int(sel)))
        sub_sel_train = sel_train.getFirst('clk',int(sel))
        # sub_graph = d2.approxGraph(sub_sel_train.union(train_tids))
        # print(sub_graph)
        for ins in sel_test.getInsList():
            print(ins.short)
            p = sub_sel_train.predictCountG(ins,{},False)
            print("Test    ins: ", ins.short, " Count: ", ins.cnt)
            print("closest ins: ", p.ins.short, " Count: ", p.ins.cnt, "Extrapolate: ", p.cnt)
            print("NN1 Error", 100*abs(p.cnt - ins.cnt) / ins.cnt)
            print("Knn5 Avg Error", 100*abs(p.avg - ins.cnt) / ins.cnt)
            if ins.col == 'l_discount':
                y.append(100*abs(p.avg - ins.cnt) / ins.cnt)
            # print(i.short)
            # print(p.ins.short)
            # print(p.ins.cnt)
            # print(i.cnt, p.cnt)

    print(y)
    ind = [int(i) for i in nins2]
    Utils.plotBar(ind, y, 'q6_discount2.pdf', 'Error perc', 'Number of instructions')

def test_test():
    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    stats = Utils.loadStatistics('tpch10_stats.txt')
    logging.info("loading training set...")
    d1 = MalDictionary.fromJsonFile("traces/tpch-sf10/20.json", blacklist, stats)
    logging.info("loading test set...")
    d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/20.json", blacklist, stats)

    (G,pG) = d2.buildApproxGraph(d1)

def test_qmem():
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStats.fromFile('config/tpch_sf10_stats.txt')

    for i in range(1,2):
        logging.info("Testing query {}".format(i))
        q = "{}".format(i)
        if i<10:
            q = "0{}".format(q)

        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran{}_200_sf10.json".format(i), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)

        (G,pG) = d2.buildApproxGraph(d1)
        # d2.testMaxMem(d1, G, pG)
        # for ins in d2.getInsList():
        #     p = ins.predictCount(d1,G)
        #     if ins.free_size != ins.approxFreeSize(pG):
        #         print("{:10} {:20} {:10.0f} {:10.0f} t:{:10.0f} p:{:10.0f}".format(ins.ret_vars[0],ins.fname, ins.ret_size , ins.approxMemSize(pG), ins.free_size , ins.approxFreeSize(pG)))
        print(len(d1.query_tags))
        print(d2.predictMaxMem(pG) / 1000000000, d2.getMaxMem() / 1000000000)

def sanity_test():
    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    # stats = Utils.loadStatistics('tpch10_stats.txt')
    col_stats = ColumnStats.fromFile('config/tpch_sf10_stats.txt')
    i=1
    logging.info("Testing query {}".format(i))
    q = "{}".format(i)
    if i<10:
        q = "0{}".format(q)

    logging.info("loading training set...")
    d1 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)
    logging.info("loading test set...")
    d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)

    (G,pG) = d2.buildApproxGraph(d1)
    for ins in d2.getInsList():
        if ins.fname != 'append':
            p = ins.predictCount(d1,G)
            # try:
                # print("{:10} {:20} {:10.0f} {:10.0f} ".format(ins.ret_vars[0],ins.fname, ins.ret_size / ins.approxMemSize(d1,G), ins.cnt / ins.predictCount(d1,G)[0].avg))
            if ins.free_size != ins.approxFreeSize(pG):
                print("{:10} {:20} {:10.0f} {:10.0f} t:{:10.0f} p:{:10.0f}".format(ins.ret_vars[0],ins.fname, ins.ret_size , ins.approxMemSize(pG), ins.free_size , ins.approxFreeSize(pG)))
            # except Exception:
                # pass
    # print(d2.predictMaxMem(d1, G) / 1000000000, d2.getMaxMem() / 1000000000)


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
    parser.add_argument('--log_level', type = str, help = '{INFO,DEBUG,WARN,ERROR}', default='INFO', required=False)
    # parser.add_argument('--runs', type = int, help = 'Number of variants', default= 3)


    return parser

if __name__ == '__main__':
    parser = init_parser()
    args   = parser.parse_args()
    init_logger(args.log_level)
    # print(args.log_level)
    # test_test()
    # sanity_test()
    test_qmem()
