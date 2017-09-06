import logging
from utils    import Utils
from utils    import Prediction
from stats    import ColumnStats
from stats    import ColumnStatsD
from mal_dict import MalDictionary

def test_max_mem():
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    qno = 19
    for qno in range(10,23):
        logging.info("Examining Query: {}".format(qno))
        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran_q{}_n200_tpch10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(qno), blacklist, col_stats)

        pG  = d2.buildApproxGraph(d1)

        pmm = d2.predictMaxMem(pG) / 1000000000
        mm  = d2.getMaxMem()      / 1000000000

        err = 100* abs((pmm -mm) / mm)

        print("query: {}, pred mem: {}, actual mem: {}, error {}".format(qno,pmm,mm,err))

def analyze_max_mem():
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    qno = 19
    for qno in range(19,20):
        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran_q{}_n200_tpch10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(qno), blacklist, col_stats)

        pG  = d2.buildApproxGraph(d1)
        # sel2 = d2.filter(lambda ins: ins.fname in ['select','thetaselect'])

        testi = d2.getInsList()
        testi.sort(key = lambda ins: ins.clk)
        for ins in testi:
            pmm = ins.approxMemSize(pG)
            mm  = ins.ret_size

            if mm > 0 and mm > 10000:
                err = 100* abs((pmm -mm) / mm)
                print(ins.short)
                print("query: {}, pred mem: {}, actual mem: {}, error {}".format(qno,pmm,mm,err))
                print("cnt: {} pred cnt: {}".format(ins.cnt, ins.predictCount(d1, pG)[0].avg))
                print("")

def plot_max_mem_error(q):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    for qno in q:
        logging.info("Testing query {}".format(qno))
        q = "{}".format(qno)
        if qno<10:
            q = "0{}".format(q)

        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran{}_200_sf10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)
        train_tags = d1.query_tags
        train_tags.sort()
        e   = []
        ind = []
        for i in [1,5,10,15,20,25,30,40,50,75,100,125,150,175,200]:
            d12 = d1.filter( lambda ins: ins.tag in train_tags[0:i])
            print(len(d12.query_tags))
            pG  = d2.buildApproxGraph(d12)
            pmm = d2.predictMaxMem(pG) / 1000000000
            mm  = d2.getMaxMem() / 1000000000
            e.append( 100* abs((pmm -mm) / mm) )
            ind.append(i)
        print(e)
        Utils.plotBar(ind,e,"results/memf_error_q{}.pdf".format(qno),'nof training queries','error perc')

def plot_select_error(q):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    for qno in q:
        logging.info("Testing query {}".format(qno))
        q = "{}".format(qno)
        if qno<10:
            q = "0{}".format(q)

        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran{}_200_sf10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)
        sel2 = d2.filter(lambda ins: ins.fname in ['select','thetaselect'])
        seli = sel2.getInsList()
        train_tags = d1.query_tags
        train_tags.sort()
        e   = []
        ind = []
        for i in [1,5,10,15,20,25,30,40,50,75,100,125,150,175,200]:
            d12 = d1.filter( lambda ins: ins.tag in train_tags[0:i])
            print(len(d12.query_tags))
            pG = d2.buildApproxGraph(d12)
            error = 0
            for ins in seli:
                p   = ins.predictCount(d12, pG)[0]
                cnt = ins.cnt
                pc  = p.cnt
                error += 100* abs((pc -cnt) / cnt)
            e.append( error / len(seli) )
            ind.append(i)
        print(e)
        # Utils.plotBar(ind,e,"results/select_error_q{}.pdf".format(qno),'nof training queries','Select error perc')



def examine_select(qno):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStats.fromFile('config/tpch_sf10_stats.txt')

    logging.info("Testing select insruction for query {}".format(qno))
    q = "{}".format(qno)
    if qno<10:
        q = "0{}".format(q)

    train = "traces/random_tpch_sf10/ran{}_200_discount_sf10.json".format(qno)
    test  = "traces/tpch-sf10/{}.json".format(q)
    logging.info("loading training set... {}".format(train))
    d1 = MalDictionary.fromJsonFile(train, blacklist, col_stats)
    logging.info("loading test set... {}".format(test))
    d2 = MalDictionary.fromJsonFile(test, blacklist, col_stats)
    train_tags = d1.query_tags
    train_tags.sort()
    # for i in [1,5,10,15,20,25,30,40,50,75,100,125,150,175,200]:
    d12 = d1.filter( lambda ins: ins.tag in train_tags[0:1])
    (G,pG) = d2.buildApproxGraph(d12)
    sel2 = d2.filter(lambda ins: ins.fname in ['select','thetaselect'])
    for i in sel2.getInsList():
        print(i.short)
        p = i.predictCount(d12, G)[0]
        print(p.avg,i.cnt)
        # print("{10.0f} {10.0f}".format(p.avg,i.cnt))
        print("kNN ", p.ins.short)
        print(p.ins.cnt)
        print("-----------------------------------")
