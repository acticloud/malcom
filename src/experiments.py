import logging
from utils    import Utils
from utils    import Prediction
from stats    import ColumnStats
from stats    import ColumnStatsD
from mal_dict import MalDictionary

def plot_select_error_air(db, qno):
    assert db=='tpch10' or db=='airtraffic'
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    e   = []
    q = "0{}".format(qno) if qno < 10 else "{}".format(qno)
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n1000_{db}.json".format(db=db,q=q)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db,q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    #filter only select instructions
    seld  = testd.filter(lambda ins: ins.fname in ['select','thetaselect'])
    seli  = seld.getInsList()

    train_tags = traind.query_tags
    train_tags.sort()
    e   = []
    ind = []
    for i in range(1,1000,25):
        d12 = traind.filter( lambda ins: ins.tag in train_tags[0:i])
        print(len(d12.query_tags))
        pG = testd.buildApproxGraph(d12)
        error = 0
        for ins in seli:
            p      = ins.predictCount(d12, pG)[0]
            cnt    = ins.ret_size
            pc     = p.getMem()
            error += 100*abs((pc -cnt) / cnt)
        e.append( error / len(seli) )
        ind.append(i)

    print("error array:",e)
    outpdf = '{}_sel{}_error.pdf'.format(db,q)
    Utils.plotLine(ind,e,outpdf,'Error perc','Nof training queries')

def plot_mem_error_air(db,qno):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    e   = []
    q = "0{}".format(qno) if qno < 10 else "{}".format(qno)
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n1000_{db}.json".format(db=db,q=q)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db,q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    train_tags = traind.query_tags
    train_tags.sort()
    e   = []
    ind = []
    for i in range(1,1000,25):
        d12 = traind.filter( lambda ins: ins.tag in train_tags[0:i])
        print(len(d12.query_tags))
        pG  = testd.buildApproxGraph(d12)
        pmm = testd.predictMaxMem(pG)
        mm  = testd.getMaxMem()
        # print(pmm / 1000000, mm / 1000000)
        e.append( 100 * abs((pmm -mm) / mm) )
        ind.append(i)

    print(e)
    outf = '{}_q{}_memerror.pdf'.format(db,q)
    Utils.plotLine(ind,e,outf,'Error perc','Nof training queries')


def analyze_mem_error_air(db,qno):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    e   = []
    q = "0{}".format(qno) if qno < 10 else "{}".format(qno)
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n1000_{db}.json".format(db=db,q=q)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db,q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    train_tags = traind.query_tags
    train_tags.sort()
    e   = []
    ind = []
    for i in range(1,1000,20):
        d12 = traind.filter( lambda ins: ins.tag in train_tags[0:i])
        print(len(d12.query_tags))
        pG  = testd.buildApproxGraph(d12)
        insl = testd.getInsList()
        insl.sort(key = lambda inst: inst.clk)
        for ins in insl:
            print(ins.short)
            p = ins.predictCount(d12, pG)[0]
            print(ins.ret_size / 1000000, p.getMem()/1000000)

def analyze_select_error_air(db, qno):
    assert db=='tpch10' or db=='airtraffic'
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    e   = []
    q = "0{}".format(qno) if qno < 10 else "{}".format(qno)
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n1000_{db}.json".format(db=db,q=q)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)
    traind.linkSelectInstructions()

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db,q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)
    testd.linkSelectInstructions()

    #filter only select instructions
    seld  = testd.filter(lambda ins: ins.fname in ['select','thetaselect'])
    seli  = seld.getInsList()

    train_tags = traind.query_tags
    train_tags.sort()
    e   = []
    ind = []
    for i in range(800,1001,200):
    # for i in [1,5,10,15,20,25,50,75,100,150,200,250,375,500,675,800,1000]:
        d12 = traind.filter( lambda ins: ins.tag in train_tags[0:i+1])
        print(len(d12.query_tags))
        pG = testd.buildApproxGraph(d12)
        error = 0
        for ins in seli:
            p      = ins.predictCount(d12, pG)[0]
            rs    = ins.ret_size
            pm     = p.getMem()
            print("TESTi: ",ins.short)
            print("NNi ",p.ins.short)
            print(rs/1000000 , pm/1000000)
            error += 100*abs((pm -rs)/rs)
        print("select error == ", error / len(seli) )
        # ind.append(i)

def predict_max_mem_tpch10():
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    e   = []
    for qno in range(1,23):
        q = "0{}".format(qno) if qno < 10 else "{}".format(qno)
        logging.info("Examining Query: {}".format(q))
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran_q{}_n200_tpch10.json".format(q), blacklist, col_stats)
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)

        pG  = d2.buildApproxGraph(d1)

        pmm = d2.predictMaxMem(pG)  / 1000000000
        mm  = d2.getMaxMem()       / 1000000000

        err = 100* abs((pmm -mm) / mm)

        print("query: {}, pred mem: {}, actual mem: {}, error {}".format(qno,pmm,mm,err))
        e.append(err)
        print(err)
    # Utils.plotBar(range(1,23), e, "mem_error_1-23.pdf",'error perc','query no')


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
