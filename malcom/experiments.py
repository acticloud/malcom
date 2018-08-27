import datetime
import logging
import numpy
import os
import sys
import yaml

from malcom.utils import Utils
from malcom.stats import ColumnStatsD
from malcom.mal_dict import MalDictionary

class Definition:
    """Data loaded from an experiment definition file

    Provides methods to extract all sorts of paths
    """

    def __init__(self, filename):
        file = open(filename)
        self.conf = yaml.safe_load(file)
        self.filename = os.path.abspath(filename)

    def get(self, field):
        value = self.conf.get(field)
        if value == None:
            raise RuntimeError('no field {} in definition file {}', field, self.filename)
        return value

    def path(self, *path_components):
        base_dir = os.path.dirname(self.filename)
        if os.path.basename(base_dir) == 'config':
            base_dir = os.path.dirname(base_dir)  # ../
        return os.path.join(
            base_dir, 
            self.conf.get('root_path', '.'),
            *path_components)

    def out_path(self, *path_components):
        return self.path(self.get('out_path'), *path_components)

    def blacklist_path(self):
        return self.path(self.get('blacklist'))

    def stats_path(self):
        return self.path(self.get('stats'))

    def result_file(self):
        return self.out_path(self.get('result_file'))

    def data_file(self):
        return self.path(self.get('data_file'))

    def query_num(self):
        return self.get('query')

    def experiment(self):
        return self.get('experiment')

    def demo_get(self, field):
        demo = self.conf.get('demo', None)
        if not demo:
            raise RuntimeError('No demo section in definition file')
        value = self.conf.get(field)
        if value == None:
            raise RuntimeError('no field demo.{} in definition file {}', field, self.filename)
        return value

    def demo_training_set(self):
        return self.path(self.demo_get('training_set'))

    def demo_model_storage(self):
        return self.path(self.demo_get('model_storage'))

    def demo_plan_file(self):
        return self.path(self.demo_get('plan_file'))


def parse_experiment_definition(filename):
    return Definition(filename)


def leave_one_out(definition):
    initial_time = datetime.datetime.now()
    blacklist = Utils.init_blacklist(definition.blacklist_path())
    col_stats = ColumnStatsD.fromFile(definition.stats_path())
    query_num = definition.query_num()

    dataset_dict = None
    # Note: the commented code below does not work because
    # writeToFile, and loadFromFile are broken. When they are fixed
    # this should speed up the whole procedure a bit, because we will
    # not need to parse a big trace file.

    # if os.path.exists(definition['model_file']) and os.path.isfile(definition['model_file']):
    #     try:
    #         dataset_dict = MalDictionary.loadFromFile(definition['model_file'])
    #     except:
    #         logging.warning('Could not load model file: {}. Rebuilding.'.format(definition['model_file']))
    #         dataset_dict = None

    if dataset_dict is None:
        print('Loading traces for query: {:02}...'.format(query_num), end='')
        sys.stdout.flush()
        load_start = datetime.datetime.now()
        dataset_dict = MalDictionary.fromJsonFile(
            self.data_file(),
            blacklist,
            col_stats
        )
        load_end = datetime.datetime.now()
        print('Done: {}'.format(load_end - load_start))
        # dataset_dict.writeToFile(definition['model_file'])

    errors = list()
    pl = open(definition.result_file(), 'w')
    cnt = 0
    total = len(dataset_dict.query_tags)
    for leaveout_tag in dataset_dict.query_tags:
        iter_start = datetime.datetime.now()
        print("\b\b\b\b", end='')
        print('{:03}%'.format(int(100 * cnt / total)), end='')
        sys.stdout.flush()
        cnt += 1
        test_dict = dataset_dict.filter(lambda x: x.tag == leaveout_tag)
        train_dict = dataset_dict.filter(lambda x: x.tag != leaveout_tag)

        graph = test_dict.buildApproxGraph(train_dict)

        predict_start = datetime.datetime.now()
        predicted_mem = test_dict.predictMaxMem(graph)
        actual_mem = test_dict.getMaxMem()
        iter_end = datetime.datetime.now()

        errors.append(100 * (predicted_mem - actual_mem) / actual_mem)
        pl.write("{} {} {}\n".format(iter_end - iter_start,
                                     iter_end - predict_start,
                                     errors[cnt - 1]))

    print("")
    outfile = definition.out_path('Q{:02}_memerror.pdf'.format(query_num))
    print()
    pl.close()
    Utils.plotLine(numpy.arange(1, cnt), errors, outfile, 'Error percent', 'Leave out query')


def plot_actual_memory(definition):
    blacklist = Utils.init_blacklist(definition.blacklist_path())
    col_stats = ColumnStatsD.fromFile(definition.stats_path())

    print('Loading traces...', end='')
    sys.stdout.flush()
    data_file = definition.data_file()
    load_start = datetime.datetime.now()
    dataset_dict = MalDictionary.fromJsonFile(
        data_file,
        blacklist,
        col_stats
    )
    load_end = datetime.datetime.now()
    print('Done: {}'.format(load_end - load_start))

    outfile = definition.result_file()
    ofl = open(outfile, 'w')

    print('Computing footprint...     ', end='')
    sys.stdout.flush()
    result = dict()
    cnt = 0
    total = len(dataset_dict.query_tags)
    for t in dataset_dict.query_tags:
        print("\b\b\b\b", end='')
        print('{:03}%'.format(int(100 * cnt / total)), end='')
        sys.stdout.flush()
        cnt += 1
        # get the total memory for a specific query
        tq = dataset_dict.filter(lambda x: x.tag == t)
        total_mem = tq.getMaxMem()
        ofl.write("{},{}\n".format(t, total_mem))

    print("")
    ofl.close()


# The functions below might be useful, but are not currently used, and
# cannot be used unless the methods writeToFile and loadFromFile are
# fixed

def train_model(definition):
    blacklist = Utils.init_blacklist(definition.blacklist_path())
    col_stats = Utils.init_blacklist(definition.stats_path())
    print('Loading traces for demo... ', end='')
    sys.stdout.flush()
    training_set = definition.demo_training_set()
    dataset_mal = MalDictionary.fromJsonFile(
        training_set,
        blacklist,
        col_stats
    )
    print('Done')
    print('Writing model to disk: {}... '.format(definition.demo_model_storage()), end='')
    dataset_mal.writeToFile(definition.demo_model_storage())
    print('Done')

    return dataset_mal


def load_model(definition):
    try:
        model = MalDictionary.loadFromFile(definition.demo_model_storage())
    except:
        logging.warning('Model not found on disk. Training')
        model = train_model(definition)

    return model


def predict(definition):
    blacklist = Utils.init_blacklist(definition.blacklist_path())
    col_stats = Utils.init_blacklist(definition.stats_path())

    model = load_model(definition)
    plan = definition.demo_plan_file()
    plan_dict = MalDictionary.fromJsonFile(plan, blacklist, col_stats)

# EVERYTHING BELOW THIS LINE IS DEPRECATED


def plot_select_error_air(db, q, trainq=None, path="", ntrain=1000, step=25, output=None):
    assert db == 'tpch10' or db == 'airtraffic'
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    if trainq is None:
        trainq = q

    e = []
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n{n}_{db}.json".format(db=db, q=trainq, n=ntrain)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db, q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    # filter only select instructions
    seld = testd.filter(lambda ins: ins.fname in ['select', 'thetaselect'])
    seli = seld.getInsList()

    train_tags = traind.query_tags
    train_tags.sort()
    e = []
    ind = []
    # kutsurak: This loop increases the queries we use to train the
    # model.
    for i in range(1, ntrain + 2, step):
        d12 = traind.filter(lambda ins: ins.tag in train_tags[0:i])
        print(len(d12.query_tags))
        pG = testd.buildApproxGraph(d12)
        error = 0
        for ins in seli:
            p = ins.predict(d12, pG)[0]
            cnt = ins.ret_size
            pc = p.getMem()
            # we use abs so that the errors do not cancel out
            if cnt > 0:
                error += 100 * abs((pc - cnt) / cnt)
        e.append(error / len(seli))
        ind.append(i)

    print("error array:", e)
    outpdf = path+'{}_sel{}_error.pdf'.format(db, q) if output is None else output
    Utils.plotLine(ind, e, outpdf, 'Error perc', 'Nof training queries')


def plot_mem_error_air(db, q, trainq=None, path="", output=None, ntrain=1000, step=25):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    if trainq is None:
        trainq = q

    e = []
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n{n}_{db}.json".format(db=db, q=trainq, n=ntrain)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db, q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    train_tags = traind.query_tags
    train_tags.sort()
    e = []
    ind = []
    for i in range(1, ntrain + 2, step):
        d12 = traind.filter(lambda ins: ins.tag in train_tags[0:i])
        print(len(d12.query_tags))
        pG = testd.buildApproxGraph(d12)
        pmm = testd.predictMaxMem(pG)
        mm = testd.getMaxMem()
        # print(pmm / 1000000, mm / 1000000)
        e.append(100 * ((pmm - mm) / mm))
        ind.append(i)

    print(e)
    outf = path+'{}_q{}_memerror.pdf'.format(db, q) if output is None else output
    Utils.plotLine(ind, e, outf, 'Error perc', 'Nof training queries')


def analyze_mem_error_air(db, q, ntrain=1000, step=25):
    """
    Useful for analyse prediction results
    """
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    # e = []
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n{n}_{db}.json".format(db=db, q=q, n=ntrain)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db, q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    train_tags = traind.query_tags
    train_tags.sort()
    # e = []
    # ind = []
    for i in range(1, ntrain, step):
        d12 = traind.filter(lambda ins: ins.tag in train_tags[0:i])
        print("Number of train queries: ", len(d12.query_tags))
        pG = testd.buildApproxGraph(d12)
        insl = testd.getInsList()
        insl.sort(key=lambda inst: inst.clk)
        for ins in insl:
            p = ins.predict(d12, pG)[0]
            actual_size_mb = ins.ret_size / 1_000_000
            predic_size_mb = p.getMem() / 1_000_000
            print("{:120} actual: {:10.1f} pred: {:10.1f}\n".format(ins.short, actual_size_mb, predic_size_mb))


def analyze_select_error_air(db, q, ntrain=1000, step=25):
    assert db == 'tpch10' or db == 'airtraffic'
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/{}_stats.txt'.format(db))

    # e = []
    logging.info("Examining Query: {}".format(q))

    logging.info("loading training set...")
    trainf = "traces/random_{db}/ran_q{q}_n{n}_{db}.json".format(db=db, q=q, n=ntrain)
    traind = MalDictionary.fromJsonFile(trainf, blacklist, col_stats)

    logging.info("loading test set...")
    testf = "traces/{}/{}.json".format(db, q)
    testd = MalDictionary.fromJsonFile(testf, blacklist, col_stats)

    # filter only select instructions
    seld = testd.filter(lambda ins: ins.fname in ['select', 'thetaselect'])
    seli = [i for i in seld.getInsList() if i.ret_size > 0]

    train_tags = traind.query_tags
    train_tags.sort()
    # e = []
    # ind = []
    f = "{:120} realm: {:10.1f} predm: {:10.1f}, argc: {:10.0f} pr_argc {:10.0f}\n"

    for i in range(1, ntrain, step):
        d12 = traind.filter(lambda ins: ins.tag in train_tags[0:i + 1])
        print(len(d12.query_tags))
        pG = testd.buildApproxGraph(d12)
        error = 0
        for ins in seli:
            p = ins.predict(d12, pG)[0]
            rs = ins.ret_size
            pm = p.getMem()
            rs_mb = rs / 1_000_000
            pm_mb = p.getMem() / 1_000_000
            print(f.format(ins.short, rs_mb, pm_mb, ins.argCnt(), ins.approxArgCnt(pG)))
            print("NNi ", p.ins.short)
            error += 100 * abs((pm - rs) / rs)
            print("local error == ", 100 * abs((pm - rs) / rs))
        print("select error == ", error / len(seli))


def plot_allmem_tpch10(path=""):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    e = []
    for qno in range(1, 23):
        q = "0{}".format(qno) if qno < 10 else "{}".format(qno)
        logging.info("Examining Query: {}".format(q))
        d1 = MalDictionary.fromJsonFile("traces/random_tpch10/ran_q{}_n200_tpch10.json".format(q), blacklist, col_stats)
        d2 = MalDictionary.fromJsonFile("traces/tpch10/{}.json".format(q), blacklist, col_stats)

        pG = d2.buildApproxGraph(d1)

        pmm = d2.predictMaxMem(pG) / 1_000_000_000
        mm = d2.getMaxMem() / 1_000_000_000

        err = 100 * abs((pmm - mm) / mm)

        print("query: {}, pred mem: {}, actual mem: {}, error {}".format(qno, pmm, mm, err))
        e.append(err)
        print(err)
        # TODO: use os.path.join for the following
        outf = path+"mem_error_1-23.pdf"
    Utils.plotBar(range(1, 23), e, outf, 'error perc', 'query no')

####################################################################################
#                                DEPRECATED SCRIPTS....
####################################################################################


def analyze_max_mem():
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    qno = 19
    for qno in range(19, 20):
        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran_q{}_n200_tpch10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(qno), blacklist, col_stats)

        pG = d2.buildApproxGraph(d1)
        # sel2 = d2.filter(lambda ins: ins.fname in ['select','thetaselect'])

        testi = d2.getInsList()
        testi.sort(key=lambda ins: ins.clk)
        for ins in testi:
            pmm = ins.approxMemSize(pG)
            mm = ins.ret_size

            if mm > 0 and mm > 10000:
                err = 100 * abs((pmm - mm) / mm)
                print(ins.short)
                print("query: {}, pred mem: {}, actual mem: {}, error {}".format(qno, pmm, mm, err))
                print("cnt: {} pred cnt: {}".format(ins.cnt, ins.predict(d1, pG)[0].avg))
                print("")


def plot_max_mem_error(q):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    for qno in q:
        logging.info("Testing query {}".format(qno))
        q = "{}".format(qno)
        if qno < 10:
            q = "0{}".format(q)

        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran{}_200_sf10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)
        train_tags = d1.query_tags
        train_tags.sort()
        e = []
        ind = []
        for i in [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]:
            d12 = d1.filter(lambda ins: ins.tag in train_tags[0:i])
            print(len(d12.query_tags))
            pG = d2.buildApproxGraph(d12)
            pmm = d2.predictMaxMem(pG) / 1000000000
            mm = d2.getMaxMem() / 1000000000
            e.append(100 * abs((pmm - mm) / mm))
            ind.append(i)
        print(e)
        Utils.plotBar(ind, e, "results/memf_error_q{}.pdf".format(qno), 'nof training queries', 'error perc')


def plot_select_error(q):
    blacklist = Utils.init_blacklist("config/mal_blacklist.txt")

    col_stats = ColumnStatsD.fromFile('config/tpch_sf10_stats.txt')

    for qno in q:
        logging.info("Testing query {}".format(qno))
        q = "{}".format(qno)
        if qno < 10:
            q = "0{}".format(q)

        logging.info("loading training set...")
        d1 = MalDictionary.fromJsonFile("traces/random_tpch_sf10/ran{}_200_sf10.json".format(qno), blacklist, col_stats)
        logging.info("loading test set...")
        d2 = MalDictionary.fromJsonFile("traces/tpch-sf10/{}.json".format(q), blacklist, col_stats)
        sel2 = d2.filter(lambda ins: ins.fname in ['select', 'thetaselect'])
        seli = sel2.getInsList()
        train_tags = d1.query_tags
        train_tags.sort()
        e = []
        ind = []
        for i in [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200]:
            d12 = d1.filter(lambda ins: ins.tag in train_tags[0:i])
            print(len(d12.query_tags))
            pG = d2.buildApproxGraph(d12)
            error = 0
            for ins in seli:
                p = ins.predict(d12, pG)[0]
                cnt = ins.cnt
                pc = p.cnt
                error += 100 * abs((pc - cnt) / cnt)
            e.append(error / len(seli))
            ind.append(i)
        print(e)
        # Utils.plotBar(ind,e,"results/select_error_q{}.pdf".format(qno),'nof training queries','Select error perc')
