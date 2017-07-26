#!/usr/bin/python3
import json
import sys
from mal_instr import MalInstruction
from pprint import pprint
from utils import Utils
from mal_dict import MalDictionary
import random

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


if __name__ == '__main__':
    trainset = sys.argv[1]
    testset  = sys.argv[2]

    print("Using dataset {} as train set".format(trainset))
    # print("Using dataset {} as test set".format(testset))

    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    stats = Utils.loadStatistics('tpch10_stats.txt')

    train_class = MalDictionary.fromJsonFile(trainset,blacklist, stats)

    # train_class.beta_dict.printStdout()
    # dict2 = train_class.beta_dict

    # test_filter = dict2.filter(lambda ins: ins.ctype != 'bat[:bit]')

    # test_filter.printStdout()
    # (test_filter1,test_filter2) = test_filter.randomSplit(0.1)
    # # test_filter1.printStdout()
    # # print("filter2")
    # # test_filter2.printStdout()
    #
    # for ti in test_filter2.getInsList():
    #     pr = test_filter1.predict(ti)
    #     print(ti.short)
    #     print(ti.lo,ti.hi,ti.cnt)
    #     print(pr.lo,pr.hi,pr.cnt)
    #     print("next")

    il = train_class.getInsList()
    il.sort(key = lambda i: -i.mem_fprint)

    for i in il:
        print(i.mem_fprint,i.short)

    print(len(il))
    #
    # for ti in dict2.getInsList():
    #     if ti.col == 'TMP':
    #         print(ti.short)
        # pr = test_filter1.predict(ti)
        # print(ti.lo,ti.hi,ti.cnt)
        # print(pr.lo,pr.hi,pr.cnt)
        # print("next")

    # for k in stats.keys():
        # print(k, stats[k])

    # train_class.printShort("select")
    # test_class  = MalDictionary.fromJsonFile(testset,blacklist)
    #
    # print("ntrain: {} ntest: {}".format(len(train_class.getInsList()),len(test_class.getInsList())))
    #
    # # test_sampling(train_class,test_class)
    # qtags = train_class.query_tags
    # print(train_class.getMaxMem()/1000000000,test_class.getMaxMem()/1000000000)
    # queries = list(range(1,23))
    # print(qtags)
    # qtags.sort()
    # tag2query = dict([(tag,i) for (i,tag) in enumerate(qtags)])
    # qtagst = test_class.query_tags
    # qtagst.sort()
    # tag2queryt = dict([(tag,i) for (i,tag) in enumerate(qtagst)])
    # print(test_class.query_tags)
    # for q in queries:
    #     # print("testint query ",q)
    #     test_q  = [q]
    #     train_q = Utils.list_diff(queries,test_q)
    #     # print(q)
    #     (split1_train,split2_train) = train_class.splitQuery(train_q,test_q,tag2query)
    #     (split1_test,split2_test) = test_class.splitQuery(train_q,test_q,tag2queryt)
    #     print("ntrain: {} ntest: {}".format(len(split2_train.getInsList()),len(split2_test.getInsList())))

        # test_sampling(split2_train, split2_test)
    #
    #     test_sampling(train_class, test_class)
    # train_class.printPredictions(test_class,ignoreScale=True)
    # for ins in test_class.getInsList():
    #     l = train_class.findInstr(ins,True)
    #     if len(l) >= 1 and ins.mem_fprint > 1000000:
    #         print("{:10d} {:10d} {:10.1f}".format(l[0].mem_fprint,int(ins.mem_fprint/10),l[0].mem_fprint/ins.mem_fprint))

    # llist = [ins for ins in test_class.getInsList() if ins.mem_fprint > 10000000]
    #
    # nhits = [ins for ins in llist if train_class.findInstr(ins,True)[0].mem_fprint*10-ins.mem_fprint < 1000000]
    #
    # print("{:10d} {:10d}".format(len(nhits),len(llist)))

        # else:
            # print("WTF {} {}".format(len(l),ins.mem_fprint))
    # exact = sum([len(train_class.findInstr(ins,True)) for ins in test_class.getInsList()])
    #
    # nexact = [ins.short for ins in test_class.getInsList() if len(train_class.findInstr(ins,True)) == 0]
    #
    # for i in nexact:
    #     print(i)
    #
    # print("Exact found: {}".format(exact))
    # qtags = train_class.query_tags
    # queries = list(range(1,45))
    # qtags.sort()
    # tag2query = dict([(tag,i) for (i,tag) in enumerate(qtags)])
    #
    # qtagst = test_class.query_tags
    # qtagst.sort()
    # tag2queryt = dict([(tag,i) for (i,tag) in enumerate(qtagst)])
    # # print(qtags)
    # # print(len(qtags))
    # # print(len(qtagst))
    # for q in queries:
    #     test_q  = [q]
    #     train_q = Utils.list_diff(queries,test_q)
    #     # print(q)
    #     (split1_train,split2_train) = train_class.splitQuery(train_q,test_q,tag2query)
    #     (split1_test,split2_test) = test_class.splitQuery(train_q,test_q,tag2queryt)
    #     try:
    #         print("{}".format(split2_train.getMaxMem() / split2_test.getMaxMem()))
    #     except:
    #         # print("{} {}".format(split2_train.getMaxMem(),split2_test.getMaxMem()))
    #         pass
    #
    # print("queries: {}".format(tag2query))
    # split_i = int(len(qtags)/8)
    #
    # test_q  = [17] #queries[0:1]
    #

    # print("train memf {}".format(train_class.getMaxMem()))
    # # (split1,split2) = train_class.splitRandom(0.9,0.1)
    #
    # print("train_q: {}".format(train_q))
    # print("test_q : {}".format(test_q))
    #
    # l = len([i.mem_fprint for i in split2.getInsList() if i.mem_fprint == 0])
    #
    # print("{}".format(l / len(split2.getInsList())))
    # # split1.printPredictions(split2)
    # # print("AvgError: {}".format(split1.avgError(split2)))
    #
    # pl = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    # for t in range(1,23):
    #     print("Testing Query {}".format(t))
    #     test_q = [t]
    #     train_q = Utils.list_diff(queries,test_q)
    #     (split1,split2) = train_class.splitQuery(train_q,test_q,tag2query)
    #     y = []
    #     for p in pl:
    #         bestp = split2.select(lambda x: -x.time,p)
    #         y.append(bestp.avgAccTimeExact2(split2))
    #         # print(len(train_class.getInsList()))
    #         # print(len(bestp.getInsList()))
    #         print("P: {:.1f} AvgAcc: {:.2f}".format(p,bestp.avgAccTimeExact2(split2)))
    #
    #     Utils.plotBar(pl,y,t,"time{}.pdf".format(t))
    # split1.printPredictionsVerbose(split1,tag2query)
    # split1.printPredictionsVerbose(split2,tag2query)
    # var2c = Utils.var2column(trainset)
    # print("Leave one out performance")
    # print("deviance: {:5.2f}%".format(split1.avgDeviance(split2)))
    # print("tags1: {}".format(split1.query_tags))
    # print("tags2: {}".format(split2.query_tags))

    # for ins in split2.getInsList():
    #     # print(ins.fname)
    #     try:
    #         mpred = split1.predictMem(ins)
    #         mem   = ins.mem_fprint
    #         if mem != 0:
    #             print("method: {:20} nargs: {:2d} actual: {:10d} pred: {:10d} perc: {:10.0f}".format(ins.fname,ins.nargs, mem,mpred,abs(100*mpred/mem)))
    #         else:
    #             print("method: {:20} nargs: {:2d} actual: {:10d} pred: {:10d}".format(ins.fname, ins.nargs, mem,mpred))
    #
    #     except Exception as err:
    #         # print("Exception: {}".format(err))
    #         print("method: {:20} nargs: {:2d}  NOT FOUND".format(ins.fname,ins.nargs))
    #         pass
    # theta3 = split2.findMethod("thetaselect")

    # for i in theta3:
    #     nn5 = split1.kNN(i, 5)
    #     ilist = [e.size for e in i.arg_list]# if e.size >0]

    #     print("s: {:<80} t: {:5d} arg_size: {:10d}, mem_fprint {:10d} args {}"
    #           .format(i.fname,i.tag,int(i.arg_size/1024), int(i.mem_fprint / 1024),ilist))
    #     print("------------------------kNN-----------------------------------=")
    #     for nn in nn5:
    #         slist = [e.size for e in nn.arg_list]# if e.size >0]
    #         print("t: {:5d} arg_size: {:10d}, mem_fprint {:10d}  argd{:10d} args {}"
    #               .format(nn.tag,int(nn.arg_size/1024), int(nn.mem_fprint / 1024), i.argDist(nn), slist))
    #     print("---------------------------------------------------------------")
    # train_class.printDiff(test_class)

    # smlist = train_class.getTopN(lambda k: -k.time,15)

    # print("-----------------------------TOP 15-------------------------------------------")
    # for e in smlist:
    #     print("time:{} instr: {}".format(e.time,e.fname))
