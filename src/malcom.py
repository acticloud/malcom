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


if __name__ == '__main__':
    trainset = sys.argv[1]
    testset  = sys.argv[2]

    print("Using dataset {} as train set".format(trainset))
    print("Using dataset {} as test set".format(testset))

    blacklist = Utils.init_blacklist("mal_blacklist.txt")

    train_class = MalDictionary.fromJsonFile(trainset,blacklist)
    test_class  = MalDictionary.fromJsonFile(testset,blacklist)


    qtags = train_class.query_tags
    qtags.sort()
    query_map = dict([(i+1,tag) for (i,tag) in enumerate(qtags)])

    print("queries: {}".format(query_map))
    split_i = int(len(qtags)/8)

    test_tags = qtags[0:split_i]
    train_tags  = qtags[split_i+1:]
    (split1,split2) = train_class.splitTag(train_tags,test_tags)
    # (split1,split2) = train_class.splitRandom(0.9,0.1)

    print("ntrain: {} ntest: {}".format(len(split1.getInsList()),len(split2.getInsList())))
    print("train_tags {}".format(train_tags))
    print("test_tags {}".format(test_tags))

    # split1.printPredictions(split2)
    print("deviance: {:5.2f}%".format(split1.avgDeviance(split2)))
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
