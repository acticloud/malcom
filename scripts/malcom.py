#!/usr/bin/python3
import json
import sys
from mal_instr import MalInstruction
from pprint import pprint
from utils import Utils
from mal_dict import MalDictionary

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
    print("tags: {}".format(train_class.query_tags))
 
    split_i = int(len(qtags)/2)
   
    (split1,split2) = train_class.split(qtags[0:split_i],qtags[split_i+1:])

    print("tags1: {}".format(split1.query_tags))

    sel3 = train_class.findMethod("thetaselect",4)
    for i in sel3:
        print("s: {:<80} arg_size: {:10d}, ret_size {:10d}".format(i.short,int(i.arg_size/1024), int(i.tot_size / 1024)))

    # train_class.printDiff(test_class)

    # smlist = train_class.getTopN(lambda k: -k.time,15)

    # print("-----------------------------TOP 15-------------------------------------------")
    # for e in smlist:
    #     print("time:{} instr: {}".format(e.time,e.stype))
