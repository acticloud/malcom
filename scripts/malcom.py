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

    train_class.printDiff(test_class)

    smlist = train_class.getTopN(lambda k: -k.time,15)

    print("-----------------------------TOP 15-------------------------------------------")
    for e in smlist:
        print("time:{} instr: {}".format(e.time,e.stype))
