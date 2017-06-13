#!/usr/bin/python3
import json
import sys
from statement import MalStatement
# import statement.Arg
from pprint import pprint


#TODO error checking
#readline until you reach '}' or EOF
def parse_single(f):
    lines = []
    rbrace = False
    while rbrace == False:
        line = f.readline()
        if line == '': #no more lines to read
            return None
        lines += line
        if line == '}\n':
            rbrace = True
    return ''.join(lines)





def get_top_N(mal_list, n):
    mal_list.sort(key = lambda k:  -k.time)
    return mal_list[0:n] #ignore the first 2(dataflow, user function)

def flatten(mal_dict):
    l = []
    for v in mal_dict.values():
        l.extend(v)
    return l

if __name__ == '__main__':
    inp = sys.argv[1]

    with open(inp) as f:
        refd = {} #TODO remove
        maldict = {}
        while 1:
            jsons = parse_single(f)
            if jsons is None:
                break
            jobj                = json.loads(jsons)
            # (comm,_)          = parse_stmt(jobj["short"]) #for debugging
            # refd[comm]    = refd.get(comm, 0) + 1
            new_mals                = MalStatement.fromJsonObj(jobj)
            if new_mals.stype != 'dataflow' and new_mals.stype != 'function user.main':
                maldict[new_mals.stype] = maldict.get(new_mals.stype,[]) + [new_mals]

    mlist  = flatten(maldict)
    smlist = get_top_N(mlist,15)
    print("-----------------------------TOP 15-------------------------------------------")
    for e in smlist:
        print("time:{} instr: {}".format(e.time,e.stype))
        print("nargs: {}".format(len(e.arg_list)))
# if len(e.arg_list) > 0:
    # pprint(e.arg_list[0])
# print("---------------------------------------------------------------------x")
# for k in refd.keys():
#     print("{}: {}".format(k,refd[k]))
