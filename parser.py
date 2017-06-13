#!/usr/bin/python3
import json
import sys
from pprint import pprint

#@arg type: type of statement(assign, thetaselect etc)
#@arg time: how much time did the statement last
#@arg size: memory footprint
#@arg list: the arguments of the query (list for now TODO change)
class MalStatement:
    def __init__(self, short, stype, time, size, alist):
        self.stype    = stype
        self.time     = time
        self.size     = size
        self.arg_list = alist
        self.short    = short
    def distance(other_mstmt):
        return 0 #TODO fix this

    @staticmethod
    def fromJsonObj(jobj):
        time          = float(jobj["usec"])
        size          = int(jobj["size"])
        short         = jobj["short"]
        (stype,_)     = parse_stmt(jobj["short"])
        # try:
        if "arg" in jobj:
            alist = [Arg.fromJsonObj(e) for e in jobj["arg"]]
            # alist = parse_stmt_args(jobj["arg"])
        else:
            alist = []
        return MalStatement(short, stype, time, size, alist)

#@attr atype: String
#@attr aval : Object
class Arg:
    def __init__(self, name, atype, val):
        self.name  = name
        self.atype = atype
        self.aval  = val

    @staticmethod
    def fromJsonObj(jobj):
        # pprint(jobj)
        name  = jobj['name']
        atype = jobj['type']
        aval  = jobj.get('value',None)
        return Arg(name,atype,aval)

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

def parse_stmt(stmt):
    # print(stmt)
    args = []
    if ":=" in stmt:
        fname = stmt.split(':=')[1].split("(")[0]
        args  = stmt.split(':=')[1].split("(")[1].split(")")[0].strip().split(" ")
    elif "(" in stmt:
        fname = stmt.split("(")[0]
        args  = stmt.split("(")[1].split(")")[0].strip().split(" ")
    else:
        fname = stmt

    return (fname.strip(),args)



def get_top_N(mal_list, n):
    mal_list.sort(key = lambda k:  -k.time)
    return mal_list[0:n] #ignore the first 2(dataflow, user function)

def flatten(mal_dict):
    l = []
    for v in mal_dict.values():
        l.extend(v)
    return l

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


# for k in maldict.keys():
#     print("{}: {}".format(k,len(maldict[k])))
#
# for k in maldict.keys():
#     print("{}: {}".format(k,maldict[k][0].arg_list))

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
