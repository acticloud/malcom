import matplotlib.pyplot as plt
from functools import reduce
from pylab import savefig
from stats import Stats
import numpy
import json
import collections

Prediction = collections.namedtuple('Prediction', ['ins', 'cnt', 'avg','retv'])

class Utils:

    @staticmethod
    #readline until you reach '}' or EOF
    def read_json_object(f):
        lines = []
        rbrace = False
        while rbrace == False:
            line = f.readline()
            if line == '': #no more lines to read
                return None
            lines += line
            if line == '}\n':
                rbrace = True

        return json.loads(''.join(lines))

    @staticmethod
    def flatten(mdict):
        l = []
        for v in mdict.values():
            l.extend(v)
        return l

    @staticmethod
    def list_diff(list1,list2):
        return list(set(list1)-set(list2))

    @staticmethod
    def cmp_arg_list(l1, l2):
        if(len(l1) != len(l2)):
            return False
        else:
            land = lambda x,y: x and y
            return reduce(land, [e1==e2 for (e1,e2) in zip(l1,l2)], True)

    @staticmethod
    def is_blacklisted(blacklist,instr):
        for mali in blacklist:
            if mali in instr:
                return True
        # print(line.strip())
        return False

    @staticmethod
    def init_blacklist(blfile):
        blacklist = []
        for line in open(blfile).readlines():
            blacklist.append(line.strip())

        return blacklist

    @staticmethod
    def loadStatistics(sfile):
        d = []
        for line in open(sfile).readlines():
            d.append(Stats.fromStr(line))
        return dict(d)

    @staticmethod
    def sumJsonList(jlist, sfield):
        sumf = lambda x,y: x+y
        return reduce(sumf, map(lambda var: var.get(sfield,0),jlist), 0)

    """ @arg stmt: str // short mal statement"""
    @staticmethod
    def extract_fname(stmt): #TODO change name
        args = []
        ret  = None
        if ":=" in stmt:
            fname = stmt.split(':=')[1].split("(")[0]
            args  = stmt.split(':=')[1].split("(")[1].split(")")[0].strip().split(",") #TODO remove
            ret   = stmt.split(':=')[0].split("[")[0].replace("(","")
        elif "(" in stmt:
            fname = stmt.split("(")[0]
            args  = stmt.split("(")[1].split(")")[0].strip().split(" ")
        else:
            fname = stmt

        return (fname.strip(),args,ret)

    @staticmethod
    def extract_operator(method, jobj):
        args  = jobj["arg"]
        nargs = len(args)
        if method == "thetaselect":
            if nargs == 3:
                return args[2]["value"].strip('\"')
            elif nargs == 4:
                return args[3]["value"].strip('\"')
            else:
                print(jobj["short"])
                raise ValueError("da??")
        elif method == "select":
            return "between"
        elif method == 'likeselect':
            return 'like'
        # elif method in ["+","-","*"]:
        #     return "batcalc"
        else:
            raise ValueError("e??")

    @staticmethod
    def hi_lo(method, op, jobj, stats):
        args = jobj["arg"]
        if method == "select":
            if len(args) == 7:
                return (args[2]["value"].strip('\"'),args[3]["value"].strip('\"'))
            elif len(args) == 6:
                if op == "between":
                    val1 = args[1]["value"].strip('\"')
                    val2 = args[2]["value"].strip('\"')
                    return (val1,val2)
                else:
                    assert op == "=="
                    # print("weird stuff")
                    val = args[1]["value"]
                    return (val,val)
            else:
                print(jobj["short"])
                print(jobj["arg"][0])
                raise ValueError("da2??")
        elif method == "thetaselect":
            val = args[-2]["value"]
            # print("VAL,OP: ", val, op)
            if op == "<=" or op == "<":
                return (stats.min,val)
            elif op == ">" or op == ">=":
                return (val,stats.max)
            elif op == "==" or op == "!=":
                return (val,val)
            else:
                print(op)
                raise ValueError("op??")
        elif method in ["+","-","*"]:
            return (stats.min,stats.max)
        elif method == 'likeselect':
            v = args[2]["value"].strip('\"')
            return (v,v)
        else:
            print(jobj["short"])
            raise ValueError("ttt")

    @staticmethod
    def extract_bounds(method, jobj):
        args  = jobj["args"]
        nargs = len(args)
        if method == "select":
            if args == 3:
                return args[2]["value"]
            elif args == 4:
                return args[3]["value"]
            else:
                raise ValueError("da??")
        else:
            raise ValueError("e??")

    @staticmethod
    def plotBar(x,y,output,ylabel,xlabel, lscale=False):
        fig, ax = plt.subplots()
        width = 0.5
        bar_width = 0.5
        # rects1 = ax.bar(ind-width, sp, width, color='b')
        ind = numpy.arange(len(x))+1
        rects1 = ax.bar(ind, y, width, color='b',log=lscale)
        ax.set_ylabel(ylabel)
        ax.set_title(xlabel)
        ax.set_xticks(ind + width)
        ax.set_xticklabels(x)

        savefig(output)#.format(sys.argv[1].split('.')[0]))

    @staticmethod
    def plotLine(x,y,output,ylabel,xlabel, lscale=False):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks(x, ind)

        plot1 = plt.plot(x, y, marker='o', color='b')

        savefig(output)#.format(sys.argv[1].split('.')[0]))
        plt.clf()
    @staticmethod
    def sizeof(type_str):
        if type_str == "int":
            return 4
        elif type_str == "bat[:oid]":
            return 8
        elif type_str == "float":
            return 8
        elif type_str == "boolean":
            return 1
        elif type_str == "date":
            return 8
        else:
            raise TypeError("Unsupported type")

    @staticmethod
    def isVar(name):
        return name.startswith("X_") or name.startswith("C_")
