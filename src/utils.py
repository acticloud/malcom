import json
import numpy
import logging
import collections
from pylab import savefig
from functools import reduce
import matplotlib.pyplot as plt

"""
@attr retv: str            //name of the returned value
@attr ins : MalInstruction //the closest instruction
@attr cnt : int            //predicted number of elements
@attr avg : int            //average element count (of 5 in case of select)
@attr t   : str            //type??
@attr mem : int            //predicted memory size (non trivial in case of str)
"""
class Prediction():
    def __init__(self,retv,ins,cnt,avg,t,mem=None):
        self.retv = retv
        self.ins  = ins
        self.cnt  = cnt
        self.avg  = avg
        self.t    = t
        self.mem  = mem

    def getMem(self):
        if self.mem != None:
            return self.mem
        else:
            return self.cnt * Utils.sizeof(self.t)

# supported_mal = ['join','thetajoin','tid','bind','bind_idxbat','new','append',
# 'sort','select','thetaselect','likeselect','==','isnil','group','subgroup',
# 'subgroupdone','groupdone','ifthenelse','hge','!=','project','substring','avg',
# '>','like','difference','and','mergecand','single','dec_round','delta','year',
# 'subavg','subsum','subcount','submin','projection','projectionpath',
# 'projectdelta','subsum','subslice','+','-','*','/','or','dbl','intersect',
# '<','firstn','hash','bulk_rotate_xor_hash','identity','mirror','sum','max']

class Utils:
    #readline until you reach '}' or EOF
    @staticmethod
    def readJsonObject(f):
        lines  = []
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
    def approxSize(g,name, typ):
        if name in g:
            c = g.get(name)
            return c*Utils.sizeof(typ)
        else:
            # logging.error("{}".format(name))
            # logging.debug("000")
            return 0

    @staticmethod
    def isVar(name):
        return name.startswith("X_") or name.startswith("C_")

    @staticmethod
    def init_blacklist(blfile):
        blacklist = []
        for line in open(blfile).readlines():
            blacklist.append(line.strip())

        return set(blacklist)

    @staticmethod
    def loadStatistics(sfile):
        return ColumnStatsD.fromFile(sfile)


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
                return (stats.minv,val)
            elif op == ">" or op == ">=":
                return (val,stats.maxv)
            elif op == "==" or op == "!=":
                return (val,val)
            else:
                print(op)
                raise ValueError("op??")
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
        width = 0.1
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
        if type_str == "bat[:int]":
            return 4
        elif type_str in ["bat[:oid]",'bat[:lng]','bat[:dbl]']:
            return 8
        elif type_str in ["float","dbl"]:
            return 8
        elif type_str in ["bat[:hge]",'hge']:
            return 16
        elif type_str in ["bat[:bit]",'bat[:bte]']:
            return 1
        elif type_str == 'bat[:date]':
            return 4
        elif type_str == 'bat[:str]':
            return 8 #TODO fix this
        elif type_str == None:
            return 0
        else:
            logging.error("Wtf type?? {}".format(type_str))
            print(type_str)
            print(type_str == 'bat[:date]')
            raise TypeError("Unsupported type")

    @staticmethod
    def isVar(name):
        return name.startswith("X_") or name.startswith("C_")
