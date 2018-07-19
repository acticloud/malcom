import json
import numpy
import logging
import matplotlib.pyplot as plt



class Prediction():
    """
    @attr retv: str            //name of the returned value
    @attr ins : MalInstruction //the closest instruction
    @attr cnt : int            //predicted number of elements
    @attr avg : int            //average element count (of 5 in case of select)
    @attr t   : str            //type??
    @attr mem : int            //predicted memory size (non trivial in case of str)
    TODO: add min,max ....
    """
    def __init__(self, retv, ins, cnt, avg, t, mem=None):
        self.retv = retv
        self.ins = ins
        self.cnt = cnt
        self.avg = avg
        self.t = t
        self.mem = mem

    def getMem(self):
        if self.mem is not None:
            return self.mem
        else:
            return self.avg * Utils.sizeof(self.t)


class Utils:
    # This function helps us reading the MonetDB JSON traces
    # readline until you reach '}' or EOF
    @staticmethod
    def readJsonObject(f):
        lines  = []
        rbrace = False
        while not rbrace:
            line = f.readline()
            if line == '':  # no more lines to read
                return None
            lines += line
            if line == '}\n':
                rbrace = True

        return json.loads(''.join(lines))

    # flatten a dictionary into a list
    @staticmethod
    def flatten(mdict):
        lst = []
        for v in mdict.values():
            lst.extend(v)
        return lst

    @staticmethod
    def listDiff(list1, list2):
        return list(set(list1) - set(list2))

    # checks if the two given lists are equal (could also have been implemented
    # with a FOR loop :))
    @staticmethod
    def cmp_arg_list(l1, l2):
        if(len(l1) != len(l2)):
            return False
        else:
            land = lambda x, y: x and y
            return reduce(land, [e1 == e2 for (e1, e2) in zip(l1, l2)], True)

    @staticmethod
    def dict2list(d):
        ilist = []
        for l in d.values():
            ilist.extend(l)
        return ilist

    @staticmethod
    def is_blacklisted(blacklist, instr):
        for mali in blacklist:
            if mali in instr:
                return True
        return False

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
    # extract the MAL function name, args and its ret from a line of MAL
    #   statement. currently args and ret are not used
    @staticmethod
    def extract_fname(stmt):
        args = []
        ret  = None
        if ":=" in stmt:
            fname = stmt.split(':=')[1].split("(")[0]
            args  = stmt.split(':=')[1].split("(")[1].split(")")[0].strip().split(",")
            ret   = stmt.split(':=')[0].split("[")[0].replace("(", "")
        elif "(" in stmt:
            fname = stmt.split("(")[0]
            args  = stmt.split("(")[1].split(")")[0].strip().split(" ")
        else:
            fname = stmt

        return (fname.strip(), args, ret)

    # find the exact operator associated with certain MAL instructions, because
    # the position of the operator depends on the MAL instruction
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

    # Because we have e.g. SELECT operators with different number of
    # parameters, we neeed to find the hi/lo value depending on how many
    # parameters this operator has
    # WARNING: if the MAL signature changes, this may break
    @staticmethod
    def hi_lo(method, op, jobj, stats):
        args = jobj["arg"]
        if method == "select":
            if len(args) == 7:
                return (args[2]["value"].strip('\"'), args[3]["value"].strip('\"'))
            elif len(args) == 6:
                if op == "between":
                    val1 = args[1]["value"].strip('\"')
                    val2 = args[2]["value"].strip('\"')
                    return (val1, val2)
                else:
                    assert op == "=="
                    # print("weird stuff")
                    val = args[1]["value"]
                    return (val, val)
            else:
                print(jobj["short"])
                print(jobj["arg"][0])
                raise ValueError("da2??")
        elif method == "thetaselect":
            val = args[-2]["value"]
            # print("VAL,OP: ", val, op)
            if op == "<=" or op == "<":
                return (stats.minv, val)
            elif op == ">" or op == ">=":
                return (val, stats.maxv)
            elif op == "==" or op == "!=":
                return (val, val)
            else:
                print(op)
                raise ValueError("op??")
        elif method == 'likeselect':
            v = args[2]["value"].strip('\"')
            return (v, v)
        else:
            print(jobj["short"])
            raise ValueError("ttt")

    @staticmethod
    def extract_bounds(method, jobj):
        args  = jobj["args"]
        # nargs = len(args)
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
    def plotBar(x, y, output, ylabel, xlabel, lscale=False):
        fig, ax = plt.subplots()
        width = 0.25
        # bar_width = 0.5
        # rects1 = ax.bar(ind-width, sp, width, color='b')
        ind = numpy.arange(len(x)) + 1
        ax.bar(ind, y, width, color='b', log=lscale)
        ax.set_ylabel(ylabel)
        ax.set_title(xlabel)
        ax.set_xticks(ind + width)
        ax.set_xticklabels(x)

        fig.savefig(output)  # .format(sys.argv[1].split('.')[0]))
        fig.clf()

    @staticmethod
    def plotLine(x, y, output, ylabel, xlabel, lscale=False):
        fig, ax = plt.subplots()
        ax.xlabel(xlabel)
        ax.ylabel(ylabel)
        # plt.xticks(x, ind)

        ax.plot(x, y, marker='o', color='b')

        fig.savefig(output)  # .format(sys.argv[1].split('.')[0]))
        fig.clf()

    @staticmethod
    def sizeof(type_str):
        if type_str == "bat[:int]":
            return 4
        elif type_str in ["bat[:oid]", 'bat[:lng]', 'bat[:dbl]']:
            return 8
        elif type_str in ["float", "dbl"]:
            return 8
        elif type_str in ["bat[:hge]", 'hge']:
            return 16
        elif type_str in ["bat[:bit]", 'bat[:bte]']:
            return 1
        elif type_str == 'bat[:date]':
            return 4
        elif type_str == 'bat[:str]':
            return 8  # TODO fix this
        elif type_str is None:
            return 0
        else:
            logging.error("Wtf type?? {}".format(type_str))
            print(type_str)
            print(type_str == 'bat[:date]')
            raise TypeError("Unsupported type")
