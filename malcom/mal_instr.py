"""
interface MalInstruction {
    def argCnt()               : List<int>

    def approxArgCnt(traind: MalDictionary, G: dict<str, Prediction>): List<int>

    def predict(traind: MalDictionary, G: dict<str,Prediction>): List<Prediction>

    def kNN(traind: MalDictionary, k: int, G) -> List<MalInstruction>
}
"""
import re
import sys
import logging
import Levenshtein
from mal_arg  import Arg
from utils    import Utils
from functools import reduce
from datetime import datetime
from utils    import Prediction
from stats    import ColumnStats


class MalInstruction:
    """ MalInstruction Class:
    stores all information of a MAL instruction
    @attr pc        : int        //mal programm counter
    @attr fname     : str        //function name
    @attr time      : int        //instruction duration(microseconds)
    @attr ret_size  : int        //total return values size (bytes)
    @attr arg_size  : int        //total argument values size (bytes)
    @attr mem_fprint: int        //instruction memory footprint
    @attr size      : int        // size field (maybe temporary vars size ?)
    @attr arg_list  : list<Arg>  // instruction arguments (list for now TODO change)
    @attr short     : str        // the short mal statement, str representation
    @attr tag       : int        // the query identifier
    @attr nargs     : int        //number of arguments
    @attr free_size : int        //amount of memory freed(args for which eol==1)
    @attr arg_vars  : list<str>  //names of the arguments that are vars
    @attr ret_vars  : list<str>  //names of the return output that are vars
    @attr cnt       : int        //the number of elements of the return var
    """
    def __init__(self, pc, clk, short, fname, size, ret_size, tag, arg_size,
                 alist, ret_args, free_size, arg_vars, ret_vars, cnt):
        self.pc = pc
        self.clk = clk
        self.fname = fname
        self.time = 0
        self.size = size
        self.ret_size = ret_size
        self.arg_list = alist
        self.ret_args = ret_args
        self.short = short
        self.tag = tag
        self.mem_fprint = self.size + self.ret_size
        self.arg_size = arg_size
        self.nargs = len(alist)
        self.free_size = free_size
        self.arg_vars = arg_vars
        self.ret_vars = ret_vars
        self.cnt = cnt

    # jobj was built in mal_dict.py, in fromJsonFile
    @staticmethod
    def fromJsonObj(jobj, stats):
        size = int(jobj["size"])
        pc = int(jobj["pc"])
        clk = int(jobj["clk"])
        short = jobj["short"]
        fname, _, _ = Utils.extract_fname(jobj["short"])
        tag = int(jobj["tag"])
        # rv = [rv.get("size", 0) for rv in jobj["ret"]]
        ro = jobj.get("ret", [])  # return object
        ret_size = sum([o.get("size", 0) for o in ro if int(o["eol"]) == 0])
        arg_size = sum([o.get("size", 0) for o in jobj.get("arg", [])])
        arg_list = [Arg.fromJsonObj(e) for e in jobj.get("arg", [])]
        ret_args = [Arg.fromJsonObj(e) for e in ro]  # if e["eol"]==0]
        # print(len(alive_ret))
        free_size = sum([arg.size for arg in arg_list if arg.eol == 1])
        arg_vars = [arg.name for arg in arg_list if arg.isVar()]
        ret_vars = [ret['name'] for ret in ro if Utils.isVar(ret['name'])]
        count = int(jobj["ret"][0].get("count", 0))

        con_args = [pc, clk, short, fname, size, ret_size, tag, arg_size,
                    arg_list, ret_args, free_size, arg_vars, ret_vars, count]

        # Select Instructions
        if fname in ['select', 'thetaselect', 'likeselect']:
            return SelectInstruction(*con_args, jobj=jobj, stats=stats)  # TODO replace jobj
        # Projections
        elif fname in ['projectionpath']:
            return DirectIntruction(*con_args, base_arg_i=0)
        elif fname in ['projection', 'projectdelta']:
            return ProjectInstruction(*con_args)
        # Joins
        elif fname in ['join', 'thetajoin', 'crossproduct']:
            return JoinInstruction(*con_args)
        # Group Instructions
        elif fname in ['group', 'subgroup', 'subgroupdone', 'groupdone']:
            a0 = arg_list[0].col.split('.')[::-1][0]
            return GroupInstruction(*con_args, base_arg_i=0, base_col=a0, col_stats=stats.get(a0, None))
        # Set Instructions: the last parameter determines how to compute the
        #     prediction for this MAL instruction
        elif fname in ['intersect']:
            return SetInstruction(*con_args, i1=0, i2=1, fun=lambda a, b: min(a, b))
        elif fname in ['mergecand']:
            return SetInstruction(*con_args, i1=0, i2=1, fun=lambda a, b: a + b)
        elif fname in ['difference']:
            return SetInstruction(*con_args, i1=0, i2=1, fun=lambda a, b: a)
        elif fname in['<', '>', '>=', '<=']:
            if arg_list[1].isVar():
                return SetInstruction(*con_args, i1=0, i2=1, fun=lambda a, b: min(a, b))
            else:
                return DirectIntruction(*con_args, base_arg_i=0)
        # Direct Intructions
        elif fname in ['+', '-', '*', '/', 'or', 'dbl', 'and', 'lng', '%']:
            if arg_list[0].isVar():
                return DirectIntruction(*con_args, base_arg_i=0)
            elif arg_list[1].isVar():
                return DirectIntruction(*con_args, base_arg_i=1)
            else:
                return ReduceInstruction(*con_args)
        elif fname in ['==', 'isnil', '!=', 'like']:
            return DirectIntruction(*con_args, base_arg_i=0)
        elif fname in ['sort']:
            return DirectIntruction(*con_args, base_arg_i=0, base_ret_i=1)
        elif fname in ['subsum', 'subavg', 'subcount', 'submin']:
            return DirectIntruction(*con_args, base_arg_i=2)
        elif fname in ['subslice']:
            return DirectIntruction(*con_args, base_arg_i=0)
        elif fname in ['firstn']:
            argl = len(arg_list)
            assert argl == 4 or argl == 6
            n = int(arg_list[3].aval) if argl == 6 else int(arg_list[1].aval)
            return DirectIntruction(*con_args, base_arg_i=0, fun=lambda v: min(n, v))
        elif fname in ['hash', 'bulk_rotate_xor_hash', 'identity', 'mirror', 'year', 'ifthenelse', 'delta', 'substring', 'project', 'int', 'floor']:
            return DirectIntruction(*con_args, base_arg_i=0)
        elif fname in ['dbl']:
            return DirectIntruction(*con_args, base_arg_i=1)
        elif fname in ['hge']:
            if arg_list[1].cnt > 0:
                return DirectIntruction(*con_args, base_arg_i=1)
            else:
                return ReduceInstruction(*con_args)
        elif fname in ['append']:
            return DirectIntruction(*con_args, base_arg_i=0, fun=lambda v: v + 1)
        elif fname in ['max', 'min']:
            if len(arg_list) == 1:
                return ReduceInstruction(*con_args)
            else:
                assert len(arg_list) == 2
                return DirectIntruction(*con_args, base_arg_i=0)
        # Aggregate Instructions (result = 1)
        elif fname in ['sum', 'avg', 'single', 'dec_round']:
            return ReduceInstruction(*con_args)
        elif fname in ['new']:
            return NullInstruction(*con_args)
        # Load stuff
        elif fname in ['tid', 'bind', 'bind_idxbat']:
            return LoadInstruction(*con_args)
        else:
            logging.error("What instruction is this ?? {}".format(fname))
            return MalInstruction(*con_args)

    def getArgVars(self):
        """ returns only the arguments that are tmp variables(C_...,X_..) """
        return [arg for arg in self.arg_list if arg.isVar()]

    def approxFreeSize(self, pG):
        """
        size of all instructions whose has reached its end-of-life (a.eol == 1)
        @arg pG: dict<str,Prediction> //prediciton graph: ret var -> prediction
        @ret int: prediction of the amount of memory that is freed after this inst.
        """
        dp = Prediction(0, 0, 0, 0, 'bat[:lng]', 0)
        return sum([pG.get(a.name, dp).getMem()  for a in self.arg_list if a.eol == 1])
        # return sum([Utils.approxSize(traind, G, a.name, a.atype))

    def approxMemSize(self, pG):
        """
        all instructions who will still be alive (a.eol == 0)
        @arg pG: dict<str,Prediction> //prediciton graph: ret_var -> prediction
        @ret int: prediction of the amount of mem the instruction uses
        """
        dp = Prediction(0, 0, 0, 0, 'bat[:lng]', 0)  # default case
        return sum([pG.get(a.name, dp).getMem()  for a in self.ret_args if a.eol == 0])

    def typeof(self, rv):
        """ search for the type of a variable """
        return next(iter([r.atype for r in self.ret_args if r.name == rv]))

    def printShort(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.fname, self.nargs, self.time, self.mem_fprint))

    def printVerbose(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.short, self.nargs, self.time, self.mem_fprint))

    def isExact(self, other, ignoreScale=False):
        """if two instructions are the same or not, with/without ignoring the
        scale of the data
        """
        if not ignoreScale:
            return self.short == other.short
        else:
            # ignore count in all variables
            self_sub = re.sub(r'\[\d+\]', '', self.short)
            other_sub = re.sub(r'\[\d+\]', '', other.short)
            return self_sub == other_sub

    # """" two instructions are equal whey they have the same method name and
    #     exactly the same arguments"""
    # def __eq__(self, o):
    #     if(self.fname == o.fname and Utils.cmp_arg_list(self.arg_list,o.arg_list)):
    #         return True
    #     else:
    #         return False
    #
    # def __ne__(self, other):
    #     return self.__ne__(other)


class DirectIntruction(MalInstruction):
    """
    @desc What goes in, goes out....(hash,identity,mirror,year,delta,substring...)
    @arg *args           // the MalInstruction arguments
    @arg base_arg_i: int // the index of the lead argument (should be one...)
    @arg base_ret_i: int // index of the lead return var (should be one...)
    @arg fun: int -> int //
    """
    def __init__(self, *args, base_arg_i, base_ret_i=0, fun=lambda v: v):
        MalInstruction.__init__(self, *args)
        self.base_arg = self.arg_list[base_arg_i]
        self.base_ret = self.ret_args[base_ret_i].name
        self.rtype = self.ret_args[base_ret_i].atype
        self.cntf = fun

    def approxArgCnt(self, G, default=None):
        return G[self.base_arg.name].avg

    def approxArgDist(self, other, G):
        return abs(self.approxArgCnt(G, sys.maxsize) - other.argCnt())

    def argCnt(self):
        return self.base_arg.cnt

    def predict(self, traind, G, default=None):
        p = self.cntf(self.approxArgCnt(G, default))
        t = self.rtype
        return [Prediction(retv=self.base_ret, ins=None, cnt=p, avg=p, t=t)]


class SetInstruction(MalInstruction):
    """
    @desc covers the intersect, mergecand, difference, <, > insructions
    @arg i1: int //argument 1 index
    @arg i2: int //argument 2 index
    @arg fun: lambda (a,b) -> int // used to do the prediction
    """
    def __init__(self, *args, i1, i2, fun):
        MalInstruction.__init__(self, *args)
        self.arg1 = self.arg_list[0]
        self.arg2 = self.arg_list[1]
        self.cntf = fun

    def approxArgCnt(self, pG):
        return [pG[self.arg1.name].avg, pG[self.arg2.name].avg]

    def argCnt(self):
        return [self.arg1.cnt, self.arg2.cnt]

    def predict(self, traind, pG, default=None):
        ac = self.cntf(*self.approxArgCnt(pG))
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=ac, avg=ac, t=t)]


class ProjectInstruction(DirectIntruction):
    """
    @desc similar to DirectIntruction, except the case that we have a "projection"
        that returns a BAT of strings.
      For strings, a kNN is done to find similar ones
    """
    def __init__(self, *args):
        DirectIntruction.__init__(self, *args, base_arg_i=0)
        self.base_col = self.arg_list[1].col

    def kNN(self, traind, k, G):
        lst = traind.mal_dict[self.fname]
        c = [[i, self.approxArgDist(i, G)] for i in lst if i.base_col == self.base_col]
        c.sort(key=lambda t: t[1])
        return [e[0] for e in c[0:k]]

    def predict(self, traind, G, default=None):
        ac = self.approxArgCnt(G, default)
        retv = self.ret_args[0]
        rtype = retv.atype
        if self.fname == 'projection' and rtype == 'bat[:str]':
            kNN = self.kNN(traind, 1, G)  # TODO maybe avg 5?
            assert len(kNN) > 0
            # nn1 = kNN[0]
            rs = kNN[0].ret_args[0].size  # return size
            return [Prediction(retv=retv.name, ins=None, cnt=ac, avg=ac, t=rtype, mem=rs)]

        return super().predict(traind, G, default)


class GroupInstruction(DirectIntruction):
    def __init__(self, *args, base_arg_i, base_col, col_stats):
        MalInstruction.__init__(self, *args)
        self.base_arg = self.arg_list[base_arg_i]
        self.base_col = base_col
        self.col_stats = col_stats
        # self.base_ret = self.ret_vars[base_ret_i]

    def approxArgCnt(self, G, default=None):
        return G.get(self.base_arg.name, default).avg

    def argCnt(self):
        return self.base_arg.cnt

    def kNN(self, traind, k, G):
        lst = traind.mal_dict[self.fname]
        c = [[i, self.approxArgDist(i, G)] for i in lst if i.base_col == self.base_col]
        c.sort(key=lambda t: t[1])
        return [e[0] for e in c[0:k]]

    def predict(self, traind, G, default=None):
        p = self.approxArgCnt(G, default)
        retl = self.ret_args
        if self.fname == 'subgroupdone':
            # TODO taking argDiv into account later
            # TODO might need a better implementation than using kNN later
            kNN = self.kNN(traind, 1, G)
            if len(kNN) > 0:
                nn1r = kNN[0].ret_args
                return [Prediction(retv=r.name, ins=None, cnt=nnr.cnt, avg=nnr.cnt, t=r.atype) for (r, nnr) in zip(retl, nn1r) if r.eol == 0]
        elif self.fname == 'groupdone' and self.col_stats is not None:
            # for groupdone three things are returned, so we need to compute 3 Predictions
            stats = self.col_stats
            r1 = retl[0]
            p1 = Prediction(retv=r1.name, ins=None, cnt=p, avg=p, t=r1.atype)
            r2 = retl[1]
            p2 = Prediction(retv=r2.name, ins=None, cnt=stats.uniq, avg=stats.uniq, t=r2.atype)
            r3 = retl[2]
            p3 = Prediction(retv=r3.name, ins=None, cnt=stats.uniq, avg=stats.uniq, t=r3.atype)
            return [p for (r, p) in [(r1, p1), (r2, p2), (r3, p3)] if r.eol == 0]

        # If we have a groupdone but no column statistics, then for each variable, just return the input size, which gives an overestimation
        return [Prediction(retv=r.name, ins=None, cnt=p, avg=p, t=r.atype) for r in retl if r.eol == 0]


class ReduceInstruction(MalInstruction):
    """
    @desc All2One instructions (sum,avg,max,min,single,dec_round)
    """
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)
        self.base_arg = self.arg_list[0]

    def argCnt(self):
        return self.base_arg.cnt

    def approxArgCnt(self, G, default=None):
        return G.get(self.base_arg.name, default).avg

    def predict(self, traind, G, default=None):
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=1, avg=1, t=t)]


class NullInstruction(MalInstruction):
    """
    @desc 0 output instruction (e.g new...)
    """
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)

    def predict(self, traind, G, default=None):
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=0, avg=0, t=None)]


class LoadInstruction(MalInstruction):
    """
    @desc instructions loading tables and columns (e.g tid,bind,bind_idxbat...)
    """
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)

    def predict(self, traind, G, default=None):
        # if self.fname in ['tid','bind_idxbat']:
            # m=0
        # else:
            # TODO replace the self.cnt information with column statistics,
            #  because the current implementation is a little bit cheating,
            #  since in reality, we won't have the information of self.cnt, but
            #  we can easily get that from the column statistics.
            # l = traind.mal_dict[self.fname] if i.arg_list[0].col==self.arg_list[0].col][0]
            # m = [i.ret_size for i in
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=self.cnt, avg=self.cnt, t=t, mem=self.ret_size)]


class JoinInstruction(MalInstruction):
    """
    @desc Join group (join, thetajoin) ret count can be bigger than input
    UNDER CONSTRUCTION
    """
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)
        self.arg1 = self.arg_list[0]
        self.arg2 = self.arg_list[1]

    def approxArgCnt(self, G):
        return [G.get(self.arg1.name, None).avg, G.get(self.arg2.name, None).avg]

    def argCnt(self):
        return [self.arg1.cnt, self.arg2.cnt]

    def approxArgDiv(self, ins, G):  # TODO rethink the order
        div = 1
        app = self.approxArgCnt(G)
        ac = ins.argCnt()
        for (a1, a2) in zip(app, ac):
            if a1 is not None:
                div = div * a1 / a2  # Hoping the order is correct...
        return div

    def approxArgDiv2(self, ins, G):  # TODO rethink the order
        div = 1
        app = ins.approxArgCnt(G)
        ac = self.argCnt()
        for (a1, a2) in zip(app, ac):
            if a1 is not None:
                div = div * a1 / a2  # Hoping the order is correct...
        # print("div == ",div)
        return div

    def approxArgDist(self, ins, G):
        assert G is not None
        lead_args = [self.arg1, self.arg2]
        self_cnt = [float(G.get(a.name, None).avg) for a in lead_args]
        ins_count = [arg.cnt for arg in [ins.arg1, ins.arg2]]
        return sum([(c1 - c2)**2 for (c1, c2) in zip(self_cnt, ins_count)])

    def extrapolate(self, other):
        return self.cnt

    def kNN(self, ilist, k, G):
        cand = [[i, self.approxArgDist(i, G)] for i in ilist]  # TODO check for columns ???
        cand.sort(key=lambda t: t[1])
        return [t[0] for t in cand[0:k]]

    def predict(self, traind, g):
        # for crossproduct, the predicted size is siz-of(x) * size-of(y)
        # for other joins, use kNN for the prediction
        if self.fname == 'crossproduct':
            p = reduce(lambda x, y: x * y, self.approxArgCnt(g))
            retv = self.ret_args
            return [Prediction(retv=r.name, ins=None, cnt=p, avg=p, t=r.atype) for r in retv]
        else:
            cand_l = traind.mal_dict[self.fname]
            knn5 = self.kNN(cand_l, 5, g)
            avg = sum([ins.cnt for ins in knn5]) / len(knn5)  # TODO add argdiv
            (i, c) = (knn5[0].short, knn5[0].cnt)
            retv = self.ret_args
            return [Prediction(retv=r.name, ins=i, cnt=c, avg=avg, t=r.atype) for r in retv]


number_types = ['bat[:int]', 'bat[:lng]', 'lng', 'bat[:hge]', 'bat[:bte]', 'bat[:sht]']


class SelectInstruction(MalInstruction):
    """
    @desc
    @attr col     : str //column name (TMP if None)
    @attr ctype   : str //column type
    @attr op      : str //selectivity operator (<.>,between if range)
    @attr lo      : obj //lower bound of select
    @attr hi      : obj //upper bound
    @attr lead_arg: Arg //
    """
    def __init__(self, *def_args, jobj, stats):
        MalInstruction.__init__(self, *def_args)
        # this is the column type
        self.ctype = jobj["arg"][0].get("type", "UNKNOWN")
        alias_iter = iter([o["alias"] for o in jobj["arg"] if "alias" in o])
        # this is the initial column
        self.col = next(alias_iter, "TMP").split('.')[-1]
        # if we have a projection argument, this is the projection column
        self.proj_col = next(alias_iter, "TMP").split('.')[-1]
        self.arg_size = [o.get("size", 0) for o in jobj.get("arg", [])]
        self.op = Utils.extract_operator(self.fname, jobj)

        a1 = self.arg_list[1]
        self.lead_arg_i = 1 if a1.isVar() and a1.cnt > 0 else 0
        self.lead_arg = self.arg_list[self.lead_arg_i]

        lo, hi = Utils.hi_lo(self.fname, self.op, jobj, stats.get(self.col, ColumnStats(0, 0, 0, 0, 0)))

        if self.ctype in ['bat[:int]', 'bat[:lng]', 'lng', 'bat[:hge]', 'bat[:bte]', 'bat[:sht]']:
            if self.op in ['>=', 'between'] and self.col in stats:
                s = stats[self.col]
                step = round((int(s.maxv) - int(s.minv)) / int(s.uniq))
                self.lo, self.hi = (int(lo), int(hi) + step)
            else:  # TODO <=
                self.lo, self.hi = (int(lo), int(hi))
        elif self.ctype == 'bat[:date]':
            self.lo = datetime.strptime(lo, '%Y-%m-%d')
            self.hi = datetime.strptime(hi, '%Y-%m-%d')
        else:
            # logging.error("Wtf type if this ?? :{}".format(self.ctype))
            self.hi = hi
            self.lo = lo

    @staticmethod
    def removeDuplicates(ins_list):
        bounds_set = set()
        uniqs = []
        for ins in ins_list:
            if (ins.hi, ins.lo, ins.col) not in bounds_set:
                bounds_set.add((ins.hi, ins.lo, ins.col))
                uniqs.append(ins)

        return uniqs

    # for the range instructions
    def isIncluded(self, other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]', 'bat[:lng]', 'bat[:date]', 'lng', 'bat[:hge]']:
            return self.lo >= other.lo and self.hi <= other.hi

        return None

    # for the range instructions
    def includes(self, other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]', 'bat[:lng]', 'bat[:date]', 'lng', 'bat[:hge]']:
            return self.lo <= other.lo and self.hi >= other.hi

        return None

    def approxArgCnt(self, G, default=None):
        # logging.error("lead arg: {}".format(self.lead_arg.name))
        return G.get(self.lead_arg.name, default).avg

    def argCnt(self):
        return self.lead_arg.cnt

    def approxArgDist(self, other, G):
        """
        @desc argument distance between self and another train instruction
        !!ASSUMES other is a known (training) instruction (we know the cnt)!!
        """
        assert G is not None
        lead_arg = self.lead_arg
        ac = G[lead_arg.name].avg if lead_arg.name in G else 'inf'
        return abs(other.lead_arg.cnt - float(ac))

    def argDist(self, other):
        assert False
        diff = [abs(a.cnt - b.cnt) for (a, b) in zip(self.arg_list[1:2], other.arg_list[1:2])]
        return sum(diff)

    def argDiv(self, other):
        return other.lead_arg.cnt / self.lead_arg.cnt


    # TODO rename range dist
    def distance(self, other, G=None):
        """ @desc distance between the range(hi,lo) of self, other """
        assert self.ctype == other.ctype
        # if self.includes(other) or self.isIncluded(other): #TODO remove this if
        if self.ctype in ['bat[:int]', 'bat[:lng]', 'lng', 'bat[:hge]', 'bat[:bte]', 'bat[:sht]']:
            # for numerical values, return square distance
            return float((self.lo - other.lo)**2 + (self.hi - other.hi)**2)
        elif self.ctype == 'bat[:date]':  # TODO implement timestamp
            (min_lo, max_lo) = (min(self.lo, other.lo), max(self.lo, other.lo))
            (min_hi, max_hi) = (min(self.hi, other.hi), max(self.hi, other.hi))
            return float((max_lo - min_lo).days + (max_hi - min_hi).days)
        elif self.ctype == 'bat[:str]':
            assert self.lo == self.hi and other.lo == other.hi
            return Levenshtein.levenshtein(self.lo, other.lo)
        elif self.ctype == 'bat[:bit]':  # just try to find something close to the argument size
            return self.approxArgDist(other, G)

        logging.error("What type is this {}".format(self.ctype))
        assert False

    def isSameType(self, ins):
        """checks if the two instructions have the same column,type,operator"""
        return self.col == ins.col and self.op == ins.op and self.ctype == ins.ctype

    def kNN(self, ilist, k, G):
        lst = [[i, self.distance(i, G)] for i in ilist if self.isSameType(i)]
        lst.sort(key=lambda t: t[1])
        return [t[0] for t in lst[0:k]]

    def predict(self, traind, approxG, default=None):
        """
        @desc run kNN to find the 5 closest instructions based on the range bounds
        range extrapolation:  (self.hi - self.lo) / (traini.hi - traini.lo)
        arg   extrapolation:  self.arg_cnt / traini.arg_cnt
        prediction(traini) = traini.cnt * range_extrapolation * arg_extrapolation
        """
        assert approxG is not None
        # First get candidate list by searching for instructions with the same name
        self_list = traind.mal_dict.get(self.fname, [])

        # prev_list = []
        #
        # tmp = self
        # while(tmp.prev_i != None):
        #     prev_list.append(tmp.prev_i)
        #     tmp = tmp.prev_i

        # prev_list.reverse()
        # curr_nn = self_list
        # maxk = 5 * (2 ** len(prev_list))
        # for node in prev_list:
        #     k = int(maxk / 2)
        #     curr_level = [ins for ins in curr_nn if node.col == ins.col]
        #     curr_nn = node.kNN(curr_level,k, approxG)
        #     maxk = maxk / 2
        # if self.proj_col != 'TMP' and self.prev_i != None:
        #     level1 = [ins for ins in self_list if self.proj_col == ins.col]
        #     logging.error("len level1: {}".format(len(level1)))
        #     logging.error("testing {}".format(self.short))
        #     prev_nn = self.prev_i.kNN(level1,100, approxG)
        #     cand_list = list([ins.next_i for ins in prev_nn])
        # else:
        #     cand_list = self_list

        # Second: filter candidate list by columns, projection columns and length of arguments
        cand_list = [ins for ins in self_list if self.col == ins.col and self.proj_col == ins.proj_col and len(ins.arg_list) == len(self.arg_list)]
        # random.shuffle(cand_list)
        nn = self.kNN(cand_list, 5, approxG)
        rt = self.ret_args[0].atype  # return type

        # DEBUG
        if self.fname == 'thetaselect' and self.op == '>':
            for ins in nn:
                logging.debug("NN: {} {}".format(ins.cnt, ins.short))

        if len(nn) == 0:
            logging.error("0 knn len in select ?? {} {} {} {}".format(self.short, self.op, self.ctype, self.col))
            # logging.error("Cand len {}".format(len(prev_nn)))
            logging.error("self col: {} proj col {}".format(self.col, self.proj_col))
            # for di in [ins for ins in self_list if self.col == ins.col]:
                # logging.error("cand: {} {} {}".format(di.short, di.col, di.proj_col))

        # still just for debugging: keep only one instruction, iso 5 instructions
        nn.sort(key=lambda ins: self.approxArgDist(ins, approxG))
        nn1 = nn[0]
        arg_cnt = self.approxArgCnt(approxG)

        if arg_cnt is not None:
            # do the proper scale up/down according to the proportion
            avg = sum([i.extrapolate(self) * (arg_cnt / i.argCnt()) for i in nn if i.argCnt() > 0]) / len(nn)
            cal_avg = min(avg, arg_cnt)  # calibration
            avgm = cal_avg * Utils.sizeof(rt)
            cnt1 = nn1.extrapolate(self) * arg_cnt / nn1.argCnt() if nn1.argCnt() > 0 else nn1.extrapolate(self)
            return [Prediction(retv=self.ret_vars[0], ins=nn1, cnt=cnt1, avg=cal_avg, t=rt, mem=avgm)]
        else:
            logging.error("None arguments ??? {}".format(self.lead_arg.name))
            avg = sum([i.extrapolate(self) for i in nn]) / len(nn)
            return [Prediction(retv=self.ret_vars[0], ins=nn1, cnt=nn1.extrapolate(self), avg=avg, t=rt)]

    def extrapolate(self, other):
        """ assumes self is training instruction!!! """
        if self.ctype in ['bat[:int]', 'bat[:lng]', 'lng', 'bat[:hge]', 'bat[:bte]', 'bat[:sht]']:
            self_dist = self.hi - self.lo
            other_dist = other.hi - other.lo

            if self_dist * other_dist != 0:
                return self.cnt * other_dist / self_dist
            else:
                return self.cnt
        elif self.ctype == 'bat[:date]':  # TODO: implement time stamp
            diff1 = (other.hi - other.lo)
            diff2 = (self.hi - self.lo)

            if diff1.days * diff2.days != 0:
                return self.cnt * (diff1.days / diff2.days)
            else:
                print("0 product", self.hi, self.lo, other.hi, other.lo)
                return self.cnt
        elif self.ctype == 'bat[:str]':
            return self.cnt
        elif self.ctype == 'bat[:bit]':
            return self.cnt
        else:
            print("weird stuff in select", self.short)
            print("type ==", self.ctype, self.lo, self.hi)
            return None
