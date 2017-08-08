from utils import Utils
from mal_arg import Arg
from stats import Stats
from datetime import datetime

# from sel_ins import SelectInstruction
from functools import reduce
import re


""" MalInstruction Class:
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
@attr free_size : int        //amount of memory freed(arguments for which eol == 1)
@attr arg_vars  : list<str>  //names of the arguments that are vars
@attr ret_vars  : list<str>  //names of the return output that are vars
@attr cnt       : int        //the number of elements of the return var
"""
class MalInstruction:
    def __init__(self, pc, clk, short, fname, size, ret_size, tag, arg_size, alist, free_size, arg_vars, ret_vars, cnt):
        self.pc         = pc
        self.clk        = clk
        self.fname      = fname
        self.time       = 0
        self.size       = size
        self.ret_size   = ret_size
        self.arg_list   = alist
        self.short      = short
        self.tag        = tag
        self.mem_fprint = self.size + self.ret_size
        self.arg_size   = arg_size
        self.nargs      = len(alist)
        self.free_size  = free_size
        self.arg_vars   = arg_vars
        self.ret_vars   = ret_vars
        self.cnt        = cnt

    @staticmethod
    def fromJsonObj(jobj, stats):
        size      = int(jobj["size"])
        pc        = int(jobj["pc"])
        clk       = int(jobj["clk"])
        short     = jobj["short"]
        fname,_,_ = Utils.extract_fname(jobj["short"])
        tag       = int(jobj["tag"])
        rv        = [rv.get("size",0) for rv in jobj["ret"]]
        ret_size  = sum([o.get("size",0) for o in jobj.get("ret",[])])
        arg_size  = sum([o.get("size",0) for o in jobj.get("arg",[])])
        arg_list  = [Arg.fromJsonObj(e) for e in jobj.get("arg",[])]
        free_size = sum([arg.size for arg in arg_list if arg.eol == 1])
        arg_vars  = [arg.name for arg in arg_list if arg.isVar()]
        ret_vars  = [ret['name'] for ret in jobj.get("ret",[]) if Utils.isVar(ret['name'])]
        count     = int(jobj["ret"][0].get("count",0))
        if fname in ['select','thetaselect']:
            return SelectInstruction(
                pc, clk, short, fname, size, ret_size, tag, arg_size, arg_list, free_size, arg_vars, ret_vars, count,jobj, stats
            )
        else :
            return MalInstruction(
                pc, clk, short, fname, size, ret_size, tag, arg_size, arg_list, free_size, arg_vars, ret_vars, count
            )

    #deprecated
    def distance(self,other):
        return self.metric.distance(other.metric) #TODO fix this

    def argDist(self, other):
        assert len(self.arg_list) == len(other.arg_list)
        diff = [abs(a.size-b.size) for (a,b) in zip(self.arg_list,other.arg_list)]
        return sum(diff)

    def similarity(self, other):
        assert len(self.arg_list) == len(other.arg_list)
        total_self  = max(sum([a.size for a in self.arg_list]),0.001)
        total_other = max(sum([a.size for a in other.arg_list]),0.001)
        return total_self / total_other


    """ @ret: string //arguments as a string"""
    def argListStr(self):
        slist = ["arg: {:10} ".format(int(a.size / 1000)) for a in self.arg_list if a.size>0]
        return ' '.join(slist)

    """ returns only the arguments that are tmp variables """
    def getArgVars(self):
        return [arg for arg in self.arg_list if arg.isVar()]

    def printVarFlow(self,varflow):
        l = [(a.name,varflow.lookup(a.name,self.tag)) for a in self.arg_list]
        v = lambda name: "table " if name.startswith("C_") else "column"
        j = ["var: {} {}: {} ".format(n,v(n),t) for (n,t) in l if t]
        if len(j) > 0:
            print('|| '.join(j))

    def kNN(self, ilist, k):
        cand = [[i,self.argDist(i)] for i in ilist if len(self.arg_list) == len(i.arg_list)]
        cand.sort(key = lambda t: t[1])
        return [ t[0] for t in cand[0:k] ]

    def predictCount(self, ilist, default=None):
        knn = self.kNN(ilist, 1)
        if len(knn) > 1:
            return self.mem_fprint
        else:
            return default

    def printShort(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.fname,self.nargs, self.time, self.mem_fprint))

    def printVerbose(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.short,self.nargs, self.time, self.mem_fprint))


    def isExact(self,other,ignoreScale=False):
        if ignoreScale == False:
            return self.short == other.short
        else:
            #ignore cardinality in arguments
            self_sub  = re.sub(r'\[\d+\]','',self.short)
            # print(self_sub)
            other_sub = re.sub(r'\[\d+\]','',other.short)
            return self_sub == other_sub

    """" two instructions are equal whey they have the same method name and
        exactly the same arguments"""
    def __eq__(self, o):
        if(self.fname == o.fname and Utils.cmp_arg_list(self.arg_list,o.arg_list)):
            return True
        else:
            return False

    def __ne__(self, other):
        return self.__ne__(other)


class SelectInstruction(MalInstruction):
    def __init__(self, pc, clk, short, fname, size, ret_size, tag, arg_size, alist, free_size, arg_vars, ret_vars, cnt, jobj, stats):
        MalInstruction.__init__(self, pc, clk, short, fname, size, ret_size, tag, arg_size, alist, free_size, arg_vars, ret_vars, cnt)
        self.ctype    = jobj["arg"][0].get("type","UNKNOWN")
        self.col      = next(iter([o["alias"] for o in jobj["arg"] if "alias" in o]),"TMP").split('.')[-1]
        self.arg_size = [o.get("size",0) for o in jobj.get("arg",[])]
        self.op       = Utils.extract_operator(fname, jobj)
        # print(method, op)
        lo, hi = Utils.hi_lo(fname, self.op, jobj, stats.get(self.col,Stats(0,0)))
        if self.ctype in ['bat[:int]','bat[:lng]','lng']:
            self.lo,self.hi    = (int(lo),int(hi))
        elif self.ctype == 'bat[:date]':
            # print(lo,hi)
            self.lo  = datetime.strptime(lo,'%Y-%m-%d')
            self.hi  = datetime.strptime(hi,'%Y-%m-%d')
        else:
            self.hi     = hi
            self.lo     = lo


    @staticmethod
    def removeDuplicates(ins_list):
        bounds_set = set()
        uniqs = []
        for ins in ins_list:
            if (ins.hi, ins.lo) not in bounds_set:
                bounds_set.add((ins.hi, ins.lo))
                uniqs.append(ins)

        return uniqs

    def isIncluded(self,other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]','bat[:lng]','bat[:date]','lng']:
            return self.lo >= other.lo and self.hi <= other.hi

        return None
    #
    def includes(self, other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]','bat[:lng]','bat[:date]','lng']:
            return self.lo <= other.lo and self.hi >= other.hi

        return None

    def approxArgCnt(self, G):
        return G.get(self.arg_list[1].name,None)

    def argCnt(self):
        return self.arg_list[1].cnt

    def approxArgDist(self, other, G):
        assert G != None
        approx_count = float(G.get(self.arg_list[1].name,'inf'))
        #TODO rethink
        return abs(other.arg_list[1].cnt-approx_count)

    def argDist(self, other):
        # assert len(self.arg_list) == len(other.arg_list)
        # print(self.arg_list[1].cnt, other.arg_list[1].cnt)
        diff = [abs(a.cnt-b.cnt) for (a,b) in zip(self.arg_list[1:2],other.arg_list[1:2])]
        return sum(diff)

    def argDiv(self, other):
        return other.arg_list[1].cnt /self.arg_list[1].cnt


    def distance(self, other):
        # print(self.col, other.col)
        assert self.ctype == other.ctype
        if self.includes(other) or self.isIncluded(other):
            if self.ctype in ['bat[:int]','bat[:lng]','lng']:
                return float((self.lo-other.lo)**2 + (self.hi-other.hi)**2)
            elif self.ctype == 'bat[:date]':
                (min_lo,max_lo) = (min(self.lo,other.lo),max(self.lo,other.lo))
                (min_hi,max_hi) = (min(self.hi,other.hi),max(self.hi,other.hi))
                return float((max_lo-min_lo).days + (max_hi-min_hi).days)
        else:
            return float('inf')
        return None

    def kNN(self, ilist, k):
        # for i in ilist:
            # print(i.short)
        # print(self.col, self.op, self.ctype)
        cand = [[i,self.distance(i)] for i in ilist if i.col == self.col and i.op == self.op and i.ctype == self.ctype]
        cand.sort( key = lambda t: t[1] )
        return [ t[0] for t in cand[0:k] ]

    def predictCount(self, ilist, default=0):
        knn = self.kNN(ilist, 1)
        if len(knn) >= 1:
            return self.extrapolate(knn[0])
        else:
            print("None found")
            return default


    def extrapolate(self, other):
        if self.ctype in ['bat[:int]','bat[:lng]','lng']:
            self_dist  = self.hi  - self.lo
            other_dist = other.hi - other.lo

            if self_dist*other_dist != 0:
                return self.cnt*other_dist/self_dist
            else:
                return self.cnt
        elif self.ctype == 'bat[:date]':
            diff1 = (other.hi-other.lo)
            diff2 = (self.hi-self.lo)

            if diff1.days * diff2.days != 0:
                return self.cnt * (diff1.days / diff2.days)
            else:
                print("0 product", self.hi, self.lo, other.hi, other.lo)
                return self.cnt
        elif self.ctype == 'bat[:str]':
            return self.cnt
        else:
            print("type ==",self.ctype,self.lo,self.hi)
            return None
