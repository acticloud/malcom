import logging
import sys
import re
from utils    import Utils
from mal_arg  import Arg
from stats    import Stats
from datetime import datetime
from utils    import Prediction

"""
interface MalInstruction {

    def argCnt()               : List<int>

    def approxArgCnt(traind, G): List<int>

    def predictCount(traind, G): List<Prediction>

    def kNN(traind, k, G)      : List<MalInstruction>
}
"""


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
@attr free_size : int        //amount of memory freed(args for which eol==1)
@attr arg_vars  : list<str>  //names of the arguments that are vars
@attr ret_vars  : list<str>  //names of the return output that are vars
@attr cnt       : int        //the number of elements of the return var
"""
class MalInstruction:
    def __init__(self, pc, clk, short, fname, size, ret_size, tag, arg_size, alist, ret_args, free_size, arg_vars, ret_vars, cnt):
        self.pc         = pc
        self.clk        = clk
        self.fname      = fname
        self.time       = 0
        self.size       = size
        self.ret_size   = ret_size
        self.arg_list   = alist
        self.ret_args   = ret_args
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
        ro        = jobj.get("ret",[]) #return object
        ret_size  = sum([o.get("size",0) for o in ro])
        arg_size  = sum([o.get("size",0) for o in jobj.get("arg",[])])
        arg_list  = [Arg.fromJsonObj(e) for e in jobj.get("arg",[])]
        ret_args  = [Arg.fromJsonObj(e) for e in ro]# if e["eol"]==0]
        # print(len(alive_ret))
        free_size = sum([arg.size for arg in arg_list if arg.eol == 1])
        arg_vars  = [arg.name for arg in arg_list if arg.isVar()]
        ret_vars  = [ret['name'] for ret in ro if Utils.isVar(ret['name'])]
        count     = int(jobj["ret"][0].get("count",0))

        con_args =  [pc, clk, short, fname, size, ret_size, tag, arg_size, arg_list, ret_args, free_size, arg_vars, ret_vars, count]

        if fname in ['select','thetaselect','likeselect']:
            return SelectInstruction(*con_args,jobj=jobj, stats=stats) #TODO replace jobj
        elif fname in ['projection','projectionpath','projectdelta']:
            return ProjectInstruction(*con_args)
        elif fname in ['+','-','*','/','or','dbl','and']:
            bi = 0 if fname in ['+','*','/'] else 1 #TODO fix this
            if arg_list[bi].isVar():
                return DirectIntruction(*con_args, base_arg_i=bi)
            else:
                return ReduceInstruction(*con_args)
        elif fname in ['join','thetajoin']:
            return JoinInstruction(*con_args)
        elif fname in ['group','subgroup','subgroupdone','groupdone']:
            return GroupInstruction(*con_args, base_arg_i=0, base_col=arg_list[0].col)
        elif fname in ['firstn']:
            return FirstnInstruction(*con_args)
        elif fname in ['hash','bulk_rotate_xor_hash','identity','mirror','year','ifthenelse','delta','substring','project']:
            return DirectIntruction(*con_args, base_arg_i = 0)
        elif fname in ['dbl']:
            return DirectIntruction(*con_args, base_arg_i = 1)
        elif fname in ['hge']:
            if arg_list[1].cnt > 0:
                return DirectIntruction(*con_args, base_arg_i = 1)
            else:
                return ReduceInstruction(*con_args)
        elif fname in ['==','isnil','!=','like']:
            #TODO fix this
            return DirectIntruction(*con_args, base_arg_i = 0)
        elif fname in ['intersect','<','>']:
            return CompareIntruction(*con_args, arg_i = [0,1])
        elif fname in ['sort']:
            return DirectIntruction(*con_args, base_arg_i = 0, base_ret_i = 1)#TODO fix this (multiple ret??)
        elif fname in ['subsum','subavg','subcount','submin']:
            return SubCalcInstruction(*con_args)
        elif fname in ['subslice']:
            return DirectIntruction(*con_args, base_arg_i = 0)
        elif fname in ['difference']:
            return DiffInstruction(*con_args)
        elif fname in ['sum','avg','single','dec_round','max']:
            return ReduceInstruction(*con_args)
        elif fname in ['mergecand']:
            return MergeInstruction(*con_args)
        elif fname in ['append']:
            return AppendInstruction(*con_args)
        elif fname in ['new']:
            return NullInstruction(*con_args)
        elif fname in ['tid','bind','bind_idxbat']:
            return LoadInstruction(*con_args)
        else :
            return MalInstruction(*con_args)

    def argDist(self, other):
        assert len(self.arg_list) == len(other.arg_list)
        diff = [abs(a.size-b.size) for (a,b) in zip(self.arg_list,other.arg_list)]
        return sum(diff)

    #TODO remove
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

    # def extrapolate(self, other):
        # return other.cnt
    #TODO remove
    def predictCount(self, ilist, default=None):
        print(self.fname)
        knn = self.kNN(ilist, 1)
        if len(knn) > 1:
            return self.mem_fprint
        else:
            return default

    def approxFreeSize(self, G, pG):
        dp = Prediction(0,0,0,0,0)
        return sum([pG.get(a.name,dp).getMem(G)  for a in self.arg_list if a.eol==1])
        # return sum([Utils.approxSize(traind, G, a.name, a.atype))

    def approxMemSize(self, traind, G):
        # if self.fname == 'bind':
            # return self.ret_args[0].size #TODO fix this
        # elif self.fname == 'tid':
            # return 0 #TODO fix this
        # else:
        return sum([r.getMem(G) for r in self.predictCount(traind, G)])

    def typeof(self, rv):
        return next(iter([r.atype for r in self.ret_args if r.name == rv]))

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
            #ignore count in all variables
            self_sub  = re.sub(r'\[\d+\]','',self.short)
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

    def extrapolate(self, other):
        return self.cnt

"""
@desc What goes in, goes out....(hash,identity,mirror,year,delta,substring...)
@arg
"""
class DirectIntruction(MalInstruction):
    def __init__(self, *args, base_arg_i, base_ret_i = 0):
        MalInstruction.__init__(self, *args)
        self.base_arg = self.arg_list[base_arg_i]
        # print(self.fname, self.short)
        self.base_ret = self.ret_args[base_ret_i].name
        self.rtype    = self.ret_args[base_ret_i].atype

    def approxArgCnt(self, G, default=None):
        return G[self.base_arg.name]

    def approxArgDist(self, other, G):
        return abs(self.approxArgCnt(G, sys.maxsize)-other.argCnt())

    def argCnt(self):
        return self.base_arg.cnt

    def predictCount(self, traind, G, default=None):
        p = self.approxArgCnt(G, default)
        t = self.rtype
        return [Prediction(retv=self.base_ret, ins=None, cnt=p, avg=p, t=t)]

"""
CompareIntruction: output is at most min of the inputs
@arg
"""
class CompareIntruction(MalInstruction):
    def __init__(self, *args, arg_i):
        MalInstruction.__init__(self, *args)
        self.base_arg_i = arg_i

    def approxArgCnt(self, G, default=None):
        return [G[self.arg_list[i].name] for i in self.base_arg_i]

    def argCnt(self):
        return [self.arg_list[i].cnt for i in self.base_arg_i]

    def predictCount(self, traind, G, default=None):
        p = min(self.approxArgCnt(G, default))
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=p, avg=p, t=t)]


""" ret count = arg1.cnt + arg2.cnt worst case """
class MergeInstruction(MalInstruction):
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)
        self.arg1 = self.arg_list[0]
        self.arg2 = self.arg_list[1]

    def approxArgCnt(self, G, default=None):
        return [G.get(self.arg1.name,default),G.get(self.arg2.name,default)]

    def argCnt(self):
        return [self.arg1.cnt,self.arg2.cnt]

    def predictCount(self, traind, G, default=None):
        ac = sum(self.approxArgCnt(G, default))
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=ac, avg=ac, t=t)]

""" ret count = arg1.cnt - arg2.cnt """
class DiffInstruction(MalInstruction):
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)
        self.arg1 = self.arg_list[0]
        self.arg2 = self.arg_list[1]

    def approxArgCnt(self, G, default=None):
        return [G.get(self.arg1.name,default),G.get(self.arg2.name,default)]

    def argCnt(self):
        return [self.arg1.cnt,self.arg2.cnt]

    def predictCount(self, traind, G, default=None):
        ac = max(self.approxArgCnt(G, default))
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0],ins=None, cnt=ac, avg=ac, t=t)]

class ProjectInstruction(DirectIntruction):
    def __init__(self, *args):
        DirectIntruction.__init__(self, *args, base_arg_i = 0)
        self.base_col = self.arg_list[1].col
        self.proj_col = self.arg_list[0].col #maybe we do not need this

    def kNN(self, traind, k, G):
        l = traind.mal_dict[self.fname]
        c = [[i,self.approxArgDist(i, G)] for i in l if i.base_col == self.base_col]
        c.sort( key = lambda t: t[1] )
        return [e[0] for e in c[0:k]]

    def predictCount(self, traind, G, default=None):
        p = self.approxArgCnt(G, default)
        retl = self.ret_args
        if self.fname == 'projection' and self.ret_args[0].atype == 'bat[:str]':
            #TODO argDiv
            # print("came")
            kNN  = self.kNN(traind, 1, G)
            if len(kNN)>0:
                nn1  = kNN[0]
                nn1r = kNN[0].ret_args[0]
                return [Prediction(retv=retl[0].name, ins=None, cnt=nn1r.cnt, avg=nn1r.cnt, t=retl[0].atype, mem=nn1.ret_size)]

        return super().predictCount(traind, G, default)

class SubCalcInstruction(DirectIntruction):
    def __init__(self, *args):
        DirectIntruction.__init__(self, *args, base_arg_i = 2)
#
# class BatCalcInstruction(DirectIntruction):
#     def __init__(self,*args):
#         DirectIntruction.__init__(self,*args, base_arg_i = bi)

class GroupInstruction(DirectIntruction):
    def __init__(self, *args, base_arg_i, base_col):
        MalInstruction.__init__(self, *args)
        self.base_arg = self.arg_list[base_arg_i]
        self.base_col = base_col
        # self.base_ret = self.ret_vars[base_ret_i]

    def approxArgCnt(self, G, default=None):
        return G.get(self.base_arg.name,default)

    def argCnt(self):
        return self.base_arg.cnt

    def kNN(self, traind, k, G):
        l = traind.mal_dict[self.fname]
        c = [[i,self.approxArgDist(i, G)] for i in l if i.base_col == self.base_col]
        c.sort( key = lambda t: t[1] )
        return [e[0] for e in c[0:k]]

    def predictCount(self, traind, G, default=None):
        p = self.approxArgCnt(G, default)
        retl = self.ret_args
        if self.fname == 'subgroupdone':
            #TODO argDiv
            kNN  = self.kNN(traind, 1, G)
            if len(kNN)>0:
                nn1r = kNN[0].ret_args
                return [Prediction(retv=r.name, ins=None, cnt=nnr.cnt, avg=nnr.cnt, t=r.atype) for (r,nnr) in zip(retl,nn1r) if r.eol==0]

        return [Prediction(retv=r.name, ins=None, cnt=p, avg=p, t=r.atype) for r in retl if r.eol==0]

class FirstnInstruction(DirectIntruction):
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)
        self.base_arg = self.arg_list[0]

        if len(self.arg_list) == 6:
            self.n = int(self.arg_list[3].aval)
        elif len(self.arg_list) == 4:
            self.n = int(self.arg_list[1].aval)
        else:
            assert False

    def predictCount(self, traind, G, default = None):
        p = min(self.approxArgCnt(G, int(sys.maxsize)),self.n)
        t = self.ret_args[0].atype
        r = self.ret_args[0].name
        return [Prediction(retv=r, ins=None, cnt=p, avg=p, t=t)]

"""
@desc All2One instructions (sum,avg,max,min,single,dec_round)
"""
class ReduceInstruction(MalInstruction):
    def __init__(self, *args):
        MalInstruction.__init__(self,*args)
        self.base_arg = self.arg_list[0]

    def argCnt(self):
        return self.base_arg.cnt

    def approxArgCnt(self, G, default = None):
        return G.get(self.base_arg.name,default)

    def predictCount(self, traind, G, default = None):
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=1, avg=1, t=t)]

class AppendInstruction(ReduceInstruction):
    def __init__(self, *args):
        ReduceInstruction.__init__(self,*args)

    def predictCount(self, traind, G, default = None):
        ac = G[self.base_arg.name]
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0], ins=None, cnt=1+ac, avg=1+ac, t=t)]

"""
@desc 0 output instruction (e.g new...)
"""
class NullInstruction(MalInstruction):
    def __init__(self, *args):
        MalInstruction.__init__(self,*args)

    def predictCount(self, traind, G, default = None):
        return [Prediction(retv=self.ret_vars[0],ins= None, cnt = 0, avg = 0, t=None)]

"""
@desc instructions loading tables and columns (e.g tid,bind,bind_idxbat...)
"""
class LoadInstruction(MalInstruction):
    def __init__(self, *args):
        MalInstruction.__init__(self,*args)

    def predictCount(self, traind, G, default = None):
        # if self.fname in ['tid','bind_idxbat']:
            # m=0
        # else:
            #TODO column statistics
            # l = traind.mal_dict[self.fname] if i.arg_list[0].col==self.arg_list[0].col][0]
            # m = [i.ret_size for i in
        t = self.ret_args[0].atype
        return [Prediction(retv=self.ret_vars[0],ins= None, cnt=self.cnt, avg=self.cnt, t=t, mem=self.ret_size)]

        # if self.ret_vars[0] == 'X_54':
            # print("came")


"""
@desc Join group (join, thetajoin) ret count can be bigger than input
"""
class JoinInstruction(MalInstruction):
    def __init__(self, *args):
        MalInstruction.__init__(self, *args)
        self.arg1 = self.arg_list[0]
        self.arg2 = self.arg_list[1]

    def approxArgCnt(self, G):
        return [G.get(self.arg1.name,None),G.get(self.arg2.name,None)]

    def argCnt(self):
        return [self.arg1.cnt,self.arg2.cnt]

    def approxArgDiv(self, ins, G): #TODO rethink the order
        div = 1
        app = self.approxArgCnt(G)
        ac  = ins.argCnt()
        for (a1,a2) in zip(app,ac):
            if a1 != None:
                div = div * a1 / a2 #Hoping the order is correct...
        return div

    def approxArgDiv2(self, ins, G): #TODO rethink the order
        div = 1
        app = ins.approxArgCnt(G)
        ac  = self.argCnt()
        for (a1,a2) in zip(app,ac):
            if a1 != None:
                div = div * a1 / a2 #Hoping the order is correct...
        # print("div == ",div)
        return div

    def approxArgDist(self, ins, G):
        assert G != None
        lead_args = [self.arg1, self.arg2]
        self_cnt  = [float(G.get(a.name,None)) for a in lead_args]
        ins_count = [arg.cnt for arg in [ins.arg1,ins.arg2]]
        return sum([(c1-c2)**2 for (c1,c2) in zip(self_cnt,ins_count)])

    def extrapolate(self, other):
        return self.cnt

    def kNN(self, ilist, k, G):
        cand = [[i,self.approxArgDist(i,G)] for i in ilist] #TODO check for columns ???
        cand.sort( key = lambda t: t[1] )
        return [ t[0] for t in cand[0:k] ]

    def predictCount(self, traind, g):
        cand_l = traind.mal_dict[self.fname]
        knn5   = self.kNN(cand_l, 5, g)
        avg    = sum([ins.cnt for ins in knn5]) / len(knn5) #TODO add argdiv
        (i,c)  = (knn5[0].short,knn5[0].cnt)
        retv   = self.ret_args
        return [Prediction(retv=r.name, ins=i, cnt=c, avg=avg, t=r.atype) for r in retv]

class SelectInstruction(MalInstruction):
    def __init__(self, *def_args, jobj, stats):
        MalInstruction.__init__(self, *def_args)
        self.ctype    = jobj["arg"][0].get("type","UNKNOWN")#TODO fix
        alias_list    = [o["alias"] for o in jobj["arg"] if "alias" in o]
        self.col      = next(iter(alias_list),"TMP").split('.')[-1]
        self.arg_size = [o.get("size",0) for o in jobj.get("arg",[])]
        self.op       = Utils.extract_operator(self.fname, jobj)

        #is it the first select (number of C_ vars == 0)??...
        nC            = len([a for a in self.arg_list if a.name.startswith("C_")])
        self.lead_arg = self.arg_list[1] if nC>0 else self.arg_list[0]

        # print(method, op)
        lo, hi = Utils.hi_lo(self.fname, self.op, jobj, stats.get(self.col,Stats(0,0)))
        if self.ctype in ['bat[:int]','bat[:lng]','lng','bat[:hge]']:
            self.lo,self.hi    = (int(lo),int(hi))
        elif self.ctype == 'bat[:date]':
            # print(lo,hi)
            self.lo  = datetime.strptime(lo,'%Y-%m-%d')
            self.hi  = datetime.strptime(hi,'%Y-%m-%d')
        else:
            self.hi  = hi
            self.lo  = lo


    @staticmethod
    def removeDuplicates(ins_list):
        bounds_set = set()
        uniqs = []
        for ins in ins_list:
            if (ins.hi, ins.lo, ins.col) not in bounds_set:
                bounds_set.add((ins.hi, ins.lo, ins.col))
                uniqs.append(ins)

        return uniqs

    def isIncluded(self,other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]','bat[:lng]','bat[:date]','lng','bat[:hge]']:
            return self.lo >= other.lo and self.hi <= other.hi

        return None
    #
    def includes(self, other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]','bat[:lng]','bat[:date]','lng','bat[:hge]']:
            return self.lo <= other.lo and self.hi >= other.hi

        return None

    def approxArgCnt(self, G, default = None):
        return G.get(self.lead_arg.name,default)

    def argCnt(self):
        return self.lead_arg.cnt

    def approxArgDist(self, other, G):
        assert G != None
        lead_arg = self.lead_arg
        # print(G[lead_arg.name])
        ac = G[lead_arg.name] if lead_arg.name in G else 'inf'
        # approx_count = float(G.get(self.lead_arg.name,'inf'))
        return abs(other.lead_arg.cnt-float(ac))

    def argDist(self, other):
        assert False
        # assert len(self.arg_list) == len(other.arg_list)
        # print(self.arg_list[1].cnt, other.arg_list[1].cnt)
        diff = [abs(a.cnt-b.cnt) for (a,b) in zip(self.arg_list[1:2],other.arg_list[1:2])]
        return sum(diff)

    def argDiv(self, other):
        return other.lead_arg.cnt /self.lead_arg.cnt


    def distance(self, other):
        # print(self.col, other.col)
        assert self.ctype == other.ctype
        if self.includes(other) or self.isIncluded(other):
            if self.ctype in ['bat[:int]','bat[:lng]','lng','bat[:hge]']:
                return float((self.lo-other.lo)**2 + (self.hi-other.hi)**2)
            elif self.ctype == 'bat[:date]':
                (min_lo,max_lo) = (min(self.lo,other.lo),max(self.lo,other.lo))
                (min_hi,max_hi) = (min(self.hi,other.hi),max(self.hi,other.hi))
                return float((max_lo-min_lo).days + (max_hi-min_hi).days)
        else:
            return float('inf')
        return None

    def kNN(self, ilist, k, G):
        cand = [[i,self.distance(i)] for i in ilist if i.col == self.col and i.op == self.op and i.ctype == self.ctype]
        cand.sort( key = lambda t: t[1] )
        return [ t[0] for t in cand[0:k] ]

    #deprecated
    def predictCount2(self, ilist, default=0):
        knn = self.kNN(ilist, 1)
        if len(knn) >= 1:
            return self.extrapolate(knn[0])
        else:
            print("None found")
            return default


    def predictCount(self, traind, approxG, default=None):
        assert approxG != None
        self_list = traind.mal_dict.get(self.fname,[])
        self_list = SelectInstruction.removeDuplicates(self_list)
        nn        = self.kNN(self_list,5, approxG)
        rt        = self.ret_args[0].atype #return type

        if len(nn) == 0:
            logger.error("0 knn len in select ?? {} {} {}", self.short, self.op, self.ctype, self.col)
            # for i in self_list:
                # logger(i.short, i.fname, i.op, i.ctype, i.col)

        nn.sort( key = lambda ins: self.approxArgDist(ins, approxG))
        nn1       = nn[0]
        arg_cnt   = self.approxArgCnt(approxG)

        # if (verbose == True):
            # print("ArgumentEstimation: real: {} estimated: {}".format(test_i.argCnt(), test_i.approxArgCnt(approxG)))

        if arg_cnt != None:
            avg  = sum([i.extrapolate(self) * ( arg_cnt / i.argCnt()) for i in nn]) / len(nn)
            avgm = sum([i.ret_size for i in nn]) / len(nn)
            return [Prediction(retv = self.ret_vars[0], ins=nn1,cnt = nn1.extrapolate(self) * ( arg_cnt / nn1.argCnt()), avg=avg, t=rt, mem=avgm)]
        else:
            logging.error("None arguments ??? {}".format(self.lead_arg.name))
            avg = sum([i.extrapolate(self) for i in nn]) / len(nn)
            return [Prediction(retv = self.ret_vars[0], ins=nn1, cnt = nn1.extrapolate(self), avg = avg, t=rt)]

    def extrapolate(self, other):
        if self.ctype in ['bat[:int]','bat[:lng]','lng','bat[:hge]']:
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
        elif self.ctype == 'bat[:bit]':
            return self.cnt
        else:
            print("weird stuff in select", self.short)
            print("type ==",self.ctype,self.lo,self.hi)
            return None
