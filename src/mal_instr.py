from utils import Utils
from mal_arg import Arg
from functools import reduce
import re


""" MalInstruction Class:
@arg type: string    // type of statement(assign, thetaselect etc)
@arg time: float     // instruction duratin(microseconds)
@arg size: int       // size field
@arg list: List<Arg>)// instruction arguments (list for now TODO change)
@arg short: string   // the short mal statement, str representation
@arg tag: int        // the query identifier
@arg arg_size: int   // total argument size(bytes)
@var metric: Metric  // var that can define a distance between two queries
"""
class MalInstruction:
    def __init__(self, pc, short, fname, size, ret_size, tag, arg_size, alist, free_size):
        self.pc         = pc
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
        self.metric     = Metric.fromMalInstruction(self.fname,self.arg_list)#TODO remove

    @staticmethod
    def fromJsonObj(jobj):
        size      = int(jobj["size"])
        pc        = int(jobj["pc"])
        short     = jobj["short"]
        fname,_,_ = Utils.extract_fname(jobj["short"])
        tag       = int(jobj["tag"])
        rv        = [rv.get("size",0) for rv in jobj["ret"]]
        sumf      = lambda x,y: x+y
        ret_size  = sum([o.get("size",0) for o in jobj.get("ret",[])])
        arg_size  = sum([o.get("size",0) for o in jobj.get("arg",[])])
        arg_list  = [Arg.fromJsonObj(e) for e in jobj.get("arg",[])]
        free_size = sum([arg.size for arg in arg_list if arg.eol == 1])

        return MalInstruction(pc, short, fname, size, ret_size, tag, arg_size, arg_list, free_size)

    #deprecated
    def distance(self,other):
        return self.metric.distance(other.metric) #TODO fix this

    def argDist(self, other):
        assert len(self.arg_list) == len(other.arg_list)
        diff = [abs(a.size-b.size) for (a,b) in zip(self.arg_list,other.arg_list)]
        return reduce(lambda x,y: x+y, diff, 0)

    def similarity(self, other):
        assert len(self.arg_list) == len(other.arg_list)
        total_self  = max(sum([a.size for a in self.arg_list]),0.001)
        total_other = max(sum([a.size for a in other.arg_list]),0.001)
        return total_self / total_other


    def argListStr(self):
        # slist = ["arg: {} {} ".format(a.name, int(a.size / 1000)) for a in self.arg_list if a.size>0]
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


    def print_short(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.fname,self.nargs, self.time, self.mem_fprint))

    def print_verbose(self):
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


#TODO rename Metric, wtf name is this ?
#TODO maybe input output operator: Arg format ???
"""
@arg itype: intstuction type: String
@arg value:
@arg op: type of operator(e.g thetaselect)
@arg vtype: type of value(short, int, boolean, date...)
"""
class Metric:
    def __init__(self, itype, op, vtype, value):
        self.itype = itype #maybe remove??
        self.op    = op
        self.vtype = vtype
        self.value = value

    def distance(self, other):
        if( self.itype == other.itype and
            self.op == other.op and
            self.vtype == other.vtype):
            if(self.vtype == "int"):
                return float((other.value-self.value) ** 2)
        else:
            return float("inf")

    @staticmethod
    def fromMalInstruction(sname, arg_list):
        if(sname == 'thetaselect'):
            if(len(arg_list) == 4):
                return None
            elif len(arg_list) == 3:
                # print("thetaselect found {} {} {} {}".format(sname, arg_list[2].aval, arg_list[1].atype, arg_list[1].aval))
                return Metric(sname, arg_list[2].aval, arg_list[1].atype, arg_list[1].aval)
            else:
                print("wtf")
                return None
        else:
            return None
