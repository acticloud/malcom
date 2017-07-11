from utils import Utils
from mal_arg import Arg
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
"""
class MalInstruction:
    def __init__(self, pc, short, fname, size, ret_size, tag, arg_size, alist, free_size, arg_vars, ret_vars):
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
        self.arg_vars   = arg_vars
        self.ret_vars   = ret_vars

    @staticmethod
    def fromJsonObj(jobj):
        size      = int(jobj["size"])
        pc        = int(jobj["pc"])
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
        return MalInstruction(pc, short, fname, size, ret_size, tag, arg_size, arg_list, free_size, arg_vars, ret_vars)

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
