from utils import Utils
from mal_arg import Arg
from functools import reduce

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
    def __init__(self, short, fname, size, ret_size, tag, arg_size, alist):
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
        self.metric     = Metric.fromMalInstruction(self.fname,self.arg_list)#TODO rethink

    #deprecated
    def distance(self,other):
        return self.metric.distance(other.metric) #TODO fix this

    def argDist(self, other):
        assert len(self.arg_list) == len(other.arg_list)
        diff = [abs(a.size-b.size) for (a,b) in zip(self.arg_list,other.arg_list)]
        return reduce(lambda x,y: x+y, diff, 0)

    @staticmethod
    def fromJsonObj(jobj):
        size     = int(jobj["size"])
        short    = jobj["short"]
        fname    = Utils.extract_fname(jobj["short"])
        tag      = int(jobj["tag"])
        rv       = [rv.get("size",0) for rv in jobj["ret"]]
        sumf     = lambda x,y: x+y
        ret_size = Utils.sumJsonList(jobj["ret"],"size")
        arg_size = Utils.sumJsonList(jobj["arg"],"size")
        arg_list = [Arg.fromJsonObj(e) for e in jobj["arg"]]

        return MalInstruction(short, fname, size, ret_size, tag, arg_size, arg_list)

    def print_short(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.fname,self.nargs, self.time, self.mem_fprint))

    def print_verbose(self):
        fmt = "Instr: {} nargs: {} time: {} mem_fprint: {}"
        print(fmt.format(self.short,self.nargs, self.time, self.mem_fprint))

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
