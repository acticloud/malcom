from functools import reduce

def parse_stmt(stmt):
    # print(stmt)
    args = []
    if ":=" in stmt:
        fname = stmt.split(':=')[1].split("(")[0]
        args  = stmt.split(':=')[1].split("(")[1].split(")")[0].strip().split(" ") #TODO remove
    elif "(" in stmt:
        fname = stmt.split("(")[0]
        args  = stmt.split("(")[1].split(")")[0].strip().split(" ")
    else:
        fname = stmt

    return (fname.strip(),args)

def cmp_arg_list(l1, l2):
    if(len(l1) != len(l2)):
        return False
    else:
        return reduce(lambda x,y: x and y, [e1==e2 for (e1,e2) in zip(l1,l2)], True)

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
        self.metric   = Metric.fromMalStatement(self.stype,self.arg_list)

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



    def __eq__(self, other):
        if(self.stype == other.stype and cmp_arg_list(self.arg_list,other.arg_list) == True):
            return True
        else:
            return False

    def __ne__(self, other):
        return self.__ne__(other)
""" Arg class """
#@attr atype: String
#@attr aval : Object
class Arg:
    def __init__(self, name, atype, val, size):
        self.name   = name
        self.atype  = atype
        self.aval   = val
        self.size   = size
    @staticmethod
    def fromJsonObj(jobj):
        # pprint(jobj)
        name  = jobj['name']
        atype = jobj['type']
        aval  = jobj.get('value',None)
        size  = jobj.get('size',0)
        return Arg(name,atype,aval,size)

    def __eq__(self, other):
        if (self.name  == other.name  and
            self.atype == other.atype and
            self.aval  == other.aval  and
            self.size  == other.size):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

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
    def fromMalStatement(sname, arg_list):
        if(sname == 'thetaselect'):
            print("thetaselect found")
            if(len(arg_list) == 4):
                return None
            else if(len(arg_list) == 3):
                return Metric(sname, arg_list[3].val, arg_list[2].atype, arg_list[2].val)
            else:
                print("wtf")
                return None
        else:
            return None
