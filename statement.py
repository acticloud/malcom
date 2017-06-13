#@arg type: type of statement(assign, thetaselect etc)
#@arg time: how much time did the statement last
#@arg size: memory footprint
#@arg list: the arguments of the query (list for now TODO change)

def parse_stmt(stmt):
    # print(stmt)
    args = []
    if ":=" in stmt:
        fname = stmt.split(':=')[1].split("(")[0]
        args  = stmt.split(':=')[1].split("(")[1].split(")")[0].strip().split(" ")
    elif "(" in stmt:
        fname = stmt.split("(")[0]
        args  = stmt.split("(")[1].split(")")[0].strip().split(" ")
    else:
        fname = stmt

    return (fname.strip(),args)

class MalStatement:
    def __init__(self, short, stype, time, size, alist):
        self.stype    = stype
        self.time     = time
        self.size     = size
        self.arg_list = alist
        self.short    = short
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


    def cmp_arg_list(l1, l2):
        if(len(l1) != len(l2)):
            return False
        else:
            # cmp =
            return reduce(lambda x,y: x and y, [e1==e2 for (e1,e2) in zip(l1,l2)])

    def __eq__(self, other):
        if(self.stype == other.stype and cmp_arg_list(self.alist,other.alist) == True):
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
        self.name  = name
        self.atype = atype
        self.aval  = val
        self.size  = size

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
