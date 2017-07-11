"""
Arg class
@attr name : str //the name of the arg
@attr atype: str //the type of the argument
@attr aval : obj //arg value
@attr size : int //argument memory size
@attr eol  : int //end of life (if == 1 the arg is freed by the server)
"""
class Arg:
    def __init__(self, name, atype, val, size, eol):
        self.name   = name
        self.atype  = atype
        self.aval   = val
        self.size   = size
        self.eol    = eol

    @staticmethod
    def fromJsonObj(jobj):
        name  = jobj['name']
        atype = jobj['type']
        aval  = jobj.get('value',None)
        size  = int(jobj.get('size',0))
        eol   = int(jobj["eol"])
        return Arg(name,atype,aval,size,eol)

    def isVar(self):
        return self.name.startswith("X_") or self.name.startswith("C_")

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
