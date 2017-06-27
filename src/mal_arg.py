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
        size  = int(jobj.get('size',0))
        return Arg(name,atype,aval,size)

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
