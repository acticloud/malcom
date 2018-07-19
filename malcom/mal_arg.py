class Arg:
    """
    Arg class
    @attr name : str //the name of the arg
    @attr atype: str //the type of the argument
    @attr aval : obj //arg value
    @attr size : int //argument memory size
    @attr eol  : int //end of life (if == 1 the arg is freed by the server)
    @attr col  : str //name of column (or TMP if None)
    @attr cnt  : int //number of elements
    """
    def __init__(self, name, atype, val, size, eol, count, col):
        self.name = name
        self.atype = atype
        self.aval = val
        self.size = size
        self.eol = eol
        self.cnt = count
        self.col = col

    @staticmethod
    def fromJsonObj(jobj):
        name = jobj['name']
        atype = jobj['type']
        aval = jobj.get('value', None)
        size = int(jobj.get('size', 0))
        eol = int(jobj["eol"])
        count = int(jobj.get("count", 0))
        col = jobj.get("alias", 'TMP')
        return Arg(name, atype, aval, size, eol, count, col)

    def isVar(self):
        return (self.name.startswith("X_") or self.name.startswith("C_")) and self.aval is None

    def __eq__(self, other):
        return bool(self.name == other.name and self.atype == other.atype and self.aval == other.aval and self.size == other.size)

    def __ne__(self, other):
        return not self.__eq__(other)
