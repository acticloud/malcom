from utils import Utils

SIZE_OF_INT   = 4
SIZE_OF_BOOL  = 1
SIZE_OF_FLOAT = 8
SIZE_OF_DATE  = 8
class BetaIns:
    def __init__(self, short, method, column, col_type, operator, hi, lo, cnt):
        self.short  = short
        self.method = method
        self.column = column
        self.ctype  = col_type
        self.op     = operator
        self.hi     = hi
        self.low    = low
        self.cnt    = cnt

    @staticmethod
    def fromJsonObj(jobj, method, varflow):
        count  = int(jobj["count"])
        ctype  = jobj["arg"][0].get("type","UNKNOWN")
        column = jobj["arg"][0].get("alias","TMP")
        op     = Utils.extract_operator(method, jobj)
