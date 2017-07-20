from utils import Utils
from stats import Stats
from datetime import datetime
SIZE_OF_INT   = 4
SIZE_OF_BOOL  = 1
SIZE_OF_FLOAT = 8
SIZE_OF_DATE  = 8


class BetaIns:
    def __init__(self, short, method, column, col_type, operator, hi, lo, cnt):
        self.short  = short
        self.method = method
        self.col    = column
        self.ctype  = col_type
        self.op     = operator
        self.hi     = hi
        self.lo     = lo
        self.cnt    = cnt

    @staticmethod
    def fromJsonObj(jobj, method, stats):
        count  = int(jobj["ret"][0]["count"])
        ctype  = jobj["arg"][0].get("type","UNKNOWN")
        column = jobj["arg"][0].get("alias","TMP").split('.')[-1]
        op     = Utils.extract_operator(method, jobj)
        # print(method, op)
        lo, hi = Utils.hi_lo(method, op, jobj, stats.get(column,Stats("UNK","UNK")))
        # print("HI,LO: ",hi,lo)
        return BetaIns(jobj["short"], method, column,ctype, op, hi, lo, count)

    def toStr(self):
        return "{:15} {:30} {:10} {:10} {:25} {:25} {:10}".format(
            self.method, self.col, self.op, self.ctype, self.lo, self.hi, self.cnt
        )

    def isIncluded(self,other):
        assert self.ctype == other.ctype
        if self.ctype == 'bat[:int]' or self.ctype == 'bat[:lng]':
            self_lo,self_hi    = (long(self.lo),long(self.lo))
            other_lo, other_hi = (long(other.lo),long(other.lo))

            if self_lo >= other_lo and self_hi <= other_hi:
                return True
            else:
                return False
        elif self.ctype == 'bat[:date]':
            self_lo,self_hi    = (datetime.strptime(self.lo,'%y-%m-%d'), datetime.strptime(self.hi,'%y-%m-%d'))
            other_lo, other_hi = (datetime.strptime(other.lo,'%y-%m-%d'),datetime.strptime(other.hi,'%y-%m-%d'))
            if self_lo >= other_lo and self_hi <= other_hi:
                return True
            else:
                return False

        return None

    def includes(self, other):
        assert self.ctype == other.ctype
        if self.ctype == 'bat[:int]' or self.ctype == 'bat[:lng]':
            self_lo,self_hi    = (long(self.lo),long(self.lo))
            other_lo, other_hi = (long(other.lo),long(other.lo))

            if self_lo <= other_lo and self_hi >= other_hi:
                return True
            else:
                return False
        elif self.ctype == 'bat[:date]':
            self_lo,self_hi    = (datetime.strptime(self.lo,'%y-%m-%d'),datetime.strptime(self.hi,'%y-%m-%d'))
            other_lo, other_hi = (datetime.strptime(other.lo,'%y-%m-%d'),datetime.strptime(other.hi,'%y-%m-%d'))
            if self_lo <= other_lo and self_hi >= other_hi:
                return True
            else:
                return False

        return None

    def distance(self, other):
        assert self.ctype == other.ctype
        if self.ctype == 'bat[:int]' or self.ctype == 'bat[:lng]':
            if self.includes(other) or self.isIncluded(other):
                return float((other.lo-other.lo)**2 + (self.hi-other.hi)**2)
            else:
                return float('inf')
        return None

    def extrapolate(self, other):
        return None
