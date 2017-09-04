from utils import Utils
from stats import Stats
from datetime import datetime


class BetaIns:
    def __init__(self, tag, short, method, column, col_type, operator, arg_size, hi, lo, cnt):
        self.tag      = tag
        self.short    = short
        self.method   = method
        self.col      = column
        self.ctype    = col_type
        self.op       = operator
        self.cnt      = cnt
        self.arg_size = arg_size
        if self.ctype in ['bat[:int]','bat[:lng]','lng']:
            self.lo,self.hi    = (int(lo),int(hi))
        elif self.ctype == 'bat[:date]':
            # print(lo,hi)
            self.lo  = datetime.strptime(lo,'%Y-%m-%d')
            self.hi  = datetime.strptime(hi,'%Y-%m-%d')
        else:
            self.hi     = hi
            self.lo     = lo

    @staticmethod
    def fromJsonObj(jobj, method, stats):
        # print(jobj["short"])
        count  = int(jobj["ret"][0].get("count",0))
        ctype  = jobj["arg"][0].get("type","UNKNOWN")
        column = next(iter([o["alias"] for o in jobj["arg"] if "alias" in o]),"TMP").split('.')[-1]
        arg_size = [o.get("size",0) for o in jobj.get("arg",[])]
        # column = jobj["arg"][0].get("alias","TMP").split('.')[-1]
        op     = Utils.extract_operator(method, jobj)
        # print(method, op)
        lo, hi = Utils.hi_lo(method, op, jobj, stats.get(column,Stats(0,0)))
        # print("HI,LO: ",hi,lo)
        return BetaIns(jobj["tag"],jobj["short"], method, column,ctype, op, arg_size, hi, lo, count)

    def toStr(self):
        return "{:15} {:30} {:10} {:10} {} {} {:10}".format(
            self.method, self.col, self.op, self.ctype, self.lo, self.hi, self.cnt
        )

    def isIncluded(self,other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]','bat[:lng]','bat[:date]','lng']:
            return self.lo >= other.lo and self.hi <= other.hi

        return None
    #
    def includes(self, other):
        assert self.ctype == other.ctype
        t = self.ctype
        if t in ['bat[:int]','bat[:lng]','bat[:date]','lng']:
            return self.lo <= other.lo and self.hi >= other.hi

        return None

    def distance(self, other):
        assert self.ctype == other.ctype
        if self.includes(other) or self.isIncluded(other):
            if self.ctype in ['bat[:int]','bat[:lng]','lng']:
                return float((self.lo-other.lo)**2 + (self.hi-other.hi)**2)
            elif self.ctype == 'bat[:date]':
                (min_lo,max_lo) = (min(self.lo,other.lo),max(self.lo,other.lo))
                (min_hi,max_hi) = (min(self.hi,other.hi),max(self.hi,other.hi))
                return float((max_lo-min_lo).days + (max_hi-min_hi).days)
        else:
            return float('inf')
        return None

    def extrapolate(self, other):
        if self.ctype in ['bat[:int]','bat[:lng]','lng']:
            self_dist  = self.hi  - self.lo
            other_dist = other.hi - other.lo

            if self_dist*other_dist != 0:
                return self.cnt*other_dist/other_dist
            else:
                return self.cnt
        elif self.ctype == 'bat[:date]':
            diff1 = (other.hi-other.lo)
            diff2 = (self.hi-self.lo)

            if diff1.days * diff2.days != 0:
                return self.cnt * (diff1.days / diff2.days)
            else:
                return self.cnt
        elif self.ctype == 'bat[:str]':
            return self.cnt
        else:
            print("type ==",self.ctype,self.lo,self.hi)
            return None
