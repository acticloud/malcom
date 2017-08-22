"""
@desc Column statistics class (for now only min and max)
"""
class Stats:
    def __init__(self, minval, maxval):
        self.min = minval
        self.max = maxval

    @staticmethod
    def fromStr(line):
        tokens = line.split("|")
        column = tokens[0].strip()
        minval = tokens[1].strip()
        maxval = tokens[2].strip()
        return (column,Stats(minval,maxval))

    def __str__(self):
        return "{:8} {:8}".format(self.min,self.max)

class ColumnStats:
    def __init__(self, cnt, minv, maxv, uniq):
        self.cnt  = int(cnt)
        self.minv = minv
        self.maxf = maxf
        self.uniq = int(uniq)

    @staticmethod
    def fromStr():
        tokens = line.split("|")
        column = tokens[0]
        return (column, *tokens[1::])

    def __str__(self):
        return "{:8} {:8} {:8} {}".format(self.cnt,self.min,self.max,self.uniq)
