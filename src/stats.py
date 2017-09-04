"""
@desc Column statistics class
@attr cnt : int //number of column elements
@attr minv: obj //minimum value of column
@attr maxf: obj //maximum value of column
@attr uniq: int //number of unique elements
@attr wid : int //width of column
"""
class ColumnStats:
    def __init__(self, width, minv, maxv, count, uniq):
        self.cnt  = int(count)
        self.minv = minv
        self.maxv = maxv
        self.uniq = int(uniq)
        self.wid  = int(width)

    @staticmethod
    def fromStr(line):
        tokens = [t.strip() for t in line.split("|")]
        column = tokens[0]
        return (column, ColumnStats(*tokens[1::]))

    """@ret dict<str, ColumnStats> """
    @staticmethod
    def fromFile(file_name):
        d = []
        for line in open(file_name).readlines():
            d.append(ColumnStats.fromStr(line))
        return dict(d)

    def __str__(self):
        return "{:8} {:8} {:8} {}".format(self.cnt,self.min,self.max,self.uniq)

class ColumnStatsD:
    def __init__(self, d):
        self.statsd = d

    @staticmethod
    def fromFile(fname):
        d = []
        for line in open(fname).readlines():
            d.append(ColumnStats.fromStr(line))
        return ColumnStatsD(dict(d))

    def __getitem__(self,c):
        return self.statsd[c]
