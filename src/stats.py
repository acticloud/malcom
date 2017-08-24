"""
@desc Column statistics class (for now only min and max)
"""
# class Stats:
#     def __init__(self, minval, maxval):
#         self.min = minval
#         self.max = maxval
#
#     @staticmethod
#     def fromStr(line):
#         tokens = line.split("|")
#         column = tokens[0].strip()
#         minval = tokens[1].strip()
#         maxval = tokens[2].strip()
#         return (column,Stats(minval,maxval))
#
#     def __str__(self):
#         return "{:8} {:8}".format(self.min,self.max)


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
