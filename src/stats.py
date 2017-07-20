
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
