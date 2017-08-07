
"""
DAG of the relationship between vars and columns for each query
"""
class Dataflow:
    def __init__(self, d = {}):
        self.dict = d

    """ single add: one var one table-column to the dag"""
    def add(self, tag, var, ins):
        assert var.startswith("X_") or var.startswith("C_")
        self.dict[tag]          = self.dict.get(tag,{})
        if ins in ['tid']:
            self.dict[tag][var] = ins.cnt
        else:
            self.dict[tag][var] = ins

    def lookup(self, var, tag, default=None):
        assert self.dict.get(tag,None) != None
        return self.dict.get(tag,{}).get(var,default)

    #deprecated
    def addIgnore(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            d = self.dict.get(tag,{})
            d[var] = table
            self.dict[tag] = d

    def union(self, other):
        return Dataflow({**self.dict, **other.dict})
