
"""
DAG of the relationship between vars and columns for each query
"""
class Dataflow:
    def __init__(self):
        self.dict = {}

    """ single add: one var one table-column to the dag"""
    def add(self, tag, var, table):
        assert var.startswith("X_") or var.startswith("C_")
        self.dict[tag] = self.dict.get(tag,{})
        self.dict[tag][var] = table

    """
    add an instruction to the DAG:
    for each return var r: dag[r] = lookup(args)
    """
    def addI(self, tag, arg_list, ret_list):
        flatl = set()
        for arg in arg_list:
            for a in self.lookup(arg,tag,default=[]):
                flatl.add(a)

        for r in ret_list:
            self.add(tag,r,list(flatl))

    def lookup(self, var, tag, default=None):
        assert self.dict.get(tag,None) != None
        return self.dict.get(tag,{}).get(var,default)

    #deprecated
    def addIgnore(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            d = self.dict.get(tag,{})
            d[var] = table
            self.dict[tag] = d
