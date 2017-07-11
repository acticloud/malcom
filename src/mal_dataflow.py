
#more like a varflow
class Dataflow:
    def __init__(self):
        self.dict = {}

    def add(self, tag, var, table):
        assert var.startswith("X_") or var.startswith("C_")
        self.dict[tag] = self.dict.get(tag,{})
        self.dict[tag][var] = table
        # print("{} {}".format(var,table))

    def addI(self, tag, arg_list, ret_list):
        # deps = [self.lookup(arg,tag) for arg in arg_list]
        flatl = set()
        # print(arg_list)
        for arg in arg_list:
            for a in self.lookup(arg,tag,default=[]):
                # print("Ilookup: ",a)
                flatl.add(a)

        for r in ret_list:
            # print("added to dic: {} {}".format(r,flatl))
            self.add(tag,r,list(flatl))

    def addIgnore(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            d = self.dict.get(tag,{})
            d[var] = table
            self.dict[tag] = d

    def lookup(self, var, tag, default=None):
        # print("looking for {}".format(var))
        assert self.dict.get(tag,None) != None
        return self.dict.get(tag,{}).get(var,default)
