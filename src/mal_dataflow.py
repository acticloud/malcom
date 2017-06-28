
#more like a varflow
class Dataflow:
    def __init__(self):
        self.dict = {}

    def add(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            self.dict[tag] = self.dict.get(tag,{})
            self.dict[tag][var] = table
            # print("added {} {} {}".format(var,tag,table))
        else:
            raise  Exception("K3y3rr0r")# coding=utf-8

    def addIgnore(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            d = self.dict.get(tag,{})
            d[var] = table
            self.dict[tag] = d

    def lookup(self, var, tag):
        # print(tag)
        # print(self.dict.keys())
        assert self.dict.get(tag,None) != None
        return self.dict.get(tag,{}).get(var,None)
