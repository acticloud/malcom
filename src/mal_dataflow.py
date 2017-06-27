
#more like a varflow
class Dataflow:
    def __init__(self):
        self.dict = {}

    def add(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            d = self.dict.get(tag,{})
            d[var] = table
        else:
            raise  Exception("K3y3rr0r")# coding=utf-8

    def addIgnore(self, tag, var, table):
        if var.startswith("X_") or var.startswith("C_"):
            d = self.dict.get(tag,{})
            d[var] = table
            self.dict[tag] = d

    def lookup(self, tag, var):
        return self.dict.get(tag,{}).get(var,None)
