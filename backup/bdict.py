import copy
import random

class BDict:
    def __init__(self, d):
        self.bd = d

    def printStdout(self):
        for l in self.bd.values():
            for i in l:
                print(i.toStr())

    def saveToFile(self, fname):
        with open(fname,'w') as f:
            for i in self.values():
                f.write(i.toStr())

    def getInsList(self):
        il = []
        for l in self.bd.values():
            il.extend(l)
        return il
    # def predictMem(self,ins):
    #     cand = [i for i in self.bd.get(ins.method,[]) if i.col == ins.col and i.op == ins.op]
    #     return None #TODO fill this

    def kNN(self, ins):
        cand = [[i,i.distance(ins)] for i in self.bd.get(ins.method,[]) if i.col == ins.col and i.op == ins.op]
        mins = min(cand, key=lambda ins: ins[1])
        return mins[0]

    def predictCount(self, ins, default=0):
        if ins.method in ['select','thetaselect']:
            cand = [[i,i.distance(ins)] for i in self.bd.get(ins.method,[]) if i.col == ins.col and i.op == ins.op]
            if len(cand)==0:
                return default
            knn = min(cand, key=lambda ins: ins[1])[0]
            return knn.extrapolate(ins)
        elif ins.method in ['join']:
            print("What do we do with joins ???")
            return None

    def avgAcc(self, test_set):
        self_list = self.getInsList()
        test_list = test_set.getInsList()
        non_zeros = [i for i in test_list if i.cnt > 0]
        acc = [1 for i in non_zeros if abs(self.predictCount(i)-i.cnt)/i.cnt < 0.1]
        # print(len(acc))
        return float(sum(acc))/len(non_zeros)

    def printPredictions(self, test_set):
        self_list = self.getInsList()
        test_list = test_set.getInsList()
        non_zeros = [i for i in test_list if i.cnt > 0]
        for test_i in non_zeros:
            pred = self.predictCount(test_i)
            # print(test_i.short)
            # print(pred.short)
            print("{} {:10.0f} {:10.0f} {:4.1f}".format(test_i.op, pred,test_i.cnt, abs(self.predictCount(test_i)-test_i.cnt)/test_i.cnt))

    def filter(self, f):
        newd = copy.deepcopy(self.bd)
        for (k,l) in newd.items():
            newd[k] = list([ins for ins in l if f(ins) == True])

        return BDict(newd)

    def randomSplit(self, p):
        s1,s2 = {}, {}
        testp = 1.0 - p
        for (k,l) in self.bd.items():
            for ins in l:
                r = random.random()
                if r < p:
                    s1[k] = s1.get(k,[]) + [ins]
                else:
                    assert r + testp >= 1
                    s2[k] = s2.get(k,[]) + [ins]
        return (BDict(s1),BDict(s2))
    #TODO load from file
