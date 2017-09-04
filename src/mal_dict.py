import sys
import json
import random
import pickle
import logging
from utils        import Utils
from utils        import Prediction
from mal_instr    import MalInstruction
from mal_instr    import SelectInstruction
#TODO add method find closest instruction

class MalDictionary:
    """
    @arg mal_dict: dict<List<MalInstruction>>
    @arg q_tags  : list<int> //list of the unique query tags
    @arg varflow: dic<tag,dic<var,table>>
    """
    def __init__(self, mal_dict, q_tags, col_stats={}):
        self.mal_dict   = mal_dict
        self.query_tags = q_tags

    @staticmethod
    def loadFromFile(file_name):
        with open(file_name,'rb') as f:
            obj = pickle.load(f)
            return obj

    def writeToFile(self, file_name):
        with open(file_name,'wb') as f:
            pickle.dump(self, f)


    def union(self, other):
        union_ilist   = self.getInsList() + other.getInsList()
        return MalDictionary.fromInsList(union_ilist)

    """
    @arg mfile    : str                   //json file containing query run
    @arg blacklist: list<str>             //list of blacklisted mal ins
    @arg col_stats: dict<str,ColumnStats> //column statistics
    """
    @staticmethod
    def fromJsonFile(mfile, blacklist, col_stats):
        with open(mfile) as f:
            maldict    = {}
            startd     = {}
            query_tags = set()

            while 1: #while not EOF
                jobj = Utils.readJsonObject(f)
                if jobj is None:
                    break
                fname,args ,ret  = Utils.extract_fname(jobj["short"])

                if not Utils.is_blacklisted(blacklist,fname):
                    if jobj["state"] == "start":
                        startd[jobj["pc"]] = jobj["clk"]
                    elif jobj["state"] == "done":
                        assert jobj["pc"] in startd
                        new_mals       = MalInstruction.fromJsonObj(jobj, col_stats)
                        new_mals.time  = int(jobj["clk"])-int(startd[jobj["pc"]])
                        new_mals.start = int(startd[jobj["pc"]])
                        maldict[fname] = maldict.get(fname,[]) + [new_mals]
                        query_tags.add(int(jobj["tag"]))

        return MalDictionary(maldict,list(query_tags), col_stats)

    """ Constructor from instruction list
    @arg ilist: List<MalInstruction>
    @arg varflow
    @ret MalDictionary
    """
    @staticmethod
    def fromInsList(ilist):
        mdict = {}
        qtags = set()
        for i in ilist:
            mdict[i.fname] = mdict.get(i.fname,[]) + [i]
            qtags.add(i.tag)
        return MalDictionary(mdict, list(qtags))

    """
    @des builds a graph that approximates the count for each variable
    @arg Maldictionary
    @ret dict<str,int> //dictionary with var name as a key,est count as val
    #TODO rename predictionGraph
    """
    def buildApproxGraph(self, traind):
        ilist = self.getInsList()
        ilist.sort( key = lambda ins: ins.clk )
        pg = {}
        for ins in ilist:
            for p in ins.predictCount(traind, pg):
                pg[p.retv] = p
                r          = p.retv
                if p.avg == sys.maxsize or p.avg == None:
                    logging.error("None in the graph: {}".format(ins.short))
                # logging.debug("m: {:20} {:5} {:10.0f}".format(ins.fname,r,pg[r]))
        return pg



    def getFirstI(self, field, N):
            ilist = self.getInsList()
            ilist.sort(key = lambda ins: getattr(ins, field) )
            return MalDictionary.fromInsList(ilist[0:N])
    """
    @arg mals: MalInstruction
    @ret: List<MalInstriction> //list of all exact matches
    """
    def findInstr(self, mals,ignoreScale=False):
        dic = self.mal_dict
        if not mals.fname in dic:
             return []
        if ignoreScale == False:
            return [x for x in dic[mals.fname] if x == mals]
        else:
            return [x for x in dic[mals.fname] if x.isExact(mals,True)]
        return ret

    """ @desc returns a list of all the instructions """
    def getInsList(self):
        ilist = []
        for l in self.mal_dict.values():
            ilist.extend(l)
        return ilist

    """
    !!Assumes we know each instruction's memory footprint!!
    @ret int //actual max bytes the query will allocate
    """
    def getMaxMem(self):
        ilist = self.getInsList()
        ilist.sort(key = lambda i: i.clk)
        max_mem  = 0
        curr_mem = 0
        for i in ilist:
            max_mem  = max(max_mem,curr_mem + i.mem_fprint)
            curr_mem = curr_mem + i.ret_size - i.free_size

        return max_mem

    """
    @arg pG: dict<str, Prediction> //graph that relates each var to a prediction
    @ret int //actual max bytes the query will allocate
    """
    def predictMaxMem(self, pG): #TODO fix this
        ilist = self.getInsList()
        ilist.sort(key = lambda i: i.clk)
        max_mem  = 0
        curr_mem = 0
        for i in ilist:
            max_mem  = max(max_mem,curr_mem + i.approxMemSize(pG))
            curr_mem = curr_mem + i.ret_size - i.approxFreeSize(pG)

        return max_mem


    """
    @arg mals: string //method name
    @arg nags: int    //nof arguments
    @ret: list<MalInstruction>
    """
    def findMethod(self, fname, nargs=None):
        dic = self.mal_dict
        if nargs == None:
            return dic[fname]

        return [x for x in dic[fname] if len(x.arg_list) == nargs]



    def getAll(self, method, nargs):
        d = self.mal_dict
        return filter(
            lambda s: s.fname == method and len(s.arg_list) == nargs, d[method]
        )

    """
    @arg f: lamdba k: MalInstruction //comparison metric
    @ret: list of topN instuctions
    """
    def getTopN(self, f, n):
        mal_list = Utils.flatten(self.mal_dict)
        mal_list.sort(key = f)
        return mal_list[0:n]

    """
    @arg f: lambda i: MalInstruction -> boolean
    retain only the mal instructions that satisfy the given function
    """
    def filter(self, f):
        mal_list = self.getInsList()
        new_ilist = list([i for i in mal_list if f(i) == True])
        return MalDictionary.fromInsList(new_ilist)

    """
    @desc splits the dictionary in two based on the query tags
    @arg: list<int>
    """
    def splitTag(self, train_tags, test_tags):
        s1,s2 = {},{}

        for (k,l) in self.mal_dict.items():
            for mali in l:
                if mali.tag in train_tags:
                    s1[k] = s1.get(k,[]) + [mali]
                if mali.tag in test_tags:
                    # assert not mali.tag in train_tags
                    s2[k] = s2.get(k,[]) + [mali]

        return (MalDictionary(s1,train_tags,self.varflow), MalDictionary(s2,test_tags,self.varflow))

    """
    @desc splits the dictionary in two based on the query id
    @arg: list<int>
    @arg tag2query: dict<int,int>
    """
    def splitQuery(self, train_q, test_q, tag2query):
        s1,s2 = {},{}
        s1_tags, s2_tags = set(), set()

        for (k,l) in self.mal_dict.items():
            for mali in l:
                if tag2query[mali.tag] in train_q:
                    s1[k] = s1.get(k,[]) + [mali]
                    s1_tags.add(mali.tag)
                if tag2query[mali.tag] in test_q:
                    s2[k] = s2.get(k,[]) + [mali]
                    s2_tags.add(mali.tag)

        return (MalDictionary(s1,list(s1_tags),self.varflow), MalDictionary(s2,list(s2_tags),self.varflow))

    """
    @desc splits the dictionary in two based on the query tags
    @arg: list<int>
    """
    def splitRandom(self, trainp):
        # assert trainp + testp == 1.0
        s1,s2 = {}, {}
        s1_tags, s2_tags = set(), set()

        for (k,l) in self.mal_dict.items():
            for mali in l:
                r = random.random()
                if r < trainp:
                    s1[k] = s1.get(k,[]) + [mali]
                    s1_tags.add(mali.tag)
                else:
                    # assert r + testp >= 1
                    s2[k] = s2.get(k,[]) + [mali]
                    s2_tags.add(mali.tag)
        return (MalDictionary(s1,list(s1_tags),self.varflow), MalDictionary(s2,list(s2_tags),self.varflow))

        """ selects sth"""
    def select(self, fun, perc):
        ilist = self.getInsList()
        ilist.sort(key = fun)#bda ins: -ins.mem_fprint)
        bestp = ilist[0:int(perc*len(ilist))]
        d = {}
        tags = set()
        for i in bestp:
            d[i.fname] = d.get(i.fname,[]) + [i]
            tags.add(i.tag)
        return MalDictionary(d,tags,self.varflow)


    def avgDeviance(self, test_dict):
        suml  = lambda x,y: x+y
        diff  = sum( map(lambda ins: abs(ins.mem_fprint-self.predictMem(ins,0)),test_dict.getInsList()) )
        total = sum( map(lambda ins: ins.mem_fprint,test_dict.getInsList()) )
        # print("dev")
        return 100 * diff / total

    def avgError(self, test_dict):
        suml   = lambda x,y: x+y
        test_l = test_dict.getInsList()
        ldiff = lambda i: abs(i.mem_fprint-self.predictMem(i,0)) / i.mem_fprint if i.mem_fprint != 0 else 0.0
        diff  = sum( map(ldiff,test_l) )
        # print(list(map(ldiff,test_l)))
        # print(max(list(map(ldiff,test_l))))
        # print("dev")
        return 100 * diff / len(test_l)

    def avgMemAcc(self, test_set, thres):
        self_list = self.getInsList()
        test_list = test_set.getInsList()
        non_zeros = [i for i in test_list if i.mem_fprint > 0]
        acc = [1 for i in non_zeros if abs(self.predictMem(i,0)-i.mem_fprint)/i.mem_fprint < thres]

        return 100*float(sum(acc))/len(non_zeros)

    def avgCountAcc(self, test_set, thres):
        self_list = self.getInsList()
        test_list = test_set.getInsList()
        non_zeros = [i for i in test_list if i.cnt > 0]
        acc = [1 for i in non_zeros if abs(i.predictCount(self_list,0)-i.cnt)/i.cnt < thres]

        return 100*float(sum(acc))/len(non_zeros)
