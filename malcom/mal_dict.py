import gzip
import sys
import random
import pickle
import logging

from collections import defaultdict

from malcom.utils import Utils
from malcom.mal_instr import MalInstruction

def _make_list():
    return []

class MalDictionary:
    """
    @arg mal_dict: dict<List<MalInstruction>>
    @arg q_tags  : list<int> //list of the unique query tags
    @arg varflow: dic<tag,dic<var,table>>
    """
    def __init__(self, mal_dict, q_tags, col_stats={}):
        self.mal_dict = mal_dict
        self.query_tags = q_tags
        self.ins_list = [i for instr_list in mal_dict.values() for i in instr_list ]

    @staticmethod
    def loadFromFile(file_name):
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
            return obj

    def writeToFile(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    # add info. from another dictionary into this one, just a simple append, no
    #   duplication elimination
    def union(self, *others):
        dict_list = [self, *others]
        union_ilist = [ i for d in dict_list for i in d.getInsList() ]
        return MalDictionary.fromInsList(union_ilist)

    @staticmethod
    def fromJsonFile(mfile, blacklist, col_stats):
        """
        @des Construct a MalDictionary object from a JSON file
        @arg mfile    : str                   //json file containing query run
        @arg blacklist: list<str>             //list of blacklisted mal ins
        @arg col_stats: dict<str,ColumnStats> //column statistics
        """
        if Utils.is_gzipped(mfile):
            open_func = gzip.open
        else:
            open_func = open
        with open_func(mfile, mode='rt', encoding='utf-8') as f:
            maldict = defaultdict(_make_list)
            startd = {}
            query_tags = set()

            while True:  # while not EOF
                jobj = Utils.readJsonObject(f)
                if jobj is None:
                    break
                fname, args, ret = Utils.extract_fname(jobj["short"])

                if not Utils.is_blacklisted(blacklist, fname):
                    if jobj["state"] == "start":
                        startd[jobj["pc"]] = jobj["clk"]
                    elif jobj["state"] == "done":
                        assert jobj["pc"] in startd
                        new_mals = MalInstruction.fromJsonObj(jobj, col_stats)
                        new_mals.time = int(jobj["clk"]) - int(startd[jobj["pc"]])
                        new_mals.start = int(startd[jobj["pc"]])
                        maldict[fname].append(new_mals)
                        query_tags.add(int(jobj["tag"]))

        return MalDictionary(maldict, list(query_tags), col_stats)

    @staticmethod
    def fromInsList(ilist):
        """ Constructor from instruction list
        @arg ilist: List<MalInstruction>
        @arg varflow
        @ret MalDictionary
        """
        mdict = defaultdict(_make_list)
        qtags = set()
        for i in ilist:
            mdict[i.fname].append(i)
            qtags.add(i.tag)
        return MalDictionary(mdict, list(qtags))

    def buildApproxGraph(self, traind):
        """
        @des builds a graph that approximates the count for each variable
        @arg self MalDictionary to be predicted
        @arg traind MalDictionary training information
        @ret dict<str,int> //dictionary with var name as a key,est count as val
        #TODO rename predictionGraph
        """
        ilist = self.getInsList()
        ilist.sort(key=lambda ins: ins.clk)
        pg = {}
        for ins in ilist:
            for p in ins.predict(traind, pg):
                pg[p.retv] = p
                if p.avg == sys.maxsize or p.avg is None:
                    logging.error("None in the graph: {}".format(ins.short))
        return pg

    def findInstr(self, mals, ignoreScale=False):
        """
        @arg self: MalDictionary
        @arg mals: MalInstruction
        @arg ignoreScale should the matching ignore the argument sizes or not (for
            different benchmark scales)
        @ret: List<MalInstriction> //list of all exact matches of 'mals' in 'self'
        """
        dic = self.mal_dict
        if mals.fname not in dic:
            return []
        if not ignoreScale:
            return [x for x in dic[mals.fname] if x == mals]

        return [x for x in dic[mals.fname] if x.isExact(mals, True)]

    def getInsList(self):
        """ @desc returns a list of all the instructions """
        return self.ins_list

    def getMaxMem(self):
        """
        !!Assumes we know each instruction's memory footprint!!
        This is to find the real max. memory usage of a query
        @ret int //actual max bytes the query will allocate
        """
        ilist = self.getInsList()
        ilist.sort(key=lambda i: i.clk)
        max_mem = 0
        curr_mem = 0
        for i in ilist:
            # Assume the worst case: all memory is only freed at the end.
            curr_mem += i.mem_fprint
            max_mem = max(max_mem, curr_mem)
            curr_mem -= i.free_size

        return max_mem

    def predictMaxMem(self, pG):  # TODO fix this
        """
        This is to predict the max. memory usage of a query
        @arg pG: dict<str, Prediction> //graph that relates each var to a
              prediction, build by buildApproxGraph
        @ret int //actual max bytes the query will allocate
        """
        ilist = self.getInsList()
        ilist.sort(key=lambda i: i.clk)
        max_mem = 0
        curr_mem = 0
        for i in ilist:
            # Assume the worst case: all memory is only freed at the end.
            curr_mem += i.approxMemSize(pG)
            max_mem = max(max_mem, curr_mem)
            curr_mem -= i.approxFreeSize(pG)

        return max_mem

    def findMethod(self, fname, nargs=None):
        """
        E.g. if you want to find all the SELECT instructions
        @arg mals: string //method name
        @arg nags: int    //nof arguments
        @ret: list<MalInstruction>
        """
        dic = self.mal_dict
        if nargs is None:
            return dic[fname]

        return [x for x in dic[fname] if len(x.arg_list) == nargs]

    def getTopN(self, f, n):
        """
        @des returning the topN depending on the given function, e.g. topN of
          memory usage or topN of exec. time
        @arg f: lamdba k: MalInstruction -> double //comparison metric
        @ret: list of topN instuctions
        """
        mal_list = self.getInsList()
        mal_list.sort(key=f)
        return mal_list[0:n]

    def filter(self, f):
        """
        @des returns a new MalDictionary containing only the MAL instructions
             selected by the given filter function
        @arg f: lambda i: MalInstruction -> boolean
        retain only the mal instructions that satisfy the given function
        """
        mal_list = self.getInsList()
        new_ilist = list([i for i in mal_list if f(i)])
        return MalDictionary.fromInsList(new_ilist)

    def linkSelectInstructions(self):
        """
        Not used yet
        An attempt to do SELECT prediction taking into account the correlation of
        two data columns
        @experimental
        """
        sel_ins = [i for i in self.getInsList() if i.fname in ['select', 'thetaselect']]
        for testi in sel_ins:
            if testi.lead_arg_i == 0:
                testi.prev_i = None
            else:
                prev = [i for i in sel_ins if i.ret_vars[0] == testi.lead_arg.name and testi.tag == i.tag]
                if len(prev) == 0:
                    testi.prev_i = None
                else:
                    assert len(prev) == 1
                    # logging.debug("Linked {} -> {}".format(testi.short, prev[0].short))
                    testi.prev_i = prev[0]
                    prev[0].next_i = testi

    def splitRandom(self, p):
        """
        @desc splits the dictionary in two randomly
        @arg: p: double //should be between 0,1, determines the sizes of l1 and l2
        """
        assert p >= 0 and p <= 1
        il = self.getInsList()
        random.shuffle(il)
        n1 = int(len(il) * p)
        l1 = MalDictionary.fromInsList(il[0:n1])
        l2 = MalDictionary.fromInsList(il[n1::])
        return (l1, l2)

    def select(self, fun, perc):
        """
        selects s-th.  very similar to the getTopN function, but returns a
        dictionary instead
        """
        ilist = self.getInsList()
        ilist.sort(key=fun)  # bda ins: -ins.mem_fprint)
        bestp = ilist[0:int(perc * len(ilist))]
        d = defaultdict(_make_list)
        tags = set()
        for i in bestp:
            d[i.fname].append(i)
            tags.add(i.tag)
        return MalDictionary(d, tags, self.varflow)
