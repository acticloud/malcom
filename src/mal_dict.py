from utils import Utils
import json
from mal_instr import MalInstruction
#TODO add method find closest instruction

class MalDictionary:
    """
    @arg mal_dict: dict<List<MalInstruction>>
    @arg q_tags  : list<int> //list of the unique query tags
    """
    def __init__(self, mal_dict, q_tags):
        self.mal_dict   = mal_dict
        self.query_tags = q_tags

    """
    @arg mals: MalInstruction
    @ret: List<MalInstriction> //list of all exact matches
    """
    def findInstr(self, mals):
        dic = self.mal_dict
        return [x for x in dic[mals.fname] if x == mals]

    def getInsList(self):
        ilist = []
        for l in self.mal_dict.values():
            ilist.extend(l)
        return ilist

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


    def findClosestSize(self, target):
        mlist     = self.mal_dict[target.fname]
        dist_list = [abs(i.arg_size-tsize) for x in mlist]
        nn_index  = dist_list.index(min(dist_list))
        return mlist[nn_index]

    #@arg
    def kNN(self, ins, k):
        mlist     = self.mal_dict[ins.fname]
        l         = len(ins.arg_list)
        dist_list = list(filter(lambda ins: len(ins.arg_list) == l, mlist))
        dist_list.sort( key = lambda i: ins.argDist(i) )
        return dist_list[0:k]

        #@arg
    def predictMem(self, ins):
        # try:
        nn1 = self.kNN(ins,1)[0].mem_fprint
        # except Exception:
            # print("method not found {}".format(ins.fname))
            # nn1 = 0
        return nn1


    #TODO too custom, needs rewritting
    def printAll(self, method, nargs):
        dic = self.mal_dict
        for s in dic[method]:
            if s.fname == method and nargs == len(s.arg_list):
                a2val = s.arg_list[2].aval
                a1val = s.arg_list[1].aval
                a1_t  = s.arg_list[1].atype
                print("mal: {}, args: {} {} {} size: {}, time: {}".format(method,a2val,a1_t, a1val,s.size,s.time))

    """ @arg other: MalDictionary"""
    def printDiff(self, other):
        for l in other.mal_dict.values():
            for m1 in l:
                assert len(self.findInstr(m1)) == 1
                try:
                    m   = self.findInstr(m1)[0]
                    ins = m.fname
                    td  = abs(m1.time-m.time)
                    sd  = abs(m1.mem_fprint-m.mem_fprint)
                    fs  = "q: {:<25s} tdiff: {:8.0f}/{:<8.0f} sdiff {:5d}/{:<5d}"
                    print(fs.format(ins,td,m.time,sd,m.mem_fprint))
                except IndexError:
                    print("Index Error: {}".format(m1.short))

    def getAll(self, method, nargs):
        d = self.mal_dict
        return filter(
            lambda s: s.fname == method and len(s.arg_list) == nargs, d[method]
        )

    """
    @arg f: lamdba k: MalInstruction
    """
    def getTopN(self, f, n):
        mal_list = Utils.flatten(self.mal_dict)
        mal_list.sort(key = f)
        return mal_list[0:n]


    """
    @arg mfile    : json file containing mal execution info (link??)
    @arg blacklist: list of black listed mal instructions
    """
    @staticmethod
    def fromJsonFile(mfile, blacklist):
        with open(mfile) as f:
            maldict    = {}
            startd     = {}
            query_tags = set()

            while 1: #while not EOF
                jobj = Utils.read_json_object(f)
                if jobj is None:
                    break
                fname    = Utils.extract_fname(jobj["short"])

                if not Utils.is_blacklisted(blacklist,fname):
                    new_mals = MalInstruction.fromJsonObj(jobj)

                    if jobj["state"] == "start":
                        startd[jobj["pc"]] = jobj["clk"]
                    elif jobj["state"] == "done":
                        assert jobj["pc"] in startd
                        new_mals.time  = float(jobj["clk"]) - float(startd[jobj["pc"]])
                        maldict[fname] = maldict.get(fname,[]) + [new_mals]
                        query_tags.add(int(jobj["tag"]))

        return MalDictionary(maldict,list(query_tags))

    """
    @desc splits the dictionary in two based on the query tags
    @arg: list<int>
    """
    def split(self, train_tags, test_tags):
        s1 = {}
        s2 = {}

        for (k,l) in self.mal_dict.items():
            for mali in l:
                if mali.tag in train_tags:
                    s1[k] = s1.get(k,[]) + [mali]
                if mali.tag in test_tags:
                    # assert not mali.tag in train_tags
                    s2[k] = s2.get(k,[]) + [mali]

        return (MalDictionary(s1,train_tags), MalDictionary(s2,test_tags))
