from utils import Utils
import json
from mal_instr import MalInstruction

#TODO add method find closest instruction

class MalDictionary:
    """ @arg mal_dict: dict<List<MalInstruction>>"""
    def __init__(self, mal_dict):
        self.mal_dict = mal_dict

    """ @arg mals: MalInstruction"""
    def findInstr(self, mals):
        dic = self.mal_dict
        return [x for x in dic[mals.stype] if x == mals]

    #TODO too custom, needs rewritting
    def printAll(self, method, nargs):
        dic = self.mal_dict
        for s in dic[method]:
            if s.stype == method and nargs == len(s.arg_list):
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
                    ins = m.stype
                    td  = abs(m1.time-m.time)
                    sd  = abs(m1.size-m.size)
                    fs  = "q: {:<25s} tdiff: {:8.0f}/{:<8.0f} sdiff {:5d}/{:<5d}"
                    print(fs.format(ins,td,m.time,sd,m.size))
                except IndexError:
                    print("Index Error: {}".format(m1.short))

    def getAll(self, method, nargs):
        d = self.mal_dict
        return filter(
            lambda s: s.stype == method and len(s.arg_list) == nargs, d[method]
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
            maldict = {}
            startd  = {}
            while 1: #while not EOF
                jsons = Utils.read_json_object(f)
                if jsons is None:
                    break
                jobj     = json.loads(jsons)
                new_mals = MalInstruction.fromJsonObj(jobj)
                fname    = new_mals.stype

                if not Utils.is_blacklisted(blacklist,fname):
                    if jobj["state"] == "start":
                        startd[jobj["pc"]] = jobj["clk"]
                    elif jobj["state"] == "done":
                        assert jobj["pc"] in startd
                        new_mals.time = float(jobj["clk"])-float(startd[jobj["pc"]])
                        maldict[fname] = maldict.get(fname,[]) + [new_mals]

        return MalDictionary(maldict)
