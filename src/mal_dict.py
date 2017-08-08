import random
import json
import pickle
import collections

from utils        import Utils
from mal_instr    import MalInstruction
from mal_bins     import BetaIns
from mal_dataflow import Dataflow
from bdict        import BDict
#TODO add method find closest instruction

Prediction = collections.namedtuple('Prediction', ['ins', 'cnt', 'avg'])

class MalDictionary:
    """
    @arg mal_dict: dict<List<MalInstruction>>
    @arg q_tags  : list<int> //list of the unique query tags
    @arg varflow: dic<tag,dic<var,table>>
    """
    def __init__(self, mal_dict, q_tags, varflow, beta_dict = {}):
        self.mal_dict   = mal_dict
        self.query_tags = q_tags
        self.varflow    = varflow
        self.beta_dict  = BDict(beta_dict) #TODO remove

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
        # union_qtags   = self.query_tags.union(other.query_tags)
        union_varflow = self.varflow.union(other.varflow)

        return MalDictionary.fromInsList(union_ilist, union_varflow)

    """
    @arg mfile    : json file containing mal execution info (link??)
    @arg blacklist: list of black listed mal instructions
    """
    @staticmethod
    def fromJsonFile(mfile, blacklist, stats):
        with open(mfile) as f:
            maldict    = {}
            startd     = {}
            query_tags = set()
            varflow    = Dataflow()
            bins       = {}
            blist      = ["thetaselect","select","+","-","*"]
            while 1: #while not EOF
                jobj = Utils.read_json_object(f)
                if jobj is None:
                    break
                fname,args ,ret  = Utils.extract_fname(jobj["short"])

                if not Utils.is_blacklisted(blacklist,fname):
                    new_mals = MalInstruction.fromJsonObj(jobj, stats)

                    if jobj["state"] == "start":
                        startd[jobj["pc"]] = jobj["clk"]
                    elif jobj["state"] == "done":
                        assert jobj["pc"] in startd
                        new_mals.time  = float(jobj["clk"]) - float(startd[jobj["pc"]])
                        new_mals.start = int(startd[jobj["pc"]])
                        maldict[fname] = maldict.get(fname,[]) + [new_mals]
                        query_tags.add(int(jobj["tag"]))
                        tag = int(jobj["tag"])

                        #deprecated
                        if fname in blist:
                            bi = BetaIns.fromJsonObj(jobj, fname, stats)
                            bins[fname] = bins.get(fname,[])
                            bins[fname].append(bi)

                        for r in jobj["ret"]:

                            if "alias" in r:
                                varflow.add(tag,r["name"],r["alias"].split('.')[-1])
                            # print(bi.toStr())
                        # if fname == "bind" or fname == "bind_idxbat":
                        #     varflow.add(tag, ret, [args[3].strip()])
                        #     # varflow.add(tag, ret, [args[2].strip(), args[3].strip()])
                        # elif fname == "tid":
                        #     # print("tid")
                        #     varflow.add(tag, ret, [args[2].strip()])
                        # else:
                        #     # print("ret_vars {}".format(new_mals.ret_vars))
                        #     varflow.addI(tag,new_mals.arg_vars,new_mals.ret_vars)
                            # for r in new_mals.arg_vars:
                            #     print("short",new_mals.short)
                            #     print("lookup: ",r,varflow.lookup(r,tag))
        return MalDictionary(maldict,list(query_tags),varflow, bins)

    """ Constructor from instruction list
    @arg ilist: List<MalInstruction>
    @arg varflow
    @ret: MalDictionary
    """
    @staticmethod
    def fromInsList(ilist, varflow):
        mdict = {}
        qtags = set()
        for i in ilist:
            mdict[i.fname] = mdict.get(i.fname,[]) + [i]
            qtags.add(i.tag)
        return MalDictionary(mdict, list(qtags), varflow)

    def approxGraph(self, traind):
        ilist = self.getInsList()
        ilist.sort( key = lambda ins: ins.clk)
        g = {}

        for ins in ilist:
            if ins.fname in ['tid']:
                g[ins.ret_vars[0]] = ins.cnt
            elif ins.fname in ['select', 'thetaselect']:
                print(ins.short)
                # print(ins.ret_vars[0], traind.predictCountG(ins,g), ins.cnt)
                g[ins.ret_vars[0]] = traind.predictCountG(ins,g).cnt
            # else: #TODO batcalc instructions
                # g[ins.ret_vars[0]] = None
        return g
    #deprecated
    def estimate_arg_size(self, ins):
        o = self.varflow.get(ins.arg,None)
        if isinstance(o, SelectInstruction):
            return estimate_arg_size(self, o)
        elif isinstance(o ,int):
            return o


    def getFirst(self, field, N):
            ilist = self.getInsList()
            ilist.sort(key = lambda ins: getattr(ins, field) )
            return MalDictionary.fromInsList(ilist[0:N], self.varflow)
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
        # if len(ret) == 0:
            # print(mals.short)
        return ret

    def getInsList(self):
        ilist = []
        for l in self.mal_dict.values():
            ilist.extend(l)
        return ilist

    def getMaxMem(self):
        ilist = self.getInsList()
        ilist.sort(key = lambda i: i.start) #clk ???
        max_mem  = 0
        curr_mem = 0
        for i in ilist:
            max_mem = max(max_mem,curr_mem + i.mem_fprint)
            curr_mem = curr_mem + i.ret_size - i.free_size
            # print("{} {} {}".format(i.pc,max_mem,curr_mem))

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


    #deprecated???
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
    def predictMem(self, ins, default=None):
        try:
            return self.predict(ins).mem_fprint
        except:
            return default

    def predict(self, ins, default=None):
        exact = self.findInstr(ins)
        if len(exact) == 1:
            return exact[0]
        else:
            try:
                nn1 = self.kNN(ins,1)[0]
            except Exception as e:
                if default != None:
                    nn1 =  default
                else:
                    raise e
            return nn1

    #TODO should rename findExactOR. ...
    def predictMemExactOr(self, ins, default):
        exact = self.findInstr(ins)
        assert len(exact) <= 1
        if len(exact) == 1:
            return exact[0].mem_fprint
        else:
            return default

    def predictTimeExactOr(self, ins, default):
        exact = self.findInstr(ins)
        assert len(exact) <= 1
        if len(exact) == 1:
            return exact[0].time
        else:
            return default
    #TODO too custom, needs rewritting
    def printAll(self, method, nargs):
        dic = self.mal_dict
        for s in dic[method]:
            if s.fname == method and nargs == len(s.arg_list):
                a2val = s.arg_list[2].aval
                a1val = s.arg_list[1].aval
                a1_t  = s.arg_list[1].atype
                print("mal: {}, args: {} {} {} size: {}, time: {}".format(method,a2val,a1_t, a1val,s.size,s.time))

    def printShort(self, method):
        dic = self.mal_dict
        for s in dic[method]:
            print(s.short)
            # if s.fname == method and nargs == len(s.arg_list):
            #     a2val = s.arg_list[2].aval
            #     a1val = s.arg_list[1].aval
            #     a1_t  = s.arg_list[1].atype
            #     print("mal: {}, args: {} {} {} size: {}, time: {}".format(method,a2val,a1_t, a1val,s.size,s.time))


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
        return MalDictionary.fromInsList(new_ilist, self.varflow)

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

    def printPredictions(self, test_dict):
        for ins in test_dict.getInsList():
            try:
                knni  = self.predict(ins)
                mpred = self.predictMem(ins)
                # knn5  = self.predictDEBUG(ins)
                mem   = ins.mem_fprint
                if mem != 0:
                    sim = knni.similarity(ins)
                    print("method: {:20} nargs: {:2d} actual: {:10d} pred: {:10d} sim: {:10.1f} perc: {:10.0f}".format(ins.fname,ins.nargs, mem,mpred,sim,abs(100*mpred/mem)))
                else:
                    print("method: {:20} nargs: {:2d} actual: {:10d} pred: {:10d}".format(ins.fname, ins.nargs, mem,mpred))
            except Exception as err:
                print("Exception: {}".format(err))
                print("method: {:20} nargs: {:2d}  NOT FOUND".format(ins.fname,ins.nargs))
                pass

    def printPredictionsVerbose(self, test_dict, tag2query):
        for ins in test_dict.getInsList():
            try:
                ipred      = self.predict(ins)
                mpred      = ipred.mem_fprint
                mem        = ins.mem_fprint
                if mem != 0  and mpred / mem > 2:
                    # print("DEBUG: {}".format(test_dict.query_tags))

                    print("INS Q: {:2d} SHORT: {:80}".format(tag2query[ins.tag],ins.short))
                    # ins.printVarFlow(self.varflow)
                    print("KNN Q: {:2d} SHORT: {:80}".format(tag2query[ipred.tag],ipred.short))
                    # ipred.printVarFlow(self.varflow)
                    print("real: {:10d} pred: {:10d}\nINS: {} \nKNN: {}\n".format(mem,mpred, ins.argListStr(), ipred.argListStr()))

                    # for alt in knn:
                    #     print("ALT Q: {:2d} SHORT: {:80}".format(tag2query[alt.tag],alt.short))
                    #     print("ALTM: {} KNN: {}\n".format(alt.mem_fprint, alt.argListStr()))


            except Exception as err:
                print("Exception: {}".format(err))
                print("method: {:20} nargs: {:2d}  NOT FOUND".format(ins.fname,ins.nargs))
                pass

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

    def predictCount(self, test_i):

        self_list = self.mal_dict.get(test_i.fname,[])# + self.mal_dict.get(alias[test_i.fname],[])
        nn        = test_i.kNN(self_list,1)[0]
        return nn

    def predictCount2(self, test_i):
        self_list = self.mal_dict.get(test_i.fname,[])# + self.mal_dict.get(alias[test_i.fname],[])
        nn        = test_i.kNN(self_list,10)
        nn.sort(key = lambda i: i.argDist(test_i))
        nn3 = nn[0:3]
        # print("nn3: ", nn3[0].short, nn3[0].cnt)
        # ex = sum([nni.extrapolate(test_i)*nni.argDiv(test_i) for nni in nn3]) / len(nn3)
        return nn3[0]

    def predictCountG(self, test_i, approxG):
        assert approxG != None
        self_list = self.mal_dict.get(test_i.fname,[])# + self.mal_dict.get(alias[test_i.fname],[])
        nn        = test_i.kNN(self_list,5)

        nn.sort( key = lambda ins: test_i.approxArgDist(ins, approxG))
        nn1       = nn[0]
        arg_cnt   = test_i.approxArgCnt(approxG)
        if test_i.fname == 'select' and test_i.col == 'l_discount':
            for i in nn:
                print(i.short, i.extrapolate(test_i) * ( arg_cnt / i.argCnt()))
        avg = sum([i.extrapolate(test_i) * ( arg_cnt / i.argCnt()) for i in nn]) / len(nn)
        if arg_cnt != None:
            # print("extrapolate", nn1.cnt, nn1.extrapolate(test_i))
            return Prediction(ins=nn1,cnt = nn1.extrapolate(test_i) * ( arg_cnt / nn1.argCnt()), avg = avg)
        else:
            return Prediction(ins=nn1, cnt = nn1.extrapolate(test_i), avg = None)

    def errorCount(self, test_i):
        nn = self.predictCount(test_i)
        return 100 * abs(nn.extrapolate(test_i)-test_i.cnt)/test_i.cnt

    def avgErrorExact(self, test_dict):
        suml   = lambda x,y: x+y
        test_l = test_dict.getInsList()
        ldiff = lambda i: abs(i.mem_fprint-self.predictMemExactOr(i,0)) / i.mem_fprint if i.mem_fprint != 0 else 0.0
        diff  = sum( map(ldiff,test_l) )
        # print(list(map(ldiff,test_l)))
        # print(max(list(map(ldiff,test_l))))
        # print("dev")
        return 100 * diff / len(test_l)

    def avgAccMemExact(self, test_dict):
        suml   = lambda x,y: x+y
        test_l = test_dict.getInsList()
        lacc = lambda i: abs(self.predictMemExactOr(i,0)) / i.mem_fprint if i.mem_fprint != 0 else 1.0
        sacc  = sum( map(lacc,test_l) )
        return 100 * sacc / len(test_l)

    def avgAccTimeExact(self, test_dict):
        suml   = lambda x,y: x+y
        test_l = test_dict.getInsList()
        lacc = lambda i: abs(self.predictTimeExactOr(i,0)) / i.time if i.time != 0 else 1.0
        sacc  = sum( map(lacc,test_l) )
        return 100 * sacc / len(test_l)

    def avgAccTimeExact2(self, test_dict):
        suml   = lambda x,y: x+y
        test_l = test_dict.getInsList()
        actual = sum(map(lambda i: i.time, test_dict.getInsList()))
        pred   = sum(map(lambda i: self.predictTimeExactOr(i,0), test_dict.getInsList()))
        return 100 * pred / actual
