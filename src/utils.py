from functools import reduce
import json

class Utils:
    @staticmethod
    #readline until you reach '}' or EOF
    def read_json_object(f):
        lines = []
        rbrace = False
        while rbrace == False:
            line = f.readline()
            if line == '': #no more lines to read
                return None
            lines += line
            if line == '}\n':
                rbrace = True

        return json.loads(''.join(lines))

    @staticmethod
    def flatten(mdict):
        l = []
        for v in mdict.values():
            l.extend(v)
        return l

    @staticmethod
    def list_diff(list1,list2):
        return list(set(list1)-set(list2))

    @staticmethod
    def cmp_arg_list(l1, l2):
        if(len(l1) != len(l2)):
            return False
        else:
            land = lambda x,y: x and y
            return reduce(land, [e1==e2 for (e1,e2) in zip(l1,l2)], True)

    @staticmethod
    def is_blacklisted(blacklist,instr):
        for mali in blacklist:
            if mali in instr:
                return True
        # print(line.strip())
        return False

    @staticmethod
    def init_blacklist(blfile):
        blacklist = []
        for line in open(blfile).readlines():
            blacklist.append(line.strip())

        return blacklist

    @staticmethod
    def sumJsonList(jlist, sfield):
        sumf = lambda x,y: x+y
        return reduce(sumf, map(lambda var: var.get(sfield,0),jlist), 0)

    """ @arg stmt: str // short mal statement"""
    @staticmethod
    def extract_fname(stmt): #TODO change name
        args = []
        ret  = None
        if ":=" in stmt:
            fname = stmt.split(':=')[1].split("(")[0]
            args  = stmt.split(':=')[1].split("(")[1].split(")")[0].strip().split(",") #TODO remove
            ret   = stmt.split(':=')[0].split("[")[0]
        elif "(" in stmt:
            fname = stmt.split("(")[0]
            args  = stmt.split("(")[1].split(")")[0].strip().split(" ")
        else:
            fname = stmt

        return (fname.strip(),args,ret)

    # @staticmethod
    # def var2columns(mfile):
    #     with open(mfile) as f:
    #
    #
    #         while 1: #while not EOF
    #             jobj = Utils.read_json_object(f)
    #             if jobj is None:
    #                 break
    #
    #             if jobj["state"] == "done":
    #                 fname,args,ret = Utils.extract_fname(jobj["short"])
    #                 if fname == "bind": #or fname == "bind_idxbat":
    #                     # print("{} {} {} {}".format(fname,ret,args[2],args[3]))
    #                     xvar_d[ret] = (args[2].strip(), args[3].strip())
    #                 elif fname == "tid": #or fname == "bind_idxbat":
    #                     # print("{} {} {}".format(fname,ret,args[2]))
    #                     cvar_d[ret] = args[2].strip()
    #
    #     return (xvar_d,cvar_d)
