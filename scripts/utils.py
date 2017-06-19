from functools import reduce

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
        return ''.join(lines)

    @staticmethod
    def flatten(mdict):
        l = []
        for v in mdict.values():
            l.extend(v)
        return l

    @staticmethod
    def cmp_arg_list(l1, l2):
        if(len(l1) != len(l2)):
            return False
        else:
            return reduce(lambda x,y: x and y, [e1==e2 for (e1,e2) in zip(l1,l2)], True)

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
