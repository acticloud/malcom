#!/usr/bin/env python3

from collections import defaultdict
import json
import re
class Var:
    __slots__ = [ 'name', 'def_clk', 'kill_clk', 'size']
    def __init__(self, name=None, size=None, def_clk=None, kill_clk=None):
        self.name = name
        self.def_clk = def_clk
        self.kill_clk = kill_clk
        self.size = size

    def __str__(self):
        return "Var(name={}, size={}, def_clk={}, kill_clk={})".format(
            self.name, self.size,
            self.def_clk, self.kill_clk
        )

    @staticmethod
    def from_json(jobj):
        return Var(name=jobj['name'], size=jobj.get('size'))

def var_lifes(events):
    events = sorted(events, key=lambda t: (t['clk'], t['clk'] == 'done'))
    
    pattern = re.compile(r':= ([a-zA-Z0-9.]+)')
    lifes = {}
    for e in events:
        clk = int(e['clk'])
        pc = int(e['pc'])
        state = e['state']
        stmt = e['stmt']
        m = pattern.search(stmt)
        # fname = m.group(1) if m else stmt.split()[0]
        # print(clk, pc, state, fname, [(r['name'], r.get('size'), r['eol']) for r in e.get('ret', [])], [(r['name'], r.get('size'), r['eol']) for r in e.get('arg', [])])
        var = None
        if state == 'start':
            for r in e.get('ret', []):
                var = Var.from_json(r)
                assert var.name not in lifes, "pc {}: var {} already alive since clk {}".format(pc, var.name, var.def_clk)
                lifes[var.name] = var
                # a var starts its life when the instruction returning it begins
                var.def_clk = clk
                # print("   ", var)
        elif state == 'done':
            for r in e.get('ret', []) + e.get('arg', []):
                varname = r['name']
                assert varname in lifes or 'value' in r, "pc {}: var {} is unknown".format(pc, var.name)
                assert not ('value' in lifes and 'size' in lifes), "pc: {} var {} has value AND size".format(pc, varname)
                if varname in lifes:
                    var = lifes[varname]
                    # A var dies when it occurs with eol=1 as either an arg or a ret.
                    # The ret case might occur if the var is never used.
                    eol = r['eol']
                    if eol:
                        assert not var.kill_clk, "var {} is killed at clk {} but also at clk {}".format(varname, var.kill_clk, clk)
                        assert var.def_clk <= clk, "var {} is killed before being defined".format(varname)
                        var.kill_clk = clk
                    size = r.get('size', 0)
                    if size:
                        assert var.size == size or not var.size, "found size {} for {} but also {}".format(size, varname, var.size)
                        var.size = size
                    # print("   ", var)
    
    return lifes.values()





def process(lines):
    jsons = [json.loads(line) for line in lines]
    
    lifes = var_lifes(jsons)
    deltas = [(v.def_clk, 1, v.name, v.size) for v in lifes if v.size]
    deltas += [(v.kill_clk, 2, v.name, -v.size) for v in lifes if v.size and v.kill_clk]

    deltas.sort()
    current = 0
    highest = 0
    for d in deltas:
        timestamp = d[0]
        name = d[2]
        delta = d[3]
        current += delta
        flag = "***" if current > highest else ""
        highest = max(current, highest)
        print("t=%10d %+10d %10d %s %s" % (timestamp, delta, current, name, flag))

    print("highest: {} ({})".format(highest, sizeof_fmt(highest)))

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

if __name__ == "__main__":
    process(open('Q06.json'))
