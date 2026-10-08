"""Independent Phase I audit using assignment tables and joint log likelihood.

Run: .venv/bin/python phase2/code/audit_quartet.py
Only outputs inside phase2/results; Phase I files are never modified.
The independent routine uses neither the Phase I partition generator nor its
entropy/Jensen-gap objective. The original evaluate routine is also rerun as
a distinct reproduction comparison, and results are compared to saved values.
"""
from itertools import product
from pathlib import Path
import importlib.util
import json
import math
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'phase2/results'

def canonical_table(t):
    labels = {}
    return tuple(labels.setdefault(v, len(labels)) for v in t)

TABLES = sorted(set(canonical_table(t) for t in product(range(4), repeat=4)))
T2 = [t for t in TABLES if len(set(t)) == 2]
T3 = [t for t in TABLES if len(set(t)) == 3]
CHAINS = [(a,b) for a in T2 for b in T3
          if all(a[i] == a[j] for i in range(4) for j in range(4) if b[i] == b[j])]

def law(L):
    w = mp.exp(-L)
    e = w*w
    d = mp.sqrt(mp.log(2)/L)
    c = (1-2*w)/2
    # Writing the eight joint entries directly preserves exact symmetry.
    return ((c*(1-e),c*e),(w*(1-d),w*d),(w*d,w*(1-d)),(c*e,c*(1-e)))

def direct_risk(joint, table):
    ans = mp.mpf(0)
    for z in set(table):
        cell = [mp.fsum(joint[x][y] for x in range(4) if table[x] == z) for y in range(2)]
        mass = mp.fsum(cell)
        ans += mp.fsum(v*mp.log(mass/v) for v in cell if v)
    return ans

def direct_channel_risk(joint, channel):
    ans = mp.mpf(0)
    for z in range(len(channel[0])):
        cell = [mp.fsum(joint[x][y]*channel[x][z] for x in range(4)) for y in range(2)]
        mass = mp.fsum(cell)
        ans += mp.fsum(v*mp.log(mass/v) for v in cell if v)
    return ans

def label(t):
    return '|'.join(''.join('ACDB'[x] for x in range(4) if t[x] == z) for z in sorted(set(t)))

def evaluate(L):
    mp.mp.dps = math.ceil(2*L/math.log(10)) + 110
    joint = law(L)
    risks = {t: direct_risk(joint,t) for t in TABLES}
    base = risks[(0,1,2,3)]
    r2 = min(risks[t] for t in T2)
    r3 = min(risks[t] for t in T3)
    excess2,excess3 = r2-base,r3-base
    vectors = [(risks[a]/r2,risks[b]/r3) for a,b in CHAINS]
    det = min(max(v) for v in vectors)
    exc = min(max((risks[a]-base)/excess2,(risks[b]-base)/excess3) for a,b in CHAINS)
    mixed = det
    for a,b in product(vectors,repeat=2):
        da,db = a[0]-a[1],b[0]-b[1]
        if da*db<0:
            weight = db/(db-da)
            mixed = min(mixed,weight*a[0]+(1-weight)*b[0])
    half = mp.mpf('0.5')
    fine = [[1,0,0],[0,1,0],[0,half,half],[0,0,1]]
    coarse = [[row[0]+row[1],row[2]] for row in fine]
    sr2,sr3 = direct_channel_risk(joint,coarse),direct_channel_risk(joint,fine)
    fields = {'opt2':excess2,'opt3':excess3,'baseline':base,'total_price':det,
              'price':exc,'randomized_total_price':mixed,'stochastic_upper':max(sr2/r2,sr3/r3),
              'stochastic_r2':sr2,'stochastic_r3':sr3}
    return {'L':L,'precision':mp.mp.dps,'flat2':[label(t) for t in T2 if mp.almosteq(risks[t],r2,rel_eps=mp.mpf('1e-80'),abs_eps=0)],
            'flat3':[label(t) for t in T3 if mp.almosteq(risks[t],r3,rel_eps=mp.mpf('1e-80'),abs_eps=0)],
            'values':{k:mp.nstr(v,70) for k,v in fields.items()},
            'partitions':[{'table':list(t),'partition':label(t),'total_risk':mp.nstr(risks[t],70)} for t in TABLES]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert (len(TABLES),len(T2),len(T3),len(CHAINS)) == (15,7,6,18)
    spec = importlib.util.spec_from_file_location('original_quartet',ROOT/'phase1/code/quartet_experiment.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    saved = {entry['row']['L']:entry for entry in json.loads((ROOT/'phase1/results/quartet_all_partitions.json').read_text())}
    reports = []
    for L in [16,64,256,2048]:
        independent = evaluate(L)
        rerun = original.evaluate(L)[1]
        errors = {}
        for key,value in independent['values'].items():
            a,b,c = mp.mpf(value),mp.mpf(rerun[key]),mp.mpf(saved[L][key])
            errors[key] = {'versus_original_rerun':mp.nstr(abs(a-b)/abs(a),8),
                           'versus_saved':mp.nstr(abs(a-c)/abs(a),8)}
            assert abs(a-b)/abs(a) < mp.mpf('1e-58'),(L,key,a,b)
            assert abs(a-c)/abs(a) < mp.mpf('1e-58'),(L,key,a,c)
        assert independent['flat2'] == ['AC|DB']
        assert independent['flat3'] == ['A|CD|B']
        independent['comparisons'] = errors
        independent['original_rerun'] = {k:rerun[k] for k in independent['values']}
        reports.append(independent)
        print(L,independent['values']['total_price'][:22],independent['flat2'],independent['flat3'],flush=True)
    (OUT/'audit_quartet.json').write_text(json.dumps({'kind':'exact population numeric evaluation; not sampling',
        'partition_counts':[15,7,6,18],'comparison_tolerance':'relative 1e-58',
        'implementation':'direct joint likelihood; exhaustive function tables; independent from original code',
        'cases':reports},indent=2))
    print('All independent and original comparisons passed. Phase I files untouched.')

if __name__ == '__main__':
    main()
