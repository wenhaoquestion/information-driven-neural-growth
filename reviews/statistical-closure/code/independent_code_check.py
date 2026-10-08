"""Read-only deterministic review of frozen inputs. No RNG, experiment or old main.

Only output files next to this script are written. Pure old functions are imported
with bytecode disabled to replay existing count inference / rational certificates.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from fractions import Fraction as Q
from itertools import product, combinations
from math import isclose
from datetime import datetime, timezone
import hashlib
import json
import ast

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PILOT = ROOT / 'studies' / 'additive-risk'
FILES = '''PROTOCOL.json
qa/protocol_freeze.json
statistics/DERIVATION.md
statistics/honest_delta.py
statistics/certify_interior2048.py
statistics/infer_confirmation.py
statistics/confirm_once.py
statistics/CONFIRMATION_REFERENCE.json
statistics/INTERIOR_2048_POWER_ENVELOPE.json
statistics/INTERIOR_2048_POWER_CERTIFICATE.json
statistics/PRE_CONFIRMATION_VALIDATION.json
theory/local_loss_oracle.py
theory/LOCAL_GRADIENT_SUPPORT.md
theory/LOCAL_ORACLE_CERTIFICATE.json
theory/rational_certificates.json
theory/FIRST_PASS_THEORY.md
statistics/confirmation_run.log
computation/first_pass_freeze.json
computation/first_pass.json
computation/precision_recheck.json
computation/runtime.json
computation/CERTIFICATE_COMPARISON.md
PILOT_RESULT_ZH.md'''.splitlines()
FILES += [str(p.relative_to(PILOT)) for p in sorted((PILOT/'statistics/confirmation').rglob('*')) if p.is_file()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = {name: {'sha256': digest(PILOT/name), 'bytes': (PILOT/name).stat().st_size}
          for name in FILES}
data = {name: json.loads((PILOT/name).read_text()) for name in FILES if name.endswith('.json')}
checks = []

def ok(name, assertion, detail=None):
    if not assertion:
        raise AssertionError(name)
    checks.append({'name': name, 'passed': True, 'detail': detail})

def canon(part):
    return tuple(sorted(tuple(sorted(block)) for block in part))

def independent_refines(fine, coarse):
    return all(any(set(block) <= set(cell) for cell in coarse) for block in fine)

# Independent enumeration via all 4^4 label assignments, not old recursion.
parts = set()
for labels in product(range(4), repeat=4):
    parts.add(canon([[i for i in range(4) if labels[i] == value] for value in set(labels)]))
p2 = {p for p in parts if len(p) <= 2}
p3 = {p for p in parts if len(p) <= 3}
hierarchies = {(a,b) for a in p2 for b in p3 if independent_refines(b,a)}
ok('independent_complete_enumeration', (len(parts),len(p2),len(p3),len(hierarchies)) == (15,8,14,39))

sys.path[:0] = [str(PILOT/'statistics'),str(PILOT/'theory')]
import honest_delta as hd
import infer_confirmation as inf
from local_loss_oracle import Interior2048Loss
ok('old_enumerations_equal_independent_sets',
   {canon(p) for p in hd.PARTITIONS} == parts and
   {(canon(a),canon(b)) for a,b in hd.HIERARCHIES} == hierarchies)

def decode(obj):
    if isinstance(obj,dict):
        if set(obj) == {'rational','decimal'}:
            return Q(obj['rational'])
        if set(obj) == {'lo','hi'}:
            return hd.Interval(decode(obj['lo']),decode(obj['hi']))
        return {k:decode(v) for k,v in obj.items()}
    if isinstance(obj,list):
        return tuple(decode(v) for v in obj)
    return obj

protocol = data['PROTOCOL.json']
ok('protocol_frozen_bytes_match', before['PROTOCOL.json']['sha256'] == data['qa/protocol_freeze.json']['sha256'] and before['PROTOCOL.json']['bytes'] == data['qa/protocol_freeze.json']['bytes'])
ok('first_pass_frozen_bytes_match', before['computation/first_pass.json']['sha256'] == data['computation/first_pass_freeze.json']['sha256'])

hash_checks = []
for name in ['statistics/INTERIOR_2048_POWER_CERTIFICATE.json','statistics/confirmation/RUNTIME.json','statistics/confirmation/INFERENCE.json']:
    for relative,expected in data[name]['source_sha256'].items():
        matched = digest(PILOT/relative) == expected
        hash_checks.append({'record':name,'source':relative,'matched':matched})
        ok('source_hash:' + name + ':' + relative, matched)
for relative,expected in data['statistics/PRE_CONFIRMATION_VALIDATION.json']['source_sha256'].items():
    ok('prevalidation_hash:' + relative, digest(PILOT/'statistics'/relative) == expected)
local = data['theory/LOCAL_ORACLE_CERTIFICATE.json']
ok('local_oracle_source_and_envelope_hashes_match', local['oracle_sha256'] == digest(PILOT/'theory/local_loss_oracle.py') and local['envelope_sha256'] == digest(PILOT/'statistics/INTERIOR_2048_POWER_ENVELOPE.json'))
theory = data['theory/rational_certificates.json']
ok('theory_script_hash_matches', digest(PILOT/'theory/certify_bounds.py') == theory['script_sha256'])

public = data['statistics/confirmation/COUNTS_PUBLIC.json']
saved = decode(data['statistics/confirmation/INFERENCE.json'])
replayed = inf.run_inference(public)
for key,value in replayed.items():
    ok('fixed_count_replay:' + key, decode(value) == saved[key])
count_hash = digest(PILOT/'statistics/confirmation/COUNTS_PUBLIC.json')
ok('counts_hash_matches_both_receipts', count_hash == data['statistics/confirmation/INFERENCE.json']['counts_sha256'] == data['statistics/confirmation/COMPLETED.json']['counts_sha256'])
ok('count_budget_and_integer_constraints', sum(n for n,k in public['counts']) == 10**6 and all(type(n) is int and type(k) is int and 0 <= k <= n for n,k in public['counts']))

rejected_fields=[]
for field in ['reference_posterior','theta','true_reports','optimal_partition']:
    attempted=dict(public);attempted[field]=[Q(1,8)]*4
    try:
        inf.run_inference(attempted)
    except ValueError:
        rejected_fields.append(field)
ok('reference_extra_keys_are_rejected_before_inference',len(rejected_fields)==4,rejected_fields)

loss=Interior2048Loss()
power=hd.certify_power((Q(1,8),Q(3,8),Q(5,8),Q(7,8)),(Q(1,4),)*4,loss,
                      m=10**6,entropy_width=loss.entropy_width,slope_width=loss.slope_width)
saved_power=decode(data['statistics/INTERIOR_2048_POWER_CERTIFICATE.json'])
for key,value in power.items():
    ok('rational_power_replay:' + key,value==saved_power[key])
ok('separate_envelope_exactly_matches',power['envelope']==decode(data['statistics/INTERIOR_2048_POWER_ENVELOPE.json']))
ok('all_local_oracle_constants_exactly_match',all(Q(entry['rational'])==getattr(loss,key) for key,entry in local['constants'].items()))
cells={tuple(c['block']):c for c in local['cells']}
nonempty={tuple(block) for size in range(1,5) for block in combinations(range(4),size)}
ok('local_certificate_has_all_15_distinct_nonempty_blocks',set(cells)==nonempty)
rect=power['envelope']['rectangle']
for block,cell in cells.items():
    lo=sum(rect[i].lo for i in block)/len(block)
    hi=sum(rect[i].hi for i in block)/len(block)
    ok('whole_mean_interval_safe:' + str(block),all(loss._safe((min(a[0],b[0]),max(a[1],b[1]))) for a,b in zip(loss._args(lo)[1],loss._args(hi)[1])))
    ok('stored_mean_interval_and_slope:' + str(block),
       (lo,hi)==tuple(Q(x['rational']) for x in cell['mean']) and
       (loss.slope(hi).lo,loss.slope(lo).hi)==tuple(Q(x['rational']) for x in cell['hprime_interval']))

# Separate exact exponential witnesses; never evaluate a floating exponential.
for c,target in [(Q(1269,250),160),(Q(3689,1000),40),(Q(4383,1000),80)]:
    total=term=Q(1)
    for j in range(1,81):
        term=term*c/j;total+=term
    ok('positive_series_probability_constant:' + str(c),total>target)

# Verify outward location and bracket error directly for each actual interval.
for (n,k),interval in zip(public['counts'],replayed['posterior_simultaneous_95_rectangle']):
    t=Q(k,n);c=Q(1269,250);pad=Q(1,2**56)
    def polynomial(q):
        d=abs(q-t);return n*d*d-c*(2*q*(1-q)+Q(2,3)*d)
    ok('actual_bernstein_outward_brackets:' + str(n),
       interval.lo<=t<=interval.hi and
       (interval.lo==0 or (polynomial(interval.lo)>=0 and polynomial(min(t,interval.lo+pad))<=0)) and
       (interval.hi==1 or (polynomial(interval.hi)>=0 and polynomial(max(t,interval.hi-pad))<=0)))

# Independently narrow log(2) using 60 rather than stored 40 terms.
loglo=2*sum((Q(1,(2*j+1)*3**(2*j+1)) for j in range(60)),Q(0))
loghi=loglo+Q(2,121*3**121)/(1-Q(1,9))
storedlog=(Q(theory['log2']['lower_rational']),Q(theory['log2']['upper_rational']))
ok('independent_60_term_log_enclosed_by_frozen_40_term_bound',storedlog[0]<=loglo<=loghi<=storedlog[1])

def interval_pair(row):
    return Q(row['lower_rational']),Q(row['upper_rational'])

for row in theory['interior']:
    n=row['n'];k=n.bit_length()-1
    el,eu=interval_pair(row['epsilon'])
    tail=Q(row['tail_error_bound'])
    al=4+64*el**2-2*tail;au=4+64*eu**2
    dl,du=interval_pair(row['Delta'])
    ok('interior_population_certificate:' + str(n),
       el<=16*k*loglo/n<=16*k*loghi/n<=eu and
       tail>=2/(n**5*k*loglo) and
       dl==(el/4-8*tail)/au and du==(eu/4+8*tail)/al and
       dl>Q(1,1000))
for row in theory['endpoint']:
    n=row['n'];k=n.bit_length()-1;lo,hi=interval_pair(row['d']);dl,du=interval_pair(row['Delta'])
    ok('endpoint_population_certificate:' + str(n),
       lo<=(4*k*loglo/n)**2<=(4*k*loghi/n)**2<=hi<Q(1,4) and
       dl==0 and du==(4-8*hi**2)*hi**3<Q(1,1000))

numeric=data['computation/first_pass.json']
rechecks=data['computation/precision_recheck.json']
for relative,expected in numeric['metadata']['source_sha256'].items():
    ok('numeric_source_hash:'+relative,digest(PILOT/'historical_snapshot'/relative)==expected)
ok('numeric_script_hash_matches',digest(PILOT/'computation/compute_pilot.py')==numeric['metadata']['script_sha256'])
numeric_summary=[]
for candidate,recheck in zip(numeric['candidates'],rechecks):
    par=candidate['parameters'];cell=candidate['cells'];rows=candidate['partitions'];hs=candidate['hierarchies'];summary=candidate['summary']
    byid={x['partition_id']:x for x in rows}
    normal=candidate['normalization'];scale=normal['cost_scale_from_raw_generator']
    expected_p={canon(r['cells']) for r in rows}
    expected_h={(canon(byid[h['coarse_id']]['cells']),canon(byid[h['fine_id']]['cells'])) for h in hs}
    ok('numeric_complete_enumeration:'+par['candidate_id'],expected_p==parts and expected_h==hierarchies and len(rows)==15 and len(hs)==39 and len(cell)==15)
    opt2=min(x['cost_normalized'] for x in rows if x['cell_count']<=2)
    opt3=min(x['cost_normalized'] for x in rows if x['cell_count']<=3)
    for row in rows:
        total=sum(cell[''.join(chr(65+i) for i in block)]['total'] for block in row['cells'])
        ok('numeric_partition_sum:'+par['candidate_id']+':'+str(row['partition_id']),isclose(total,row['cost_raw'],rel_tol=1e-13,abs_tol=1e-28) and isclose(total*scale,row['cost_normalized'],rel_tol=1e-13,abs_tol=1e-28))
    for h in hs:
        d2=byid[h['coarse_id']]['cost_normalized'];d3=byid[h['fine_id']]['cost_normalized']
        ok('numeric_hierarchy_arithmetic:'+par['candidate_id']+':'+str(h['hierarchy_id']),h['D2']==d2 and h['D3']==d3 and h['delta']==max(d2-opt2,d3-opt3) and h['rho']==max(d2/opt2,d3/opt3))
    ok('numeric_optimizer:'+par['candidate_id'],summary['delta']==min(h['delta'] for h in hs) and summary['rho']==min(h['rho'] for h in hs) and summary['opt2']==opt2 and summary['opt3']==opt3)
    differences=recheck['cell_relative_differences']
    ok('precision_recheck_complete:'+par['candidate_id'],recheck['candidate_id']==par['candidate_id'] and set(recheck['check_cells'])==set(cell)==set(differences) and max(differences.values())==recheck['max_cell_relative_difference'])
    for key,row in recheck['check_cells'].items():
        ok('recheck_cell_sum:'+par['candidate_id']+':'+key,isclose(row['total'],row['needles']+row['constant'],rel_tol=1e-13,abs_tol=1e-28))
    numeric_summary.append({'candidate':par['candidate_id'],'delta':summary['delta'],'max_recheck_cell_relative_difference':recheck['max_cell_relative_difference']})
ok('numeric_ten_candidate_count',len(numeric['candidates'])==len(rechecks)==10)
ok('recheck_runtime_max_consistent',max(r['max_cell_relative_difference'] for r in rechecks)==data['computation/runtime.json']['max_recheck_cell_relative_difference'])

runner=ast.parse((PILOT/'statistics/confirm_once.py').read_text())
functions={n.name:n for n in runner.body if isinstance(n,ast.FunctionDef)}
main=functions['main']
random_calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(main) if isinstance(n,ast.Call) and ('rng.' in ast.unparse(n.func) or 'random.' in ast.unparse(n.func))]
ok('runner_rng_sites_only_expected',random_calls==[(57,'np.random.default_rng'),(58,'rng.multinomial'),(59,'rng.binomial')],random_calls)
ok('lock_creation_textually_precedes_rng',(PILOT/'statistics/confirm_once.py').read_text().index('os.open(lock,os.O_CREAT|os.O_EXCL')<(PILOT/'statistics/confirm_once.py').read_text().index('rng=np.random.default_rng'))
ok('no_rng_modules_loaded_in_this_check',not any(name=='numpy' or name=='random' for name in sys.modules))
ok('all_frozen_review_inputs_unchanged',{name:digest(PILOT/name) for name in FILES}=={name:item['sha256'] for name,item in before.items()})

def encode(obj):
    if isinstance(obj,Q):return {'rational':str(obj),'decimal':float(obj)}
    if isinstance(obj,hd.Interval):return {'lo':encode(obj.lo),'hi':encode(obj.hi)}
    if isinstance(obj,dict):return {key:encode(value) for key,value in obj.items()}
    if isinstance(obj,(tuple,list)):return [encode(x) for x in obj]
    return obj

report={'reviewed_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Independent deterministic review; existing counts and certificates only. No RNG, no old main, no integrations, no new experiment.',
        'check_count':len(checks),'all_passed':True,'checks':checks,'inputs':before,
        'numeric_summaries':numeric_summary,'delta_95_interval':replayed['delta_95_interval'],
        'power_output_lower':power['inference_lower_bound_on_power_event'],
        'population_report_error_upper_range':[min(x['population_report_error_interval'].hi for x in replayed['all_15_partition_report_error_bounds']),max(x['population_report_error_interval'].hi for x in replayed['all_15_partition_report_error_bounds'])],
        'known_limitation':'Local receipts and source inspection establish recorded one attempt, not an absolute proof against unrecorded external seed trials. Public input contract is not process isolation or a security sandbox.'}
(OUT/'PUBLICATION_CODE_CHECK.json').write_text(json.dumps(encode(report),ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all_passed':True,'check_count':len(checks),'delta_95_interval':encode(replayed['delta_95_interval']),'power_output_lower':encode(power['inference_lower_bound_on_power_event']),'rng_invoked':False,'old_main_invoked':False,'frozen_input_count':len(FILES),'all_review_inputs_unchanged':True},ensure_ascii=False,indent=2))
