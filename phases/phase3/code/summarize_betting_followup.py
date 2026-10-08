"""Paired, prespecified fresh-seed comparison of two valid acceptance gates."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

root=Path(__file__).resolve().parents[1]
bet=json.loads((root/'results/betting_followup4096.json').read_text())
eb=json.loads((root/'results/eb_followup4096.json').read_text())
assert len(bet['runs'])==len(eb['runs'])==64, 'Wait for both complete 64-run follow-ups'
B={(r['task'],r['seed']):r for r in bet['runs']}
E={(r['task'],r['seed']):r for r in eb['runs']}
assert B.keys()==E.keys()
for key in B:
    assert B[key]['start_parameters']==E[key]['start_parameters']
    assert B[key]['initial_risk']==E[key]['initial_risk']

def interval(values):
    a=np.asarray(values,float);m=float(a.mean());h=float(student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/math.sqrt(len(a)))
    return dict(mean=m,lower=m-h,upper=m+h,n=len(a))

rows=[]
for task in ['null','linear','additive','interaction']:
    keys=sorted(k for k in B if k[0]==task)
    for method in ['residual','random']:
        b=[B[k]['methods'][method] for k in keys];e=[E[k]['methods'][method] for k in keys]
        item={'task':task,'method':method,'seeds':[k[1] for k in keys]}
        for field in ['final_risk','final_width','tail_risk','training_parameter_examples']:
            item['bet_'+field]=float(np.mean([x[field] for x in b]))
            item['eb_'+field]=float(np.mean([x[field] for x in e]))
            item[field+'_paired_bet_minus_eb']=interval([x[field]-y[field] for x,y in zip(b,e)])
        item['training_cost_ratio_mean']=float(np.mean([x['training_parameter_examples']/y['training_parameter_examples'] for x,y in zip(b,e)]))
        item['bet_fixed_matched_risk']=float(np.mean([x['fixed_matched_risk'] for x in b]))
        item['bet_fixed_pool_large_risk']=float(np.mean([B[k]['baselines']['large_pool_matched']['risk'] for k in keys]))
        item['bet_fixed_matched_paired']=interval([x['final_risk']-x['fixed_matched_risk'] for x in b])
        item['bet_actions']={a:sum(ev['action']==a for x in b for ev in x['events']) for a in ['hold','continue','grow']}
        item['eb_actions']={a:sum(ev['action']==a for x in e for ev in x['events']) for a in ['hold','continue','grow']}
        item['bet_betting_factor_evaluations_per_run']=sum(ev[p]['betting_factor_evaluations'] for ev in b[0]['events'] for p in ['continuation_vs_incumbent','growth_vs_incumbent','growth_vs_continuation'])
        item['audit_labels_per_run']=b[0]['audit_labels']
        item['bet_evaluator_risk_increase_events']=sum(ev['retained_risk']>ev['incumbent_risk']+1e-12 for x in b for ev in x['events'])
        rows.append(item)
selector_contrasts={task:interval([r['methods']['residual']['final_risk']-r['methods']['random']['final_risk'] for r in bet['runs'] if r['task']==task]) for task in ['null','linear','additive','interaction']}
out={'evidence_type':'fresh-seed follow-up; paired independent runs; pointwise t intervals, no familywise comparative-performance claim',
     'rows':rows,'bet_wall_seconds':bet['wall_seconds'],'eb_wall_seconds':eb['wall_seconds'],
     'betting_residual_minus_random_final_risk':selector_contrasts,
     'overlapping_execution':'two processes run concurrently; aggregate runtime is not a per-method hardware comparison',
     'code_sha256':bet['code_sha256'],'same_initial_states_verified':True}
(root/'results/betting_followup_summary.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['\\begin{tabular}{llrrrr}','\\toprule','Task & Proposal & EB risk & Betting risk & Paired difference [95\\% CI] & Width (EB / bet)\\\\','\\midrule']
for r in rows:
    d=r['final_risk_paired_bet_minus_eb']
    lines.append(f"{r['task']} & {r['method']} & {r['eb_final_risk']:.5f} & {r['bet_final_risk']:.5f} & {d['mean']:+.5f} [{d['lower']:+.5f}, {d['upper']:+.5f}] & {r['eb_final_width']:.2f} / {r['bet_final_width']:.2f}\\\\")
lines+=['\\bottomrule','\\end{tabular}']
(root/'manuscript/betting_followup_table.tex').write_text('\n'.join(lines)+'\n')
for r in rows:
    d=r['final_risk_paired_bet_minus_eb']
    print(r['task'],r['method'],'risk',round(r['eb_final_risk'],5),'->',round(r['bet_final_risk'],5),
          'difference',round(d['mean'],5),(round(d['lower'],5),round(d['upper'],5)),
          'width',r['eb_final_width'],'->',r['bet_final_width'],'cost ratio',round(r['training_cost_ratio_mean'],3))
