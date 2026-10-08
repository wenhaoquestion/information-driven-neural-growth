"""Post-hoc matched-capacity/full-label-pool diagnostic for Phase III.

Chosen after reviewing the primary confirmation, not a new primary comparison.
Uses every EB16384 residual run's realized final width and recorded total
training/probe parameter-example budget. Full-pool SGD settings are inherited
unchanged from the frozen pool control: persistent Adam, lr .015, batch 256.
The fixed model starts from the same initialization as that run's existing
fixed_matched control. Width is retrospective and pool access is offline.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.stats import t
from sequential_growth import init, sample, train_pool, params, pack, risk

ROOT=Path(__file__).resolve().parents[1]
TASK_INDEX={'null':0,'linear':1,'additive':2,'interaction':3}


def summarize(runs):
    out={}
    for task in TASK_INDEX:
        rr=[r for r in runs if r['task']==task]
        if not rr:continue
        diff=np.array([r['residual_risk']-r['pool_matched_capacity_risk'] for r in rr])
        out[task]=dict(n=len(rr),residual_mean=float(np.mean([r['residual_risk'] for r in rr])),
            pool_matched_capacity_mean=float(np.mean([r['pool_matched_capacity_risk'] for r in rr])),
            residual_minus_pool_mean=float(diff.mean()),
            paired_t95_halfwidth=float(t.ppf(.975,len(rr)-1)*diff.std(ddof=1)/np.sqrt(len(rr))) if len(rr)>1 else None,
            mean_width=float(np.mean([r['final_width'] for r in rr])),
            min_unique_labels=min(r['unique_labels'] for r in rr),max_unique_labels=max(r['unique_labels'] for r in rr),
            mean_parameter_examples=float(np.mean([r['parameter_examples'] for r in rr])))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='confirm_eb16384.json')
    ap.add_argument('--output',default='pool_capacity_diagnostic.json');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();source=ROOT/'results'/args.input;target=ROOT/'results'/args.output
    inherited=json.loads(source.read_text());cfg=inherited['config']
    assert cfg['gate']=='eb' and cfg['naudit']==16384
    assert len(inherited['runs'])==cfg['seeds']*len(cfg['tasks'])
    output=dict(kind='post_hoc_resource_allocation_diagnostic',
        design='Fixed realized final capacity, same initial parameters as inherited fixed_matched; all fitting/audit labels pooled offline; residual branch/probe parameter-example budget.',
        status='in_progress',source_file=args.input,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        learner_sha256=hashlib.sha256((ROOT/'code/sequential_growth.py').read_bytes()).hexdigest(),
        optimizer=dict(name='persistent Adam',lr=.015,batch=256),
        initialization_seed='SeedSequence([seed,937,realized_final_width]), exactly inherited fixed_matched initialization',
        pool_shuffle_seed='SeedSequence([seed,331,2]); fixed for this diagnostic before execution',
        caveats=['Chosen after inspecting confirmation; descriptive paired intervals, no new confirmatory discovery claim.',
            'Final width is retrospective, not a deployable advance choice.',
            'All future audit labels are available from the outset; this is an offline end-of-lifetime allocation.',
            'Optimizer and batch size differ from full-batch neural branches; no pure data-only attribution.',
            'Matched parameter-example proxy is not matched hardware time or peak memory.'],runs=[])
    if args.resume and target.exists():
        old=json.loads(target.read_text())
        assert old['source_sha256']==output['source_sha256'] and old['code_sha256']==output['code_sha256']
        output['runs']=old['runs']
    done={(r['task'],r['seed']) for r in output['runs']};start=time.perf_counter()
    for r in inherited['runs']:
        task,seed=r['task'],r['seed']
        if (task,seed) in done:continue
        m=r['methods']['residual'];width=m['final_width'];budget=m['training_parameter_examples']
        streams=np.random.SeedSequence([seed,TASK_INDEX[task]]).spawn(5)
        data_rng=np.random.default_rng(streams[0]);audit_rng=np.random.default_rng(streams[2]);test_rng=np.random.default_rng(streams[3])
        x,y,_=sample(data_rng,cfg['ntrain'],task);xt,_,qt=sample(test_rng,cfg['ntest'],task)
        audits=[sample(audit_rng,cfg['naudit'],task)[:2] for _ in range(cfg['events'])]
        poolx=np.concatenate([x]+[a[0] for a in audits]);pooly=np.concatenate([y]+[a[1] for a in audits])
        p0=init(np.random.default_rng(np.random.SeedSequence([seed,937,width])),6,width)
        begin=time.perf_counter()
        p,account=train_pool(poolx,pooly,p0,budget,np.random.default_rng(np.random.SeedSequence([seed,331,2])),batch=256,lr=.015)
        elapsed=time.perf_counter()-begin
        assert params(p)==m['events'][-1]['parameters']==8*width+1
        assert 0<=budget-account['parameter_examples']<params(p)
        assert len(pooly)==cfg['ntrain']+cfg['events']*cfg['naudit']==account['available_labels']
        output['runs'].append(dict(task=task,seed=seed,final_width=width,parameter_count=params(p),
            residual_risk=m['final_risk'],inherited_fixed_matched_risk=m['fixed_matched_risk'],
            pool_matched_capacity_risk=risk(xt,qt,p),parameters=pack(p),initial_parameters=pack(p0),
            residual_budget=budget,elapsed_seconds=elapsed,**account))
        output['summary']=summarize(output['runs']);output['wall_seconds_current_invocation']=time.perf_counter()-start
        temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(output,indent=2));temp.replace(target)
        print(task,seed,'width',width,'residual',round(m['final_risk'],6),'pool',round(output['runs'][-1]['pool_matched_capacity_risk'],6),flush=True)
    output['status']='complete';output['summary']=summarize(output['runs'])
    output['wall_seconds_current_invocation']=time.perf_counter()-start
    target.write_text(json.dumps(output,indent=2))
    print(json.dumps(output['summary'],indent=2),flush=True)


if __name__=='__main__':main()
