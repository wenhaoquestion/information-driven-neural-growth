"""Separate fresh-seed adaptive-width extension; heuristic validation, no theorem."""
from __future__ import annotations
import argparse,copy,json,time,sys,platform,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import quadratic_experiment as q

ROOT=Path(__file__).resolve().parents[1]
METHODS=['fresh_adaptive','random_adaptive','fixed2_gated','fixed7_gated','fixed7_ungated']
def gate(old,new,width_increase):
    d=old-new;mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)))
    penalty=.002+.001*width_increase
    return {'mean_gain':mean,'sample_se':se,'penalty':penalty,'threshold':2*se+penalty,
            'qualified':bool(mean>2*se+penalty),'penalized_mean_gain':mean-penalty,
            'heuristic_not_coverage_guaranteed':True}

def learner(seed,task,method,cfg,xstream,ystream):
    fixed=method.startswith('fixed');k0=(2 if method=='fixed2_gated' else 7) if fixed else 1
    s=q.init([seed,q.TASKS[task],409,k0],cfg['d'],k0);initial=q.pack(s['p'])
    reservoir=q.Reservoir(cfg['matched_reservoir'],cfg['d'],np.random.default_rng([seed,q.TASKS[task],500]))
    ledger=q.Ledger(len(ystream));ledger.counts['validation']=np.zeros(len(ystream),dtype=np.int32);ledger.proxy['validation']=0
    events=[];start=time.perf_counter()
    for t in range(cfg['arrivals']):
        left=t*cfg['block'];right=left+cfg['block'];fitright=left+cfg['fit_per_block'];ids=np.arange(left,fitright);vids=np.arange(fitright,right)
        xx=xstream[left:fitright];yy=ystream[left:fitright];old=(reservoir.x.copy(),reservoir.y.copy(),reservoir.ids.copy())
        poolids=np.r_[old[2],ids];poolx=np.vstack([old[0],xx]);pooly=np.r_[old[1],yy]
        before=copy.deepcopy(s);old_counts={k:v.copy() for k,v in ledger.counts.items()};proposal=None
        width=len(s['p']['a']);can_grow=t>0 and not fixed and width<cfg['max_width']
        kind='fresh_residual' if method=='fresh_adaptive' else 'random_matched138'
        if can_grow:
            w,proposal=q.direction(kind,s,old,(xx,yy,ids),None,right,np.random.default_rng([seed,q.TASKS[task],701,t]),ledger,cfg)
            g0=q.birth(s,w)
            assert np.max(np.abs(q.predict(xx,s['p'])-q.predict(xx,g0['p'])))<1e-10
        gated=t>0 and method!='fixed7_ungated'
        branch_widths=[width,width]+([width+1] if can_grow else [])
        validation_cost=len(vids)*sum(cfg['d']*k+3*k+1 for k in branch_widths) if gated else 0
        proposal_cost=proposal['proxy'] if proposal else 0
        available=cfg['event_budget']-validation_cost-proposal_cost;assert available>0
        continuation=copy.deepcopy(s);cc=q.fit(continuation,poolx,pooly,poolids,available//2 if can_grow else available,
              np.random.default_rng([seed,q.TASKS[task],809,t,0]),ledger,cfg)
        branches={'incumbent':before,'continuation':continuation};costs={'continuation':cc}
        if can_grow:
            gc=q.fit(g0,poolx,pooly,poolids,available-available//2,
                      np.random.default_rng([seed,q.TASKS[task],809,t,1]),ledger,cfg)
            branches['growth']=g0;costs['growth']=gc
        # Candidate fitting and proposal searches finish before current validation access.
        assert all(not np.any((ledger.counts[k]-old_counts[k])[fitright:]) for k in ['fit','proposal','moment'])
        tests={}
        if gated:
            losses={name:(q.predict(xstream[vids],st['p'])-ystream[vids])**2 for name,st in branches.items()}
            ledger.add(vids,'validation',validation_cost)
            for name in costs:tests[name]=gate(losses['incumbent'],losses[name],len(branches[name]['p']['a'])-width)
            eligible=[name for name,v in tests.items() if v['qualified']]
            selected=max(eligible,key=lambda name:tests[name]['penalized_mean_gain']) if eligible else 'incumbent'
        else:selected='continuation'
        s=copy.deepcopy(branches[selected]);action='grow' if selected=='growth' else ('hold' if selected=='incumbent' else ('warmup' if t==0 else 'continue'))
        rejected=sum(v['proxy'] for name,v in costs.items() if name!=selected)
        diff={k:ledger.counts[k]-old_counts[k] for k in ledger.counts}
        assert all(not np.any(v[right:]) for v in diff.values())
        event={'t':t,'arrival_range':[left,right],'fit_current_range':[left,fitright],'validation_range':[fitright,right],
               'width':len(s['p']['a']),'action':action,'selected_branch':selected,'proposal':proposal,'tests':tests,
               'before_parameters':q.pack(before['p']),'after_parameters':q.pack(s['p']),
               'branch_parameters':{name:q.pack(st['p']) for name,st in branches.items()},
               'before_optimizer':{'m':q.pack(before['m']),'v':q.pack(before['v']),'step':before['step']},
               'branch_fit_accounts':costs,'rejected_branch_proxy':rejected,'proposal_proxy':proposal_cost,
               'validation_proxy':validation_cost,'total_event_proxy':sum(v['proxy'] for v in costs.values())+proposal_cost+validation_cost,
               'fit_pool_ids':poolids.tolist(),'retained_ids_before':old[2].tolist(),
               'event_accesses':{k:int(v.sum()) for k,v in diff.items()},
               'event_unique_accesses':{k:int(np.count_nonzero(v)) for k,v in diff.items()},
               'current_validation_fitted_before_decision':False,
               'persistent_model_optimizer_words':3*q.nparams(s['p']),
               'persistent_retention_words':cfg['matched_reservoir']*(cfg['d']+2),
               'branch_parameter_optimizer_words':sum(3*q.nparams(st['p']) for st in branches.values()),
               'fit_pool_workspace_words':len(pooly)*(cfg['d']+2),
               'candidate_workspace_word_bound':4*cfg['d']**2+len(yy)*(cfg['d']+2)+10*cfg['d']*cfg['probes'],
               'ledger':ledger.report(),'elapsed_seconds':time.perf_counter()-start}
        # Completed validation labels are released only after the chosen state is fixed.
        reservoir.update(xstream[left:right],ystream[left:right],np.arange(left,right));ledger.extra['reservoir_seen']+=right-left
        event['retained_ids_after']=reservoir.ids.tolist();event['release_after_decision']=True
        events.append(event)
    return {'method':method,'initial_parameters':initial,'final_parameters':q.pack(s['p']),'events':events,
            'ledger':ledger.report(),'elapsed_seconds':time.perf_counter()-start},ledger.counts

def run_one(seed,task,cfg,outdir):
    outdir=Path(outdir).resolve();start=time.perf_counter();x,y,ex,eq,spec=q.generate(seed,task,cfg)
    datadir=outdir.parent/'data';datadir.mkdir(parents=True,exist_ok=True);data=datadir/f'{task}_{seed}.npz'
    np.savez_compressed(data,x=x,y=y,evaluator_x=ex,evaluator_mean=eq,teacher_matrix=spec['matrix'],teacher_rotation=spec['rotation'],noise=spec['noise'])
    results={};counts={}
    for method in cfg['methods']:
        r,cnt=learner(seed,task,method,cfg,x,y);results[method]=q.evaluate(r,task,spec,ex,eq)
        for key,value in cnt.items():counts[method+'_'+key]=value
    access=datadir/f'{task}_{seed}_access.npz';np.savez_compressed(access,**counts)
    row={'seed':seed,'task':task,'config':cfg,'methods':results,'runner_sha256':q.digest(Path(__file__).read_bytes()),
         'dependency_sha256':q.digest(Path(q.__file__).read_bytes()),'config_sha256':q.jsonhash(cfg),
         'data_file':str(data.relative_to(ROOT)),'data_sha256':q.digest(data.read_bytes()),
         'access_file':str(access.relative_to(ROOT)),'access_sha256':q.digest(access.read_bytes()),'wall_seconds':time.perf_counter()-start}
    path=outdir/f'{task}_{seed}.json';path.write_text(json.dumps(row,indent=2))
    return {'seed':seed,'task':task,'file':str(path.relative_to(ROOT)),'sha256':q.digest(path.read_bytes()),
            'wall_seconds':row['wall_seconds'],'risks':{m:r['final_risk'] for m,r in results.items()},
            'widths':{m:r['events'][-1]['width'] for m,r in results.items()}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    env={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'command':sys.argv,'runner_sha256':q.digest(Path(__file__).read_bytes()),'dependency_sha256':q.digest(Path(q.__file__).read_bytes())}
    (outdir/'environment.json').write_text(json.dumps(env,indent=2));rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs=[executor.submit(run_one,s,t,cfg,outdir) for t in cfg['tasks'] for s in cfg['seeds']]
        for job in as_completed(jobs):
            row=job.result();rows.append(row);print(json.dumps(row),flush=True)
            (outdir/'run_index.json').write_text(json.dumps({'runs':rows,'completed':len(rows),'planned':len(jobs),'wall_seconds':time.perf_counter()-start},indent=2))
    print('COMPLETE',len(rows),time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
