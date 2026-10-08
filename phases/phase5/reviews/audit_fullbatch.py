"""Independent full-minibatch follow-up audit; preserves all study outputs."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, math, time
import numpy as np
from audit_empirical import risk,stats,strip_times

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reviews';RAW=ROOT/'experiments/fullbatch_confirmation/results'
errors=[];checks=0;maximum={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok,msg):
    global checks
    checks+=1
    if not ok:errors.append(msg)
def close(a,b,name):
    e=float(np.max(abs(np.asarray(a)-np.asarray(b))));maximum[name]=max(maximum.get(name,0.),e);check(e<2e-11,name+str(e))
def fitcount(ids,seed,tid,t,n,N):
    rng=np.random.default_rng([seed,tid,809,t]);out=np.zeros(N,dtype=np.int64)
    while n:
        order=rng.permutation(len(ids));take=min(n,len(ids));np.add.at(out,ids[order[:take]],1);n-=take
    return out
def main():
    start=time.perf_counter();cfg=json.loads((ROOT/'experiments/fullbatch_confirmation.json').read_text());oldcfg=json.loads((ROOT/'experiments/confirmation.json').read_text())
    for k in cfg:
        if k not in ['methods','seeds']:check(cfg[k]==oldcfg[k],'unchanged config '+k)
    for freeze in ['frozen.json','adaptive_frozen.json','fullbatch_frozen.json']:
        for rel,h in json.loads((ROOT/'experiments'/freeze).read_text())['files'].items():check(sha(ROOT.parent/rel)==h,'frozen '+rel)
    for item in json.loads((OUT/'empirical_audit.json').read_text())['raw_file_hashes']:check(sha(ROOT/item['path'])==item['sha256'],'original raw preserved')
    old=ast.parse((ROOT/'code/quadratic_experiment.py').read_text());new=ast.parse((ROOT/'code/quadratic_fullbatch.py').read_text())
    olddefs={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};newdefs={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    changed=[k for k in olddefs if ast.dump(olddefs[k])!=ast.dump(newdefs[k])];check(changed==['fit','run_one'],'only fit/run_one changed')
    cleaned=copy.deepcopy(newdefs['run_one']);original=olddefs['run_one']
    for node in cleaned.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paired' for t in node.targets):
            source=next(n for n in original.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paired' for t in n.targets));node.value=copy.deepcopy(source.value)
    check(ast.dump(cleaned)==ast.dump(original),'run_one only diagnostics disabled')
    index=json.loads((RAW/'run_index.json').read_text());check(index['completed']==index['planned']==48,'all48 completed')
    for r in index['runs']:check(sha(ROOT/r['file'])==r['sha256'],'result hash')
    runs=[json.loads(p.read_text()) for p in sorted(RAW.glob('*.json')) if p.name not in ['environment.json','run_index.json']]
    tasks={'null':0,'lowrank_noiseless':1,'indefinite_noisy':2,'nonlinear':3};vals={};resources={};hashes=[]
    for r in runs:
        task=r['task'];seed=r['seed'];N=7168
        check(r['config']==cfg and r['runner_sha256']==sha(ROOT/'code/quadratic_fullbatch.py'),'config/runner')
        check(r['paired_checkpoint_diagnostics']==[],'no diagnostic branches executed')
        for key in ['data','access']:
            p=ROOT/r[key+'_file'];h=sha(p);check(h==r[key+'_sha256'],'data/access hash');hashes.append({'path':str(p.relative_to(ROOT)),'sha256':h})
        data=np.load(ROOT/r['data_file']);access=np.load(ROOT/r['access_file'])
        for method,m in r['methods'].items():
            v=risk(m['final_parameters'],task,data);close(v,m['final_risk'],'final_risk');vals[(task,seed,method)]=v
            total=np.zeros(N,dtype=np.int64);steps=[];ns=[];sizes=[];cost=0
            for e in m['events']:
                t=e['t'];left=t*1024;right=left+1024;ids=np.asarray(e['fit_pool_ids']);k=e['width'];per=40*k+3;pc=e['proposal']['proxy'] if e['proposal'] else 0;mc=e['moment_proxy'];available=2_000_000-pc-mc;n=available//per
                check(np.all(ids<right),'no future IDs in fit pool')
                if t:check(e['before_parameters']==m['events'][t-1]['after_parameters'],'model chain')
                check(e['fit']['proxy']==n*per and e['fit']['remainder']==available-n*per,'fitting budget')
                check(e['fit']['steps']==math.ceil(n/128),'fullbatch steps')
                check(e['fit']['full_minibatches']==n//128,'fullbatch count')
                check(e['fit']['terminal_budget_batch_size']==n%128,'terminal remainder batch')
                check(e['event_label_accesses']['fit']==n,'fitting accesses')
                fc=fitcount(ids,seed,tasks[task],t,n,N);total+=fc
                check(int(np.count_nonzero(fc))==e['event_unique_labels']['fit'],'unique fitting IDs')
                check(pc+mc+n*per==e['total_event_proxy']<=2_000_000,'event proxy')
                cost+=e['total_event_proxy'];steps.append(e['fit']['steps']);ns.append(n);sizes.append(n%128)
                close(risk(e['before_parameters'],task,data),e['risk_before'],'before_risk');close(risk(e['after_parameters'],task,data),e['risk_after'],'after_risk')
            check(np.array_equal(total,access[method+'_fit']),'all per-ID fitting counts')
            check(int(total.sum())==m['ledger']['label_accesses']['fit'],'total fitting count')
            resources[(task,seed,method)]={'steps_by_arrival':steps,'fit_accesses_by_arrival':ns,'terminal_batch_sizes':sizes,'steps':sum(steps),'fit_accesses':sum(ns),'total_proxy':cost}
        a=resources[(task,seed,'random_replay128')];b=resources[(task,seed,'random_matched138')]
        check(a==b,'R128/R138 steps/accesses/batch sizes/proxy identical')
    pub=json.loads((ROOT/'results/fullbatch/summary.json').read_text());means=[];contrasts=[]
    for row in pub['summary']:
        st=stats([vals[(row['task'],s,row['method'])] for s in cfg['seeds']])
        for k in ['n','mean','ci95_low','ci95_high']:close(st[k],row[k],'summary_'+k)
        rs=[resources[(row['task'],s,row['method'])] for s in cfg['seeds']]
        check(all(a==rs[0] for a in rs),'resource counts constant')
        check(row['fit_accesses']==rs[0]['fit_accesses'] and row['optimizer_steps']==rs[0]['steps'],'resource summary')
        means.append({**row,**st})
    for row in pub['paired']:
        st=stats([vals[(row['task'],s,row['first'])]-vals[(row['task'],s,row['second'])] for s in cfg['seeds']])
        for k in ['n','mean','ci95_low','ci95_high']:close(st[k],row[k],'paired_'+k)
        contrasts.append({**row,**st})
    # Instrument real gradient calls in one full four-arm fresh replay.
    spec=importlib.util.spec_from_file_location('fullbatch_review',ROOT/'code/quadratic_fullbatch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    fit0,grad0=mod.fit,mod.grad;traces=[];current=[]
    def traced_grad(x,y,p):current.append(len(y));return grad0(x,y,p)
    def traced_fit(*args,**kwargs):
        current.clear();answer=fit0(*args,**kwargs);seen=current.copy()
        check(all(n==128 for n in seen[:-1]) and 1<=seen[-1]<=128,'actual gradient minibatch sizes')
        check(len(seen)==answer['steps'],'actual gradient step count')
        traces.append({'actual_batch_count':len(seen),'actual_full_batches':sum(n==128 for n in seen),'actual_final_batch':seen[-1],'record':answer})
        return answer
    mod.grad=traced_grad;mod.fit=traced_fit
    target=OUT/'fullbatch_replay/results';target.mkdir(parents=True,exist_ok=True)
    if (target/'indefinite_noisy_850000.json').exists():raise RuntimeError('Refusing to overwrite fullbatch replay')
    mod.run_one(850000,'indefinite_noisy',cfg,target)
    replay=json.loads((target/'indefinite_noisy_850000.json').read_text());original=json.loads((RAW/'indefinite_noisy_850000.json').read_text())
    exact=strip_times(replay)==strip_times(original);check(exact,'exact instrumented replay')
    out={'checks':checks,'errors':errors,'max_abs_errors':maximum,'runs':len(runs),'trajectories':len(vals),'events':len(vals)*7,'source_sha256':sha(ROOT/'code/quadratic_fullbatch.py'),'original_source_sha256':sha(ROOT/'code/quadratic_experiment.py'),'changed_ast_definitions':changed,'original_frozen_artifacts_preserved':True,'raw_hashes':hashes,'method_summary':means,'paired_contrasts':contrasts,'resource_counts':{m:resources[('indefinite_noisy',850000,m)] for m in cfg['methods']},'instrumented_replay_exact':exact,'actual_fit_call_traces':traces,'elapsed_seconds':time.perf_counter()-start}
    (OUT/'fullbatch_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['raw_hashes','method_summary','paired_contrasts','actual_fit_call_traces']},indent=2),flush=True)
    if errors:raise AssertionError(errors[:5])

if __name__=='__main__':main()
