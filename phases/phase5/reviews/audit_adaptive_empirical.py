"""Independent gate reconstruction and current-validation perturbation audit."""
from pathlib import Path
import hashlib, importlib.util, json, math, sys, time
import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reviews';RAW=ROOT/'experiments/adaptive_confirmation/results'
TASKID={'null':0,'lowrank_noiseless':1,'indefinite_noisy':2,'nonlinear':3}
errors=[];checks=0;maximum={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok,msg):
    global checks
    checks+=1
    if not ok:errors.append(msg)
def close(a,b,name,tol=2e-11):
    err=float(np.max(abs(np.asarray(a)-np.asarray(b))));maximum[name]=max(maximum.get(name,0.),err);check(err<=tol,name+': '+str(err))
def pred(x,p):
    w=np.asarray(p['w']);a=np.asarray(p['a']);b=np.einsum('ik,k,jk->ij',w,a,w)
    return np.einsum('ni,ij,nj->n',x,b,x)-np.trace(b)+p['c']
def risk(p,task,data):
    w=np.asarray(p['w']);b=np.einsum('ik,k,jk->ij',w,p['a'],w)
    if task=='nonlinear':return float(np.mean((pred(data['evaluator_x'],p)-data['evaluator_mean'])**2))+float(data['noise'])**2
    return 2*float(np.sum((b-data['teacher_matrix'])**2))+float(p['c'])**2+float(data['noise'])**2
def fitcounts(ids,rngseed,budget,k,N):
    per=40*k+3;rng=np.random.default_rng(rngseed);v=np.zeros(N,dtype=np.int64);todo=budget//per;steps=0
    while todo:
        ix=rng.permutation(len(ids));n=min(todo,len(ids));np.add.at(v,ids[ix[:n]],1);todo-=n;steps+=math.ceil(n/128)
    return v,steps,int(v.sum())*per
def stats(v):
    v=np.asarray(v);mu=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)));h=float(student_t.ppf(.975,len(v)-1)*se)
    return dict(n=len(v),mean=mu,ci95_low=mu-h,ci95_high=mu+h)

def main():
    started=time.perf_counter();cfg=json.loads((ROOT/'experiments/adaptive_confirmation.json').read_text());N=7168
    frozen=json.loads((ROOT/'experiments/adaptive_frozen.json').read_text())
    for rel,h in frozen['files'].items():check(sha(ROOT.parent/rel)==h,'frozen '+rel)
    index=json.loads((RAW/'run_index.json').read_text());check(index['completed']==index['planned']==48,'completed48')
    for r in index['runs']:check(sha(ROOT/r['file'])==r['sha256'],'raw run hash')
    runs=[json.loads(p.read_text()) for p in sorted(RAW.glob('*.json')) if p.name not in ['environment.json','run_index.json']]
    metrics={};hashes=[];decision_rows=[]
    for run in runs:
        seed,task=run['seed'],run['task'];tid=TASKID[task]
        check(run['runner_sha256']==sha(ROOT/'code/adaptive_quadratic.py'),'runner hash')
        check(run['dependency_sha256']==sha(ROOT/'code/quadratic_experiment.py'),'dependency hash')
        check(run['config']==cfg,'config')
        for k in ['data','access']:
            p=ROOT/run[k+'_file'];h=sha(p);check(h==run[k+'_sha256'],'data hash '+k);hashes.append({'path':str(p.relative_to(ROOT)),'sha256':h})
        data=np.load(ROOT/run['data_file']);access=np.load(ROOT/run['access_file']);x,y=data['x'],data['y']
        for method,m in run['methods'].items():
            totals={k:np.zeros(N,dtype=np.int64) for k in ['fit','proposal','moment','validation']};retained=[];seen=0;resrng=np.random.default_rng([seed,tid,500]);grow=hold=cont=increases=0;cost=rejectcost=0
            for e in m['events']:
                t=e['t'];left=1024*t;middle=left+768;right=left+1024;fresh=np.arange(left,middle);val=np.arange(middle,right);k=len(e['before_parameters']['a'])
                check(e['retained_ids_before']==retained,'retained before')
                ids=np.r_[retained,fresh].astype(int);check(np.array_equal(ids,e['fit_pool_ids']),'eligible fit pool')
                check(not np.any((ids>=middle)&(ids<right)),'current validation exclusion')
                check(np.max(ids)<right,'future exclusion')
                branches=e['branch_parameters'];can_grow='growth' in branches;gated=t>0 and method!='fixed7_ungated'
                vcost=256*sum(15*len(p['a'])+1 for p in branches.values()) if gated else 0
                pcost=768*(144+12*k+3*k+12)+10*12**3 if can_grow and method=='fresh_adaptive' else 0
                close(vcost,e['validation_proxy'],'validation proxy',0);close(pcost,e['proposal_proxy'],'proposal proxy',0)
                available=2_000_000-vcost-pcost;counts={kk:np.zeros(N,dtype=np.int64) for kk in totals}
                if pcost:counts['proposal'][fresh]=1
                if gated:counts['validation'][val]=1
                fitcost=0
                for bi,name in enumerate(['continuation']+(['growth'] if can_grow else [])):
                    budget=(available//2 if bi==0 else available-available//2) if can_grow else available
                    fc,steps,spent=fitcounts(ids,[seed,tid,809,t,bi],budget,k+bi,N);counts['fit']+=fc;fitcost+=spent
                    check(spent==e['branch_fit_accounts'][name]['proxy'],'branch fit cost')
                    check(steps==e['branch_fit_accounts'][name]['steps'],'branch fit steps')
                    check(budget-spent==e['branch_fit_accounts'][name]['remainder'],'branch budget remainder')
                check(fitcost+vcost+pcost==e['total_event_proxy']<=2_000_000,'event budget')
                cost+=e['total_event_proxy']
                tests={}
                if gated:
                    base=(pred(x[val],branches['incumbent'])-y[val])**2
                    for name in e['branch_fit_accounts']:
                        diff=base-(pred(x[val],branches[name])-y[val])**2;mu=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(256));pen=.002+.001*(len(branches[name]['a'])-k)
                        tests[name]={'mean_gain':mu,'sample_se':se,'penalty':pen,'threshold':2*se+pen,'qualified':mu>2*se+pen,'penalized_mean_gain':mu-pen}
                        for kk,v in tests[name].items():
                            if isinstance(v,(bool,np.bool_)):check(v==e['tests'][name][kk],'gate qualification')
                            else:close(v,e['tests'][name][kk],'gate_'+kk)
                    allowed=[name for name in tests if tests[name]['qualified']]
                    selected=max(allowed,key=lambda name:tests[name]['penalized_mean_gain']) if allowed else 'incumbent'
                else:selected='continuation'
                check(selected==e['selected_branch'],'selected branch')
                check(e['after_parameters']==branches[selected],'committed exact selected parameters')
                action='grow' if selected=='growth' else ('hold' if selected=='incumbent' else ('warmup' if not t else 'continue'))
                check(action==e['action'],'action')
                rejected=sum(v['proxy'] for name,v in e['branch_fit_accounts'].items() if name!=selected);check(rejected==e['rejected_branch_proxy'],'rejected cost');rejectcost+=rejected
                for kk,v in counts.items():
                    check(int(v.sum())==e['event_accesses'][kk],'event ID access total '+kk)
                    check(int(np.count_nonzero(v))==e['event_unique_accesses'][kk],'event unique '+kk)
                    check(not np.any(v[right:]),'no future access '+kk);totals[kk]+=v
                br,ar=risk(e['before_parameters'],task,data),risk(e['after_parameters'],task,data)
                close(br,e['risk_before'],'before_risk');close(ar,e['risk_after'],'after_risk')
                if t:
                    grow+=action=='grow';hold+=action=='hold';cont+=action=='continue';increases+=ar>br+1e-12
                    check(e['before_parameters']==m['events'][t-1]['after_parameters'],'model chain')
                decision_rows.append({'task':task,'seed':seed,'method':method,'event':t,'selected_branch':selected,'action':action,'width':len(branches[selected]['a']),'risk_before':br,'risk_after':ar})
                for ii in range(left,right):
                    seen+=1
                    if len(retained)<138:retained.append(ii)
                    else:
                        j=int(resrng.integers(seen))
                        if j<138:retained[j]=ii
                check(e['retained_ids_after']==retained,'reservoir after')
            for kk,v in totals.items():
                check(np.array_equal(v,access[method+'_'+kk]),'all ID counts '+kk)
                check(int(v.sum())==m['ledger']['label_accesses'][kk],'final total '+kk)
            fr=risk(m['final_parameters'],task,data);close(fr,m['final_risk'],'final_risk')
            metrics[(task,seed,method)]={'risk':fr,'width':m['events'][-1]['width'],'grow':grow,'hold':hold,'continue':cont,'increases':increases,'cost':cost,'rejected':rejectcost}
    published=json.loads((ROOT/'results/adaptive/summary.json').read_text());summary=[];paired=[]
    for row in published['summary']:
        vals=[metrics[(row['task'],s,row['method'])] for s in cfg['seeds']];st=stats([r['risk'] for r in vals])
        for k,v in st.items():close(v,row[k],'summary_'+k)
        for key,col in [('grow','grow'),('hold','hold'),('continue','continue'),('evaluator_risk_increases','increases')]:check(sum(r[col] for r in vals)==row[key],'summary action '+key)
        for key,col in [('mean_total_proxy','cost'),('mean_rejected_proxy','rejected'),('mean_width','width')]:close(np.mean([r[col] for r in vals]),row[key],'summary_'+key)
        summary.append({**row,**st})
    for row in published['paired']:
        st=stats([metrics[(row['task'],s,row['first'])]['risk']-metrics[(row['task'],s,row['second'])]['risk'] for s in cfg['seeds']])
        for k,v in st.items():close(v,row[k],'paired_'+k)
        paired.append({**row,**st})
    out={'checks':checks,'errors':errors,'max_abs_errors':maximum,'runs':len(runs),'trajectories':len(metrics),'events':len(decision_rows),'hashes':hashes,'summary':summary,'paired':paired,'decision_reconstruction':decision_rows,'elapsed_seconds':time.perf_counter()-started}
    (OUT/'adaptive_empirical_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['hashes','summary','paired','decision_reconstruction']},indent=2),flush=True)
    sys.path.insert(0,str(ROOT/'code'));import adaptive_quadratic as mod
    run=json.loads((RAW/'indefinite_noisy_840000.json').read_text());data=np.load(ROOT/run['data_file']);x=data['x'];y=data['y'].copy();y[1792:2048]=10*np.random.default_rng(4371).normal(size=256)
    short={**cfg,'arrivals':2};perturb=[];target=OUT/'adaptive_validation_perturbation';target.mkdir(exist_ok=True)
    for method in cfg['methods']:
        p=target/(method+'.json')
        if p.exists():raise RuntimeError('Refusing to overwrite '+str(p))
        rr,_=mod.learner(840000,'indefinite_noisy',method,short,x,y);orig=run['methods'][method]
        e=rr['events'][1];base=orig['events'][1]
        same=e['branch_parameters']==base['branch_parameters'] and e['proposal']==base['proposal'] and e['before_optimizer']==base['before_optimizer']
        check(same,'current validation branch independence '+method)
        changed=e['tests']!=base['tests'];check(changed or method=='fixed7_ungated','validation perturbation effective '+method)
        perturb.append({'method':method,'branch_parameters_proposal_optimizer_identical':same,'validation_stats_changed':changed,'original_selection':base['selected_branch'],'perturbed_selection':e['selected_branch']})
        p.write_text(json.dumps(rr,indent=2))
    pcheck={'checks':checks,'errors':errors,'task':'indefinite_noisy','seed':840000,'changed_label_indices':[1792,2048],'new_labels':'10*Normal(seed4371)','prefix_events_run':2,'results':perturb,'elapsed_seconds':time.perf_counter()-started}
    (OUT/'adaptive_validation_perturbation_check.json').write_text(json.dumps(pcheck,indent=2));print(json.dumps(pcheck,indent=2),flush=True)
    if errors:raise AssertionError(errors[:5])

if __name__=='__main__':main()
