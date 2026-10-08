"""Independent reconstruction of frozen main experiment; never alters its artifacts.

Run with .venv/bin/python, OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1.
--replay also runs one full case and a future-input perturbation experiment.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, time
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reviews'
RAW = ROOT/'experiments/confirmation/results'
CODE = ROOT/'code/quadratic_experiment.py'
TASK_ID = {'null': 0, 'lowrank_noiseless': 1, 'indefinite_noisy': 2, 'nonlinear': 3}
ERRORS = []
CHECKS = 0
MAX = {}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok, msg):
    global CHECKS
    CHECKS += 1
    if not ok: ERRORS.append(msg)
def close(a,b,name,tol=2e-11):
    delta = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    MAX[name] = max(MAX.get(name,0.),delta)
    check(delta<=tol, f'{name}: error {delta}')
def stats(v):
    v=np.asarray(v,float);mean=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(len(v)))
    width=float(student_t.ppf(.975,len(v)-1))*se
    return dict(n=len(v),mean=mean,se=se,ci95_low=mean-width,ci95_high=mean+width)
def matrix(p):
    w=np.asarray(p['w']);a=np.asarray(p['a'])
    return np.einsum('ik,k,jk->ij',w,a,w)
def risk(p,task,data):
    b=matrix(p);c=float(p['c']);noise=float(data['noise'])
    if task!='nonlinear':
        err=b-data['teacher_matrix'];return 2*float(np.einsum('ij,ij->',err,err))+c*c+noise*noise
    x=data['evaluator_x'];pred=np.einsum('ni,ij,nj->n',x,b,x)-np.trace(b)+c
    return float(np.mean((pred-data['evaluator_mean'])**2))+noise*noise
def cmpstats(want,got,kind):
    for key in ['n','mean','se','ci95_low','ci95_high']: close(want[key],got[key],kind+'_'+key)
def fcounts(ids,seed,t,per,budget,batch):
    rng=np.random.default_rng([seed,TASK_ID[CURRENT_TASK],809,t]);todo=budget//per
    counts=np.zeros(CURRENT_N,dtype=np.int64);steps=0
    # One permutation is exhausted before the next one is generated; final batch can be partial.
    while todo:
        order=rng.permutation(len(ids));num=min(todo,len(ids));chosen=ids[order[:num]]
        np.add.at(counts,chosen,1);steps+=math.ceil(num/batch);todo-=num
    return counts,steps
def strip_times(obj):
    if isinstance(obj,dict):
        return {k:strip_times(v) for k,v in obj.items() if k not in ['elapsed_seconds','wall_seconds','data_file','access_file']}
    if isinstance(obj,list): return [strip_times(v) for v in obj]
    return obj

def main(replay=False):
    global CURRENT_TASK,CURRENT_N
    begin=time.perf_counter()
    frozen=json.loads((ROOT/'experiments/frozen.json').read_text())
    for rel,digest in frozen['files'].items():check(sha(ROOT.parent/rel)==digest,'frozen hash '+rel)
    cfg=json.loads((ROOT/'experiments/confirmation.json').read_text())
    configsha=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    codesha=sha(CODE);index=json.loads((RAW/'run_index.json').read_text())
    check(index['completed']==index['planned']==48,'run completion count')
    for r in index['runs']:check(sha(ROOT/r['file'])==r['sha256'],'run hash '+r['file'])
    published=json.loads((ROOT/'results/quadratic/summary.json').read_text())
    runs=[json.loads(p.read_text()) for p in sorted(RAW.glob('*.json')) if p.name not in ['environment.json','run_index.json']]
    check(len(runs)==48,'48 raw configurations')
    finals={};checkpoint={};resources={};all_events=[];raw_hashes=[]
    for run in runs:
        task=run['task'];seed=run['seed'];CURRENT_TASK=task;CURRENT_N=cfg['block']*cfg['arrivals']
        check(run['runner_sha256']==codesha,'runner hash '+str(seed)+task)
        check(run['config_sha256']==configsha and run['config']==cfg,'configuration '+str(seed)+task)
        for name,key in [('data','data'),('access','access')]:
            p=ROOT/run[key+'_file'];actual=sha(p);check(actual==run[key+'_sha256'],name+' hash '+str(seed)+task)
            raw_hashes.append({'path':str(p.relative_to(ROOT)),'sha256':actual})
        data=np.load(ROOT/run['data_file']);access=np.load(ROOT/run['access_file'])
        check(len(data['y'])==CURRENT_N and data['x'].shape==(CURRENT_N,cfg['d']),'data shape')
        q=data['teacher_rotation'];close(q.T@q,np.eye(cfg['d']),'teacher_orthogonality')
        expectedvals=np.zeros(cfg['d'])
        if task=='lowrank_noiseless':expectedvals[:2]=[.8,.45]
        if task=='indefinite_noisy':expectedvals[:4]=[.65,-.5,.3,-.2]
        close(data['teacher_matrix'],(q*expectedvals)@q.T,'teacher_matrix')
        ex=data['evaluator_x'];z=ex@q
        eq=(.8*np.tanh(1.5*z[:,0])*np.tanh(1.5*z[:,1])+.4*np.sin(z[:,2])) if task=='nonlinear' else np.einsum('ni,ij,nj->n',ex,data['teacher_matrix'],ex)-np.trace(data['teacher_matrix'])
        close(eq,data['evaluator_mean'],'evaluator_mean')
        for method,m in run['methods'].items():
            check(len(m['events'])==7,'event count');rr=risk(m['final_parameters'],task,data)
            close(rr,m['final_risk'],'final_risk');close(risk(m['initial_parameters'],task,data),m['initial_risk'],'initial_risk')
            finals[(task,seed,method)]=rr
            capacity=128 if method in ['historical_moment','random_replay128'] else 138
            counts={k:np.zeros(CURRENT_N,dtype=np.int64) for k in ['fit','proposal','moment']}
            resrng=np.random.default_rng([seed,TASK_ID[task],500]);retained=[];seen=0;old=[];cumproxy=0
            for e in m['events']:
                t=e['t'];left=t*1024;right=left+1024;ids=np.arange(left,right);k=e['width'];kold=k-(t>0 and not method.startswith('fixed'))
                check(e['arrival_range']==[left,right],'arrival range')
                check(e['width']==(7 if method.startswith('fixed') else t+1),'width')
                fitids=np.r_[old,ids].astype(int);check(np.array_equal(fitids,e['fit_pool_ids']),'fit eligible ID pool')
                before=risk(e['before_parameters'],task,data);after=risk(e['after_parameters'],task,data)
                close(before,e['risk_before'],'before_risk');close(after,e['risk_after'],'after_risk')
                if t:check(e['before_parameters']==m['events'][t-1]['after_parameters'],'model event chain')
                momentcost=1024*(12**2+12) if method=='historical_moment' else 0
                propcost=0;pc=np.zeros(CURRENT_N,dtype=np.int64);mc=np.zeros(CURRENT_N,dtype=np.int64)
                if momentcost:mc[ids]=1
                if t and not method.startswith('fixed'):
                    if method=='historical_moment':propcost=12**2*kold+12*kold+10*12**3
                    elif method in ['fresh_residual','reservoir_residual']:
                        pis=ids if method=='fresh_residual' else np.asarray(old,dtype=int);pc[pis]+=1
                        propcost=len(pis)*(12**2+12*kold+3*kold+12)+10*12**3
                    elif method=='learned_probe':
                        gen=np.random.default_rng([seed,TASK_ID[task],701,t]);gen.normal(size=(12,4))
                        for _ in range(20):
                            chosen=gen.choice(len(old),min(128,len(old)),replace=False);np.add.at(pc,np.asarray(old)[chosen],1)
                        pc[np.asarray(old)]+=1
                        propcost=20*min(128,len(old))*(5*12*4+12*kold+8*4)+len(old)*(12*4+12*kold+5*4)
                    close(propcost,e['proposal']['proxy'],'candidate_proxy',0)
                    if task!='nonlinear':
                        w=np.asarray(e['proposal']['direction']);D=data['teacher_matrix']-matrix(e['before_parameters']);v=float(w@D@w);oracle=float(np.max(abs(np.linalg.eigvalsh(D))))
                        close(v,e['evaluator_direction_coefficient'],'direction_coefficient')
                        close(oracle,e['evaluator_best_rank1_coefficient_magnitude'],'direction_oracle')
                per=3*12*k+4*k+3;budget=2_000_000-momentcost-propcost
                fc,steps=fcounts(fitids,seed,t,per,budget,128);fcost=int(fc.sum())*per
                check(steps==e['fit']['steps'],'fit step reconstruction')
                check(fcost==e['fit']['proxy'] and budget-fcost==e['fit']['remainder'],'fit proxy reconstruction')
                check(momentcost==e['moment_proxy'],'moment proxy')
                check(momentcost+propcost+fcost==e['total_event_proxy']<=2_000_000,'event budget')
                cumproxy+=momentcost+propcost+fcost
                for kind,cc in [('fit',fc),('proposal',pc),('moment',mc)]:
                    check(not np.any(cc[right:]),'no future IDs reconstructed')
                    check(int(cc.sum())==e['event_label_accesses'][kind],'event access conservation '+kind)
                    check(int(np.count_nonzero(cc))==e['event_unique_labels'][kind],'event unique access '+kind)
                    counts[kind]+=cc
                for ii in ids:
                    seen+=1
                    if len(retained)<capacity:retained.append(int(ii))
                    else:
                        j=int(resrng.integers(seen))
                        if j<capacity:retained[j]=int(ii)
                check(retained==e['retained_ids_after'],'independent reservoir reconstruction')
                check(e['parameters']==13*k+1 and e['persistent_model_optimizer_words']==3*(13*k+1),'model memory')
                check(e['persistent_retention_words']==capacity*14+(144 if method=='historical_moment' else 0),'retention memory')
                old=retained.copy()
                all_events.append({'task':task,'seed':seed,'method':method,'event':t,'risk':after,'width':k,'proxy':e['total_event_proxy'],'fit_accesses':int(fc.sum()),'proposal_accesses':int(pc.sum()),'moment_accesses':int(mc.sum())})
            check(m['final_parameters']==m['events'][-1]['after_parameters'],'final model chain')
            for kind,v in counts.items():
                check(np.array_equal(v,access[method+'_'+kind]),'ID count exact reconstruction '+task+str(seed)+method+kind)
                check(int(v.sum())==m['ledger']['label_accesses'][kind],'final access conservation')
                check(int(np.count_nonzero(v))==m['ledger']['unique_label_accesses'][kind],'final unique conservation')
            check(cumproxy==sum(m['ledger']['proxy'].values())+m['ledger']['extra']['eigendecomposition'],'cumulative proxy')
            resources[(task,seed,method)]={'proxy':cumproxy,'fit':int(counts['fit'].sum()),'proposal':int(counts['proposal'].sum()),'persistent':e['persistent_retention_words']+e['persistent_model_optimizer_words']+6}
        ref=run['methods']['random_matched138']
        for d in run['paired_checkpoint_diagnostics']:
            t=d['t'];right=(t+1)*1024;left=t*1024
            check(d['before_parameters']==ref['events'][t]['before_parameters'],'same checkpoint model')
            check(d['old_replay_ids']==ref['events'][t-1]['retained_ids_after'],'same checkpoint retention')
            check(d['available_through_exclusive']==right,'checkpoint prefix')
            close(risk(d['before_parameters'],task,data),d['risk_before'],'checkpoint_before_risk')
            for method,item in d['choices'].items():
                rr=risk(item['trained_parameters'],task,data);close(rr,item['risk_after_identical_refit'],'checkpoint_refit_risk')
                checkpoint[(task,seed,t,method,'risk_after_identical_refit')]=rr
                histcost=right*156 if method=='historical_moment' else 0
                check(item['historical_reconstruction_proxy']==histcost,'diagnostic historical work charged')
                check(item['historical_reconstruction_label_accesses']==(right if histcost else 0),'diagnostic historical labels charged')
                check(item['total_diagnostic_proxy']==histcost+item['proposal']['proxy']+item['fit_account']['proxy'],'diagnostic total charge')
                per=3*12*(t+1)+4*(t+1)+3
                check(item['fit_account']['proxy']==(2_000_000//per)*per,'identical refit budget')
                if task!='nonlinear':
                    w=np.asarray(item['proposal']['direction']);D=data['teacher_matrix']-matrix(d['before_parameters']);v=float(w@D@w);gain=2*v*v;oracle=float(np.max(abs(np.linalg.eigvalsh(D))))
                    close(gain,item['population_best_gain_along_direction'],'checkpoint_gain')
                    close(2*oracle*oracle,item['population_oracle_rank1_gain'],'checkpoint_oracle')
                    checkpoint[(task,seed,t,method,'population_best_gain_along_direction')]=gain
    reconstructed={'summary':[],'primary_contrasts':[],'checkpoint_contrasts':[],'extra_same_reservoir_contrasts':[]}
    for row in published['summary']:
        task,method=row['task'],row['method'];ss=stats([finals[(task,s,method)] for s in cfg['seeds']]);cmpstats(ss,row,'method_summary')
        rsrc=[resources[(task,s,method)] for s in cfg['seeds']]
        for key,col in [('mean_total_proxy','proxy'),('mean_fit_accesses','fit'),('mean_proposal_accesses','proposal')]:close(row[key],np.mean([r[col] for r in rsrc]),'resource_summary')
        check(row['persistent_words']==rsrc[0]['persistent'],'persistent summary')
        reconstructed['summary'].append({**row,**ss})
    for row in published['primary_contrasts']:
        vals=[finals[(row['task'],s,row['first'])]-finals[(row['task'],s,row['second'])] for s in cfg['seeds']];ss=stats(vals);cmpstats(ss,row,'paired_summary');reconstructed['primary_contrasts'].append({**row,**ss})
    for row in published['checkpoint_contrasts']:
        metric=row['metric'].split('_first_minus_second')[0]
        vals=[np.mean([checkpoint[(row['task'],s,t,row['first'],metric)]-checkpoint[(row['task'],s,t,row['second'],metric)] for t in range(1,7)]) for s in cfg['seeds']]
        ss=stats(vals);cmpstats(ss,row,'checkpoint_summary');reconstructed['checkpoint_contrasts'].append({**row,**ss})
    for task in cfg['tasks']:
        vals=[finals[(task,s,'historical_moment')]-finals[(task,s,'random_replay128')] for s in cfg['seeds']]
        reconstructed['extra_same_reservoir_contrasts'].append({'task':task,'first':'historical_moment','second':'random_replay128',**stats(vals)})
    with (ROOT/'results/quadratic/events.csv').open() as f:csvrows=list(csv.DictReader(f))
    bykey={(r['task'],int(r['seed']),r['method'],int(r['event'])):r for r in csvrows}
    for r in all_events:
        saved=bykey[(r['task'],r['seed'],r['method'],r['event'])]
        for k in ['risk','width','proxy','fit_accesses','proposal_accesses','moment_accesses']:close(float(saved[k]),r[k],'event_csv_'+k)
    for name,key in [('method_summary.csv','summary'),('paired_contrasts.csv','primary_contrasts'),('checkpoint_contrasts.csv','checkpoint_contrasts')]:
        with (ROOT/'results/quadratic'/name).open() as f:rows=list(csv.DictReader(f))
        check(len(rows)==len(reconstructed[key]),'CSV row count '+name)
        for a,b in zip(rows,reconstructed[key]):
            for k,v in b.items():
                if isinstance(v,(float,int)):close(float(a[k]),v,'csv_'+name+'_'+k)
                else:check(a[k]==v,'CSV text '+name+k)
    out={'checks':CHECKS,'errors':ERRORS,'max_abs_errors':MAX,'runner_sha256':codesha,'config_canonical_sha256':configsha,'runs':len(runs),'trajectories':len(finals),'events':len(all_events),'checkpoint_choices':48*6*5,'raw_file_hashes':raw_hashes,'reconstructed':reconstructed,'elapsed_seconds':time.perf_counter()-begin}
    (OUT/'empirical_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['raw_file_hashes','reconstructed']},indent=2),flush=True)
    if replay:
        spec=importlib.util.spec_from_file_location('quadratic_review_import',CODE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        target=OUT/'empirical_replay/results';target.mkdir(exist_ok=True,parents=True)
        check(not (target/'indefinite_noisy_820000.json').exists(),'replay destination new')
        if (target/'indefinite_noisy_820000.json').exists():raise RuntimeError('Refusing to overwrite replay')
        mod.run_one(820000,'indefinite_noisy',cfg,target)
        replayed=json.loads((target/'indefinite_noisy_820000.json').read_text());original=json.loads((RAW/'indefinite_noisy_820000.json').read_text())
        eq=strip_times(replayed)==strip_times(original);check(eq,'exact full replay excluding paths and timings')
        data=np.load(ROOT/original['data_file']);x=data['x'].copy();y=data['y'].copy();cut=3*1024
        x[cut:]=-x[cut:];y[cut:]=-y[cut:]+1.
        prefix=[]
        for method in cfg['methods']:
            rr,_=mod.learner(820000,'indefinite_noisy',method,cfg,x,y)
            def selection(event):return {k:strip_times(event[k]) for k in ['before_parameters','after_parameters','before_optimizer','proposal','fit','fit_pool_ids','retained_ids_after','event_label_accesses','ledger']}
            same=all(selection(a)==selection(b) for a,b in zip(rr['events'][:3],original['methods'][method]['events'][:3]))
            different=rr['final_parameters']!=original['methods'][method]['final_parameters']
            check(same,'future perturbation prefix '+method);check(different,'future perturbation effective '+method)
            prefix.append({'method':method,'identical_first_three_events':same,'different_final_parameters':different})
            (OUT/'empirical_replay'/('perturbed_'+method+'.json')).write_text(json.dumps(rr,indent=2))
        replay_report={'exact_full_replay':eq,'task':'indefinite_noisy','seed':820000,'future_cut_exclusive':cut,'future_change':'x_suffix -> -x_suffix; y_suffix -> 1-y_suffix','prefix_results':prefix,'errors':ERRORS,'checks':CHECKS,'code_sha256':codesha,'elapsed_seconds':time.perf_counter()-begin}
        (OUT/'empirical_replay_check.json').write_text(json.dumps(replay_report,indent=2));print(json.dumps(replay_report,indent=2),flush=True)
    if ERRORS:raise AssertionError(f'{len(ERRORS)} audit failures: '+str(ERRORS[:5]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');main(p.parse_args().replay)
