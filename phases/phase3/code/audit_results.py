"""Independent confirmation-output audit and full-sequence diagnostics.

Run while work continues: .venv/bin/python phase3/code/audit_results.py
Require every planned result: .venv/bin/python phase3/code/audit_results.py --require-complete
No fitting is performed. The script verifies saved decisions, accounting,
history consistency, and terminal risks, and computes all primary paired
effects and gate-conservatism diagnostics from saved outputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
TASKS={'null':0,'linear':1,'additive':2,'interaction':3}


def estimate(v):
    a=np.asarray(v,dtype=float);n=len(a)
    hw=float(student_t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=float(a.mean()) if n else None,t95_halfwidth=hw,
                min=float(a.min()) if n else None,max=float(a.max()) if n else None)


def size(p):return sum(np.asarray(v).size for v in p.values())


def evaluation(seed,task,n):
    ss=np.random.SeedSequence([seed,TASKS[task]]).spawn(5)
    x=np.random.default_rng(ss[3]).normal(size=(n,6))
    if task=='null':g=np.zeros(n)
    elif task=='linear':g=2*x[:,0]
    elif task=='additive':g=2.5*np.tanh(2*x[:,0]+.6)-2*np.tanh(2*x[:,1]-.5)+1.7*np.tanh(2*x[:,2]+.3)-1.5*np.tanh(2*x[:,3])
    else:g=4*np.tanh(2*x[:,0])*np.tanh(2*x[:,1])+2*np.tanh(2*x[:,2])*np.tanh(2*x[:,3])
    return x,.05+.9*expit(g)


def risk(x,q,p):
    p={k:np.asarray(v) for k,v in p.items()}
    pr=expit(np.tanh(x@p['w']+p['b'])@p['a']+p['c'])
    return float(np.mean(pr*pr-2*pr*q+q))


def check_file(path):
    data=json.loads(path.read_text());cfg=data['config'];runs=data['runs']
    expected={(task,seed) for task in cfg['tasks'] for seed in range(cfg['seed_base'],cfg['seed_base']+cfg['seeds'])}
    actual=[(r['task'],r['seed']) for r in runs]
    assert len(set(actual))==len(actual), 'duplicate seed/task run'
    assert set(actual)<=expected, 'unplanned seed/task run'
    max_terminal_error=0.;method_diagnostics={};histories=0
    for r in runs:
        assert set(r['methods'])==set(cfg['methods'])
        assert r['config']==cfg
        x,q=evaluation(r['seed'],r['task'],cfg['ntest'])
        max_terminal_error=max(max_terminal_error,abs(r['initial_risk']-risk(x,q,r['start_parameters'])))
        for method,m in r['methods'].items():
            events=m['events'];assert len(events)==cfg['events']
            previous=r['start_parameters'];previous_risk=r['initial_risk']
            work=cfg['ntrain']*cfg['presteps']*size(previous);audit=0;geometry=0
            key=r['task']+'/'+method
            d=method_diagnostics.setdefault(key,dict(events=0,grow=0,continue_=0,hold=0,
                evaluator_positive=0,evaluator_positive_rejected=0,empirical_positive=0,
                evaluator_positive_rejected_despite_empirical_positive=0,additive_term_alone_blocks=0,
                retained_evaluator_risk_increases=0,growth_evaluator_harm=0,
                limiting_additive_share=[],final_widths=[],labels=[],pool_large_unique=[]))
            for number,e in enumerate(events,1):
                histories+=1;d['events']+=1
                assert e['t']==number and e['incumbent_parameters']==previous
                assert abs(e['incumbent_risk']-previous_risk)<1e-12
                inc,cont,grown=[e[k+'_parameters'] for k in ['incumbent','continuation','growth']]
                assert size(cont)==size(inc) and size(grown)==size(inc)+8
                pairs=[e[k] for k in ['continuation_vs_incumbent','growth_vs_incumbent','growth_vs_continuation']]
                for c in pairs:
                    assert c['n']==cfg['naudit'] and abs(c['delta']-cfg['alpha']/(3*cfg['events']))<1e-15
                    additive=14*np.log(2/c['delta'])/(3*(c['n']-1))
                    rad=np.sqrt(2*c['variance']*np.log(2/c['delta'])/c['n'])+additive
                    assert abs(c['radius']-rad)<1e-12 and abs(c['lcb']-(c['mean']-rad))<1e-12
                ci,gi,gc=pairs
                assert abs((gi['mean']-ci['mean'])-gc['mean'])<1e-12
                if cfg['gate']=='forced':expected_action='grow'
                elif cfg['gate']=='betting':
                    expected_action='grow' if gi['betting_accept'] and gc['betting_accept'] else ('continue' if ci['betting_accept'] else 'hold')
                else:
                    criterion='lcb' if cfg['gate']=='eb' else 'mean'
                    expected_action='grow' if gi[criterion]>cfg['tau'] and gc[criterion]>cfg['tau'] else ('continue' if ci[criterion]>0 else 'hold')
                assert e['action']==expected_action
                pchosen={'grow':grown,'continue':cont,'hold':inc}[expected_action]
                chosenrisk=e[{'grow':'growth_risk','continue':'continuation_risk','hold':'incumbent_risk'}[expected_action]]
                assert e['retained_parameters']==pchosen and e['retained_risk']==chosenrisk
                assert e['parameters']==size(pchosen)==1+8*e['width']
                d[expected_action if expected_action!='continue' else 'continue_']+=1
                meta=e['proposal']
                work+=cfg['ntrain']*(cfg['branch_steps']*(size(cont)+size(grown))+
                     meta['candidate_count']*meta['candidate_steps']*7)
                if method=='ssd':geometry+=cfg['ntrain']*len(inc['a'])*36+216*len(inc['a'])
                audit+=cfg['naudit']*(size(inc)+size(cont)+size(grown))
                assert e['training_parameter_examples']==work and e['audit_parameter_examples']==audit
                assert e['geometry_arithmetic_proxy']==geometry
                assert e['audit_labels']==number*cfg['naudit']
                evaluator_positive=e['incumbent_risk']-e['growth_risk']>cfg['tau'] and e['continuation_risk']-e['growth_risk']>cfg['tau']
                empirical_positive=gi['mean']>cfg['tau'] and gc['mean']>cfg['tau']
                d['evaluator_positive']+=int(evaluator_positive);d['empirical_positive']+=int(empirical_positive)
                rejected=e['action']!='grow'
                d['evaluator_positive_rejected']+=int(evaluator_positive and rejected)
                d['evaluator_positive_rejected_despite_empirical_positive']+=int(evaluator_positive and rejected and empirical_positive)
                addpart=14*np.log(2/gi['delta'])/(3*(gi['n']-1))
                d['additive_term_alone_blocks']+=int(cfg['gate']=='eb' and evaluator_positive and rejected and empirical_positive and
                                                    min(gi['mean'],gc['mean'])-addpart<=cfg['tau'])
                limiting=min([gi,gc],key=lambda c:c['lcb']-cfg['tau'])
                d['limiting_additive_share'].append(addpart/limiting['radius'])
                d['retained_evaluator_risk_increases']+=int(e['retained_risk']>e['incumbent_risk']+1e-10)
                d['growth_evaluator_harm']+=int(e['action']=='grow' and e['growth_risk']>e['incumbent_risk']+1e-10)
                previous=pchosen;previous_risk=chosenrisk
            d['final_widths'].append(m['final_width']);d['labels'].append(m['audit_labels'])
            assert m['final_risk']==events[-1]['retained_risk'] and m['final_width']==events[-1]['width']
            assert m['training_parameter_examples']==work and m['audit_parameter_examples']==audit
            assert m['tail_parameter_examples']==work+cfg['ntrain']*cfg['tail_steps']*size(previous)
            assert m['fixed_matched_steps']==work//(cfg['ntrain']*size(m['fixed_parameters']))
            for val,par in [('final_risk',events[-1]['retained_parameters']),('tail_risk',m['tail_parameters']),
                            ('fixed_matched_risk',m['fixed_parameters']),('fixed_matched_tail_risk',m['fixed_tail_parameters'])]:
                max_terminal_error=max(max_terminal_error,abs(m[val]-risk(x,q,par)))
        for name in ['small','large']:
            p=r['baselines'][name+'_pool_matched'];pp=size(p['parameters'])
            budget=r['methods'].get('residual',next(iter(r['methods'].values())))['training_parameter_examples']
            assert 0<=budget-p['parameter_examples']<pp
            assert p['available_labels']==cfg['ntrain']+cfg['events']*cfg['naudit']
            assert 0<=p['unique_labels']<=p['available_labels']
            max_terminal_error=max(max_terminal_error,abs(p['risk']-risk(x,q,p['parameters'])))
    assert max_terminal_error<1e-12
    summaries={}
    for task in cfg['tasks']:
        rr=[r for r in runs if r['task']==task]
        if not rr:continue
        contrasts={}
        if 'residual' in cfg['methods']:
            for other in cfg['methods']:
                if other!='residual':
                    contrasts['residual_minus_'+other]=estimate([r['methods']['residual']['final_risk']-r['methods'][other]['final_risk'] for r in rr])
                    contrasts['residual_tail_minus_'+other+'_tail']=estimate([r['methods']['residual']['tail_risk']-r['methods'][other]['tail_risk'] for r in rr])
            contrasts['residual_minus_fixed_matched']=estimate([r['methods']['residual']['final_risk']-r['methods']['residual']['fixed_matched_risk'] for r in rr])
            contrasts['residual_tail_minus_fixed_matched_tail']=estimate([r['methods']['residual']['tail_risk']-r['methods']['residual']['fixed_matched_tail_risk'] for r in rr])
            for base in ['small','large']:
                contrasts['residual_minus_'+base+'_pool_matched']=estimate([r['methods']['residual']['final_risk']-r['baselines'][base+'_pool_matched']['risk'] for r in rr])
        summaries[task]=dict(contrasts=contrasts,
            means={m:estimate([r['methods'][m]['final_risk'] for r in rr]) for m in cfg['methods']},
            pool_unique_labels={b:estimate([r['baselines'][b+'_pool_matched']['unique_labels'] for r in rr]) for b in ['small','large']})
    for d in method_diagnostics.values():
        d['limiting_additive_share']=estimate(d['limiting_additive_share'])
        d['runs_with_multiple_growths']=sum(w>=4 for w in d['final_widths'])
        d['final_width_summary']=estimate(d['final_widths'])
        del d['pool_large_unique']
    return dict(file=path.name,source_sha256=data['code_sha256'],config=cfg,
                complete=set(actual)==expected,completed_runs=len(runs),planned_runs=len(expected),
                missing_runs=sorted(expected-set(actual)),checked_events=histories,
                maximum_independent_terminal_risk_error=max_terminal_error,
                summaries=summaries,sequence_diagnostics=method_diagnostics)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    results=[]
    for path in sorted((ROOT/'results').glob('confirm_*.json')):
        try:results.append(check_file(path))
        except json.JSONDecodeError:
            if args.require_complete:raise
            results.append(dict(file=path.name,complete=False,status='file being written; retry'))
    complete=len(results)==3 and all(r['complete'] for r in results)
    hashes={r['source_sha256'] for r in results if 'source_sha256' in r}
    assert len(hashes)<=1,'confirmation batches used different code versions'
    out=dict(status='complete confirmation audit' if complete else 'partial audit; confirmation still in progress',
             all_planned_files_complete=complete,source_versions=sorted(hashes),files=results,
             interpretation=['Empirical positive means evaluator Monte Carlo X risk exceeds tau against both incumbent and continuation.',
              'Positive evaluator gain is not an exact population certificate; uncertainty remains for tiny differences.',
              'EB rejection due its additive term is certificate conservatism, not evidence of a matching information lower bound.',
              'Paired t intervals are descriptive across independent seeds, without multiple-comparison control.',
              'Saved-statistic gate checks plus separate evaluator-isolation tests support absence of oracle decision access; they are not a security proof.'])
    (ROOT/'results/confirmation_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'status':out['status'],'files':[{k:r.get(k) for k in ['file','complete','completed_runs','checked_events','maximum_independent_terminal_risk_error']} for r in results]},indent=2))
    if args.require_complete:assert complete,'confirmation files incomplete'


if __name__=='__main__':main()
