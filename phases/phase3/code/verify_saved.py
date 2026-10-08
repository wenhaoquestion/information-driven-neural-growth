"""Saved-record arithmetic and history checks; no RNG, fitting, or writes."""
from pathlib import Path
import csv,gzip,hashlib,json,math,statistics
from restore_evidence import load
ROOT=Path(__file__).resolve().parents[1]
def near(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11),(a,b)
def size(p):return sum(sum(len(x) if isinstance(x,list) else 1 for x in v) if isinstance(v,list) else 1 for v in p.values())
def main():
    for r in json.loads((ROOT/'evidence/manifest.json').read_text())['files']:
        z=(ROOT/r['archive']).read_bytes();b=gzip.decompress(z)
        assert hashlib.sha256(z).hexdigest()==r['archive_sha256']
        assert hashlib.sha256(b).hexdigest()==r['uncompressed_sha256']
    totals={};main={};histories=events=0
    names=['confirm_eb4096.json','confirm_eb16384.json','confirm_mean16384.json','betting_followup4096.json','eb_followup4096.json']
    for name in names:
        data=load(name);cfg=data['config'];runs=data['runs']
        expected={(t,s) for t in cfg['tasks'] for s in range(cfg['seed_base'],cfg['seed_base']+cfg['seeds'])}
        assert {(r['task'],r['seed']) for r in runs}==expected and len(runs)==len(expected)
        count=0
        for r in runs:
            assert set(r['methods'])==set(cfg['methods'])
            for method,m in r['methods'].items():
                histories+=1;previous=r['start_parameters'];previous_risk=r['initial_risk']
                cost=cfg['presteps']*cfg['ntrain']*size(previous);audit=0
                assert len(m['events'])==cfg['events']
                for t,e in enumerate(m['events'],1):
                    count+=1;events+=1
                    assert e['t']==t and e['incumbent_parameters']==previous
                    near(e['incumbent_risk'],previous_risk)
                    pairs=[e[k] for k in ['continuation_vs_incumbent','growth_vs_incumbent','growth_vs_continuation']]
                    for c in pairs:
                        assert c['n']==cfg['naudit'];near(c['delta'],cfg['alpha']/(3*cfg['events']))
                        rad=math.sqrt(2*c['variance']*math.log(2/c['delta'])/c['n'])+14*math.log(2/c['delta'])/(3*(c['n']-1))
                        near(c['radius'],rad);near(c['lcb'],c['mean']-rad)
                        if cfg['gate']=='betting':
                            maximum=max(c['log_products']);loge=maximum+math.log(sum(math.exp(x-maximum) for x in c['log_products']))-math.log(5)
                            near(c['log_e'],loge);near(c['log_threshold'],-math.log(c['delta']))
                            assert c['betting_accept']==(c['log_e']>=c['log_threshold'])
                    ci,gi,gc=pairs;near(gi['mean']-ci['mean'],gc['mean'])
                    if cfg['gate']=='betting':action='grow' if gi['betting_accept'] and gc['betting_accept'] else ('continue' if ci['betting_accept'] else 'hold')
                    else:
                        key='lcb' if cfg['gate']=='eb' else 'mean'
                        action='grow' if gi[key]>cfg['tau'] and gc[key]>cfg['tau'] else ('continue' if ci[key]>0 else 'hold')
                    assert action==e['action'];chosen={'grow':'growth','continue':'continuation','hold':'incumbent'}[action]
                    assert e['retained_parameters']==e[chosen+'_parameters'];near(e['retained_risk'],e[chosen+'_risk'])
                    assert e['parameters']==size(e['retained_parameters'])==1+8*e['width']
                    meta=e['proposal']
                    cost+=cfg['ntrain']*(cfg['branch_steps']*(size(e['continuation_parameters'])+size(e['growth_parameters']))+meta['candidate_count']*meta['candidate_steps']*7)
                    audit+=cfg['naudit']*sum(size(e[k+'_parameters']) for k in ['incumbent','continuation','growth'])
                    assert e['training_parameter_examples']==cost and e['audit_parameter_examples']==audit
                    assert e['audit_labels']==t*cfg['naudit']
                    previous=e['retained_parameters'];previous_risk=e['retained_risk']
                near(m['final_risk'],previous_risk);assert m['final_width']==m['events'][-1]['width']
                assert m['training_parameter_examples']==cost and m['audit_parameter_examples']==audit
        totals[name]=dict(runs=len(runs),events=count)
        if name.startswith('confirm_'):main[name.removeprefix('confirm_').removesuffix('.json')]=runs
    for row in csv.DictReader((ROOT/'results/summary.csv').open()):
        rr=[r['methods'][row['method']] for r in main[row['protocol']] if r['task']==row['task']]
        assert len(rr)==int(row['n']);near(statistics.mean(r['final_risk'] for r in rr),float(row['risk']))
        near(statistics.mean(r['final_width'] for r in rr),float(row['width']))
    assert histories==1088 and events==8704
    print(json.dumps(dict(status='passed saved evidence checks',batches=totals,trajectories=histories,events=events,scope='Saved gate arithmetic, history, costs, and summary means. No RNG, regenerated examples, fitted-risk reevaluation, training, or proof certification.'),indent=2))
if __name__=='__main__':main()
