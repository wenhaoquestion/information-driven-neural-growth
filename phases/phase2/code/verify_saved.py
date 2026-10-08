"""Pure saved-record checks: no RNG, fitting, or writes. Stdlib only."""
from pathlib import Path
import csv,json,math,statistics,gzip,hashlib
from restore_evidence import load
ROOT=Path(__file__).resolve().parents[1]
def near(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11),(a,b)
def main():
    manifest=json.loads((ROOT/'evidence/manifest.json').read_text())
    for r in manifest['files']:
        z=(ROOT/r['archive']).read_bytes();b=gzip.decompress(z)
        assert hashlib.sha256(z).hexdigest()==r['archive_sha256']
        assert hashlib.sha256(b).hexdigest()==r['uncompressed_sha256']
    data=load('learning_curvature_bridge64.json');runs=data['runs']
    assert len(runs)==1280
    keys={(r['seed'],r['regime'],r['ntrain'],r['poststeps']) for r in runs}
    assert len(keys)==len(runs)
    groups={}
    maximum=0.
    for r in runs:
        assert 27100<=r['seed']<27164
        groups.setdefault((r['regime'],r['ntrain'],r['poststeps']),[]).append(r)
        for event in r['events'].values():
            assert event['accepted']==(event['validation_paired_gain']>1.96*event['validation_se'])
            near(event['gated_risk'],event['post_risk'] if event['accepted'] else r['continued_width1'])
            par=event['post_params'];losses=[]
            for x,q in zip(r['population_inputs'],r['population_probability']):
                h=[math.tanh(sum(x[i]*par['w'][i][j] for i in range(len(x)))+par['b'][j]) for j in range(len(par['v']))]
                logit=par['c']+sum(a*b for a,b in zip(h,par['v']))
                prob=min(1-1e-9,max(1e-9,1/(1+math.exp(-logit))))
                losses.append(-q*math.log(prob)-(1-q)*math.log1p(-prob))
            measured=statistics.mean(losses);near(measured,event['post_risk'])
            maximum=max(maximum,abs(measured-event['post_risk']))
    table=list(csv.DictReader((ROOT/'results/learning_curvature_bridge64_summary.csv').open()))
    assert len(groups)==len(table)==20
    for row in table:
        rr=groups[row['regime'],int(row['ntrain']),int(row['poststeps'])];assert len(rr)==64
        near(statistics.mean(r['events']['spectral']['post_risk']-r['events']['random']['post_risk'] for r in rr),float(row['spectral_minus_random']))
        for method in ['spectral','random','oracle']:
            near(statistics.mean(r['events'][method]['post_risk'] for r in rr),float(row[method+'_post_risk']))
    for name in ['learning_neural_probe32.json','learning_neural_aligned32.json']:
        d=load(name);assert len(d['runs'])==96
        assert len({(r['seed'],r['regime']) for r in d['runs']})==96
        for r in d['runs']:
            assert len(r['candidates'])==8
            for selection in r['selected'].values():near(selection['risk'],r['candidates'][selection['index']]['post_risk'])
    d=load('score_sampling.json');assert sum(r['reps'] for r in d['rows'])==20000
    for r in d['rows']:
        assert len(r['raw_scores'])==r['reps']
        assert sum(x<=0 for x in r['raw_scores'])==r['raw_score_nonpositive_count']
        assert sum(x<=0 for x in r['parity_feature_scores'])==r['parity_feature_nonpositive_count']
    print(json.dumps(dict(status='passed saved evidence checks',bridge_runs=1280,bridge_events=3840,probe_runs=192,score_trials=20000,maximum_saved_parameter_risk_error=maximum,scope='Historical saved values only; no regeneration, RNG, training, or theorem certification.'),indent=2))
if __name__=='__main__':main()
