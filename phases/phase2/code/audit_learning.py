"""Second implementation checks for learner derivatives and saved-run integrity.
Run .venv/bin/python phase2/code/audit_learning.py.
Writes only phase2/results/audit_learning.json.
"""
from pathlib import Path
import json
import numpy as np
from learning_neural_probe import train,init,expand
from learning_curvature_bridge import spectral,run

ROOT=Path(__file__).resolve().parents[2]

def direct(x,y,p):
    z=np.tanh(x@p['w']+p['b'])@p['v']+p['c']
    return np.mean(np.logaddexp(0,z)-y*z)

def main():
    rng=np.random.default_rng(372012)
    x=rng.normal(size=(29,3));y=rng.integers(2,size=29)
    par=init(rng,3,1)
    fd={k:np.zeros_like(v) for k,v in par.items()}
    for k in par:
        for idx in np.ndindex(par[k].shape):
            up={a:b.copy() for a,b in par.items()};dn={a:b.copy() for a,b in par.items()}
            up[k][idx]+=1e-5;dn[k][idx]-=1e-5
            fd[k][idx]=(direct(x,y,up)-direct(x,y,dn))/2e-5
    trained=train(x,y,par,1,lr=.025)
    expected={k:par[k]-.025*fd[k]/(np.abs(fd[k])+1e-8) for k in par}
    errors={k:float(np.max(np.abs(trained[k]-expected[k]))) for k in par}
    assert max(errors.values())<5e-15
    v,_,S=spectral(x,y,par)
    r=.0003
    actual=float(direct(x,y,expand(par,0,v,delta=r))-direct(x,y,par))
    predicted=float(r*r*(v@S@v)/2)
    assert abs(actual-predicted)<1e-12
    assert abs(direct(x,y,expand(par,0,v,delta=0))-direct(x,y,par))<1e-14
    raw=json.loads((ROOT/'phase2/results/learning_curvature_bridge64.json').read_text())
    checks=[]
    for regime in ['four_state','rotated_nuisance','sample_pretrained','random_start','no_signal']:
        old=next(r for r in raw['runs'] if r['regime']==regime and r['ntrain']==256 and r['poststeps']==30)
        new=run(old['seed'],regime,256,30,old['presteps'])
        keys=['initial_risk','continued_width1','fixed_width2_equal_steps','fixed_width2_approx_equal_compute']
        diffs={k:abs(old[k]-new[k]) for k in keys}
        for method in ['spectral','random','oracle']:
            for k in ['post_risk','initial_gain','gated_risk','validation_paired_gain']:
                diffs[method+'_'+k]=abs(old['events'][method][k]-new['events'][method][k])
        assert max(diffs.values())<1e-12
        checks.append({'regime':regime,'seed':old['seed'],'maximum_reproduction_difference':max(diffs.values())})
    out={'gradient_check':'single Adam update reconstructed from independently finite-differenced direct logistic risk',
         'maximum_parameter_errors':errors,'random_nonstationary_split_check':{'radius':r,'direct_loss_change':actual,
         'spectral_quadratic_prediction':predicted},'zero_split_function_preservation':True,
         'saved_run_reproduction':checks,'assertions':'all passed','seed':372012}
    (ROOT/'phase2/results/audit_learning.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
