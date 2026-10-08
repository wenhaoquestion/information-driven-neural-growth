"""Independent finite-population checks of the Phase II controller witness.
Run .venv/bin/python phase2/code/audit_controller_witness.py
Direct logistic risks, central finite differences, and edge-case challenges.
This is exact-law numerical evaluation, not a neural training experiment.
"""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
X=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
b=.5
eta=.1
p=np.array([.5-eta,.5-eta,.5+eta,.5+eta])

def loss(logits,posterior):
    return np.mean(np.logaddexp(0,logits)-posterior*logits)

def split(v,r,bias=b):
    u=r*(X@v)
    return .5*(np.tanh(bias+u)+np.tanh(bias-u))-np.tanh(bias)

def main():
    first=1-np.tanh(b)**2
    second=-2*np.tanh(b)*first
    # Reconstruct expected loss Hessian from joint-label expectation.
    S=sum((.5-pi)*second*np.outer(x,x) for pi,x in zip(p,X))/4
    H=first**2*(X.T@X)/16+S
    assert np.linalg.eigvalsh(H).min()>0
    assert np.allclose(S,-sum((.5-(1-pi))*second*np.outer(x,x) for pi,x in zip(p,X))/4)
    rows=[]
    for r in [1e-3,.02,.1,.4,1.]:
        for angle in np.linspace(0,2*np.pi,721):
            v=np.array([np.cos(angle),np.sin(angle)])
            z=split(v,r)
            a,c=loss(z,p),loss(z,1-p)
            expected=np.log(2)+np.mean(np.logaddexp(z/2,-z/2)-np.log(2))
            assert abs((a+c)/2-expected)<1e-14
            assert (a+c)/2>=np.log(2)-1e-14
            if angle in [0.,np.pi/2,np.pi,3*np.pi/2,2*np.pi]:
                rows.append({'r':r,'angle':float(angle),'risk_plus':float(a),'risk_minus':float(c),
                             'gain_plus':float(np.log(2)-a),'gain_minus':float(np.log(2)-c),
                             'quadratic_prediction_gain_plus':float(-r*r*(v@S@v)/2)})
    rng=np.random.default_rng(809241)
    for _ in range(10000):
        z=rng.normal(size=4)*rng.uniform(0,20)
        assert (loss(z,p)+loss(z,1-p))/2>=np.log(2)-1e-14
    # Parent Hessian independently via loss finite differences.
    step=1e-4
    Hfd=np.empty((2,2))
    def parent(w):
        return loss(np.tanh(b+X@w)-np.tanh(b),p)
    for i in range(2):
        for j in range(2):
            ei,ej=np.eye(2)[i]*step,np.eye(2)[j]*step
            Hfd[i,j]=(parent(ei+ej)-parent(ei-ej)-parent(-ei+ej)+parent(-ei-ej))/(4*step**2)
    assert np.max(np.abs(H-Hfd))<1e-7
    # Explicit failures if bias and signal constraints are silently dropped.
    assert np.max(np.abs(split(np.array([.6,.8]),.4,bias=0)))==0
    high_signal_S=S*(.3/eta)
    high_signal_H=first**2*(X.T@X)/16+high_signal_S
    assert np.linalg.eigvalsh(high_signal_H).min()<0
    result={'kind':'finite known-population numerical checks; no fitting or sampling claims',
            'X':X.tolist(),'bias':b,'eta':eta,'posterior_plus':p.tolist(),'posterior_minus':(1-p).tolist(),
            'S':S.tolist(),'parent_H':H.tolist(),'parent_H_finite_difference':Hfd.tolist(),
            'parent_eigenvalues':np.linalg.eigvalsh(H).tolist(),'local_stability_eta_upper':float(first**2/(4*abs(second))),
            'angles_tested_per_radius':721,'radii':[1e-3,.02,.1,.4,1.],
            'arbitrary_logits_tested':10000,'arbitrary_logits_seed':809241,
            'rows':rows,'edge_cases':{'b0':'symmetric split exactly zero for all directions and radii',
                                    'eta0.3_parent_eigenvalues':np.linalg.eigvalsh(high_signal_H).tolist()},
            'assertions':'all passed'}
    (ROOT/'phase2/results/audit_controller_witness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['parent_eigenvalues','local_stability_eta_upper','edge_cases','assertions']},indent=2))

if __name__=='__main__':
    main()
