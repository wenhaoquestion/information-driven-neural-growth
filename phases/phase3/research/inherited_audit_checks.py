"""Read-only Phase II reproduction plus independent paired raw-result checks.

Run from project root: .venv/bin/python phase3/research/inherited_audit_checks.py
All generated evidence is written under phase3/results; Phase II is imported only.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2/code"))
from learning_curvature_bridge import run, spectral
from learning_neural_probe import train, init, expand


def direct(x, y, par):
    z = np.tanh(x @ par["w"] + par["b"]) @ par["v"] + par["c"]
    return float(np.mean(np.logaddexp(0, z) - y * z))


def paired(values):
    a = np.array(values)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return {"n": len(a), "mean": float(a.mean()),
            "t95_halfwidth": float(t.ppf(.975, len(a)-1)*se)}


def main():
    relevant = [ROOT / "phase2/code/learning_neural_probe.py",
                ROOT / "phase2/code/learning_curvature_bridge.py",
                ROOT / "phase2/results/learning_curvature_bridge64.json",
                ROOT / "phase2/results/learning_neural_aligned32.json"]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in relevant}
    raw = json.loads(relevant[2].read_text())
    replay = []
    for regime in ["four_state", "rotated_nuisance", "sample_pretrained", "random_start", "no_signal"]:
        old = next(r for r in raw["runs"] if r["regime"] == regime
                   and r["ntrain"] == 256 and r["poststeps"] == 30)
        new = run(old["seed"], regime, 256, 30, old["presteps"])
        diffs = [abs(old[k]-new[k]) for k in
                 ["initial_risk", "continued_width1", "fixed_width2_equal_steps",
                  "fixed_width2_approx_equal_compute"]]
        for method in ["spectral", "random", "oracle"]:
            for k in ["post_risk", "initial_gain", "gated_risk", "validation_paired_gain"]:
                diffs.append(abs(old["events"][method][k]-new["events"][method][k]))
        assert max(diffs) < 1e-12
        replay.append({"regime": regime, "seed": old["seed"], "max_abs_error": max(diffs)})

    rng = np.random.default_rng(930771)
    x = rng.normal(size=(31, 3)); y = rng.integers(2, size=31)
    par = init(rng, 3, 2)
    fd = {k: np.zeros_like(v) for k,v in par.items()}
    for k in par:
        for idx in np.ndindex(par[k].shape):
            up={a:b.copy() for a,b in par.items()}; dn={a:b.copy() for a,b in par.items()}
            up[k][idx] += 1e-5; dn[k][idx] -= 1e-5
            fd[k][idx] = (direct(x,y,up)-direct(x,y,dn))/2e-5
    actual = train(x,y,par,1,lr=.025)
    expected = {k:par[k]-.025*fd[k]/(np.abs(fd[k])+1e-8) for k in par}
    adam_errors={k:float(np.max(np.abs(actual[k]-expected[k]))) for k in par}
    assert max(adam_errors.values()) < 1e-12
    direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
    split0 = expand(par,0,direction,delta=0)
    logits = lambda p: np.tanh(x@p['w']+p['b'])@p['v']+p['c']
    identity=float(np.max(np.abs(logits(split0)-logits(par))))
    assert identity < 1e-14

    bridge=[]
    for regime in ["four_state", "random_start", "no_signal"]:
        for steps in [30, 300]:
            rr=[r for r in raw['runs'] if r['regime']==regime
                and r['ntrain']==1024 and r['poststeps']==steps]
            bridge.append({'regime':regime,'steps':steps,
                'spectral_mean':float(np.mean([r['events']['spectral']['post_risk'] for r in rr])),
                'random_mean':float(np.mean([r['events']['random']['post_risk'] for r in rr])),
                'spectral_minus_random':paired([r['events']['spectral']['post_risk']-r['events']['random']['post_risk'] for r in rr]),
                'spectral_accepted':sum(r['events']['spectral']['accepted'] for r in rr)})
    aligned=json.loads(relevant[3].read_text())
    cmi=[]
    for regime in ['teacher','localized','linear_null']:
        rr=[r for r in aligned['runs'] if r['regime']==regime]
        cmi.append({'regime':regime,
            'aligned_minus_random':paired([r['selected']['aligned_information']['risk']-r['selected']['random']['risk'] for r in rr]),
            'entropy_minus_random':paired([r['selected']['activation_entropy']['risk']-r['selected']['random']['risk'] for r in rr]),
            'continuation_minus_aligned':paired([r['continued_width2']-r['selected']['aligned_information']['risk'] for r in rr])})
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant}
    assert before == after
    result={'kind':'Phase III independent inherited audit',
            'method':'original-code replay; independent direct-logit finite differences; independently recomputed paired raw summaries',
            'inherited_files_hashes':before,'prior_files_unchanged':True,
            'representative_replays':replay,'adam_finite_difference_max_errors':adam_errors,
            'function_preservation_logit_max_error':identity,'bridge':bridge,'aligned_cmi':cmi,
            'status':'all checks passed'}
    path=ROOT/'phase3/results/inherited_audit.json'
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
