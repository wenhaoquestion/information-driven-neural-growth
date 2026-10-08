"""Frozen, finite-state evidence-retention study. No neural training.

Run: python phases/phase4/code/retention_experiment.py
All observed outcomes are saved as sufficient input-label category counts.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def radius(n, second, maximum, alpha):
    # A variable W has |W|<=maximum and Var(W)<=second.
    u = np.log(1 / alpha)
    return np.sqrt(2 * second * u / n) + 4 * maximum * u / (3 * n)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    c = (p + z*z/(2*n)) / (1+z*z/n)
    h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(c-h), float(c+h)]


def main():
    OUT.mkdir(exist_ok=True)
    start = time.perf_counter()
    reps = 1000
    ns = [128, 512, 2048, 8192, 32768, 131072]
    historical_n = 32768
    states = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    f1, f2 = states.T
    ys = np.tile([0,1], 4)
    rows, counts_archive, checks = [], {}, []
    for alternative in [False, True]:
        for zi, z in enumerate([1, 1/4, 1/16, 1/64]):
            seed = 941000 + 10*int(alternative) + zi
            rng = np.random.default_rng(seed)
            coef = .2*np.sqrt(1-z)
            h = coef*f1 + .2*np.sqrt(z)*f2
            r = .2*np.sqrt(z)*f2
            eta = .5 + (0.003*f2/(.4*np.sqrt(z)) if alternative else 0)
            eta = np.broadcast_to(eta, (4,)).copy()
            probabilities = np.stack([1-eta, eta], axis=1).ravel()/4
            score = np.repeat(h, 2)*(2*ys-1)
            residual_score = np.repeat(r, 2)*(2*ys-1)
            historical_score = np.repeat(f1, 2)*(ys-.5)
            gain = float(np.dot(probabilities, score))
            S, S_perp = float(np.mean(h*h)), float(np.mean(r*r))
            assert np.all((eta >= .25) & (eta <= .75))
            assert np.max(np.abs(h)) < 1
            assert np.isclose(gain, .003 if alternative else 0)
            assert np.isclose(np.dot(probabilities, historical_score), 0)
            assert np.isclose(S, .04) and np.isclose(S_perp, .04*z)
            checks.append(dict(seed=seed, alternative=alternative, z=z, gain=gain,
                               S=S, S_perp=S_perp, eta=eta.tolist(), h=h.tolist()))
            old = rng.multinomial(historical_n, probabilities, size=reps)
            counts_archive[f"R1_hist_{seed}"] = old
            old_moment = old @ historical_score / historical_n
            old_radius = np.sqrt(np.log(2/.025)/(2*historical_n))
            cumulative = np.zeros((reps, 8), dtype=np.int64)
            previous = 0
            for n in ns:
                cumulative += rng.multinomial(n-previous, probabilities, size=reps)
                previous = n
                counts_archive[f"R1_fresh_{seed}_{n}"] = cumulative.copy()
                direct = cumulative @ score / n
                residual = cumulative @ residual_score / n
                finite = residual + 2*coef*old_moment
                values = {
                    "direct": (direct, radius(n,S,np.max(np.abs(h)),.05), n),
                    "oracle_moment": (residual, radius(n,S_perp,np.max(np.abs(r)),.05), n),
                    "finite_history": (finite, radius(n,S_perp,np.max(np.abs(r)),.025)
                                       + 2*abs(coef)*old_radius, n+historical_n),
                }
                for method, (est, rad, unique) in values.items():
                    for j in range(reps):
                        rows.append(dict(experiment="R1",seed=seed,replicate=j,
                            alternative=alternative,z=z,n=n,method=method,estimate=float(est[j]),
                            radius=float(rad),gain=gain,accepted=bool(est[j]>rad),unique_labels=unique))
    for m in [4,16,64]:
        n, seed = 4096, 942000+m
        rng = np.random.default_rng(seed)
        probabilities = np.ones(2*m)/(2*m)
        old = rng.multinomial(n, probabilities, size=reps)
        fresh = rng.multinomial(n, probabilities, size=reps)
        counts_archive[f"R2_hist_{seed}"] = old
        counts_archive[f"R2_fresh_{seed}"] = fresh
        # joint moment E[1_{X=j}(Y-.5)], not conditional frequency
        moments = (old[:,1::2]-old[:,::2])/(2*n)
        h = .4*np.where(moments >= 0,1.,-1.)
        adapt_est = 2*np.sum(h*moments,axis=1)
        fresh_est = np.sum(h*(fresh[:,1::2]-fresh[:,::2]),axis=1)/n
        # Each joint variable has range length 1; all m bounds hold jointly.
        beta = np.sqrt(np.log(2*m/.05)/(2*n))
        uniform_rad = 2*np.sum(np.abs(h),axis=1)*beta
        simple_rad = radius(n,.16,.4,.05)
        methods = {"naive_reuse": (adapt_est, np.full(reps,simple_rad),n),
                   "uniform_moments": (adapt_est, uniform_rad,n),
                   "fresh_audit": (fresh_est,np.full(reps,simple_rad),2*n)}
        for method,(est,rad,unique) in methods.items():
            for j in range(reps):
                rows.append(dict(experiment="R2",seed=seed,replicate=j,m=m,n=n,
                    method=method,estimate=float(est[j]),radius=float(rad[j]),gain=0.,
                    accepted=bool(est[j]>rad[j]),unique_labels=unique))
    summary = []
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = tuple((k,row[k]) for k in ["experiment","alternative","z","m","n","method"] if k in row)
        groups[key].append(row)
    for key, values in groups.items():
        k = sum(r["accepted"] for r in values)
        summary.append(dict(key,replicates=len(values),accepted=k,rate=k/len(values),
            wilson95=wilson(k,len(values)),mean_estimate=float(np.mean([r["estimate"] for r in values])),
            radius=values[0]["radius"],unique_labels=values[0]["unique_labels"]))
    (OUT/"retention_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    np.savez_compressed(OUT/"retention_counts.npz", **counts_archive)
    (OUT/"retention_summary.json").write_text(json.dumps(summary,indent=2))
    report = dict(analytic_checks=checks, rows=len(rows), count_arrays=len(counts_archive),
        protocol_sha256=hashlib.sha256((ROOT/"research/RETENTION_EXPERIMENT_PROTOCOL.md").read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        elapsed_seconds=time.perf_counter()-start)
    (OUT/"retention_execution.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="analytic_checks"},indent=2))
    for x in summary:
        if x["experiment"]=="R2" or (x.get("alternative") and x["n"]==8192): print(x)


if __name__ == "__main__":
    main()
