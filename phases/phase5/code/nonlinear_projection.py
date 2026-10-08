"""Post-hoc exact evaluation; never used to select or train any learner.

Original frozen Monte Carlo outcomes remain unchanged. This script evaluates
the analytical Hermite projection in manuscript/nonlinear_projection.tex.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.integrate import quad
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def main():
    phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
    a, ea = quad(lambda z: z*np.tanh(1.5*z)*phi(z), -np.inf, np.inf,
                 epsabs=1e-13, epsrel=1e-13)
    b, eb = quad(lambda z: np.tanh(1.5*z)**2*phi(z), -np.inf, np.inf,
                 epsabs=1e-13, epsrel=1e-13)
    coefficient = .8*a*a
    floor = .64*b*b + .08*(1-np.exp(-2)) - coefficient**2
    rows, aggregates = [], []
    for study in ['confirmation', 'adaptive_confirmation', 'fullbatch_confirmation']:
        directory = ROOT/'experiments'/study
        for path in sorted((directory/'results').glob('nonlinear_*.json')):
            record = json.loads(path.read_text())
            data = np.load(ROOT/record['data_file'])
            rotation = data['teacher_rotation']
            u, v = rotation[:, 0], rotation[:, 1]
            projection = coefficient/2*(np.outer(u, v)+np.outer(v, u))
            for method, result in record['methods'].items():
                p = result['final_parameters']
                w, weights = np.asarray(p['w']), np.asarray(p['a'])
                matrix = (w*weights)@w.T
                exact = floor+2*np.sum((matrix-projection)**2)+p['c']**2+float(data['noise'])**2
                rows.append(dict(study=study, seed=record['seed'], method=method,
                                 exact_total_risk=float(exact),
                                 original_monte_carlo_risk=result['final_risk'],
                                 mc_minus_exact=result['final_risk']-float(exact)))
        study_rows = [r for r in rows if r['study'] == study]
        for method in sorted({r['method'] for r in study_rows}):
            subset = [r for r in study_rows if r['method'] == method]
            values = np.array([r['exact_total_risk'] for r in subset])
            half = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
            aggregates.append(dict(study=study, method=method, seeds=len(values),
                                   exact_mean=float(values.mean()),
                                   descriptive_ci95=[float(values.mean()-half), float(values.mean()+half)]))
    output = ROOT/'results/nonlinear_projection'
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status='post_hoc_analytic_diagnostic_no_retraining_or_selection',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  a=a, b=b, numerical_integration_error_estimates=[ea, eb],
                  projected_coefficient=coefficient, approximation_floor=floor,
                  total_risk_floor=floor+.04,
                  sine_only_floor=.08*(1-np.exp(-2)), final_endpoints=rows,
                  summary=aggregates)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['a','b','projected_coefficient','total_risk_floor']}))
    print(f'Evaluated {len(rows)} saved final endpoints; original outputs untouched.')


if __name__ == '__main__':
    main()
