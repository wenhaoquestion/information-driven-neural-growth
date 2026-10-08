"""Independent Phase 6 audit; deliberately tiny synthetic fixtures only.

This script writes only under reviews/. It neither runs confirmation seeds nor
imports the runner's test module. Gaussian risk is checked by exact quadrature
rather than by repeating its matrix formula.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import phase5_reference as reference


def gaussian_grid(d):
    # Three nodes integrate each univariate polynomial through degree five.
    nodes = np.asarray([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
    weights = np.asarray([1 / 6, 2 / 3, 1 / 6])
    indices = np.asarray(list(itertools.product(range(3), repeat=d)))
    return nodes[indices], weights[indices].prod(axis=1)


def math_audit():
    rng = np.random.default_rng(703193)
    d, k = 4, 3
    p = {"w": rng.normal(size=(d, k)),
         "a": np.asarray([0.3, -0.9, 1.2]), "c": np.asarray(0.73)}
    matrix = rng.normal(size=(d, d))
    teacher = (matrix + matrix.T) / 2
    x, probability = gaussian_grid(d)
    teacher_mean = np.einsum("ni,ij,nj->n", x, teacher, x) - np.trace(teacher)
    explicit_predict = sum(
        p["a"][j] * ((x @ p["w"][:, j]) ** 2 - p["w"][:, j] @ p["w"][:, j])
        for j in range(k)
    ) + p["c"]
    np.testing.assert_allclose(reference.predict(x, p), explicit_predict,
                               atol=1e-12, rtol=1e-13)
    exact_by_quadrature = float(probability @ (explicit_predict - teacher_mean) ** 2)
    matrix_formula = float(2 * np.square(reference.effective(p) - teacher).sum() + p["c"] ** 2)
    np.testing.assert_allclose(exact_by_quadrature, matrix_formula,
                               rtol=1e-13, atol=1e-12)
    # Deliberately retain a nonzero intercept: omitting c² must be detectable.
    without_intercept = float(2 * np.square(reference.effective(p) - teacher).sum())
    np.testing.assert_allclose(exact_by_quadrature - without_intercept, 0.73 ** 2,
                               atol=1e-12)
    sample_x = rng.normal(size=(9, d))
    sample_y = rng.normal(size=9)
    analytic = reference.grad(sample_x, sample_y, p)
    discrepancies = {}
    epsilon = 2e-6
    for key, value in p.items():
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            pp = {name: array.copy() for name, array in p.items()}
            pm = {name: array.copy() for name, array in p.items()}
            pp[key][index] += epsilon
            pm[key][index] -= epsilon
            numerical[index] = (
                np.mean((reference.predict(sample_x, pp) - sample_y) ** 2)
                - np.mean((reference.predict(sample_x, pm) - sample_y) ** 2)
            ) / (2 * epsilon)
        np.testing.assert_allclose(numerical, analytic[key], rtol=2e-6, atol=2e-6)
        discrepancies[key] = float(np.max(np.abs(numerical - analytic[key])))
    return {"gaussian_quadrature_nodes": len(x),
            "quadrature_excess_risk": exact_by_quadrature,
            "matrix_formula_excess_risk": matrix_formula,
            "nonzero_intercept_contribution": 0.73 ** 2,
            "max_absolute_gradient_errors": discrepancies}


def config_audit():
    result = {}
    for path in sorted((ROOT / "configs").glob("*.json")):
        cfg = json.loads(path.read_text())
        assert len(cfg["seeds"]) == len(set(cfg["seeds"]))
        assert cfg["budgets"] == sorted(set(cfg["budgets"]))
        result[path.name] = {
            "config_bytes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed_count": len(cfg["seeds"]),
            "seed_range": [min(cfg["seeds"]), max(cfg["seeds"])],
            "planned_trajectories": int(np.prod([len(cfg[key]) for key in
                ("seeds", "tasks", "methods", "budgets", "age_modes")])),
        }
    primary = json.loads((ROOT / "configs/confirmation_primary.json").read_text())
    secondary = json.loads((ROOT / "configs/confirmation_secondary.json").read_text())
    age = json.loads((ROOT / "configs/age_sensitivity.json").read_text())
    smoke = json.loads((ROOT / "configs/smoke.json").read_text())
    assert len(primary["seeds"]) == 256
    assert primary["age_modes"] == ["global"]
    assert primary["tasks"] == ["indefinite_noisy"]
    assert primary["budgets"] == [3200000, 5000000, 6400000, 8000000, 10000000]
    assert set(primary["seeds"]).isdisjoint(smoke["seeds"])
    assert set(primary["seeds"]).isdisjoint(secondary["seeds"])
    assert set(age["seeds"]).issubset(primary["seeds"])
    result["sensitivity_is_paired_subset_not_extra_independent_replicates"] = True
    return result


def main():
    result = {"status": "passed", "math": math_audit(), "configs": config_audit()}
    result["audited_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "code/phase5_reference.py", Path(__file__))
    }
    target = ROOT / "reviews/first_principles_results.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
