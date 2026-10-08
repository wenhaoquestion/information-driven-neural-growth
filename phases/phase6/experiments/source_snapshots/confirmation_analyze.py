"""Frozen-endpoint analysis for phase-six scalar-work experiments.

Development selection:
  python analyze.py --select-development DIR0075 DIR015 DIR03 --output-dir OUT
Confirmation analysis (each DIR contains manifest.json and results/):
  python analyze.py --primary DIR --secondary DIR --age DIR \
    --development-selection OUT/selected_lrs.json --output-dir ANALYSIS

No model fitting occurs here. Missing/incomplete/duplicate observations are errors,
never omitted observations. The confirmatory unit is an independent teacher/stream.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy.stats import t as student_t

METHODS = ("historical_moment", "random_growth128", "random_growth138", "fixed7")
LABELS = {"historical_moment": "Historical moment", "random_growth128": "Random R128",
          "random_growth138": "Random R138", "fixed7": "Fixed width 7"}
COLORS = dict(zip(METHODS, ("#245b84", "#cf8924", "#ae517a", "#637e41")))
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D")))
PRIMARY_TASK = "indefinite_noisy"
PRIMARY_N = 256
GRID = (3200000, 5000000, 6400000, 8000000, 10000000)
BOOTSTRAP_SEED = 96070031
BOOTSTRAP_REPLICATES = 10000
EPSILON = .01
ROOT = Path(__file__).resolve().parents[1]


def path_reference(path):
    """Artifact paths remain relocatable with the phase6_work directory."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def auc(risks, budgets, arrivals=7):
    """Normalized trapezoidal integral of raw risk against log allocated work."""
    risks, budgets = np.asarray(risks, dtype=float), np.asarray(budgets, dtype=float)
    if len(budgets) < 2 or risks.shape[-1] != len(budgets):
        raise ValueError("AUC requires a common grid of at least two budgets")
    if not np.all(np.diff(budgets) > 0) or budgets[0] <= 0:
        raise ValueError("Budgets must be strictly increasing and positive")
    u = np.log(arrivals * budgets)
    return np.sum((risks[..., 1:] + risks[..., :-1]) * np.diff(u) / 2, axis=-1) / (u[-1] - u[0])


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Intervals require at least two finite independent observations")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, se=se,
                ci95_low=mean - float(student_t.ppf(.975, n-1))*se,
                ci95_high=mean + float(student_t.ppf(.975, n-1))*se)


def paired_inference(difference):
    result = mean_interval(difference)
    mean, se, n = result["mean"], result["se"], result["n"]
    if se == 0:
        p = 0. if mean < 0 else (1. if mean > 0 else .5)
    else:
        p = float(student_t.cdf(mean / se, n-1))
    result.update(upper95_one_sided=mean + float(student_t.ppf(.95, n-1))*se,
                  lower95_one_sided=mean - float(student_t.ppf(.95, n-1))*se,
                  simultaneous_lower95_bonferroni3=mean - float(student_t.ppf(1-.05/3, n-1))*se,
                  p_one_sided_historical_lower=p)
    return result


def bootstrap_differences(differences):
    """Resample entire paired teachers, retaining all comparator correlations."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap expects finite N by comparator differences")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Chunking bounds memory while preserving the fixed sequence of resampled IDs.
    samples = []
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        size = min(250, BOOTSTRAP_REPLICATES-start)
        ids = rng.integers(0, len(values), size=(size, len(values)))
        samples.append(values[ids].mean(axis=1))
    means = np.concatenate(samples, axis=0)
    return np.quantile(means, [.025, .975], axis=0)


def first_grid_target_index(risks, epsilon=EPSILON):
    """First evaluated final-budget grid point meeting the frozen target.

    None is an explicit not-attained result, never an imputed maximum cost.
    This is a discrete terminal-risk curve, not first passage along a trajectory;
    larger budget points may fail again. No monotonic envelope is imposed.
    """
    success = np.asarray(risks, dtype=float) <= epsilon
    found = np.flatnonzero(success)
    return int(found[0]) if len(found) else None


def wilson(successes, n):
    z = NormalDist().inv_cdf(.975)
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return dict(successes=int(successes), n=int(n), rate=p,
                wilson95_low=center-half, wilson95_high=center+half)


class Dataset:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.config = cfg = self.manifest["config"]
        computed_config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if computed_config_sha != self.manifest["config_sha256"]:
            raise ValueError("Manifest configuration content/hash mismatch")
        self.records, self.files = {}, []
        self.budgets = tuple(sorted(map(int, cfg["budgets"])))
        self.seeds = tuple(sorted(map(int, cfg["seeds"])))
        self.methods = tuple(cfg["methods"])
        self.ages = tuple(cfg["age_modes"])
        self.tasks = tuple(cfg["tasks"])
        self.arrivals = int(cfg["arrivals"])
        self.complete_seed_seconds = 0.
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate configured seed")
        expected = set(itertools.product(self.tasks, self.seeds, self.ages, self.methods, self.budgets))
        indexed_hashes = {}
        for index_path in sorted(self.directory.glob("completion_*.json")) + sorted(self.directory.glob("failure_*.json")):
            index = json.loads(index_path.read_text())
            for row in index.get("runs", []):
                relative, digest = row["result_file"], row["result_sha256"]
                if relative in indexed_hashes and indexed_hashes[relative] != digest:
                    raise ValueError(f"Inconsistent completed-result hashes across invocations: {relative}")
                indexed_hashes[relative] = digest
        paths = sorted((self.directory / "results").glob("*.json"))
        for path in paths:
            relative = str(path.relative_to(self.directory))
            if relative not in indexed_hashes or sha(path) != indexed_hashes[relative]:
                raise ValueError(f"Result missing from invocation index or content hash differs: {path}")
            raw = json.loads(path.read_text())
            if raw.get("complete") is not True:
                raise ValueError(f"Incomplete result: {path}")
            if raw["config_sha256"] != self.manifest["config_sha256"] or raw["code_sha256"] != self.manifest["code_sha256"]:
                raise ValueError(f"Source/config identity mismatch: {path}")
            data_path = (self.directory / raw["data_file"]).resolve()
            if not data_path.is_relative_to(self.directory) or sha(data_path) != raw["data_sha256"]:
                raise ValueError(f"Raw dataset identity mismatch: {path}")
            with np.load(data_path, allow_pickle=False) as data:
                teacher_matrix = np.asarray(data["teacher_matrix"], dtype=float)
                expected_noise = float(data["noise"])**2
            task, seed = raw["task"], int(raw["seed"])
            self.files.append(dict(path=path_reference(path), sha256=sha(path), task=task, seed=seed))
            self.complete_seed_seconds += float(raw["complete_seed_seconds"])
            for trajectory in raw["trajectories"]:
                age, method, budget = trajectory["age_mode"], trajectory["method"], int(trajectory["budget"])
                key = (task, seed, age, method, budget)
                if key in self.records or key not in expected:
                    raise ValueError(f"Duplicate/unplanned trajectory: {key}")
                events = trajectory["events"]
                if len(events) != self.arrivals or [e["t"] for e in events] != list(range(self.arrivals)):
                    raise ValueError(f"Incomplete online trajectory: {key}")
                risk = float(trajectory["final_excess_risk"])
                total_risk, noise = float(trajectory["final_risk"]), float(trajectory["noise_variance"])
                if not np.isfinite([risk, total_risk, noise]).all() or risk < -1e-12:
                    raise ValueError(f"Numerical failure; cannot declare primary success: {key}")
                if not np.isclose(total_risk-noise, risk, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Excess-risk identity fails: {key}")
                parameters = trajectory["final_parameters"]
                weights, coefficients = np.asarray(parameters["w"]), np.asarray(parameters["a"])
                # Independent contraction, not an import of the runner evaluator.
                reconstructed_matrix = np.einsum("ik,jk,k->ij", weights, weights, coefficients)
                reconstructed_risk = float(2*np.sum((reconstructed_matrix-teacher_matrix)**2) + float(parameters["c"])**2)
                if not np.isclose(noise, expected_noise, rtol=1e-12, atol=1e-14) or not np.isclose(risk, reconstructed_risk, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"Independent population-risk reconstruction fails: {key}")
                work = int(trajectory["total_proxy"])
                if work != sum(int(e["total_event_proxy"]) for e in events) or work != sum(trajectory["cumulative_costs"].values()):
                    raise ValueError(f"Work ledger does not add up: {key}")
                if any(int(e["total_event_proxy"]) > budget for e in events):
                    raise ValueError(f"Budget exceeded: {key}")
                lr = float(trajectory.get("effective_lr", cfg.get("method_lrs", {}).get(method, cfg["lr"])))
                self.records[key] = dict(task=task, seed=seed, age_mode=age, method=method,
                    event_budget=budget, allocated_total_work=budget*self.arrivals,
                    actual_total_work=work, unspent_work=budget*self.arrivals-work,
                    learner_seconds=float(trajectory["learner_seconds"]), effective_lr=lr,
                    noise_variance=noise, final_risk=total_risk, final_excess_risk=risk,
                    reaches_epsilon_001=int(risk <= EPSILON),
                    fit_steps=sum(int(e["fit"]["steps"]) for e in events),
                    source_file=path_reference(path), costs=trajectory["cumulative_costs"])
        missing = expected - set(self.records)
        if missing:
            raise ValueError(f"Missing {len(missing)} planned trajectories; no complete-case analysis allowed. Examples: {sorted(missing)[:3]}")
        if not self.records:
            raise ValueError("No trajectories")

    def record(self, task, seed, age, method, budget):
        return self.records[task, seed, age, method, int(budget)]

    def matrix(self, task, age, method, field="final_excess_risk", seeds=None):
        use = self.seeds if seeds is None else seeds
        return np.asarray([[self.record(task, seed, age, method, b)[field] for b in self.budgets] for seed in use], dtype=float)

    def provenance(self):
        invocations = []
        for path in sorted(self.directory.glob("completion_*.json")) + sorted(self.directory.glob("failure_*.json")):
            row = json.loads(path.read_text())
            invocations.append(dict(path=path_reference(path), complete=row.get("complete"), elapsed_seconds=row.get("elapsed_seconds"),
                                    error_type=row.get("error_type")))
        return dict(directory=path_reference(self.directory), config_sha256=self.manifest["config_sha256"],
                    code_sha256=self.manifest["code_sha256"], manifest_sha256=sha(self.directory/"manifest.json"),
                    trajectories=len(self.records), datasets=len(self.files),
                    allocated_work=sum(r["allocated_total_work"] for r in self.records.values()),
                    actual_proxy_work=sum(r["actual_total_work"] for r in self.records.values()),
                    learner_seconds_sum=sum(r["learner_seconds"] for r in self.records.values()),
                    complete_seed_seconds_sum=self.complete_seed_seconds,
                    invocation_elapsed_seconds_sum=sum(float(r["elapsed_seconds"] or 0) for r in invocations),
                    invocation_time_note="Elapsed invocation wall time; sums can include failed/resumed attempts. Learner sums are stored completed trajectories only.",
                    invocations=invocations, result_files=self.files)


def output_dir(path):
    out = Path(path).resolve()
    # Never overwrite existing analysis/selection output silently.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty/new output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def select_development(directories, out):
    datasets = [Dataset(directory) for directory in directories]
    if len(datasets) != 3:
        raise ValueError("Exactly three predeclared development learning rates required")
    reference = datasets[0]
    for ds in datasets:
        if ds.seeds != tuple(range(96010000, 96010016)) or ds.tasks != (PRIMARY_TASK,) or ds.ages != ("global",):
            raise ValueError("Development must use the frozen 16 seeds, indefinite task and global Adam age")
        if ds.budgets != GRID or set(ds.methods) != set(METHODS):
            raise ValueError("Unexpected development budget grid or methods")
        if ds.manifest["code_sha256"] != reference.manifest["code_sha256"]:
            raise ValueError("Development runs use different source versions")
    rates = sorted(float(ds.config["lr"]) for ds in datasets)
    if rates != [.0075, .015, .03]:
        raise ValueError(f"Expected frozen learning-rate grid, got {rates}")
    scores, per_seed = [], []
    for ds in datasets:
        lr = float(ds.config["lr"])
        for method in METHODS:
            if any(r["effective_lr"] != lr for r in ds.records.values() if r["method"] == method):
                raise ValueError("Method LR override in development changes the specified candidate grid")
            values = auc(ds.matrix(PRIMARY_TASK, "global", method), ds.budgets, ds.arrivals)
            scores.append(dict(method=method, lr=lr, **mean_interval(values)))
            per_seed.extend(dict(method=method, lr=lr, seed=seed, auc=float(value)) for seed, value in zip(ds.seeds, values))
    selected = {method: min((r for r in scores if r["method"] == method), key=lambda r: (r["mean"], r["lr"]))["lr"] for method in METHODS}
    provenance = [ds.provenance() for ds in datasets]
    result = dict(method_lrs=selected, selection_rule="Smallest mean normalized raw-excess-risk/log-allocated-work AUC over all 16 development seeds and all five budgets; exact ties choose smaller LR.",
                  eligible_lrs=rates, scores=scores, development_seeds=list(reference.seeds),
                  budgets=list(GRID), development_runs=provenance,
                  development_total_proxy_work=sum(p["actual_proxy_work"] for p in provenance),
                  development_total_allocated_work=sum(p["allocated_work"] for p in provenance),
                  development_learner_seconds_sum=sum(p["learner_seconds_sum"] for p in provenance),
                  development_invocation_elapsed_seconds_sum=sum(p["invocation_elapsed_seconds_sum"] for p in provenance),
                  analysis_source_sha256=sha(__file__))
    write_json(out/"selected_lrs.json", result)
    write_csv(out/"development_scores.csv", scores)
    write_csv(out/"development_seed_aucs.csv", per_seed)
    print(json.dumps(dict(selected_lrs=selected, output=str(out/"selected_lrs.json")), sort_keys=True))


def analyze_group(ds, task, age, primary=False):
    risks = {m: ds.matrix(task, age, m) for m in METHODS}
    au = {m: auc(risks[m], ds.budgets, ds.arrivals) for m in METHODS}
    group = f"{task}_{age}"
    seed_auc, auc_summary, contrasts, point, point_differences, seed_differences = [], [], [], [], [], []
    for m in METHODS:
        auc_summary.append(dict(group=group, method=m, **mean_interval(au[m])))
        seed_auc.extend(dict(group=group, task=task, age_mode=age, seed=s, method=m, auc=float(v)) for s, v in zip(ds.seeds, au[m]))
        for k, b in enumerate(ds.budgets):
            point.append(dict(group=group, method=m, event_budget=b, allocated_total_work=b*ds.arrivals,
                              **mean_interval(risks[m][:, k]),
                              **{f"target_{key}": val for key, val in wilson(int(np.sum(risks[m][:, k] <= EPSILON)), len(ds.seeds)).items()}))
    differences = np.column_stack([au[METHODS[0]]-au[m] for m in METHODS[1:]])
    bootstrap = bootstrap_differences(differences)
    for j, m in enumerate(METHODS[1:]):
        stats = paired_inference(differences[:, j])
        if not primary:
            # Secondary groups are descriptive: no additional success tests.
            for key in ("upper95_one_sided", "lower95_one_sided", "simultaneous_lower95_bonferroni3", "p_one_sided_historical_lower"):
                stats[key] = None
        contrasts.append(dict(group=group, comparator=m, confirmatory=primary, **stats,
                              bootstrap95_low=float(bootstrap[0, j]), bootstrap95_high=float(bootstrap[1, j]),
                              bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED))
        for i, seed in enumerate(ds.seeds):
            seed_differences.append(dict(group=group, seed=seed, comparator=m,
                                        historical_auc=float(au[METHODS[0]][i]), comparator_auc=float(au[m][i]),
                                        difference=float(differences[i, j])))
        for k, b in enumerate(ds.budgets):
            point_differences.append(dict(group=group, comparator=m, event_budget=b,
                                          allocated_total_work=b*ds.arrivals,
                                          **mean_interval(risks[METHODS[0]][:, k]-risks[m][:, k])))
    iut = dict(group=group, confirmatory=primary, n=len(ds.seeds),
               theta_hat=max(r["mean"] for r in contrasts),
               max_upper95=max(r["upper95_one_sided"] for r in contrasts) if primary else None,
               iut_p=max(r["p_one_sided_historical_lower"] for r in contrasts) if primary else None,
               success=bool(primary and all(r["upper95_one_sided"] < 0 for r in contrasts)),
               any_simultaneous_lower_nonnegative=bool(primary and any(r["simultaneous_lower95_bonferroni3"] >= 0 for r in contrasts)),
               interpretation="Primary success requires all three one-sided upper 95% paired-t bounds below zero. Bootstrap is sensitivity only; secondary groups cannot establish primary success.")
    return dict(iut=iut, seed_auc=seed_auc, auc_summary=auc_summary, contrasts=contrasts, points=point,
                point_contrasts=point_differences, seed_differences=seed_differences)


def targets_and_compression(ds, task, age):
    group = f"{task}_{age}"
    target, target_summary, compression, compression_summary = [], [], [], []
    for m in METHODS:
        matrix = ds.matrix(task, age, m)
        for seed, values in zip(ds.seeds, matrix):
            k = first_grid_target_index(values)
            target.append(dict(group=group, seed=seed, method=m, epsilon=EPSILON, attained=k is not None,
                status=("not_attained_within_grid" if k is None else "lowest_grid_point_already_attains" if k == 0 else "first_evaluated_final_grid_attainment"),
                any_risk_increase_across_grid=bool(np.any(np.diff(values)>0)),
                later_grid_failure_after_first_success=bool(k is not None and np.any(values[k+1:]>EPSILON)),
                event_budget="" if k is None else ds.budgets[k],
                allocated_total_work="" if k is None else ds.budgets[k]*ds.arrivals,
                actual_work_at_selected_grid="" if k is None else ds.record(task, seed, age, m, ds.budgets[k])["actual_total_work"]))
        current = [r for r in target if r["method"] == m]
        target_summary.append(dict(group=group, method=m, epsilon=EPSILON,
            **wilson(sum(r["attained"] for r in current), len(current)),
            not_attained=sum(not r["attained"] for r in current),
            later_grid_failure_after_first_success=sum(r["later_grid_failure_after_first_success"] for r in current),
            lowest_grid_already_attains=sum(r["status"] == "lowest_grid_point_already_attains" for r in current)))
    for low, high in ((6400000, 8000000), (8000000, 10000000)):
        if low not in ds.budgets or high not in ds.budgets:
            continue
        for m in METHODS[1:]:
            rows = []
            for seed in ds.seeds:
                h = ds.record(task, seed, age, METHODS[0], low)
                ref = ds.record(task, seed, age, m, high)
                same = ds.record(task, seed, age, m, low)
                row = dict(group=group, seed=seed, comparator=m, historical_event_budget=low, comparator_event_budget=high,
                    allocated_work_ratio=low/high, actual_work_ratio=h["actual_total_work"]/ref["actual_total_work"],
                    historical_risk=h["final_excess_risk"], comparator_risk=ref["final_excess_risk"],
                    comparator_same_low_budget_risk=same["final_excess_risk"],
                    risk_difference=h["final_excess_risk"]-ref["final_excess_risk"],
                    same_budget_risk_difference=h["final_excess_risk"]-same["final_excess_risk"],
                    historical_target_success=h["reaches_epsilon_001"], comparator_target_success=ref["reaches_epsilon_001"],
                    comparator_same_low_budget_success=same["reaches_epsilon_001"])
                rows.append(row)
            compression.extend(rows)
            compression_summary.append(dict(group=group, comparator=m, historical_event_budget=low, comparator_event_budget=high,
                **mean_interval([r["risk_difference"] for r in rows]),
                historical_successes=sum(r["historical_target_success"] for r in rows),
                comparator_successes=sum(r["comparator_target_success"] for r in rows),
                comparator_same_low_budget_successes=sum(r["comparator_same_low_budget_success"] for r in rows),
                mean_actual_work_ratio=float(np.mean([r["actual_work_ratio"] for r in rows])),
                max_actual_work_ratio=max(r["actual_work_ratio"] for r in rows),
                interpretation="Descriptive fixed-budget comparison only; does not identify minimum work or prove 20% minimum-cost savings."))
    return dict(target_seeds=target, target_summary=target_summary,
                compression_seeds=compression, compression_summary=compression_summary)


def age_comparisons(primary, local):
    rows, seeds = [], []
    if not set(local.seeds).issubset(primary.seeds):
        raise ValueError("Local-age sensitivity must pair with the same primary teachers")
    for m in METHODS:
        global_auc = auc(primary.matrix(PRIMARY_TASK, "global", m, seeds=local.seeds), primary.budgets, primary.arrivals)
        local_auc = auc(local.matrix(PRIMARY_TASK, "local", m), local.budgets, local.arrivals)
        diff = local_auc-global_auc
        rows.append(dict(method=m, comparison="local minus global on identical teachers", **mean_interval(diff)))
        seeds.extend(dict(seed=s, method=m, global_auc=float(g), local_auc=float(l), difference=float(d))
                     for s, g, l, d in zip(local.seeds, global_auc, local_auc, diff))
    return rows, seeds


def latex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def make_latex(out, groups, resources, selection):
    primary = next(g for g in groups if g["iut"]["confirmatory"])
    lines = [r"% Generated by analysis/analyze.py; do not edit numerical entries.",
             r"\begin{table}[t]\centering", r"\small",
             r"\begin{tabular}{lrrrr}", r"\hline",
             r"Comparator & Mean diff. & \shortstack{One-sided\\upper 95\%} & $p$ & Bootstrap 95\% CI \\", r"\hline"]
    for row in primary["contrasts"]:
        lines.append(f"{latex_escape(LABELS[row['comparator']])} & {row['mean']:.5g} & {row['upper95_one_sided']:.5g} & {row['p_one_sided_historical_lower']:.4g} & [{row['bootstrap95_low']:.5g}, {row['bootstrap95_high']:.5g}] " + r"\\")
    lines += [r"\hline\end{tabular}",
              r"\caption{Primary normalized excess-risk/log-allocated-work AUC differences (historical moment minus comparator), paired over 256 independent teachers and streams. Negative values favor historical moments. Bootstrap intervals use 10,000 paired resamples and are sensitivity analyses only.}",
              r"\label{tab:p6primary}\end{table}"]
    write_text = "\n".join(lines)+"\n"
    (out/"primary_table.tex").write_text(write_text)
    iut = primary["iut"]
    outcome = "met" if iut["success"] else "did not meet"
    text = (f"The prespecified conjunction {outcome} the primary criterion "
            f"(intersection--union $p={iut['iut_p']:.5g}$; maximum one-sided upper bound "
            f"${iut['max_upper95']:.5g}$). The endpoint integrates raw excess risk "
            "over the fixed log allocated-work grid; it is not a work-reduction percentage. "
            "Pointwise intervals and compressed-budget comparisons are descriptive.\n")
    if selection:
        lrs = ", ".join(f"{LABELS[m]}: {selection['method_lrs'][m]:g}" for m in METHODS)
        text += "Development selected the following learning rates: " + latex_escape(lrs) + ". "
        text += (f"All development trials consumed {selection['development_total_proxy_work']:,} "
                 f"scalar-work proxy units and {selection['development_learner_seconds_sum']:.3f} summed learner seconds. "
                 "These costs are disclosed separately from per-instance deployment work.\n")
    (out/"results_text.tex").write_text(text)
    table = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{llrrr}", r"\hline",
             r"Task / age & Method & $N$ & Mean AUC & Descriptive 95\% CI \\", r"\hline"]
    for group in groups:
        for row in group["auc_summary"]:
            table.append(f"{latex_escape(row['group'])} & {latex_escape(LABELS[row['method']])} & {row['n']} & {row['mean']:.5g} & [{row['ci95_low']:.5g}, {row['ci95_high']:.5g}] " + r"\\")
    table += [r"\hline\end{tabular}", r"\caption{Integrated terminal excess risks. Intervals are ordinary two-sided paired-unit $t$ intervals for each method mean, not simultaneous confidence bands.}", r"\label{tab:p6auc}\end{table}"]
    (out/"auc_table.tex").write_text("\n".join(table)+"\n")


def make_figures(out, groups):
    # Explicit static outputs requested for the English manuscript.
    os.environ.setdefault("MPLCONFIGDIR", str(out/".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out/".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#252a30",
        "text.color": "#252a30", "axes.edgecolor": "#555b61", "savefig.facecolor": "white"})
    figures = out/"figures"
    figures.mkdir()
    for group in groups:
        name = group["iut"]["group"]
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)
        for m in METHODS:
            rows = [r for r in group["points"] if r["method"] == m]
            x = np.array([r["allocated_total_work"] for r in rows])/1e6
            mean = np.array([r["mean"] for r in rows])
            half = np.array([r["ci95_high"]-r["mean"] for r in rows])
            axes[0].errorbar(x, mean, yerr=half, color=COLORS[m], marker=MARKERS[m],
                             lw=1.5, markersize=5, capsize=2, label=LABELS[m])
        axes[0].axhline(EPSILON, color="#555b61", ls=":", lw=1, label=r"Target $\epsilon=0.01$")
        axes[0].set_ylabel("Final population excess risk")
        axes[0].set_title("Full seven-block terminal risk")
        for m in METHODS[1:]:
            rows = [r for r in group["point_contrasts"] if r["comparator"] == m]
            x = np.array([r["allocated_total_work"] for r in rows])/1e6
            mean = np.array([r["mean"] for r in rows])
            half = np.array([r["ci95_high"]-r["mean"] for r in rows])
            axes[1].errorbar(x, mean, yerr=half, color=COLORS[m], marker=MARKERS[m],
                             lw=1.5, markersize=5, capsize=2, label="H − "+LABELS[m])
        axes[1].axhline(0, color="#555b61", lw=1)
        axes[1].set_ylabel("Paired excess-risk difference")
        axes[1].set_title("Historical moment minus comparator")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(x)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()
            ax.set_xlabel("Allocated total scalar-work budget (millions; log axis)")
            ax.grid(axis="y", color="#e4e6e8", linewidth=.6)
            ax.legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle(f"{name.replace('_', ' ')} · N={group['iut']['n']} · pointwise two-sided 95% t intervals", fontsize=11)
        for ext in ("pdf", "png"):
            fig.savefig(figures/f"{name}_frontier.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    primary = next(g for g in groups if g["iut"]["confirmatory"])
    fig, ax = plt.subplots(figsize=(7.3, 3.4), constrained_layout=True)
    for j, row in enumerate(primary["contrasts"]):
        ax.errorbar(row["mean"], j, xerr=[[row["mean"]-row["ci95_low"]], [row["ci95_high"]-row["mean"]]],
                    marker=MARKERS[row["comparator"]], color=COLORS[row["comparator"]], capsize=3,
                    label="Two-sided descriptive 95% t CI" if j == 0 else None)
        ax.plot(row["upper95_one_sided"], j+.09, marker="|", markersize=13, color="#252a30",
                label="One-sided 95% upper bound (primary rule)" if j == 0 else None)
    ax.axvline(0, color="#555b61", linewidth=1)
    ax.set_yticks(range(3), [LABELS[m] for m in METHODS[1:]])
    ax.set_xlabel("Paired normalized AUC difference: historical moment − comparator")
    ax.set_title(f"Primary integrated-risk comparisons · N={primary['iut']['n']}")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(axis="x", color="#e4e6e8", linewidth=.6)
    for ext in ("pdf", "png"):
        fig.savefig(figures/f"primary_auc_contrasts.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_confirmation(primary_dir, secondary_dir, age_dir, selection_path, out):
    primary = Dataset(primary_dir)
    if primary.tasks != (PRIMARY_TASK,) or primary.ages != ("global",) or primary.seeds != tuple(range(96100000, 96100256)):
        raise ValueError("Primary requires exactly 256 indefinite/global independent datasets")
    datasets = [primary]
    if secondary_dir:
        ds = Dataset(secondary_dir)
        if set(ds.tasks) != {"null", "lowrank_noiseless"} or ds.ages != ("global",) or ds.seeds != tuple(range(96200000, 96200032)):
            raise ValueError("Secondary tasks require exactly 32 seeds per task")
        datasets.append(ds)
    local = Dataset(age_dir) if age_dir else None
    if local:
        if local.tasks != (PRIMARY_TASK,) or local.ages != ("local",) or local.seeds != tuple(range(96100000, 96100032)):
            raise ValueError("Age sensitivity requires 32 indefinite/local datasets")
        datasets.append(local)
    if selection_path is None:
        raise ValueError("Confirmation requires the frozen development-selection JSON")
    selection = json.loads(Path(selection_path).read_text())
    if any(run["code_sha256"] != primary.manifest["code_sha256"] for run in selection["development_runs"]):
        raise ValueError("Development selection and confirmation use different frozen runner sources")
    groups, target_results = [], []
    for ds in datasets:
        if ds.budgets != GRID or set(ds.methods) != set(METHODS) or ds.arrivals != 7:
            raise ValueError("Frozen budget grid, four methods and seven arrivals required")
        if ds.manifest["code_sha256"] != primary.manifest["code_sha256"]:
            raise ValueError("Confirmation groups must share the same frozen runner source")
        if selection:
            if set(ds.seeds).intersection(selection["development_seeds"]):
                raise ValueError("Development/confirmation seed overlap")
            for record in ds.records.values():
                if record["effective_lr"] != selection["method_lrs"][record["method"]]:
                    raise ValueError("Confirmation LR differs from the development selection")
        for task in ds.tasks:
            for age in ds.ages:
                group = analyze_group(ds, task, age, primary=ds is primary)
                groups.append(group)
                target_results.append(targets_and_compression(ds, task, age))
    for name in ("seed_auc", "auc_summary", "contrasts", "points", "point_contrasts", "seed_differences"):
        write_csv(out/f"{name}.csv", [row for group in groups for row in group[name]])
    for name in ("target_seeds", "target_summary", "compression_seeds", "compression_summary"):
        write_csv(out/f"{name}.csv", [row for group in target_results for row in group[name]])
    records = []
    costs = []
    for ds in datasets:
        for row in ds.records.values():
            records.append({k: v for k, v in row.items() if k != "costs"})
            for kind, work in row["costs"].items():
                costs.append({**{key: row[key] for key in ("task", "seed", "age_mode", "method", "event_budget")},
                              "cost_category": kind, "proxy_work": work})
    write_csv(out/"all_seed_results.csv", records)
    write_csv(out/"all_seed_costs.csv", costs)
    if local:
        age_rows, age_seed_rows = age_comparisons(primary, local)
        write_csv(out/"age_paired_summary.csv", age_rows)
        write_csv(out/"age_paired_seeds.csv", age_seed_rows)
    provenance = [ds.provenance() for ds in datasets]
    result = dict(primary=groups[0]["iut"], groups=[g["iut"] for g in groups],
                  primary_comparisons=groups[0]["contrasts"],
                  analysis_source_sha256=sha(__file__), bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_replicates=BOOTSTRAP_REPLICATES, target_epsilon=EPSILON,
                  analysis_definition="Normalized trapezoidal integral of raw final population excess risk against log allocated total scalar-work budget; all five budgets and all 256 primary teachers retained.",
                  primary_units="independent paired teacher/data seeds, not events/budgets/methods",
                  interval_note="t inference is a large-sample approximation, not a finite-sample guarantee for unbounded loss; bootstrap does not replace the prespecified decision.",
                  work_note="Scalar-work proxy is not exact FLOPs or wall time; development work is disclosed separately and is not charged repeatedly to each deployed trajectory.",
                  development_selection=selection, input_provenance=provenance)
    write_json(out/"analysis_summary.json", result)
    make_latex(out, groups, provenance, selection)
    make_figures(out, groups)
    print(json.dumps(result["primary"], sort_keys=True))


def self_test():
    # Analytical fixtures only: no simulated training or experimental seed use.
    np.testing.assert_allclose(auc([1, 2, 4], [1, 2, 4]), 2.25)
    np.testing.assert_allclose(auc([1, 2, 4], [4, 8, 16]), 2.25)
    np.testing.assert_allclose(auc([0, 4, 2], [1, 2, 8]), 8/3)
    np.testing.assert_allclose(auc([3, 7, 5], [1, 2, 8]), 8/3+3)
    np.testing.assert_allclose(auc([[2, 2, 2], [1, 2, 4]], [1, 2, 4]), [2, 2.25])
    assert first_grid_target_index([.02, .005, .03]) == 1
    assert first_grid_target_index([.02, .03, .03]) is None
    assert first_grid_target_index([.001, .002, .003]) == 0
    assert first_grid_target_index([.01, .01]) == 0
    assert first_grid_target_index([1.1, .8, 1.2, .7, .6], epsilon=1.) == 1
    assert paired_inference([-1., -1., -1.])["p_one_sided_historical_lower"] == 0
    assert paired_inference([0., 0., 0.])["upper95_one_sided"] == 0
    # Negative but noisy values cannot be certified merely from their sign.
    assert paired_inference([-3., 0., 1.])["upper95_one_sided"] > 0
    known = paired_inference([-1., -2., -1., -2.])
    np.testing.assert_allclose(known["mean"], -1.5)
    np.testing.assert_allclose(known["sd"]**2, 1/3)
    np.testing.assert_allclose(known["p_one_sided_historical_lower"], student_t.cdf(-3*math.sqrt(3), 3))
    fixtures = np.column_stack([np.arange(8.), -np.arange(8.), np.zeros(8)])
    one, two = bootstrap_differences(fixtures), bootstrap_differences(fixtures)
    np.testing.assert_array_equal(one, two)
    np.testing.assert_allclose(one[:, 0], -one[::-1, 1])
    assert wilson(0, 10)["wilson95_low"] >= -1e-15
    print("analysis self-test passed: analytical AUC, target preservation, paired inference and paired bootstrap")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select-development", nargs="+", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--age", type=Path)
    parser.add_argument("--development-selection", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output_dir or bool(args.select_development) == bool(args.primary):
        parser.error("Choose exactly one of --select-development or --primary, plus --output-dir")
    out = output_dir(args.output_dir)
    try:
        if args.select_development:
            select_development(args.select_development, out)
        else:
            analyze_confirmation(args.primary, args.secondary, args.age, args.development_selection, out)
    except Exception as error:
        write_json(out/"analysis_failure.json", dict(primary_success=False, error_type=type(error).__name__,
                   error=str(error), analysis_source_sha256=sha(__file__),
                   interpretation="Analysis did not complete. No missing seed was dropped and no primary success can be declared."))
        raise


if __name__ == "__main__":
    main()
