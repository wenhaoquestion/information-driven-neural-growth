#!/usr/bin/env python3
"""Exact-law Monte Carlo for the prescribed-module audit theorem.

This is NOT neural training. Knots and candidate amplitude are supplied.
Negative-binomial and multinomial sufficient-statistic sampling are exactly
equivalent to the stated iid input/label experiment, with no oracle labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[1]


def kl(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def checks() -> dict:
    errors = []
    records = []
    for k in [2, 4, 8, 16]:
        x = np.arange(1, k + 1, dtype=float)
        relu = lambda z: np.maximum(z, 0)
        phi = np.stack([relu(x - t + 1) - 2 * relu(x - t) + relu(x - t - 1)
                        for t in range(1, k + 1)])
        errors.append(float(np.max(np.abs(phi - np.eye(k)))))
        eps = 0.2
        signs = (-1.) ** np.arange(1, k + 1)
        eta = 0.5 + eps * signs
        p = np.full(k, 0.5)
        gains = []
        for t in range(k):
            q = p + eps * signs[t] * phi[t]
            risk_p = np.mean(eta * (1 - p) ** 2 + (1 - eta) * p ** 2)
            risk_q = np.mean(eta * (1 - q) ** 2 + (1 - eta) * q ** 2)
            dp = (1 - p) ** 2 - (1 - q) ** 2
            dn = p ** 2 - q ** 2
            mean = np.mean(eta * dp + (1 - eta) * dn)
            variance = np.mean(eta * dp ** 2 + (1 - eta) * dn ** 2) - mean ** 2
            h = q - p
            identity = 4 * np.mean(h ** 2 * eta * (1 - eta)) + np.var(h * (2 * eta - p - q))
            errors.extend([abs(risk_p - risk_q - eps ** 2 / k), abs(mean - eps ** 2 / k),
                           abs(variance - identity), abs(np.mean(h ** 2) - eps ** 2 / k)])
            gains.append(float(mean))
            p = q
        records.append({"k": k, "signs": signs.tolist(), "initial_excess": eps ** 2, "gains": gains,
                        "final_excess": float(np.mean((eta - p) ** 2))})
    # Constant-prediction witness, including movement much larger than gain.
    witnesses = []
    for d in [0.1, 0.25, 0.5]:
        for gain in [d / 100, d / 10, d / 2]:
            p, q = (1 - d) / 2, (1 + d) / 2
            eta = 0.5 + gain / (2 * d)
            dp, dn = (1 - p) ** 2 - (1 - q) ** 2, p ** 2 - q ** 2
            actual_gain = eta * dp + (1 - eta) * dn
            divergence = kl(eta, 0.5)
            errors.append(abs(actual_gain - gain))
            assert divergence <= gain ** 2 / d ** 2 + 1e-14
            witnesses.append({"d": d, "gain": gain, "movement": d * d,
                              "kl": divergence, "kl_upper": gain ** 2 / d ** 2})
    max_error = max(errors)
    assert max_error < 1e-12, max_error
    return {"max_absolute_identity_error": max_error, "module_sequences": records,
            "constant_pair_witnesses": witnesses,
            "evidence_type": "independent direct finite-population calculation"}


def run(repetitions: int, seed: int) -> None:
    outdir = ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    eps, delta = 0.2, 0.05
    rng = np.random.default_rng(seed)
    summary = {
        "evidence_type": "Monte Carlo with supplied fixed neural modules; no training",
        "seed": seed, "repetitions": repetitions, "epsilon": eps, "delta": delta,
        "sampling": "NB waiting times + Binomial signed-label sufficient counts; Multinomial shared counts + Binomial signed-label counts; ordinary labels reconstructed by fixed sign map",
        "revision": "Known alternating module signs make all-useful target nonconstant. Earlier unsigned artifacts preserved under research/unsigned_diagnostic.",
        "rows": [], "checks": checks(),
    }
    with (outdir / "evidence_diagnostic_raw.jsonl").open("w") as handle:
        for k in [2, 4, 8, 16]:
            alpha = beta = delta / k
            m = math.ceil(8 / eps ** 2 * math.log(1 / min(alpha, beta)))
            n_shared = math.ceil(16 * k / eps ** 2 * math.log(2 * k / delta))
            threshold = 0.5 + 3 * eps / 4
            # Exact fixed-m tail probabilities (as a supplement to Hoeffding).
            cut = math.floor(m * threshold)
            false_accept_boundary = float(binom.sf(cut, m, 0.5 + eps / 2))
            false_reject_signal = float(binom.cdf(cut, m, 0.5 + eps))
            for condition in ["signal", "mixed", "null", "boundary"]:
                theta = np.full(k, eps)
                if condition == "mixed":
                    theta[::2] = 0
                elif condition == "null":
                    theta[:] = 0
                elif condition == "boundary":
                    theta[:] = eps / 2
                signs = (-1.) ** np.arange(1, k + 1)
                eta = 0.5 + signs * theta
                signed_eta = 0.5 + theta
                expected_decisions = theta == eps
                gain_vector = (2 * eps * theta - eps ** 2) / k
                # Number of failures before m successes in iid event X=t.
                reset_cost = rng.negative_binomial(m, 1 / k, size=(repetitions, k)) + m
                reset_success = rng.binomial(m, signed_eta, size=(repetitions, k))
                reset_actual_ones = np.where(signs[None, :] > 0, reset_success, m - reset_success)
                reset_accept = reset_success / m > threshold
                shared_count = rng.multinomial(n_shared, np.full(k, 1 / k), size=repetitions)
                shared_success = rng.binomial(shared_count, signed_eta)
                shared_actual_ones = np.where(signs[None, :] > 0, shared_success, shared_count - shared_success)
                shared_accept = np.divide(shared_success, shared_count,
                                          out=np.zeros_like(shared_success, dtype=float),
                                          where=shared_count > 0) > threshold
                reset_correct = np.all(reset_accept == expected_decisions, axis=1)
                shared_correct = np.all(shared_accept == expected_decisions, axis=1)
                reset_total = np.sum(reset_cost, axis=1)
                for i in range(repetitions):
                    record = {
                        "k": k, "condition": condition, "replicate": i,
                        "epsilon": eps, "delta": delta, "theta": theta.tolist(),
                        "signs": signs.tolist(), "label_probabilities": eta.tolist(),
                        "reset_m_per_coordinate": m,
                        "reset_label_cost_by_event": reset_cost[i].tolist(),
                        "reset_ones_by_event": reset_actual_ones[i].tolist(),
                        "reset_signed_ones_by_event": reset_success[i].tolist(),
                        "reset_accept": reset_accept[i].tolist(),
                        "reset_family_correct": bool(reset_correct[i]),
                        "shared_n": n_shared,
                        "shared_count_by_coordinate": shared_count[i].tolist(),
                        "shared_ones_by_coordinate": shared_actual_ones[i].tolist(),
                        "shared_signed_ones_by_coordinate": shared_success[i].tolist(),
                        "shared_accept": shared_accept[i].tolist(),
                        "shared_family_correct": bool(shared_correct[i]),
                        "reset_risk_gains": (gain_vector * reset_accept[i]).tolist(),
                        "shared_risk_gains": (gain_vector * shared_accept[i]).tolist(),
                    }
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                row = {
                    "k": k, "condition": condition, "alpha": alpha, "beta": beta,
                    "m": m, "n_shared": n_shared,
                    "reset_cost_mean": float(np.mean(reset_total)),
                    "reset_cost_sd": float(np.std(reset_total, ddof=1)),
                    "reset_cost_sem": float(np.std(reset_total, ddof=1) / math.sqrt(repetitions)),
                    "reset_cost_expected_exact": k * k * m,
                    "reset_lower_bound": k * k / (2 * eps ** 2) * kl(1 - beta, alpha),
                    "reset_family_success_rate": float(np.mean(reset_correct)),
                    "shared_family_success_rate": float(np.mean(shared_correct)),
                    "reset_accepted_mean": float(np.mean(np.sum(reset_accept, axis=1))),
                    "shared_accepted_mean": float(np.mean(np.sum(shared_accept, axis=1))),
                    "reset_risk_gain_mean": float(np.mean(reset_accept @ gain_vector)),
                    "shared_risk_gain_mean": float(np.mean(shared_accept @ gain_vector)),
                    "exact_per_event_false_accept_boundary": false_accept_boundary,
                    "exact_per_event_false_reject_signal": false_reject_signal,
                }
                summary["rows"].append(row)
    (outdir / "evidence_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"repetitions": repetitions, "rows": len(summary["rows"]),
                      "max_identity_error": summary["checks"]["max_absolute_identity_error"],
                      "signal_cost_ratios": [{"k": r["k"], "reset_over_shared": r["reset_cost_mean"] / r["n_shared"]}
                                              for r in summary["rows"] if r["condition"] == "signal"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=431900)
    args = parser.parse_args()
    run(args.repetitions, args.seed)
