#!/usr/bin/env python3
"""Numerical check of a proved low-rate cardinality expansion.

Only NumPy and Matplotlib are required. This evaluates explicit feasible
encoders; it does not claim numerical global optimization of the IB problem.

Run: python research/onset_experiment.py
Outputs: research/onset_results.json and research/onset_curvature.{pdf,png}.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".onset_cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".onset_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def kl_uniform(q: np.ndarray) -> float:
    """Stable KL(q || uniform), assuming q has normalized sum."""
    x = len(q) * q - 1.0
    terms = np.empty_like(x)
    small = np.abs(x) < 1e-3
    # (1+x) log(1+x) - x = sum_{j>=2} (-1)^j x^j/[j(j-1)].
    terms[small] = sum(
        (-1.0) ** j * x[small] ** j / (j * (j - 1)) for j in range(2, 12)
    )
    terms[~small] = (1 + x[~small]) * np.log1p(x[~small]) - x[~small]
    return float(np.sum(terms) / len(q))


def ray_q(n: int, z: float, index: int = 0) -> np.ndarray:
    q = np.full(n, (1 - z) / n)
    q[index] += z
    return q


def ray_a(n: int, z: float) -> float:
    return kl_uniform(ray_q(n, z))


def ray_derivative(n: int, z: float) -> float:
    return (n - 1) / n * (math.log1p((n - 1) * z) - math.log1p(-z))


def maximizing_z(n: int, a: float) -> tuple[float, float]:
    """Locate tangency after a grid verifies one change of derivative sign."""
    def numerator(z: float) -> float:
        return a * ray_derivative(n, a * z) * ray_a(n, z) - ray_a(n, a * z) * ray_derivative(n, z)

    grid = np.linspace(1e-3, 1 - 1e-10, 10001)
    vals = np.asarray([numerator(float(z)) for z in grid])
    crossings = np.flatnonzero((vals[:-1] > 0) & (vals[1:] <= 0))
    assert len(crossings) == 1, crossings
    lo, hi = float(grid[crossings[0]]), float(grid[crossings[0] + 1])
    for _ in range(70):
        mid = (lo + hi) / 2
        if numerator(mid) > 0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    eta = ray_a(n, a * z) / ray_a(n, z)
    assert a * a < eta < a
    return z, eta


def encoder(n: int, a: float, k: int, z: float, rate: float) -> dict:
    """Equal cost weights on k distinct maximizing rays plus one background."""
    uniform = np.full(n, 1 / n)
    rare = np.asarray([ray_q(n, z, i) for i in range(k)])
    ar = kl_uniform(rare[0])
    d = np.mean(rare - uniform, axis=0) / ar

    def at(t: float) -> tuple[float, np.ndarray, np.ndarray]:
        mass = t / ar
        background = uniform - t * d / (1 - mass)
        weights = np.r_[1 - mass, np.full(k, mass / k)]
        posteriors = np.vstack([background, rare])
        source = sum(w * kl_uniform(q) for w, q in zip(weights, posteriors))
        return source, weights, posteriors

    # J(t) >= t. Thus t <= rate brackets the local solution, when feasible.
    lo, hi = 0.0, rate
    assert np.min(at(hi)[2]) > 0
    for _ in range(80):
        mid = (lo + hi) / 2
        if at(mid)[0] < rate:
            lo = mid
        else:
            hi = mid
    source, weights, posteriors = at((lo + hi) / 2)
    output = sum(w * kl_uniform(a * q + (1 - a) * uniform) for w, q in zip(weights, posteriors))
    barycenter_error = float(np.max(np.abs(weights @ posteriors - uniform)))
    assert barycenter_error < 2e-14
    assert abs(source - rate) < 2e-14
    return dict(
        source_information=source,
        output_information=output,
        weights=weights.tolist(),
        posteriors=posteriors.tolist(),
        barycenter_error=barycenter_error,
    )


def main() -> None:
    rates = np.geomspace(1e-5, 1e-2, 31)
    experiments = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, n in zip(axes, [3, 4]):
        a = 0.5
        z, eta = maximizing_z(n, a)
        c = z / ray_a(n, z)
        for k in range(1, n + 1):
            predicted = 0.5 * n * (eta - a * a) * c * c * (1 / k - 1 / n)
            rows = []
            for rate in rates:
                result = encoder(n, a, k, z, float(rate))
                deficit = eta * rate - result["output_information"]
                scaled = float(deficit / rate**2)
                rows.append(dict(rate=float(rate), scaled_deficit=scaled, **result))
            color = f"C{k-1}"
            ax.plot(rates, [r["scaled_deficit"] for r in rows], color=color, label=f"{k+1} symbols")
            ax.axhline(predicted, color=color, ls=":", lw=1)
            if k < n:
                relative_error = abs(rows[0]["scaled_deficit"] / predicted - 1)
                assert relative_error < 0.002, (n, k, relative_error)
            else:
                assert max(abs(r["scaled_deficit"]) for r in rows) < 1e-8
            experiments.append(dict(n=n, a=a, z=z, eta=eta, c=c, k=k, predicted_coefficient=predicted, rows=rows))
        ax.set_xscale("log")
        ax.set_xlabel("Input information R (nats)")
        ax.set_ylabel(r"$(\eta R-I(Y;T))/R^2$")
        ax.set_title(f"{n}-state symmetric channel, a = {a}")
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "onset_curvature.pdf")
    fig.savefig(OUT / "onset_curvature.png", dpi=200)
    payload = dict(
        description="Explicit feasible encoders, not a numerical global optimizer. Dotted lines are proved limiting coefficients.",
        units="nats",
        seed=None,
        dependencies={"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        experiments=experiments,
    )
    (OUT / "onset_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    for item in experiments:
        print(f"n={item['n']} labels={item['k']+1}: predicted {item['predicted_coefficient']:.9f}; observed at R=1e-5 {item['rows'][0]['scaled_deficit']:.9f}")


if __name__ == "__main__":
    main()
