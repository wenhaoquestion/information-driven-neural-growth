#!/usr/bin/env python3
"""Render the supplied-module evidence diagnostic from saved raw summaries."""
from pathlib import Path
import json
import math
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "results" / "evidence_diagnostic_summary.json").read_text())
rows = [r for r in data["rows"] if r["condition"] == "signal"]
k = np.array([r["k"] for r in rows])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
ax = axes[0]
ax.plot(k, [r["reset_lower_bound"] for r in rows], ":", color="0.45", label="Fresh-audit lower bound")
ax.plot(k, [r["reset_cost_expected_exact"] for r in rows], "--", color="#3d64a8", label="Fresh audit: exact expected cost")
ax.errorbar(k, [r["reset_cost_mean"] for r in rows],
            yerr=[1.96 * r["reset_cost_sem"] for r in rows], fmt="o", color="#3d64a8",
            capsize=3, label="Fresh audit: Monte Carlo")
ax.plot(k, [r["n_shared"] for r in rows], "s-", color="#bf6238", label="Shared precommitted batch")
ax.set(xscale="log", yscale="log", xlabel="Number of proposed local modules K",
       ylabel="Labelled population examples", title="Same supplied candidates; different evidence use")
ax.set_xticks(k, [str(i) for i in k])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(fontsize=8)

def wilson(success_rate, n):
    z = 1.959963984540054
    c = (success_rate + z*z/(2*n)) / (1+z*z/n)
    h = z * math.sqrt(success_rate*(1-success_rate)/n + z*z/(4*n*n)) / (1+z*z/n)
    return c-h, c+h

ax = axes[1]
conds = ["signal", "mixed", "null", "boundary"]
for offset, prefix, color, label in [(-.12, "reset", "#3d64a8", "Fresh audit"), (.12, "shared", "#bf6238", "Shared batch")]:
    selected = [next(r for r in data["rows"] if r["k"] == 16 and r["condition"] == c) for c in conds]
    y = np.array([r[f"{prefix}_family_success_rate"] for r in selected])
    intervals = [wilson(v, data["repetitions"]) for v in y]
    err = np.array([[v-lo for v,(lo,hi) in zip(y,intervals)], [hi-v for v,(lo,hi) in zip(y,intervals)]])
    ax.errorbar(np.arange(4)+offset, y, yerr=np.maximum(err, 0), fmt="o", capsize=3, color=color, label=label)
ax.axhline(1-data["delta"], color="0.4", ls=":", label="Guaranteed target")
ax.set_xticks(np.arange(4), ["All useful", "Half useful", "No signal", "Risk-neutral\nboundary"])
ax.set(ylim=(.94, 1.002), ylabel="Fraction with every decision correct",
       title=f"K = 16; {data['repetitions']:,} independent replications")
ax.legend(fontsize=8, loc="lower right")
out = root / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "evidence_diagnostic.pdf")
fig.savefig(out / "evidence_diagnostic.png", dpi=180)
print(out / "evidence_diagnostic.pdf")
