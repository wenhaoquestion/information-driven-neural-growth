"""Bounded floating-point consistency checks; not an interval certificate."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi, roots_legendre


def main() -> None:
    rows = []
    for m in range(17):
        for odd in (False, True):
            if odd:
                raw_nodes, _ = roots_jacobi(m + 1, 1, 0)
            else:
                raw_nodes, _ = roots_legendre(m + 1)
            nodes = (raw_nodes + 1) / 2
            z, weights = roots_legendre(max(32, 2 * m + 4))
            t, weights = (z + 1) / 2, weights / 2
            q = np.prod(t[:, None] - nodes[None, 1:], axis=1)
            g = q * q * ((1 - t) if odd else 1)
            mass = float(weights @ g)
            moment = float(weights @ (t * g))
            value = moment / mass
            predicted = float(nodes[0])
            relative_error = abs(value - predicted) / predicted
            assert relative_error < 1e-9
            rows.append({
                "degree": 2 * m + int(odd),
                "endpoint_ratio": value,
                "smallest_shifted_root": predicted,
                "relative_error": relative_error,
                "asymmetry": (1 - value) / value,
            })
    root = Path(__file__).parent
    source = root / "exact_polynomial_asymmetry.tex"
    result = {
        "method": "Independent Gauss-Legendre integration of explicit extremizers",
        "qualification": "Observed floating-point agreement; not a rigorous numerical certificate",
        "python": platform.python_version(),
        "instances": len(rows),
        "max_relative_error": max(row["relative_error"] for row in rows),
        "theorem_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    path = root / "exact_asymmetry_check.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
