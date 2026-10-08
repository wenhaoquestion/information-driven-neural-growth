#!/usr/bin/env python3
"""Build catalogued English papers in fresh isolated copies; no experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="append", help="Catalogue id; repeat to select papers")
    parser.add_argument("--output", type=Path, default=ROOT / "_build" / "papers")
    args = parser.parse_args()
    papers = json.loads((ROOT / "docs" / "papers.json").read_text())
    if args.paper:
        unknown = set(args.paper) - {p["id"] for p in papers}
        if unknown:
            parser.error("Unknown paper ids: " + ", ".join(sorted(unknown)))
        papers = [p for p in papers if p["id"] in args.paper]
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing output is never overwritten")
    for binary in {"pdflatex", "bibtex"}:
        if shutil.which(binary) is None:
            parser.error(f"Required existing executable not found: {binary}")
    output.mkdir(parents=True)
    results = []
    for paper in papers:
        dest = output / paper["id"]
        source = dest / "source"
        phase = Path(paper["tex"]).parts[1]
        shutil.copytree(ROOT / "phases" / phase, source,
                        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk", "_build", "_reproduce"))
        tex = source / Path(paper["tex"]).relative_to(Path("phases") / phase)
        build = dest / "tex"
        build.mkdir()
        latex = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build), tex.name]
        commands = [latex]
        if paper["bibliography"] == "bibtex":
            commands.append(["bibtex", tex.stem])
        commands.extend([latex, latex])
        passes = []
        for index, command in enumerate(commands):
            env = os.environ.copy()
            cwd = tex.parent
            if command[0] == "bibtex":
                cwd = build
                env["BIBINPUTS"] = str(tex.parent) + os.pathsep + env.get("BIBINPUTS", "")
                env["BSTINPUTS"] = str(tex.parent) + os.pathsep + env.get("BSTINPUTS", "")
            result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180, env=env)
            (dest / f"pass-{index + 1}.txt").write_text(result.stdout)
            passes.append(result.returncode)
            if result.returncode:
                break
        log = (build / (tex.stem + ".log"))
        logtext = log.read_text(errors="replace") if log.exists() else ""
        unresolved = [line for line in logtext.splitlines() if "undefined" in line.lower() or "Rerun to get cross-references right" in line]
        pdf = build / (tex.stem + ".pdf")
        ok = len(passes) == len(commands) and all(c == 0 for c in passes) and pdf.exists() and not unresolved
        if pdf.exists():
            shutil.copy2(pdf, dest / Path(paper["pdf"]).name)
        results.append({"id": paper["id"], "passes": passes, "unresolved": unresolved, "passed": ok})
        print(f'{paper["id"]}: {"PASS" if ok else "FAIL"}', flush=True)
    (output / "BUILD_RESULTS.json").write_text(json.dumps({"papers": results, "passed": all(r["passed"] for r in results)}, indent=2) + "\n")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
