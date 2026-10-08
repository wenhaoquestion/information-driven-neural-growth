"""Compile the copied Phase 7 manuscript without installing dependencies.

Use --output-dir for an inspectable, new build directory. The final PDF is
written only after all LaTeX/BibTeX passes succeed. Research sources and
earlier project phases are never changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pdf', type=Path,
                        default=root / 'output/pdf/phase7_working_paper.pdf')
    args = parser.parse_args()
    if args.output_dir is None:
        out = Path(tempfile.mkdtemp(prefix='phase7-paper-build-'))
    else:
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
    manuscript = root / 'manuscript'
    env = dict(os.environ)
    env['BIBINPUTS'] = str(manuscript) + os.pathsep + env.get('BIBINPUTS', '')
    latex = ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
             '-halt-on-error', '-output-directory', str(out), 'main.tex']
    commands = [(latex, manuscript), (['bibtex', 'main'], out),
                (latex, manuscript), (latex, manuscript)]
    for number, (command, cwd) in enumerate(commands, 1):
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / f'build_{number}.txt').write_text(result.stdout)
        if result.returncode:
            print(result.stdout[-6000:])
            raise SystemExit(result.returncode)
    log = (out / 'main.log').read_text(errors='replace')
    if 'There were undefined references' in log or 'There were undefined citations' in log:
        raise RuntimeError(f'Unresolved references: inspect {out / "main.log"}')
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(manuscript.iterdir()) if p.suffix in ('.tex', '.bib')}
    target = args.pdf.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / 'main.pdf', target)
    summary = {'pdf': str(target), 'build_directory': str(out),
               'pdf_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'sources_sha256': sources,
               'layout_warnings': [line for line in log.splitlines()
                                   if 'Overfull' in line or 'Underfull' in line]}
    (out / 'build_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
