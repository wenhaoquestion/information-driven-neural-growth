"""Reproduce the main calculations, figures, table, audits, and compiled paper.
Requires Python dependencies in requirements.txt, latexmk, pdflatex, and bibtex.
"""
from pathlib import Path
import os, subprocess, sys, shutil, json, re, platform
ROOT=Path(__file__).resolve().parents[1]

def main():
    tmp=ROOT/'tmp';tmp.mkdir(exist_ok=True)
    env=dict(os.environ,MPLCONFIGDIR=str(tmp/'matplotlib'),XDG_CACHE_HOME=str(tmp/'cache'))
    commands=[([sys.executable,'code/quartet_experiment.py'],ROOT),
              ([sys.executable,'code/plot_results.py'],ROOT),
              ([sys.executable,'research/main_review_checks.py'],ROOT),
              (['latexmk','-pdf','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex'],ROOT/'manuscript')]
    records=[]
    for i,(cmd,cwd) in enumerate(commands):
        print('Running:', ' '.join(cmd), flush=True)
        r=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (tmp/f'reproduce_{i}.log').write_text(r.stdout)
        records.append(dict(command=cmd,cwd=str(cwd.relative_to(ROOT)),returncode=r.returncode))
        if r.returncode:
            print(r.stdout[-8000:]);raise SystemExit(r.returncode)
    log=(ROOT/'manuscript/main.log').read_text()
    problems=[line for line in log.splitlines() if re.search(r'undefined|Overfull|Missing character|LaTeX Error',line,re.I)]
    if problems:
        print('\n'.join(problems));raise SystemExit('LaTeX quality check failed')
    dest=ROOT/'output/pdf/irreversible_representation_growth.pdf';dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'manuscript/main.pdf',dest)
    report=dict(status='passed',python=platform.python_version(),commands=records,
                latex_problems=problems,pdf=str(dest.relative_to(ROOT)))
    (ROOT/'results/reproduction.json').write_text(json.dumps(report,indent=2))
    print('Main reproduction passed; PDF:',dest)
if __name__=='__main__':main()
