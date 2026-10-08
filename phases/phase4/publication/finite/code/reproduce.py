"""Reproduce the isolated finite candidate without touching historical files."""
from pathlib import Path
import hashlib,json,os,platform,re,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main():
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    env=dict(os.environ,MPLCONFIGDIR=str(qa/'matplotlib'),XDG_CACHE_HOME=str(qa/'cache'))
    steps=[([sys.executable,'code/quartet_experiment.py'],ROOT),
           ([sys.executable,'code/plot_results.py'],ROOT),
           ([sys.executable,'research/main_review_checks.py'],ROOT),
           ([sys.executable,'research/sharp_rate_check.py'],ROOT),
           ([sys.executable,'research/plot_sharp_rate.py'],ROOT),
           (['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],ROOT/'manuscript')]
    records=[]
    for i,(cmd,cwd) in enumerate(steps):
        start=time.monotonic();r=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (qa/f'reproduce_{i}.log').write_text(r.stdout)
        records.append(dict(command=cmd,cwd=str(cwd.relative_to(ROOT)),returncode=r.returncode,seconds=time.monotonic()-start))
        print('Completed:',cmd[-1],r.returncode,flush=True)
        if r.returncode: raise RuntimeError(r.stdout[-5000:])
    log=(ROOT/'manuscript/main.log').read_text()
    errors=[x for x in log.splitlines() if re.search('undefined|Overfull|Missing character|LaTeX Error',x,re.I)]
    if errors:raise RuntimeError('\n'.join(errors))
    pdf=ROOT/'output/pdf/finite_representation_growth.pdf';pdf.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'manuscript/main.pdf',pdf)
    report=dict(status='passed',python=platform.python_version(),commands=records,latex_errors=errors,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    (ROOT/'results/reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
