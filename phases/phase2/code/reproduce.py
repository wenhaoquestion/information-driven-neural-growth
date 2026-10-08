"""Publication reproduction: saved-evidence rebuild by default; --full retrains."""
from pathlib import Path
import argparse,json,math,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def command(args,cwd=None):subprocess.run(list(map(str,args)),cwd=cwd or ROOT.parent,check=True)
def compare(a,b,path='root'):
    if isinstance(a,dict):
        for k,v in a.items():
            if k not in ['elapsed_seconds','config']:compare(v,b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11),(path,a,b)
    else:assert a==b,(path,a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true',help='Intentionally retrain the principal ensembles into results/replayed')
    p.add_argument('--no-build',action='store_true');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'tmp/cache'))
    command([sys.executable,ROOT/'code/restore_evidence.py'])
    command([sys.executable,ROOT/'code/verify_saved.py'])
    phase=ROOT.name
    if a.full:
        (ROOT/'results/replayed').mkdir(parents=True,exist_ok=True)
        if phase=='phase2':
            jobs=[('learning_neural_probe.py','learning_neural_probe32.json',['--seeds','32','--seed-base','17000']),('learning_neural_probe.py','learning_neural_aligned32.json',['--seeds','32','--seed-base','41000']),('learning_curvature_bridge.py','learning_curvature_bridge64.json',['--seeds','64'])]
        else:
            jobs=[]
            for name in ['confirm_eb4096.json','confirm_eb16384.json','confirm_mean16384.json','betting_followup4096.json','eb_followup4096.json']:
                old=json.loads((ROOT/'results'/name).read_text());args=old['command'][1:];args[args.index('--output')+1]='replayed/'+name
                jobs.append((Path(old['command'][0]).name,name,args))
        for script,name,args in jobs:
            if phase=='phase2':args=args+['--output','replayed/'+name]
            command([sys.executable,ROOT/'code'/script,*args])
            compare(json.loads((ROOT/'results'/name).read_text())['runs'],json.loads((ROOT/'results/replayed'/name).read_text())['runs'])
    if not a.no_plots:
        scripts=['learning_summarize.py','plot_learning.py','plot_theory.py'] if phase=='phase2' else ['summarize.py','plot_evidence_diagnostic.py','plot_complementarity.py','summarize_betting_followup.py','plot_betting_followup.py']
        for name in scripts:command([sys.executable,ROOT/'code'/name])
    if not a.no_build:
        tex=ROOT/'manuscript'
        for cmd in [['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex']]:command(cmd,tex)
    print('Publication rebuild complete. Full training replay:',a.full)
if __name__=='__main__':main()
