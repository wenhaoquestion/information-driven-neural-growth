"""Build the public English paper with existing TeX tools; no experiments."""
from pathlib import Path
import argparse
import os
import subprocess

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
out=args.output_dir.resolve()
out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,BIBINPUTS=str(root/'manuscript')+os.pathsep+os.environ.get('BIBINPUTS',''))
latex=['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(out),str(root/'manuscript/main.tex')]
for number,(command,cwd) in enumerate([(latex,root/'manuscript'),(['bibtex','main'],out),(latex,root/'manuscript'),(latex,root/'manuscript')],1):
    result=subprocess.run(command,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f'build_{number}.txt').write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-6000:])
        raise SystemExit(result.returncode)
print(out/'main.pdf')
