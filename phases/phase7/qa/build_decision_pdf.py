"""Render the complete Chinese decision note using existing Pandoc/XeLaTeX."""
from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[1]
out=root/'output'/'pdf';out.mkdir(parents=True,exist_ok=True)
ast=json.loads(subprocess.check_output(['pandoc',str(root/'PHASE7_DECISION_ZH.md'),'-f','markdown','-t','json']))
def transform(x):
    if isinstance(x,list):return [transform(v) for v in x]
    if not isinstance(x,dict):return x
    if x.get('t')=='Code':return {'t':'RawInline','c':['latex',r'\nolinkurl{'+x['c'][1]+'}']}
    return {k:transform(v) for k,v in x.items()}
body=subprocess.check_output(['pandoc','-f','json','-t','latex','--wrap=none'],input=json.dumps(transform(ast)).encode()).decode()
preamble=r'''\documentclass[11pt]{article}
\usepackage[paperwidth=148mm,paperheight=210mm,top=15mm,bottom=15mm,left=13mm,right=13mm,headheight=12pt,footskip=8mm]{geometry}
\usepackage{fontspec,xeCJK}
\setmainfont{texgyrepagella-regular.otf}[BoldFont=texgyrepagella-bold.otf,ItalicFont=texgyrepagella-italic.otf]
\setsansfont{texgyreheros-regular.otf}[BoldFont=texgyreheros-bold.otf]
\setCJKmainfont{Songti SC}[BoldFont=Heiti SC]
\setCJKsansfont{Heiti SC}
\setCJKmonofont{Songti SC}
\usepackage{amsmath,amssymb,unicode-math}
\setmathfont{latinmodern-math.otf}
\usepackage{xurl,xcolor,graphicx}
\definecolor{ink}{HTML}{18354A}
\definecolor{linkblue}{HTML}{176B86}
\usepackage[unicode,colorlinks=true,linkcolor=linkblue,urlcolor=linkblue,pdftitle={Phase 7 纯数学与信息论研究决策},pdfauthor={Wenyu Huang}]{hyperref}
\usepackage{bookmark,titlesec,fancyhdr,enumitem}
\titleformat{\section}{\Large\sffamily\bfseries\color{ink}}{}{0pt}{}
\titleformat{\subsection}{\large\sffamily\bfseries\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{12pt}{7pt}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\scriptsize\sffamily Phase 7 · 纯数学与信息论}
\fancyfoot[L]{\scriptsize 2026-10-07 · 本地工作稿}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.2pt}
\setlist{leftmargin=1.4em,itemsep=4pt,topsep=4pt}
\providecommand{\tightlist}{\setlength{\itemsep}{3pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{0}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt plus 1pt}
\linespread{1.12}\emergencystretch=2em\tolerance=1600\widowpenalty=10000\clubpenalty=10000\raggedbottom\urlstyle{same}
\begin{document}
'''
tex=root/'qa'/'phase7_decision_zh.tex';tex.write_text(preamble+body+'\n\\end{document}\n')
env=dict(os.environ)
for key,sub in [('TEXMFVAR','texmf-var'),('TEXMFCONFIG','texmf-config'),('XDG_CACHE_HOME','cache')]:
    p=root/'qa'/sub;p.mkdir(exist_ok=True);env[key]=str(p)
command=['xelatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(out),str(tex)]
for i in range(2):
    r=subprocess.run(command,cwd=root/'qa',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (root/'qa'/f'chinese_pdf_compile_{i+1}.txt').write_text(r.stdout)
    if r.returncode:print(r.stdout[-6000:]);raise SystemExit(r.returncode)
print(out/'phase7_decision_zh.pdf')
