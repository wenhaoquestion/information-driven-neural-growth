"""Regenerate neural study figures and manuscript tables from saved raw runs."""
from pathlib import Path
from learning_summarize import summarize_bridge, plot_bridge, summarize_probe, R
M=R.parent/'manuscript';M.mkdir(exist_ok=True)
rows,groups=summarize_bridge(R/'learning_curvature_bridge64.json')
plot_bridge(rows,groups)
summarize_probe()
probe=summarize_probe('learning_neural_aligned32.json','learning_information_aligned')

out=[r'\begin{table}[tb]',r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',r'\caption{Actual finite-sample neural training at $n=1024$. Entries are exact population log loss after fitting; $\Delta$ is the paired spectral-minus-random mean with a 95\% Student-$t$ interval half-width across 64 independent seeds. $T$ is the post-growth update budget. Fixed width uses the approximate parameter-update budget described in the text. Lower is better.}',r'\label{tab:neural-results}',r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Regime & $T$ & Spectral & Random & Fixed 2 & Continue 1 & $\Delta\pm$ half-width\\',r'\midrule']
for rg,label in [('four_state','Four states'),('rotated_nuisance','Rotated 8D'),('random_start','Random start'),('no_signal','No signal')]:
 for st in [30,300]:
  z=next(a for a in rows if a['regime']==rg and a['ntrain']==1024 and a['poststeps']==st)
  v=[float(z[k]) for k in ['spectral_post_risk','random_post_risk','fixed_width2_approx_equal_compute','continued_width1','spectral_minus_random','spectral_minus_random_95half']]
  out.append(f"{label} & {st} & {v[0]:.5f} & {v[1]:.5f} & {v[2]:.5f} & {v[3]:.5f} & ${v[4]:+.6f} \\pm {v[5]:.6f}$ "+r'\\')
out+= [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(M/'learning_tables.tex').write_text('\n'.join(out)+'\n')

out=[r'\begin{table}[tb]',r'\centering\footnotesize\setlength{\tabcolsep}{5pt}',r'\caption{Fresh-seed operation-aligned information-score ablation. Risks integrate the teacher label probability on 8192 held-out Gaussian inputs, and are averaged across 32 seeds. The last column gives aligned-CMI minus random-growth risk with a paired 95\% Student-$t$ interval half-width. Fixed width 3 uses the leading-width compute proxy; its initialization differs from the grown model.}',r'\label{tab:aligned-probe}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Task & Aligned CMI & Random & Entropy & Fixed 3 & Difference $\pm$ half-width\\',r'\midrule']
for task in ['teacher','localized','linear_null']:
 rr={r['method']:r for r in probe if r['regime']==task}
 vals=[rr[m]['mean'] for m in ['aligned_information','random','activation_entropy','fixed_width3_parameter_compute']]
 d=rr['aligned_information']['minus_random'];hw=rr['aligned_information']['paired95_halfwidth']
 out.append(f"{task.replace('_',' ')} & "+' & '.join(f'{v:.5f}' for v in vals)+f' & ${d:+.5f} \\pm {hw:.5f}$ '+r'\\')
 print(task,'aligned minus random',d,hw)
out+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(M/'learning_aligned_table.tex').write_text('\n'.join(out)+'\n')
