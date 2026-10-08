"""Produce frozen delivery metadata from completed outputs only."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def csvrows(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))
endpoint=csvrows('endpoint_polynomial.csv');interior=csvrows('finite_degree.csv')
timing={}
for stage in ['correctness','weighted','polynomial','endpoint','endpoint_mp']:
    lines=(ROOT/'logs'/f'{stage}.stdout.log').read_text().splitlines()
    objs=[json.loads(line) for line in lines if line.startswith('{')]
    last=objs[-1]
    timing[stage]=last.get('seconds',last.get('elapsed_seconds'))
summary={'status':'complete','scope':'finite numerical constructions and implementation checks; no neural training; no interval certificate','row_counts':{'endpoint_polynomial':len(endpoint),'interior_degree_width':len(interior),'weighted_hinge':len(csvrows('weighted_endpoint_hinge.csv')),'spacing_weight':len(csvrows('spacing_weight_screen.csv'))},'highest_endpoint_instance':endpoint[-1],'v1_sequence_rows':[r for r in interior if r['v1_sequence']=='True'],'completed_stage_seconds':timing,'normalization_endpoint':'On original [0,1]: divide by M_bound=2*P_n(0)/Z_n+delta^2, an analytic global curvature upper bound. Do not reuse absolute costs for an affine-interior construction requiring global bounds outside the original hull.','proof_threshold_caveat':'The finite endpoint n<=8192 points do not satisfy every coarse sufficient inequality used in the asymptotic proof; n>=65536 is an audited conservative analytic threshold. Costs here computed directly.','checks':{'correctness':json.loads((ROOT/'raw'/'correctness.json').read_text())['status'],'endpoint_mp_max_relative_error':json.loads((ROOT/'raw'/'endpoint_mpmath_crosscheck.json').read_text())['max_relative_error'],'endpoint_two_order_max_relative_error':max(float(r['quadrature_relative_crosscheck']) for r in endpoint),'interior_two_order_max_relative_error':max(r['max_cell_relative_error'] for r in json.loads((ROOT/'raw'/'quadrature_crosscheck.json').read_text()))},'deliverables':{'report':'results/EXPERIMENT_REPORT.md','figures':['figures/endpoint_construction.pdf','figures/equal_mass_degree_width.pdf'],'reproduce':'code/reproduce.sh','configuration':'configs/run_config.json'},'retained_initial_failures':['logs/polynomial_initial_serialization_failure.stdout.log','logs/endpoint_initial_quadrature_failure.stdout.log']}
(ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and 'matplotlib_cache' not in str(p) and p.name!='SHA256SUMS.json'}
(ROOT/'logs'/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files_hashed':len(manifest),'row_counts':summary['row_counts'],'stage_seconds':timing,'status':'complete'}))
