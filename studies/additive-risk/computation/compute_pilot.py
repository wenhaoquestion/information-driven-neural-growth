#!/usr/bin/env python3
"""Independent, bounded, deterministic population pilot. No training or sampling.

Only writes beside this file. Does not import or execute historical project code.
Use `compute`, then `compare` after the first pass has been frozen.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
import sys, time, json, math, csv, hashlib, platform, resource, signal
from pathlib import Path
from itertools import combinations
START = time.perf_counter()
import numpy as np
import scipy
from scipy.special import roots_legendre

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "historical_snapshot"
SOURCE_FILES = [
 'cross_stage_audit_work/reviews/NEXT_QUESTION_RECOMMENDATION.md',
 'phase8_research_work/research/experiments/code/hierarchy_core.py',
 'phase8_research_work/research/experiments/code/run_experiments.py',
 'phase8_research_work/manuscript/curvature_hierarchy.tex',
 'phase8_research_work/manuscript/unequal_weights_endpoint.tex',
]
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                         allow_nan=False)+'\n', encoding='utf-8')

def save_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def log(event, **kwargs):
    record = {'elapsed_seconds': time.perf_counter()-START, 'event':event, **kwargs}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with (OUT/'run.log').open('a', encoding='utf-8') as handle:
        handle.write(line+'\n')

def make_partitions():
    """Restricted-growth strings enumerate each set partition exactly once."""
    rows=[]
    def extend(labels):
        if len(labels)==4:
            rows.append(tuple(tuple(i for i in range(4) if labels[i]==j)
                              for j in range(max(labels)+1)))
            return
        for value in range(max(labels)+2):
            extend(labels+[value])
    extend([0])
    return sorted(rows, key=lambda p:(len(p),p))

PARTS=make_partitions()
COARSE=[i for i,p in enumerate(PARTS) if len(p)<=2]
FINE=[i for i,p in enumerate(PARTS) if len(p)<=3]
HIER=[(c,f) for c in COARSE for f in FINE
      if all(any(set(b)<=set(a) for a in PARTS[c]) for b in PARTS[f])]
assert len(PARTS)==15 and len(HIER)==39
SUBSETS=[c for size in range(1,5) for c in combinations(range(4),size)]
NODES={}

def part_name(part):
    return '|'.join(''.join('ABCD'[i] for i in block) for block in part)

def cell_name(cell):
    return ''.join('ABCD'[i] for i in cell)

def gl_nodes(order):
    if order not in NODES:
        NODES[order]=roots_legendre(order)
    return NODES[order]

def positive_kernel(t, x, mass):
    """Original-mass Jensen kernel, evaluated from its nonnegative sides."""
    center=math.fsum(float(a*b) for a,b in zip(x,mass))/math.fsum(mass)
    left=np.sum(mass[:,None]*np.maximum(t[None,:]-x[:,None],0.0),axis=0)
    right=np.sum(mass[:,None]*np.maximum(x[:,None]-t[None,:],0.0),axis=0)
    return np.where(t<=center,left,right)

class Curvature:
    def __init__(self, family, n, order):
        self.family,self.n,self.order=family,n,order
        self.width=16*math.log(n)/n if family=='interior_v1' else (4*math.log(n)/n)**2
        self.domain=(-1.5,2.5) if family=='interior_v1' else (0.,1.)
        self.constant=8*self.width**2 if family=='interior_v1' else self.width**2
        self.coordinate_slope=4. if family=='interior_v1' else 1.
        self.coordinate_shift=-1.5 if family=='interior_v1' else 0.
        self.log_peak=float(self.log_polynomial(np.array([0.]))[0])
        if family=='interior_v1':
            self.kernel_domain=(-4.,4.)
            self.needle_centers=(-self.width,1+self.width)
            self.needle_directions=(1.,1.) # squared polynomial is even
            self.peak_knots=[self.width*v for v in (-1,-.5,-.25,0,.25,.5,1)]
        else:
            self.kernel_domain=(0.,1.)
            self.needle_centers=(0.,1.)
            self.needle_directions=(1.,-1.)
            self.peak_knots=[self.width*v for v in (0,1/64,1/16,1/4,1/2,1)]
        self.scaled_Z=self.quad(lambda s:self.scaled_poly(s),
                                *self.kernel_domain,self.peak_knots)
        self.log_Z=self.log_peak+math.log(self.scaled_Z)
        self.tail_density_upper=math.exp(-self.log_peak)/self.scaled_Z

    def log_polynomial(self, t):
        if self.family=='interior_v1':
            v=(self.width**2-t*t)/16
        else:
            # T_n(-x)^2=T_n(x)^2; choose a positive external argument.
            v=2*(self.width-t)/(1-self.width)
        result=np.empty_like(v)
        outside=v>0
        a=2*self.n*np.arcsinh(np.sqrt(v[outside]/2))
        result[outside]=2*(np.logaddexp(a,-a)-math.log(2))
        b=2*self.n*np.arcsin(np.sqrt(np.clip(-v[~outside]/2,0.,1.)))
        with np.errstate(divide='ignore'):
            result[~outside]=2*np.log(np.abs(np.cos(b)))
        return result

    def scaled_poly(self, t):
        return np.exp(self.log_polynomial(np.asarray(t))-self.log_peak)

    def quad(self, integrand, lower, upper, extra=()):
        knots=sorted(set([lower,upper]+[float(v) for v in extra if lower<v<upper]))
        nodes,weights=gl_nodes(self.order)
        values=[]
        for a,b in zip(knots,knots[1:]):
            t=a+(b-a)*(nodes+1)/2
            values.append(float((b-a)/2*np.dot(weights,integrand(t))))
        return math.fsum(values)

    def needle_integral(self, weight, lower, upper, extra=()):
        values=[]
        for center,direction in zip(self.needle_centers,self.needle_directions):
            sa=(lower-center)*direction
            sb=(upper-center)*direction
            lo,hi=min(sa,sb),max(sa,sb)
            knots=list(self.peak_knots)+[(v-center)*direction for v in extra]
            values.append(self.quad(lambda s:weight(center+direction*s)*
                                    self.scaled_poly(s)/self.scaled_Z,
                                    lo,hi,knots))
        return math.fsum(values)

    def full_integral(self, weight):
        a,b=self.domain
        return self.needle_integral(weight,a,b)+self.constant*self.quad(weight,a,b)

    def normalization(self):
        slope,shift=self.coordinate_slope,self.coordinate_shift
        m0=slope*self.full_integral(lambda z:(z-shift)/slope)
        m1=slope*self.full_integral(lambda z:1-(z-shift)/slope)
        A=max(m0,m1)
        return dict(A_unscaled_composed_generator=A,loss0_max_unscaled=m0,
                    loss1_max_unscaled=m1,loss0_max_normalized=m0/A,
                    loss1_max_normalized=m1/A,cost_scale_from_raw_generator=1/A,
                    full_raw_curvature_integral=self.full_integral(lambda z:np.ones_like(z)),
                    symmetry_relative_error=abs(m0-m1)/A,
                    canonical_report_domain=[0.,1.],raw_full_report_domain=list(self.domain),
                    affine_raw_coordinate_slope=slope,affine_raw_coordinate_shift=shift,
                    raw_curvature_constant=self.constant,scaled_kernel_normalizer=self.scaled_Z,
                    log_kernel_normalizer=self.log_Z,log_kernel_peak=self.log_peak,
                    pointwise_oscillatory_tail_density_upper=self.tail_density_upper)

    def cell(self, x, masses):
        if len(x)==1:
            return dict(total=0.,needles=0.,constant=0.)
        mean=math.fsum(float(a*b) for a,b in zip(x,masses))/math.fsum(masses)
        needles=self.needle_integral(lambda t:positive_kernel(t,x,masses),
                                    float(min(x)),float(max(x)),list(x)+[mean])
        const=self.constant*.5*math.fsum(float(p*(t-mean)**2) for t,p in zip(x,masses))
        return dict(total=needles+const,needles=needles,constant=const)

def close_ties(values, best):
    scale=max(abs(best),np.finfo(float).tiny)
    return [i for i,v in enumerate(values) if abs(v-best)<=1e-9*scale]

def compute_one(family,n,order):
    clock=time.perf_counter()
    g=Curvature(family,n,order)
    norm=g.normalization()
    if family=='interior_v1':
        x=np.array([-1.,0.,1.,2.]); masses=np.full(4,.25)
        posteriors=(x+1.5)/4
    else:
        d=g.width; x=np.array([0.,2*d,1-2*d,1.])
        masses=np.array([.5-d*d,d*d,d*d,.5-d*d]); posteriors=x.copy()
    cells={cell_name(c):g.cell(x[list(c)],masses[list(c)]) for c in SUBSETS}
    A=norm['A_unscaled_composed_generator']
    raw_costs=[math.fsum(cells[cell_name(c)]['total'] for c in part) for part in PARTS]
    costs=[v/A for v in raw_costs]
    opt2=min(costs[i] for i in COARSE); opt3=min(costs[i] for i in FINE)
    rho=[max(costs[c]/opt2,costs[f]/opt3) for c,f in HIER]
    delta=[max(costs[c]-opt2,costs[f]-opt3) for c,f in HIER]
    best_rho=min(rho);best_delta=min(delta)
    assert opt2>0 and opt3>0 and best_delta>=0
    partitions=[dict(partition_id=i,partition=part_name(p),cells=[list(c) for c in p],
                     cell_count=len(p),cost_raw=raw_costs[i],cost_normalized=costs[i])
                for i,p in enumerate(PARTS)]
    hierarchies=[dict(hierarchy_id=i,coarse_id=c,fine_id=f,coarse=part_name(PARTS[c]),
                     fine=part_name(PARTS[f]),D2=costs[c],D3=costs[f],
                     ratio2=costs[c]/opt2,ratio3=costs[f]/opt3,rho=rho[i],
                     excess2=costs[c]-opt2,excess3=costs[f]-opt3,delta=delta[i])
                 for i,(c,f) in enumerate(HIER)]
    ir=rho.index(best_rho);ia=delta.index(best_delta)
    params=dict(candidate_id=f'{family}_n{n}',family=family,n=n,
                curvature_degree=(4 if family=='interior_v1' else 2)*n,
                epsilon=g.width if family=='interior_v1' else None,
                d=g.width if family=='endpoint' else None,quadrature_order=order,
                raw_locations=x.tolist(),posteriors=posteriors.tolist(),source_masses=masses.tolist(),
                state_naming='A,B,C,D mean increasing posterior position; IDs 0,1,2,3',
                width_rule='16 log(n)/n' if family=='interior_v1' else '(4 log(n)/n)^2')
    summary=dict(candidate_id=params['candidate_id'],family=family,n=n,
                 opt2=opt2,opt3=opt3,rho=best_rho,delta=best_delta,
                 rho_hierarchy_id=ir,delta_hierarchy_id=ia,
                 rho_coarse=hierarchies[ir]['coarse'],rho_fine=hierarchies[ir]['fine'],
                 delta_coarse=hierarchies[ia]['coarse'],delta_fine=hierarchies[ia]['fine'],
                 delta_rho_optimizer=delta[ir],rho_delta_optimizer=rho[ia],
                 flat2=';'.join(part_name(PARTS[i]) for i in COARSE if abs(costs[i]-opt2)<=1e-9*opt2),
                 flat3=';'.join(part_name(PARTS[i]) for i in FINE if abs(costs[i]-opt3)<=1e-9*opt3),
                 A=norm['A_unscaled_composed_generator'],cost_scale=1/A,
                 D1=costs[0],delta_over_1e_minus3=best_delta/1e-3,
                 rho_optimal_ids=close_ties(rho,best_rho),delta_optimal_ids=close_ties(delta,best_delta),
                 elapsed_seconds=time.perf_counter()-clock)
    return dict(parameters=params,normalization=norm,cells=cells,partitions=partitions,
                hierarchies=hierarchies,summary=summary)

def metadata():
    return dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                platform=platform.platform(),thread_environment={k:os.environ[k] for k in THREAD_KEYS},
                source_sha256={name:digest(ROOT/name) for name in SOURCE_FILES},
                script_sha256=digest(Path(__file__)),no_old_results_read_before_first_pass=True,
                kind='floating-point pilot; not a rigorous interval certificate',
                numerical_method='nonnegative curvature kernels; piecewise 64-point Gauss-Legendre; constant curvature integrated analytically',
                tail_note='GL does not resolve every tail oscillation. Pointwise tail bound is recorded, tails are negligible for these candidates. No full numerical certification is asserted.',
                sampling=False,training=False,network=False,new_dependencies=False,
                target_wall_seconds=120,target_peak_memory_bytes=1024**3)

def run_compute():
    if (OUT/'first_pass.json').exists():
        raise RuntimeError('Frozen first pass exists; refusing to overwrite or silently rerun.')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second compute cap')))
    signal.alarm(120)
    log('start',mode='compute',order=64,candidate_count=10)
    candidates=[('interior_v1',n) for n in (2048,4096,8192)]+[
                 ('endpoint',n) for n in (128,256,512,1024,2048,4096,8192)]
    results=[]
    for family,n in candidates:
        result=compute_one(family,n,64);results.append(result)
        log('candidate',**result['summary'])
    save_json('first_pass.json',dict(metadata=metadata(),candidates=results))
    first_hash=digest(OUT/'first_pass.json')
    save_json('first_pass_freeze.json',dict(sha256=first_hash,old_saved_results_read=False,
               wall_seconds_at_freeze=time.perf_counter()-START,
               timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    log('first_pass_frozen',sha256=first_hash)
    checks=[]
    for first in results:
        par=first['parameters']; check=compute_one(par['family'],par['n'],128)
        relative={key:abs(first['cells'][key]['total']-check['cells'][key]['total'])/
                     max(abs(check['cells'][key]['total']),1e-300) for key in first['cells']}
        checks.append(dict(candidate_id=par['candidate_id'],order1=64,order2=128,
                      max_cell_relative_difference=max(relative.values()),cell_relative_differences=relative,
                      A_relative_difference=abs(first['summary']['A']/check['summary']['A']-1),
                      rho_relative_difference=abs(first['summary']['rho']/check['summary']['rho']-1),
                      delta_relative_difference=abs(first['summary']['delta']/check['summary']['delta']-1),
                      check_summary=check['summary'],check_normalization=check['normalization'],
                      check_cells=check['cells']))
    save_json('precision_recheck.json',checks)
    save_csv('summary.csv',[r['summary'] for r in results])
    save_csv('normalization.csv',[dict(candidate_id=r['parameters']['candidate_id'],**r['normalization']) for r in results])
    save_csv('cells.csv',[dict(candidate_id=r['parameters']['candidate_id'],cell=key,
                   raw_total=v['total'],raw_needles=v['needles'],raw_constant=v['constant'],
                   normalized_total=v['total']/r['summary']['A']) for r in results for key,v in r['cells'].items()])
    save_csv('partitions.csv',[dict(candidate_id=r['parameters']['candidate_id'],**p) for r in results for p in r['partitions']])
    save_csv('hierarchies.csv',[dict(candidate_id=r['parameters']['candidate_id'],**h) for r in results for h in r['hierarchies']])
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=rss if sys.platform=='darwin' else rss*1024
    save_json('runtime.json',dict(elapsed_seconds=time.perf_counter()-START,peak_rss_bytes=peak_bytes,
              candidate_count=10,partitions_per_candidate=15,hierarchies_per_candidate=39,
              first_order=64,recheck_order=128,single_process=True,thread_environment={k:os.environ[k] for k in THREAD_KEYS},
              within_time_target=time.perf_counter()-START<=120,within_memory_target=peak_bytes<=1024**3,
              max_recheck_cell_relative_difference=max(x['max_cell_relative_difference'] for x in checks)))
    log('complete',elapsed_seconds_total=time.perf_counter()-START,peak_rss_bytes=peak_bytes,
        max_cell_recheck_relative_difference=max(x['max_cell_relative_difference'] for x in checks))
    signal.alarm(0)

def run_compare():
    frozen=json.loads((OUT/'first_pass_freeze.json').read_text())
    assert digest(OUT/'first_pass.json')==frozen['sha256']
    current=json.loads((OUT/'first_pass.json').read_text())['candidates']
    old_paths={family:ROOT/'phase8_research_work/research/experiments/raw'/name for family,name in
               [('interior_v1','finite_degree.json'),('endpoint','endpoint_polynomial.json')]}
    old={family:json.loads(path.read_text()) for family,path in old_paths.items()}
    comparisons=[]
    for row in current:
        p=row['parameters'];s=row['summary'];family=p['family']
        choices=[v for v in old[family] if v['n']==p['n'] and
                 (family=='endpoint' or v.get('v1_sequence',False))]
        assert len(choices)==1
        prior=choices[0]; enumeration=prior['enumeration']
        differences=[]
        for labels,value in enumeration['cells'].items():
            name=cell_name(tuple(map(int,labels.split(','))))
            actual=row['cells'][name]['total']
            differences.append(abs(actual-value)/max(abs(value),1e-300))
        comparisons.append(dict(candidate_id=p['candidate_id'],source_file=str(old_paths[family].relative_to(ROOT)),
           source_sha256=digest(old_paths[family]),max_raw_cell_relative_difference=max(differences),
           new_rho=s['rho'],old_rho=prior['rho_det'],rho_relative_difference=abs(s['rho']/prior['rho_det']-1),
           old_opt2_raw=enumeration['opt2'],new_opt2_raw=s['opt2']*s['A'],
           old_opt3_raw=enumeration['opt3'],new_opt3_raw=s['opt3']*s['A'],
           old_saved_opt2_normalized=prior['opt2_normalized'],new_unit_amplitude_opt2=s['opt2'],
           old_saved_opt3_normalized=prior['opt3_normalized'],new_unit_amplitude_opt3=s['opt3'],
           old_saved_to_new_risk_factor=s['opt2']/prior['opt2_normalized'],
           note='Old normalization was a curvature upper bound; new losses have actual outcome-wise maximum amplitude exactly one.'))
    save_json('historical_comparison.json',dict(first_pass_sha256=frozen['sha256'],comparisons=comparisons))
    save_csv('historical_comparison.csv',comparisons)
    log('historical_comparison_complete',max_raw_cell_relative_difference=max(c['max_raw_cell_relative_difference'] for c in comparisons))

if __name__=='__main__':
    try:
        if len(sys.argv)!=2 or sys.argv[1] not in ('compute','compare'):
            raise ValueError('usage: compute_pilot.py {compute|compare}')
        run_compute() if sys.argv[1]=='compute' else run_compare()
    except Exception as exc:
        log('failure',type=type(exc).__name__,message=str(exc))
        raise
