"""Verify frozen files, complete run inventory and access-account conservation."""
from pathlib import Path
import json
import numpy as np
import recycling as r

def main():
    freeze=json.loads((r.BASE/'protocol_frozen.json').read_text());cfg=json.loads((r.BASE/'config.json').read_text())
    for name,h in freeze['files'].items():assert r.sha((r.BASE/name).read_bytes())==h,name
    index=json.loads((r.BASE/'results/confirmation/run_index.json').read_text())
    expected={(t,s) for t in cfg['tasks'] for s in cfg['seeds']}
    actual={(a['task'],a['seed']) for a in index['runs']};assert actual==expected and len(index['runs'])==len(expected)
    assert index['completed']==index['planned']==len(expected)
    checked=0;counts_checked=0
    for a in index['runs']:
        path=r.BASE/'results/confirmation'/a['file'];assert r.sha(path.read_bytes())==a['sha256']
        rr=json.loads(path.read_text());assert rr['runner_sha256']==freeze['files']['code/recycling.py']
        assert rr['config']==cfg and set(rr['methods'])==set(cfg['methods'])
        counts=np.load(r.BASE/rr['access_file'])
        for name,m in rr['methods'].items():
            expected_blocks=np.array(m['events'][0]['fit_accesses_by_arrival_block'])*0
            # Warmup uses only the initial pool; every parameter-example update
            # is one label access per parameter and there is no separate warmup read.
            pcount=m['initial_cost']['parameter_count'];assert m['initial_cost']['parameter_examples']%pcount==0
            expected_blocks[0]=m['initial_cost']['parameter_examples']//pcount
            for e in m['events']:expected_blocks+=np.array(e['fit_accesses_by_arrival_block'])
            assert expected_blocks.tolist()==m['pre_tail_access']['fit_accesses_by_arrival_block']
            cc=counts[name+'_pre_tail'];ac=r.Access(len(cc),cfg['ntrain'],cfg['naudit'])
            assert ac.blocks(cc)==expected_blocks.tolist()
            assert np.all(counts[name+'_all']>=cc)
            # Labels released at the final event cannot already be fitted during the online horizon.
            assert not np.any(cc[-cfg['naudit']:])
            counts_checked+=1
        checked+=1
    result=dict(status='passed',frozen_files_unchanged=len(freeze['files']),run_files_checked=checked,
        access_ledgers_conserved=counts_checked,no_missing_or_duplicate_task_seeds=True,
        final_audit_never_fitted_before_online_horizon_ends=True,
        confirmation_wall_seconds=index['wall_seconds'])
    (r.BASE/'results/integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
