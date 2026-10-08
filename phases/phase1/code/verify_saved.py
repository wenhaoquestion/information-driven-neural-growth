"""Inspect archived finite comparisons without RNG, recomputation, or writes."""
from pathlib import Path
import csv
import json
from decimal import Decimal, localcontext

ROOT = Path(__file__).resolve().parents[1]


def close(a, b):
    a, b = Decimal(str(a)), Decimal(str(b))
    assert abs(a-b) <= Decimal('1e-54') * max(abs(a), abs(b)), (a, b)


def main():
    data = json.loads((ROOT/'results/quartet_all_partitions.json').read_text())
    scaling = list(csv.DictReader((ROOT/'results/quartet_scaling.csv').open()))
    sensitivity = list(csv.DictReader((ROOT/'results/quartet_sensitivity.csv').open()))
    assert len(data) == len(scaling) == 15 and len(sensitivity) == 10
    assert [r['row']['L'] for r in data] == [int(r['L']) for r in scaling]
    with localcontext() as ctx:
        ctx.prec = 100
        for record in data:
            partitions, chains = record['partitions'], record['chains']
            assert len(partitions) == 15 and len(chains) == 18
            assert sum(p['states'] == 2 for p in partitions) == 7
            assert sum(p['states'] == 3 for p in partitions) == 6
            for k in [2, 3]:
                close(min(Decimal(p['cost']) for p in partitions if p['states'] == k), record['opt'+str(k)])
            close(min(Decimal(c['ratio']) for c in chains), record['price'])
            close(min(Decimal(c['total_ratio']) for c in chains), record['total_price'])
            for c in chains:
                close(max(Decimal(c['total_level2']), Decimal(c['total_level3'])), c['total_ratio'])
            assert Decimal(record['randomized_total_price']) <= Decimal(record['stochastic_upper'])
            assert abs(record['row']['brier_price']-1) < 1e-12
    verification = json.loads((ROOT/'results/verification.json').read_text())
    assert len(verification['precision_checks']) == 4
    for check in verification['precision_checks']:
        assert check['stored_digits'] == 60
        assert all(Decimal(v) == 0 for v in check['relative_differences_of_stored_values'].values())
    print(json.dumps(dict(status='passed saved evidence checks', scaling_cases=15,
        sensitivity_cases=10, partition_records=225, chain_records=270,
        historical_precision_checks=4,
        scope='Consistency of stored finite comparisons; no new population evaluation, RNG, proof certification, or writes.'), indent=2))


if __name__ == '__main__':
    main()
