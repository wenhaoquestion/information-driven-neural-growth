#!/usr/bin/env python3
"""Independent finite bounds; standard library and rational arithmetic only.

Run from any directory. Outputs remain beside this file. No old result is read.
See FIRST_PASS_THEORY.md for the analytic inequalities certified here.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import platform

ROOT = Path(__file__).resolve().parent
EPS0 = Q(1, 1000)
TERMS = 40


def bounds_ln2(terms=TERMS):
    lo = 2 * sum((Q(1, (2*j+1)*3**(2*j+1)) for j in range(terms)), Q())
    # log(2) = 2 atanh(1/3); first omitted term j=terms.
    remainder = Q(2, (2*terms+1)*3**(2*terms+1)) / (1-Q(1,9))
    return lo, lo+remainder


def decimal_out(x, places, upper=False):
    scale = 10**places
    scaled = x * scale
    k = scaled.numerator // scaled.denominator
    if upper and scaled.denominator != 1:
        k += 1
    sign = "-" if k < 0 else ""
    k = abs(k)
    return sign + str(k//scale) + "." + str(k%scale).zfill(places)


def frac(x):
    return f"{x.numerator}/{x.denominator}"


def interval(lo, hi, places=18):
    assert lo <= hi
    return {"lower_rational": frac(lo), "upper_rational": frac(hi),
            "lower_decimal_outward": decimal_out(lo, places),
            "upper_decimal_outward": decimal_out(hi, places, True)}


def main():
    l2, u2 = bounds_ln2()
    result = {"method": "proved analytic inequalities + exact Fraction arithmetic",
              "python": platform.python_version(), "log2_terms": TERMS,
              "log2": interval(l2, u2, 40), "threshold": frac(EPS0),
              "inputs": "Only the ten predeclared candidates; no stored costs read",
              "interior": [], "endpoint": []}
    for n in (2048, 4096, 8192):
        k = n.bit_length()-1
        assert n == 2**k
        L, U = k*l2, k*u2
        el, eu = 16*L/n, 16*U/n
        delta = 2/(n**5*L)  # dominates source 32/e exp(-sqrt(3)n e/4)
        assert 0 < el <= eu <= Q(1,16)
        jlo, jhi = 2+32*el**2-delta, 2+32*eu**2
        alo, ahi = 2*jlo, 2*jhi
        dlo, dhi = (el/4-8*delta)/ahi, (eu/4+8*delta)/alo
        opt2lo, opt2hi = (el/2+el**2-4*delta)/ahi, (eu/2+eu**2+4*delta)/alo
        opt3lo, opt3hi = (el**2/2-4*delta)/ahi, (eu**2/2+4*delta)/alo
        assert jlo > 0 and dlo > EPS0 and opt3lo > 0
        # Certifies the exact polynomial keeps the reference flat optimizers
        # and the additive Pareto alternative: all strict reference margins
        # are >= min(e/4, 1/4-e+e^2) in the relevant comparisons.
        ordering_margin_lo = min(el/4, Q(1,4)-eu+el**2)
        assert ordering_margin_lo > 16*delta
        result["interior"].append({"n": n, "epsilon": interval(el, eu),
          "tail_error_bound": frac(delta), "tail_error_decimal_upper": decimal_out(delta,30,True),
          "J": interval(jlo,jhi), "A_star_for_F_composed": interval(alo,ahi),
          "A_original": "J/(8 M_n); M_n cancels exactly from normalized loss",
          "OPT2": interval(opt2lo,opt2hi), "OPT3": interval(opt3lo,opt3hi),
          "Delta": interval(dlo,dhi), "Delta_above_threshold": dlo > EPS0,
          "normalized_upper_witness": "{0,1}|{2,3}; then {0}|{1}|{2,3}",
          "all_finite_conditions": True})
    for n in (128,256,512,1024,2048,4096,8192):
        k = n.bit_length()-1
        L,U = k*l2,k*u2
        dl,du = (4*L/n)**2,(4*U/n)**2
        assert 0 < dl <= du < Q(1,4)
        # For d <= 1/4, u(d)=(4-8d^2)d^3 is increasing:
        # u'(d)=4d^2(3-10d^2)>0.
        upper = (4-8*du**2)*du**3
        assert upper < EPS0
        result["endpoint"].append({"n": n,"d":interval(dl,du),
          "A":interval(1+dl**2/2,1+du**2/2),
          "Delta": interval(Q(0),upper),"Delta_below_threshold":upper < EPS0,
          "normalized_upper_witness":"{0,1}|{2,3}; then {0}|{1}|{2,3}",
          "upper_formula":"(4-8d^2)d^3", "all_finite_conditions":True})
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/"rational_certificates.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Independent rational certificate table", "",
           "Natural logarithms; canonical maximum outcome oscillation = 1.", "",
           "| family | n | parameter interval | certified Delta interval | threshold 0.001 |",
           "|---|---:|---|---|---|"]
    for family,key in (("interior","epsilon"),("endpoint","d")):
        for row in result[family]:
            p=row[key]; d=row["Delta"]
            lines.append(f"| {family} | {row['n']} | [{p['lower_decimal_outward']}, {p['upper_decimal_outward']}] | [{d['lower_decimal_outward']}, {d['upper_decimal_outward']}] | {'above' if family=='interior' else 'below'} |")
    (ROOT/"CERTIFIED_TABLE.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    print("\nAll exact-rational assertions passed; no floating-point quadrature used.")
    print("Inputs: 3 fixed interior + 7 original-coordinate endpoint candidates.")
    print("Script SHA-256:",result["script_sha256"])


if __name__ == "__main__":
    main()
