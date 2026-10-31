
"""Parametric baselines under the pessimistic objective: the paper's power family w=(R/C)^k,
a collective step family w=1[R>=theta*C], and a logistic family w=sigmoid((R/C-theta)/tau)."""
import json, sys
import numpy as np
from multiprocessing import Pool
import sim, rules, evaluate
P0 = json.load(open("calib_params.json"))
def specs():
    out = []
    for k in np.exp(np.linspace(np.log(0.25), np.log(200), 36)):
        out.append(("power", float(k), None))
    for th in np.linspace(0.5, 1.0, 26):
        out.append(("step", float(th), None))
    for th in np.linspace(0.6, 1.0, 17):
        for tau in (0.005, 0.01, 0.02, 0.04, 0.08):
            out.append(("logistic", float(th), tau))
    return out
def rule_of(sp):
    fam, a, b = sp
    if fam == "power":
        return rules.mixture(lambda R, cap: (R / cap) ** a)
    if fam == "step":
        return rules.mixture(lambda R, cap: (R >= a * cap - 1e-9).astype(float))
    return rules.mixture(lambda R, cap: 1.0 / (1.0 + np.exp(-(R / cap - a) / b)))
def run(sp):
    card = evaluate.screen(rule_of(sp), P0, seed=0)
    card2 = evaluate.screen(rule_of(sp), P0, seed=1000)
    return dict(family=sp[0], a=sp[1], b=sp[2], fit=evaluate.fitness(card), fit_fresh=evaluate.fitness(card2), card=card, card_fresh=card2)
if __name__ == "__main__":
    with Pool(int(sys.argv[1])) as pl:
        res = pl.map(run, specs())
    json.dump(res, open("param_grid.json", "w"))
