
"""Simulated-method-of-moments calibration of the structural families SR and SR2 to the
Exp. 1 human games (equal, mixed, proportional; 40 games each) of Koster et al. (2025).
27 moments (5 outcome + 4 behavioural per mechanism), diagonal weights = 1/bootstrap SE^2
(SE floored). The interpolating rule and the Exp. 2/3 games are held out for validation."""
import json, sys, time
import numpy as np
from scipy.optimize import differential_evolution
import sim, rules
HUM = json.load(open("human_moments.json"))
KEYS_ALL = ["spr", "gini", "sust", "all4", "active", "phi_early", "phi_late", "phi_low", "phi_ac"]
import os
KEYS = KEYS_ALL if os.environ.get("MOMENTS", "all") == "all" else ["spr", "gini", "sust", "all4", "active"]
FLOOR = dict(spr=0.2, gini=0.01, sust=0.04, all4=0.04, active=0.03, phi_early=0.015, phi_late=0.01, phi_low=0.01, phi_ac=0.03)
MECH = {"equal": rules.equal, "mixed": rules.mixed, "prop": rules.proportional}
FAMS = {
 "SR":  (sim.SR,  ["mu_a","sd_a","pi_F","b_rel","b_soc","b_pool","rho","sd_eps"],
         [(-1,3),(0.05,2.5),(0,0.5),(-1,3),(0,6),(-2,3),(0,0.97),(0.01,0.3)]),
 "SR2": (sim.SR2, ["lam","sd_lam","kappa","pi_F","pi_U","eta","sd_eps","phi0"],
         [(0,1.3),(0,0.6),(-0.3,0.6),(0,0.5),(0,0.5),(-2,6),(0.01,0.3),(0.2,1.0)]),
}
def moments(cls, names, x, G=400, seed=11):
    fam = cls(**dict(zip(names, x)))
    return {m: sim.full_moments(ru, fam, G=G, seed=seed) for m, ru in MECH.items()}
def loss_fn(cls, names):
    def f(x):
        mo = moments(cls, names, x)
        return float(sum(((mo[m][k] - HUM[m]["mean"][k]) / max(HUM[m]["se"][k], FLOOR[k])) ** 2 for m in MECH for k in KEYS))
    return f
if __name__ == "__main__":
    fam_name, seed = sys.argv[1], int(sys.argv[2])
    cls, names, bounds = FAMS[fam_name]
    t0 = time.time()
    res = differential_evolution(loss_fn(cls, names), bounds, maxiter=int(os.environ.get("MAXITER", "150")), popsize=15, seed=seed, tol=1e-7, polish=False)
    x = res.x
    fam = cls(**dict(zip(names, x)))
    fit = {m: sim.full_moments(ru, fam, G=4000, seed=99) for m, ru in MECH.items()}
    held = {nm: sim.full_moments(ru, fam, G=4000, seed=7) for nm, ru in [("interp", rules.heuristic22), ("prop", rules.proportional)]}
    out = dict(family=fam_name, seed=seed, loss=float(res.fun), params=dict(zip(names, map(float, x))), fit=fit,
               heldout=held, secs=time.time() - t0, nfev=int(res.nfev))
    json.dump(out, open(f"calib{os.environ.get('TAG', '2')}_{fam_name}_{seed}.json", "w"), indent=1)
