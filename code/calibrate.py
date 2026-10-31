
"""Simulated-method-of-moments calibration of the two structural families (SR, SR2) to the
published Exp. 1 aggregate outcomes of the equal, mixed and proportional baselines
(Koster et al. 2025; means digitised from Fig. 2A, sustained shares from the main text).
The interpolating rule (k = 22) is NOT used: it is the held-out prediction target."""
import json, sys, time
import numpy as np
from scipy.optimize import differential_evolution
import sim, rules
TARGET = {"equal": dict(spr=2.2, gini=0.15, sust=0.05, all4=0.05),
          "mixed": dict(spr=4.5, gini=0.09, sust=0.30, all4=0.30),
          "prop":  dict(spr=6.0, gini=0.36, sust=0.60, all4=0.00)}
SCALE = dict(spr=0.5, gini=0.03, sust=0.08, all4=0.08)
MECH = {"equal": rules.equal, "mixed": rules.mixed, "prop": rules.proportional}
FAMS = {
 "SR":  (sim.SR,  ["mu_a","sd_a","pi_F","b_rel","b_soc","b_pool","rho","sd_eps"],
         [(-1,3),(0.05,2.5),(0,0.5),(-1,3),(0,6),(-2,3),(0,0.95),(0.01,0.3)]),
 "SR2": (sim.SR2, ["lam","sd_lam","kappa","pi_F","pi_U","eta","sd_eps","phi0"],
         [(0,1.3),(0,0.6),(-0.3,0.6),(0,0.5),(0,0.5),(-2,3),(0.01,0.3),(0.2,1.0)]),
}
def moments(cls, names, x, G=400, seed=11):
    fam = cls(**dict(zip(names, x)))
    return {m: sim.summarize(sim.play(ru, fam, G=G, seed=seed)[0]) for m, ru in MECH.items()}
def loss_fn(cls, names):
    def f(x):
        mo = moments(cls, names, x)
        return sum(((mo[m][k] - TARGET[m][k]) / SCALE[k]) ** 2 for m in TARGET for k in TARGET[m])
    return f
if __name__ == "__main__":
    fam_name, seed = sys.argv[1], int(sys.argv[2])
    cls, names, bounds = FAMS[fam_name]
    t0 = time.time()
    res = differential_evolution(loss_fn(cls, names), bounds, maxiter=150, popsize=15, seed=seed,
                                 tol=1e-7, polish=False, workers=1)
    x = res.x
    mo = moments(cls, names, x, G=4000, seed=99)
    heldout = {nm: sim.summarize(sim.play(ru, cls(**dict(zip(names, x))), G=4000, seed=7)[0])
               for nm, ru in [("k22", rules.heuristic22), ("prop", rules.proportional)]}
    out = dict(family=fam_name, seed=seed, loss=float(res.fun), params=dict(zip(names, map(float, x))),
               fit=mo, heldout=heldout, secs=time.time() - t0, nfev=int(res.nfev))
    json.dump(out, open(f"calib_{fam_name}_{seed}.json", "w"), indent=1)
    print(fam_name, seed, res.fun)
