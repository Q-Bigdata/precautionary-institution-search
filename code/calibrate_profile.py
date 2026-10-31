"""Profile re-calibration: same 15 outcome moments as the retained fits (calibrate2.py, MOMENTS=outcome),
with one bound widened or one parameter fixed. Usage: python calibrate_profile.py FAMILY SEED TAG '{"eta":[-2,15]}' """
import os, sys, json, time
os.environ.setdefault("MOMENTS", "outcome")
import numpy as np
from scipy.optimize import differential_evolution
import calibrate2 as c2, sim, rules
fam_name, seed, tag, ov = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
cls, names, bounds = c2.FAMS[fam_name]
bounds = [tuple(ov[n]) if n in ov else b for n, b in zip(names, bounds)]
fixed = {n: b[0] for n, b in zip(names, bounds) if b[0] == b[1]}
free = [n for n in names if n not in fixed]; fb = [b for n, b in zip(names, bounds) if n not in fixed]
base = c2.loss_fn(cls, names)
def full(z):
    d = dict(zip(free, z)); d.update(fixed); return [d[n] for n in names]
t0 = time.time()
res = differential_evolution(lambda z: base(full(z)), fb, maxiter=int(os.environ.get("MAXITER", "120")), popsize=15, seed=seed, tol=1e-7, polish=False)
x = full(res.x); fam = cls(**dict(zip(names, x)))
fit = {m: sim.full_moments(ru, fam, G=4000, seed=99) for m, ru in c2.MECH.items()}
held = {nm: sim.full_moments(ru, fam, G=4000, seed=7) for nm, ru in [("interp", rules.heuristic22), ("prop", rules.proportional)]}
json.dump(dict(family=fam_name, seed=seed, tag=tag, override=ov, loss=float(res.fun), params=dict(zip(names, map(float, x))), fit=fit,
               heldout=held, secs=time.time() - t0, nfev=int(res.nfev)), open(f"prof_{tag}.json", "w"), indent=1)
