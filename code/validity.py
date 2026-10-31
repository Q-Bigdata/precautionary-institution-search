
"""Retrospective validity of each behavioural family against the human games (Koster et al. 2025).
Families are run on the four hand-designed mechanisms that were played by humans. SR and SR2 are
calibrated to Exp.1 (equal/mixed/proportional) only; EMP_ni is trained on all human games except the
interpolating-rule games. The interpolating rule (Exp.2, Exp.3) is therefore out of sample for all three."""
import json
import numpy as np
import sim, rules
P0 = json.load(open("calib_params.json"))
MECH = {"equal": rules.equal, "mixed": rules.mixed, "prop": rules.proportional, "interp": rules.heuristic22}
fams = {"SR": lambda: sim.SR(**P0["SR"]), "SR2": lambda: sim.SR2(**P0["SR2"]),
        "EMP_ni": lambda: sim.EMP("emp_train_nointerp.npz"), "EMP_all": lambda: sim.EMP("emp_train_all.npz")}
out = {}
for fn, mk in fams.items():
    fam = mk()
    for m, ru in MECH.items():
        out[f"{fn}|{m}"] = sim.full_moments(ru, fam, G=2000, seed=77)
    # AL and EV are theoretical stress tests; include their predictions for completeness
import evaluate
for m, ru in MECH.items():
    c = evaluate.screen(ru, P0, seed=500)
    for f in ("AL", "EV"):
        out[f"{f}|{m}"] = {k: v for k, v in c[f].items() if k != "tele"}
json.dump(out, open("validity.json", "w"), indent=1)
