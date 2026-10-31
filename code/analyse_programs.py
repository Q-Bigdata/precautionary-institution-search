"""Behavioural probes and proposal-level statistics for all generated programs (Fig. 3, Fig. 4b).
Usage: python analyse_programs.py all_programs.json   (list of {id, run, code, wrap})"""
import sys, json, re
import numpy as np, pandas as pd
import sim, progs, final_eval, rules
P = json.load(open("calib_params.json"))
items = json.load(open(sys.argv[1]))
CM = re.compile(r"(reserve|retain|retention|keep (?:a|some|part)|buffer|sustainab|throttle|release|budget|withh|save|cautio|safety)", re.I)
def comments(code):
    return " ".join(re.findall(r"#(.*)", code)) + " " + " ".join(re.findall(r'"""([\s\S]*?)"""', code))
bins = np.linspace(0, 200, 11)
def inplay_alloc(rule, G=128, seed=5):
    o, _ = sim.play(rule, sim.SR(**P["SR"]), G=G, seed=seed, return_traj=True)
    Rs = np.concatenate([x[0] for x in o["traj"][1:]]); Es = np.concatenate([x[1].sum(1) for x in o["traj"][1:]]); m = Rs >= 1
    fr = Es[m] / Rs[m]; b = np.digitize(Rs[m], bins) - 1
    return [float(fr[b == i].mean()) if (b == i).sum() > 20 else None for i in range(10)]
rows = []
for it in items:
    try:
        rule = progs.compile_item(it); pw = final_eval.probe_weights(rule); pf = final_eval.probe_forgiveness(rule)
    except Exception:
        continue
    al = np.array(pw["alloc"]); w = np.array(pw["w"])
    rows.append(dict(id=it["id"], run=it["run"], withheld=float(1 - al.min()), withholds=bool(1 - al.min() > 0.05), w50=w[24], w200=w[99],
                     resource_contingent=bool(w[99] - w[24] > 0.3), precautionary_wording=bool(CM.search(comments(it["code"]))),
                     half_capacity_target=bool(re.search(r"(cap\s*/\s*2|0\.5\s*\*\s*cap|half (?:the |of the )?(?:capacity|cap|reservoir|pool))", it["code"], re.I)),
                     exclusion_rounds=pf["excl_rounds"], inplay_offer_fraction=(inplay_alloc(rule) if it.get("inplay") else None)))
pd.DataFrame(rows).to_csv("program_probes.csv", index=False)
print(pd.DataFrame(rows).groupby("run")[["withholds", "precautionary_wording"]].mean().round(3))
