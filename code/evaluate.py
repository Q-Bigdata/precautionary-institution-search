
"""Screening and full evaluation of allocation programs against behavioural families."""
import json, sys, time, traceback
import numpy as np
import sim, progs

def load_params(path="calib_params.json"):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None

def fams(params, r=0.4, fr_share=None):
    return dict(SR=sim.SR(**params["SR"], fr_share=fr_share), SR2=sim.SR2(**params["SR2"], fr_share=fr_share))


def telemetry(o, P=4, blocks=5, T=40):
    """Block-averaged digest of play: pool, return fraction among included agents, included agents."""
    R = np.stack([x[0] for x in o["traj"]], 1)            # (G, T)
    E = np.stack([x[1] for x in o["traj"]], 1); C = np.stack([x[2] for x in o["traj"]], 1)
    use_ = E >= 1.0
    phi = np.where(use_, C / np.maximum(E, 1e-9), np.nan)
    b = np.array_split(np.arange(R.shape[1]), blocks)
    with np.errstate(all="ignore"):
        pool = [float(np.mean(R[:, ix])) for ix in b]
        ret = [float(np.nan_to_num(np.nanmean(phi[:, ix]))) for ix in b]
        inc = [float(np.mean(use_[:, ix].sum(-1))) for ix in b]
    return dict(pool=[round(x) for x in pool], ret=[round(x, 2) for x in ret], inc=[round(x, 1) for x in inc])

SCREEN = dict(SR_G=512, AL_G=48, AL_EP=150, AL_LAST=30, EV_N=400, EV_GENS=300, EV_TAIL=100)

def screen(rule, params, r=0.4, seed=0, cfg=SCREEN, families=("SR", "AL", "EV")):
    T, P, R0 = 40, 4, 200.0
    out = {}
    if "SR" in families:
        o, _ = sim.play(rule, sim.SR(**params["SR"]), G=cfg["SR_G"], r=r, seed=seed + 11, return_traj=True)
        out["SR"] = sim.summarize(o, r=r); out["SR"]["tele"] = telemetry(o)
    if "SR2" in families:
        o, _ = sim.play(rule, sim.SR2(**params["SR2"]), G=cfg["SR_G"], r=r, seed=seed + 13)
        out["SR2"] = sim.summarize(o, r=r)
    if "AL" in families:
        fam = sim.QL(alpha=0.1, gamma=0.95, eps0=0.2, eps1=0.01, q0=150.0, episodes=cfg["AL_EP"], eval_last=cfg["AL_LAST"])
        agg = sim.eval_QL(rule, G=cfg["AL_G"], r=r, seed=seed + 17, fam=fam, keep_traj=True)
        out["AL"] = sim.summarize(agg, r=r); out["AL"]["tele"] = telemetry(agg)
    if "EV" in families:
        ev = sim.eval_EV(rule, N=cfg["EV_N"], gens=cfg["EV_GENS"], tail=cfg["EV_TAIL"], r=r, seed=seed + 19, keep_traj=True)
        s = sim.summarize(ev, r=r); s["phi_bar"] = ev["phi_bar"]; s["tele"] = telemetry(ev); out["EV"] = s
    return out

def fitness(card, families=("SR", "AL", "EV")):
    return min(card[f]["eff"] for f in families)

def run_batch(inp, outp):
    job = json.load(open(inp))
    params = job["params"]; r = job.get("r", 0.4); families = tuple(job.get("families", ["SR", "AL", "EV"]))
    cfg = dict(SCREEN, **job.get("cfg", {}))
    res = []
    for item in job["items"]:
        t0 = time.time()
        rec = dict(id=item["id"])
        try:
            rule = progs.compile_item(dict(item, wrap=job.get("wrap") or item.get("wrap")))
            progs.smoke(rule)
            card = screen(rule, params, r=r, seed=job.get("seed", 0), cfg=cfg, families=families)
            rec.update(ok=True, card=card, fit=fitness(card, families),
                       length=progs.description_length(item["code"]), lines=progs.logical_lines(item["code"]))
        except Exception as ex:
            rec.update(ok=False, error=f"{type(ex).__name__}: {str(ex)[:200]}")
        rec["secs"] = time.time() - t0
        res.append(rec)
        json.dump(res, open(outp, "w"))
    return res

if __name__ == "__main__":
    run_batch(sys.argv[1], sys.argv[2])
