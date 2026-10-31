
"""Full evaluation battery for allocation programs (run on the compute server).
Usage: python3 final_eval.py programs.json out_prefix shard nshards"""
import json, sys, time, os
import numpy as np
import sim, progs

P0 = json.load(open("calib_params.json"))
CONF = dict(SR_G=1024, SR_SEEDS=2, AL_SEEDS=3, AL_EP=200, AL_LAST=40, EV_N=1000, EV_GENS=600, EV_TAIL=200, EV_SEEDS=3,
            EV_AUX_GENS=400, EV_AUX_TAIL=150, EMP_G=512)
if os.environ.get("FINAL_CONF"):
    CONF.update(json.loads(os.environ["FINAL_CONF"]))
EMP_PATH = "emp_train_all.npz"
_EMP = {}
def emp(fr=None):
    k = fr or 0
    if k not in _EMP:
        _EMP[k] = sim.EMP(EMP_PATH, fr_share=fr)
    return _EMP[k]

def al(gamma=0.95, q0=150.0):
    return sim.QL(alpha=0.1, gamma=gamma, eps0=0.2, eps1=0.01, q0=q0, episodes=CONF["AL_EP"], eval_last=CONF["AL_LAST"])

def one(rule, fam, G, seeds, **kw):
    outs = [sim.play(rule, fam, G=G, seed=s, **kw)[0] for s in seeds]
    agg = {k: np.concatenate([o[k] for o in outs]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    T = kw.get("T", 40); P = kw.get("P", 4); r = kw.get("r", 0.4)
    return sim.summarize(agg, T=T, r=r, P=P)

def one_al(rule, seeds, G=64, **kw):
    fam_kw = {k: kw.pop(k) for k in list(kw) if k in ("gamma", "q0")}
    outs = [sim.eval_QL(rule, G=G, seed=s, fam=al(**fam_kw), **kw) for s in seeds]
    agg = {k: np.concatenate([o[k] for o in outs]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    return sim.summarize(agg, T=kw.get("T", 40), r=kw.get("r", 0.4), P=kw.get("P", 4))

def one_ev(rule, seeds, N=None, gens=None, tail=None, zealot=0.2, exec_sd=0.05, **kw):
    N = N or CONF["EV_N"]; gens = gens or CONF["EV_GENS"]; tail = tail or CONF["EV_TAIL"]
    res = [sim.eval_EV(rule, N=N, gens=gens, tail=tail, zealot=zealot, exec_sd=exec_sd, seed=s, **kw) for s in seeds]
    agg = {k: np.concatenate([o[k] for o in res]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    s = sim.summarize(agg, r=kw.get("r", 0.4), P=kw.get("P", 4)); s["phi_bar"] = float(np.mean([o["phi_bar"] for o in res]))
    return s

def free_rider_advantage(rule, G=2000, seed=5):
    """SR groups in which player 0 is a committed free rider: its surplus relative to the others' mean."""
    fam = sim.SR(**P0["SR"])
    rng = np.random.default_rng(seed)
    st = fam.init(G, 4, rng); st["committed"][:] = False; st["committed"][:, 0] = True
    o, _ = sim.play(rule, fam, G=G, seed=seed, st=st)
    fr = o["cum"][:, 0]; oth = o["cum"][:, 1:].mean(1)
    return dict(fr_adv=float(fr.mean() / max(oth.mean(), 1e-9)), fr_eff=float(o["surplus"].mean() / sim.ceiling()))

def battery(rule, level="full"):
    R = {}
    t0 = time.time()
    # --- search families, fresh seeds (winner's-curse correction)
    R["SR"] = one(rule, sim.SR(**P0["SR"]), CONF["SR_G"], [101, 102][:CONF["SR_SEEDS"]])
    R["AL"] = one_al(rule, [201, 202, 203][:CONF["AL_SEEDS"]])
    R["EV"] = one_ev(rule, [301, 302, 303][:CONF["EV_SEEDS"]])
    R["worst3"] = min(R[f]["eff"] for f in ("SR", "AL", "EV"))
    if level == "search":
        R["secs"] = time.time() - t0
        return R
    # --- held-out behavioural families
    R["SR2"] = one(rule, sim.SR2(**P0["SR2"]), 1024, [111, 112])
    R["EMP"] = one(rule, emp(), CONF["EMP_G"], [121])
    R["AL_g08"] = one_al(rule, [211, 212], gamma=0.8, q0=60.0)
    R["EV_clean"] = one_ev(rule, [311, 312], zealot=0.0, exec_sd=0.0, gens=CONF["EV_AUX_GENS"], tail=CONF["EV_AUX_TAIL"])
    R["SR_fr25"] = one(rule, sim.SR(**P0["SR"], fr_share=0.25), 1024, [131])
    R["EMP_fr25"] = one(rule, emp(0.25), CONF["EMP_G"], [141])
    R.update({f"FR_{k}": v for k, v in free_rider_advantage(rule).items()})
    # --- environment transfer (SR, EMP, EV)
    for r_ in (0.2, 0.6, 0.8):
        R[f"SR_r{r_}"] = one(rule, sim.SR(**P0["SR"]), 1024, [151], r=r_)
        R[f"EMP_r{r_}"] = one(rule, emp(), CONF["EMP_G"], [152], r=r_)
        R[f"EV_r{r_}"] = one_ev(rule, [153], r=r_, gens=CONF["EV_AUX_GENS"], tail=CONF["EV_AUX_TAIL"])
    for P_ in (3, 6, 8):
        R[f"SR_p{P_}"] = one(rule, sim.SR(**P0["SR"]), 1024, [161], P=P_)
        R[f"EMP_p{P_}"] = one(rule, emp(), CONF["EMP_G"], [162], P=P_)
    R["SR_shock"] = one(rule, sim.SR(**P0["SR"]), 1024, [171], shock=(15, 0.5))
    R["EMP_shock"] = one(rule, emp(), CONF["EMP_G"], [172], shock=(15, 0.5))
    R["SR_T80"] = one(rule, sim.SR(**P0["SR"]), 1024, [181], T=80)
    R["EMP_T80"] = one(rule, emp(), CONF["EMP_G"], [182], T=80)
    heldout_keys = ["SR2", "EMP", "AL_g08", "EV_clean"]
    R["worst_heldout"] = min(R[k]["eff"] for k in heldout_keys)
    transfer_keys = [k for k in R if isinstance(R[k], dict) and any(s in k for s in ("_r0", "_p", "_shock", "_T80", "_fr25"))]
    R["worst_transfer"] = min(R[k]["eff"] for k in transfer_keys)
    R["secs"] = time.time() - t0
    return R

# ------------------------------------------------------------------ behavioural probes
def probe_weights(rule, P=4, cap=200.0, g=0.4, warm=6):
    """Effective equality weight w_eff(R) in a stationary probe: each probe game holds the reservoir at R
    and the agents return a fixed 40/30/20/10 split of total returns R/(1+g) for `warm` rounds (so that
    programs with memory are past their initialisation); offers are read at the last round."""
    Rg = np.linspace(2, 200, 100)
    share = np.array([0.4, 0.3, 0.2, 0.1])[:P] if P == 4 else np.linspace(P, 1, P) / np.linspace(P, 1, P).sum()
    c = (Rg / (1 + g))[:, None] * share[None, :]
    e_prev = np.repeat((Rg / P)[:, None], P, 1); mem = {}
    for t in range(1, warm + 1):
        off = np.asarray(rule(Rg.copy(), e_prev.copy(), c.copy(), t, mem, cap, g), float)
        off = sim.sanitize(np.broadcast_to(off, (len(Rg), P)), Rg)
        e_prev = off
    tot = off.sum(1)
    frac_alloc = np.minimum(tot / Rg, 1.0)
    sh = off / np.maximum(tot[:, None], 1e-12)
    a = 1.0 / P - share
    w = ((sh - share) * a).sum(1) / (a * a).sum()
    return dict(R=Rg.tolist(), w=np.clip(w, -1.5, 1.5).tolist(), alloc=frac_alloc.tolist(), top_share=sh[:, 0].tolist())

def probe_forgiveness(rule, P=4, cap=200.0, g=0.4, T=30, t_def=5, phi=0.8):
    """Scripted game: all return phi; player 0 returns nothing in round t_def only (then phi again).
    Returns offers to the defector relative to the others over time."""
    R = np.array([cap]); e = np.full((1, P), cap / P); mem = {}
    rel = []; offs = []
    for t in range(T):
        if t > 0:
            e = np.asarray(rule(R.copy(), e_prev.copy(), c_prev.copy(), t, mem, cap, g), float)
            e = sim.sanitize(np.broadcast_to(e, (1, P)), R)
        ph = np.full((1, P), phi)
        if t == t_def:
            ph[0, 0] = 0.0
        c_ = np.where(e >= 1, ph * e, 0.0)
        offs.append(e[0].tolist())
        rel.append(float(e[0, 0] / max(e[0, 1:].mean(), 1e-9)))
        R = np.minimum(cap, R - e.sum(1) + (1 + g) * c_.sum(1))
        e_prev, c_prev = e, c_
        if R[0] < 1:
            break
    excl = [t for t in range(t_def + 1, len(offs)) if offs[t][0] < 1.0]
    return dict(rel=rel, offers=offs, excl_rounds=len(excl), readmitted=bool(len(offs) > t_def + 1 and offs[-1][0] >= 1.0))

if __name__ == "__main__":
    progs_in = json.load(open(sys.argv[1])); pre = sys.argv[2]; shard, nsh = int(sys.argv[3]), int(sys.argv[4])
    level = sys.argv[5] if len(sys.argv) > 5 else "full"
    items = progs_in[shard::nsh]
    out = []
    for it in items:
        rec = dict(id=it["id"])
        try:
            if "theta" in it:
                import es_blackbox
                rule = es_blackbox.make_rule(np.array(it["theta"], float))
            elif "param" in it:
                import param_grid
                rule = param_grid.rule_of(tuple(it["param"]))
            else:
                rule = progs.compile_item(it)
            rec["res"] = battery(rule, level=level)
            rec["probe_w"] = probe_weights(rule)
            rec["probe_f"] = probe_forgiveness(rule)
            rec["ok"] = True
        except Exception as ex:
            rec.update(ok=False, error=f"{type(ex).__name__}: {str(ex)[:300]}")
        out.append(rec)
        json.dump(out, open(f"{pre}_{shard}.json", "w"))
