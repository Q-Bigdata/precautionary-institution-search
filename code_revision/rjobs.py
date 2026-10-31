"""Targeted re-evaluations for the revision (Idea 5).
Usage: python code_revision/rjobs.py tasks.json output_prefix shard nshards
Each task: {tid, rule: {...}, evals: [{name, fam, ...}]}.  Rules:
  {type: prog, code, wrap, normalize}  - a generated program (normalize: whole pool split in proportion to its offers)
  {type: es, theta, noret}             - DeepSets policy (noret: retention slot removed, offers renormalised)
  {type: param, param}                 - parametric reference rule from param_grid
Evaluations (fam): SR, SR2 (G, seeds, r, P, T, shock, fr_share, params), AL (seeds, G, episodes, eval_last, gamma, q0,
  r, P, curve, frozen), EV (seeds, N, gens, tail, zealot, exec_sd, K, r, P, traj), EMP (path, k, G, seeds, r, P, T, fr_share),
  SCREEN (seed, cfg), FR (G, seed)."""
import json, sys, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import sim, progs, evaluate
P0 = json.load(open(os.path.join(HERE, "..", "code", "calib_params.json")))

def make_rule(spec):
    t = spec["type"]
    if t == "prog":
        rule = progs.compile_rule(spec["code"])
        if spec.get("wrap") == "shares" or spec.get("normalize"):
            rule = progs.shares_wrapper(rule)
        return rule
    if t == "es":
        _c = os.getcwd(); os.chdir(os.path.join(HERE, "..", "code")); import es_blackbox; os.chdir(_c)
        rule = es_blackbox.make_rule(np.array(spec["theta"], float))
        return progs.shares_wrapper(rule) if spec.get("noret") else rule
    if t == "param":
        _c = os.getcwd(); os.chdir(os.path.join(HERE, "..", "code")); import param_grid; os.chdir(_c)
        return param_grid.rule_of(tuple(spec["param"]))
    raise ValueError(t)

_EMP = {}
def emp(path, k=25, fr=None):
    key = (path, k, fr)
    if key not in _EMP:
        _EMP[key] = sim.EMP(os.path.join(HERE, "..", "code", path), k=k, fr_share=fr)
    return _EMP[key]

def agg_summary(outs, T=40, r=0.4, P=4, R0=200.0):
    agg = {k: np.concatenate([o[k] for o in outs]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    s = sim.summarize(agg, T=T, r=r, P=P, R0=R0)
    s["raw_surplus"] = float(agg["surplus"].mean())
    s["per_seed_eff"] = [float(o["surplus"].mean() / sim.ceiling(T, R0, r)) for o in outs]
    return s

def al_run(rule, seed, G=64, episodes=200, eval_last=40, gamma=0.95, q0=150.0, alpha=0.1, eps0=0.2, eps1=0.01,
           r=0.4, P=4, T=40, curve=False, frozen=0, nact=11):
    fam = sim.QL(alpha=alpha, gamma=gamma, eps0=eps0, eps1=eps1, q0=q0, episodes=episodes, eval_last=eval_last)
    if nact != 11:
        fam.ACTS = np.linspace(0, 1, nact)
    rng = np.random.default_rng(seed)
    st = fam.init(G, P, rng)
    res, cur = [], []
    ceil = sim.ceiling(T, 200.0, r)
    for ep in range(episodes):
        st["eps"] = fam.eps0 + (fam.eps1 - fam.eps0) * min(1.0, ep / (0.7 * episodes))
        o, st = sim.play(rule, fam, G=G, T=T, P=P, r=r, seed=seed * 1000 + ep, st=st)
        if ep >= episodes - eval_last:
            res.append(o)
        if curve:
            cur.append(float(o["surplus"].mean() / ceil))
    out = dict(online=res)
    if frozen:
        fam.alpha = 0.0; st["eps"] = 0.0
        fr = [sim.play(rule, fam, G=G, T=T, P=P, r=r, seed=seed * 1000 + 50000 + j, st=st)[0] for j in range(frozen)]
        out["frozen"] = fr
    if curve:
        out["curve"] = cur
    return out

def ev_run(rule, seed, N=600, gens=400, tail=150, zealot=0.2, exec_sd=0.05, K=5.0, mu=0.01, r=0.4, P=4, T=40, traj=False):
    if not traj:
        return sim.eval_EV(rule, N=N, gens=gens, tail=tail, zealot=zealot, exec_sd=exec_sd, K=K, mu=mu, r=r, P=P, T=T, seed=seed), None
    # trajectory version (same dynamics as sim.eval_EV, records per-generation summaries)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 1, N)
    nz = int(round(zealot * N)); fixed = np.zeros(N, bool); fixed[:nz] = True; phi[fixed] = 0.0
    class _Const:
        def init(s, G, P_, rng_): return dict(rng=rng_)
        def act(s, st_, e, R, R0_):
            ph = st_["phi"] if exec_sd == 0 else np.clip(st_["phi"] + st_["rng"].normal(0, exec_sd, e.shape), 0, 1)
            return np.where(st_["fixed"], 0.0, ph)
        def observe(s, *a, **k): pass
    fam = _Const(); keep = []; tr = []
    ceil = sim.ceiling(T, 200.0, r)
    for g in range(gens):
        perm = rng.permutation(N)
        st = dict(rng=rng, phi=phi[perm].reshape(-1, P), fixed=fixed[perm].reshape(-1, P))
        o, _ = sim.play(rule, fam, G=N // P, T=T, P=P, r=r, seed=int(rng.integers(1 << 30)), st=st)
        pay = np.empty(N); pay[perm] = o["cum"].ravel()
        if g >= gens - tail:
            keep.append(o)
        if g % 5 == 0 or g == gens - 1:
            tr.append(dict(g=g, phi=float(phi[~fixed].mean()), phi_q=[float(x) for x in np.quantile(phi[~fixed], [.1, .5, .9])],
                           eff=float(o["surplus"].mean() / ceil), sust=float(o["sustained"].mean())))
        model = rng.integers(0, N, N)
        prob = 1.0 / (1.0 + np.exp(-(pay[model] - pay) / K))
        adopt = rng.random(N) < prob
        phi = np.where(adopt, phi[model], phi)
        phi = np.clip(phi + rng.normal(0, mu, N), 0, 1)
        phi[fixed] = 0.0
    agg = {k: np.concatenate([x[k] for x in keep]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    return agg, tr

def fr_test(rule, G=2048, seed=5):
    """Same-seat comparison under common random numbers: seat 0 as a committed free rider vs. as an ordinary SR player."""
    fam = sim.SR(**P0["SR"])
    out = {}
    for mode in ("fr", "base"):
        rng = np.random.default_rng(seed)
        st = fam.init(G, 4, rng); st["committed"][:] = False
        if mode == "fr":
            st["committed"][:, 0] = True
        o, _ = sim.play(rule, fam, G=G, seed=seed, st=st)
        out[mode] = dict(seat0=float(o["cum"][:, 0].mean()), others=float(o["cum"][:, 1:].mean()),
                         eff=float(o["surplus"].mean() / sim.ceiling()), sust=float(o["sustained"].mean()))
    out["fr_adv"] = out["fr"]["seat0"] / max(out["fr"]["others"], 1e-9)
    out["deviation_gain"] = out["fr"]["seat0"] - out["base"]["seat0"]
    return out

def run_eval(rule, ev):
    f = ev["fam"]; kw = {k: ev[k] for k in ("r", "P", "T") if k in ev}
    T = ev.get("T", 40); r = ev.get("r", 0.4); P = ev.get("P", 4)
    if f in ("SR", "SR2"):
        params = ev.get("params") or P0[f]
        fam = (sim.SR if f == "SR" else sim.SR2)(**params, fr_share=ev.get("fr_share"))
        outs = [sim.play(rule, fam, G=ev.get("G", 1024), seed=s, shock=tuple(ev["shock"]) if ev.get("shock") else None, **kw)[0]
                for s in ev["seeds"]]
        return agg_summary(outs, T=T, r=r, P=P)
    if f == "EMP":
        fam = emp(ev.get("path", "emp_train_all.npz"), ev.get("k", 25), ev.get("fr_share"))
        outs = [sim.play(rule, fam, G=ev.get("G", 512), seed=s, shock=tuple(ev["shock"]) if ev.get("shock") else None, **kw)[0]
                for s in ev["seeds"]]
        return agg_summary(outs, T=T, r=r, P=P)
    if f == "AL":
        res = [al_run(rule, s, G=ev.get("G", 64), episodes=ev.get("episodes", 200), eval_last=ev.get("eval_last", 40),
                      gamma=ev.get("gamma", 0.95), q0=ev.get("q0", 150.0), r=r, P=P, T=T, curve=ev.get("curve", False),
                      frozen=ev.get("frozen", 0), nact=ev.get("nact", 11)) for s in ev["seeds"]]
        out = agg_summary([o for x in res for o in x["online"]], T=T, r=r, P=P)
        out["per_seed_eff"] = [float(np.mean([o["surplus"].mean() for o in x["online"]]) / sim.ceiling(T, 200.0, r)) for x in res]
        if ev.get("frozen"):
            fz = agg_summary([o for x in res for o in x["frozen"]], T=T, r=r, P=P)
            fz["per_seed_eff"] = [float(np.mean([o["surplus"].mean() for o in x["frozen"]]) / sim.ceiling(T, 200.0, r)) for x in res]
            out["frozen"] = fz
        if ev.get("curve"):
            out["curve"] = [x["curve"] for x in res]
        return out
    if f == "EV":
        res = [ev_run(rule, s, N=ev.get("N", 600), gens=ev.get("gens", 400), tail=ev.get("tail", 150),
                      zealot=ev.get("zealot", 0.2), exec_sd=ev.get("exec_sd", 0.05), K=ev.get("K", 5.0), r=r, P=P, T=T,
                      traj=ev.get("traj", False)) for s in ev["seeds"]]
        out = agg_summary([x[0] for x in res], T=T, r=r, P=P)
        if ev.get("traj"):
            out["traj"] = [x[1] for x in res]
        return out
    if f == "SCREEN":
        card = evaluate.screen(rule, P0, seed=ev.get("seed", 1000), cfg=dict(evaluate.SCREEN, **ev.get("cfg", {})))
        for k in card:
            card[k].pop("tele", None)
        card["fit"] = evaluate.fitness(card)
        return card
    if f == "FR":
        return fr_test(rule, G=ev.get("G", 2048), seed=ev.get("seed", 5))
    raise ValueError(f)

def do_task(task):
    t0 = time.time()
    rec = dict(tid=task["tid"], res={})
    try:
        rule = make_rule(task["rule"])
        for ev in task["evals"]:
            rec["res"][ev["name"]] = run_eval(rule, ev)
        rec["ok"] = True
    except Exception as ex:
        import traceback
        rec.update(ok=False, error=f"{type(ex).__name__}: {str(ex)[:300]}", tb=traceback.format_exc()[-800:])
    rec["secs"] = time.time() - t0
    return rec

if __name__ == "__main__":
    tasks = json.load(open(sys.argv[1])); outp = sys.argv[2]; shard, nsh = int(sys.argv[3]), int(sys.argv[4])
    outp = f"{outp}_{shard}.jsonl"
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["tid"])
            except Exception: pass
    todo = [t for t in tasks[shard::nsh] if t["tid"] not in done]
    with open(outp, "a") as fh:
        for t in todo:
            rec = do_task(t)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
