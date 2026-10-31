"""Build the human-data inputs used by the study from the public per-round file of Koster et al. (2025).
Usage: python build_human_data.py path/to/sustainable_behavior.csv
Writes human_games.csv (per-game outcomes), human_moments.json (Exp. 1 moments with bootstrap s.e.; Exp. 2/3 held-out),
emp_train_nointerp.npz and emp_train_all.npz (case-based family training sets)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import sim
use = ["mech_name_by_player", "round_id", "episode_id", "launch_id", "mechanism_observation.pool"] + [f"offer_{i}" for i in range(4)] + [f"player_action_{i}" for i in range(4)]
df = pd.read_csv(sys.argv[1], usecols=use, low_memory=False).rename(columns={"mech_name_by_player": "cond", "mechanism_observation.pool": "pool"})
key = ["cond", "launch_id", "episode_id"]
df = df.sort_values(key + ["round_id"]).reset_index(drop=True)
O = df[[f"offer_{i}" for i in range(4)]].to_numpy(float); A = np.minimum(np.maximum(df[[f"player_action_{i}" for i in range(4)]].to_numpy(float), 0), O)
rows = []
for k, g in df.groupby(key, sort=False):
    ix = g.index.to_numpy(); T = len(ix); cum = (O[ix] - A[ix]).sum(0)
    nxt = min(200, g["pool"].iloc[-1] - O[ix[-1]].sum() + 1.4 * A[ix[-1]].sum()); sus = nxt >= 1.0
    nact = (O[ix] >= 1).sum(1)
    rows.append(dict(cond=k[0], launch=k[1], episode=k[2], T=T, spr=cum.sum() / (T * 4), gini=float(sim.gini_rows(cum[None])[0]),
                     active=nact.mean() / 4, sust=bool(sus), all4=bool(sus and nact[-1] == 4)))
games = pd.DataFrame(rows); games.to_csv("human_games.csv", index=False)
def arrays_for(cond):
    gs = [g.index.to_numpy() for _, g in df[df.cond == cond].groupby(key, sort=False)]
    T = min(len(i) for i in gs)
    return np.stack([O[i[:T]] for i in gs]), np.stack([A[i[:T]] for i in gs])
HUM = {}; rng = np.random.default_rng(0)
for m, cond in [("equal", "Equal Baseline Exp 1"), ("mixed", "Mixed Baseline Exp 1"), ("prop", "Proportional Baseline Exp 1"),
                ("interp2", "Interpolating Baseline Exp 2"), ("interp3", "Interpolating Baseline Exp 3"), ("prop2", "Proportional Baseline Exp 2")]:
    E, C = arrays_for(cond); gm = games[games.cond == cond]
    base = dict(spr=gm.spr.mean(), gini=gm.gini.mean(), sust=gm.sust.mean(), all4=gm.all4.mean(), active=gm.active.mean()); base.update(sim.beh_moments(E, C))
    boots = []
    for b in range(300):
        ix = rng.integers(0, len(gm), len(gm)); gb = gm.iloc[ix]
        d = dict(spr=gb.spr.mean(), gini=gb.gini.mean(), sust=gb.sust.mean(), all4=gb.all4.mean(), active=gb.active.mean()); d.update(sim.beh_moments(E[ix], C[ix])); boots.append(d)
    HUM[m] = dict(mean={k: float(v) for k, v in base.items()}, se={k: float(v) for k, v in pd.DataFrame(boots).std().items()}, n=len(gm))
json.dump(HUM, open("human_moments.json", "w"), indent=1)
def build(filt):
    X, Y = [], []
    for k, g in df[df.cond.map(filt)].groupby(key, sort=False):
        ix = g.index.to_numpy(); E, C, R = O[ix], A[ix], g["pool"].to_numpy(float)
        for t in range(len(R)):
            Ep = E[t-1:t] if t > 0 else np.zeros((1, 4)); Cp = C[t-1:t] if t > 0 else np.zeros((1, 4))
            F = sim.emp_features(E[t:t+1], Cp, Ep, R[t:t+1], t)[0]; F[:, 6] = t / 40.0  # training used the unclipped round index
            u = E[t] >= 1
            X.append(F[u]); Y.append(C[t][u] / E[t][u])
    return np.concatenate(X), np.clip(np.concatenate(Y), 0, 1)
human = lambda c: " Exp " in c
X, Y = build(lambda c: human(c) and "Interpolating" not in c); np.savez_compressed("emp_train_nointerp.npz", X=X, Y=Y)
X, Y = build(human); np.savez_compressed("emp_train_all.npz", X=X, Y=Y)
print(games.groupby("cond")[["spr", "gini", "sust", "all4", "active"]].mean().round(3))
