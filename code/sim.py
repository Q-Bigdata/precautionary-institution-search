
"""Vectorised engine for the Koster et al. (2025) common-pool trust game and three
behavioural model families used to score allocation programs.

Game: p players, pool cap R0, growth r, T rounds. Round 0 allocation is equal (as in the
paper). Each round the rule offers e_{i,t} >= 0 with sum_i e_{i,t} <= R_t; any unallocated
remainder is retained in the pool. Players return c_{i,t} = phi_{i,t} e_{i,t}; offers < 1 cannot
be reciprocated (integer slider in the human interface). Pool update:
R_{t+1} = min(R0, R_t - sum_i e_{i,t} + (1+r) sum_i c_{i,t}); a pool below 1 is depleted for good.
"""
import numpy as np

R0_DEFAULT = 200.0

def ceiling(T=40, R0=200.0, r=0.4):
    return T * R0 * r / (1.0 + r)

def sanitize(offers, pool):
    o = np.nan_to_num(np.asarray(offers, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    o = np.maximum(o, 0.0)
    s = o.sum(1)
    scale = np.where(s > pool, pool / np.maximum(s, 1e-12), 1.0)
    return o * scale[:, None]

def gini_rows(x):
    x = np.sort(np.maximum(x, 0), 1); n = x.shape[1]
    s = x.sum(1); idx = np.arange(1, n + 1)
    g = (2 * (idx * x).sum(1) / (n * np.maximum(s, 1e-12))) - (n + 1) / n
    return np.where(s > 1e-9, g, 0.0)

# ---------------------------------------------------------------- behavioural families
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))

class SR:
    """Structural reciprocity mixture: persistent return fractions pulled towards a target
    that depends on the relative offer, on others' previous return fractions and on the pool."""
    name = "SR"
    def __init__(self, mu_a=1.0, sd_a=0.8, pi_F=0.15, b_rel=0.8, b_soc=2.0, b_pool=0.5,
                 rho=0.5, sd_eps=0.08, a_F=-2.5, fr_share=None):
        self.p = dict(mu_a=mu_a, sd_a=sd_a, pi_F=pi_F, b_rel=b_rel, b_soc=b_soc, b_pool=b_pool,
                      rho=rho, sd_eps=sd_eps, a_F=a_F)
        self.fr_share = fr_share   # if set, this share of players are committed free riders (phi=0)
    def init(self, G, P, rng):
        q = self.p
        a = rng.normal(q["mu_a"], q["sd_a"], (G, P))
        fr = rng.random((G, P)) < q["pi_F"]
        a = np.where(fr, q["a_F"], a)
        committed = np.zeros((G, P), bool)
        if self.fr_share:
            committed = rng.random((G, P)) < self.fr_share
        return dict(a=a, phi=_sig(a), soc=np.full((G, P), 0.7), committed=committed, rng=rng)
    def act(self, st, e, R, R0):
        q = self.p; P = e.shape[1]
        rel = e / np.maximum(R[:, None] / P, 1e-9)
        lr = np.log(np.clip(rel, 0.05, 20.0))
        tgt = _sig(st["a"] + q["b_rel"] * lr + q["b_soc"] * (st["soc"] - 0.7) + q["b_pool"] * (R[:, None] / R0 - 1.0))
        phi = q["rho"] * st["phi"] + (1 - q["rho"]) * tgt + st["rng"].normal(0, q["sd_eps"], e.shape)
        phi = np.clip(phi, 0.0, 1.0)
        phi = np.where(st["committed"], 0.0, phi)
        return phi
    def observe(self, st, e, c, phi, surplus, R_next, done):
        # others' mean return fraction among players who received a usable offer
        act = e >= 1.0
        frac = np.where(act, c / np.maximum(e, 1e-9), 0.0)
        tot = frac.sum(1, keepdims=True); n = act.sum(1, keepdims=True)
        others = np.where(n - act > 0, (tot - frac) / np.maximum(n - act, 1), st["soc"])
        st["soc"] = others
        st["phi"] = np.where(act, phi, st["phi"])   # excluded players keep their disposition

class QL:
    """Tabular Q-learning players that learn across repeated games with fixed partners.
    State = (relative-offer bin, pool bin); action = return fraction in {0, .1, ..., 1}."""
    name = "AL"
    ACTS = np.linspace(0, 1, 11)
    def __init__(self, alpha=0.15, gamma=0.9, eps0=0.3, eps1=0.02, q0=8.0, episodes=60, eval_last=15, fr_share=None):
        self.alpha, self.gamma, self.eps0, self.eps1, self.q0 = alpha, gamma, eps0, eps1, q0
        self.episodes, self.eval_last, self.fr_share = episodes, eval_last, fr_share
    @staticmethod
    def state(e, R, R0):
        P = e.shape[1]
        rel = e / np.maximum(R[:, None] / P, 1e-9)
        ob = np.where(e < 1.0, 0, np.where(rel < 0.5, 1, np.where(rel < 1.5, 2, 3)))
        f = R / R0
        pb = np.where(f < 0.25, 0, np.where(f < 0.5, 1, np.where(f < 0.9, 2, 3)))
        return ob * 4 + pb[:, None]
    def init(self, G, P, rng):
        Q = np.full((G, P, 16, len(self.ACTS)), float(self.q0))
        committed = np.zeros((G, P), bool)
        if self.fr_share:
            committed = rng.random((G, P)) < self.fr_share
        return dict(Q=Q, rng=rng, eps=self.eps0, committed=committed, s=None, a=None)
    def act(self, st, e, R, R0):
        s = self.state(e, R, R0)
        G, P = e.shape
        Qs = np.take_along_axis(st["Q"], s[:, :, None, None], 2)[:, :, 0, :]
        greedy = Qs.argmax(-1)
        rnd = st["rng"].integers(0, len(self.ACTS), (G, P))
        a = np.where(st["rng"].random((G, P)) < st["eps"], rnd, greedy)
        st["s"], st["a"] = s, a
        phi = self.ACTS[a]
        return np.where(st["committed"], 0.0, phi)
    def observe(self, st, e, c, phi, surplus, R_next, done, e_next=None):
        pass  # updated inside the engine because the TD target needs the next state

def _ql_update(fam, st, s_next, reward, terminal):
    Q = st["Q"]; s, a = st["s"], st["a"]
    G, P = s.shape
    gi, pi = np.meshgrid(np.arange(G), np.arange(P), indexing="ij")
    if s_next is None:
        target = reward
    else:
        target = reward + fam.gamma * np.where(terminal[:, None], 0.0, Q[gi, pi, s_next].max(-1))
    Q[gi, pi, s, a] += fam.alpha * (target - Q[gi, pi, s, a])



class SR2:
    """Conditional-cooperation matching model (held-out structural family): a player's return
    fraction matches a type-specific slope times others' previous return fraction plus an
    intercept, and responds to the change in its own offer (reciprocity to generosity)."""
    name = "SR2"
    def __init__(self, lam=0.8, sd_lam=0.3, kappa=0.1, pi_F=0.15, pi_U=0.1, eta=0.5, sd_eps=0.08, phi0=0.6,
                 fr_share=None):
        self.p = dict(lam=lam, sd_lam=sd_lam, kappa=kappa, pi_F=pi_F, pi_U=pi_U, eta=eta, sd_eps=sd_eps, phi0=phi0)
        self.fr_share = fr_share
    def init(self, G, P, rng):
        q = self.p
        lam = np.clip(rng.normal(q["lam"], q["sd_lam"], (G, P)), 0, 1.5)
        u = rng.random((G, P))
        typ = np.where(u < q["pi_F"], 0, np.where(u < q["pi_F"] + q["pi_U"], 2, 1))  # 0 free rider, 1 conditional, 2 unconditional
        committed = np.zeros((G, P), bool)
        if self.fr_share:
            committed = rng.random((G, P)) < self.fr_share
        return dict(lam=lam, typ=typ, soc=np.full((G, P), q["phi0"]), e_last=None, rng=rng, committed=committed)
    def act(self, st, e, R, R0):
        q = self.p; P = e.shape[1]
        d = np.zeros_like(e) if st["e_last"] is None else (e - st["e_last"]) / (R0 / P)
        cc = st["lam"] * st["soc"] + q["kappa"] + q["eta"] * d
        phi = np.where(st["typ"] == 0, 0.02, np.where(st["typ"] == 2, 0.72, cc))
        phi = np.clip(phi + st["rng"].normal(0, q["sd_eps"], e.shape), 0, 1)
        return np.where(st["committed"], 0.0, phi)
    def observe(self, st, e, c, phi, surplus, R_next, done):
        act = e >= 1.0
        frac = np.where(act, c / np.maximum(e, 1e-9), 0.0)
        tot = frac.sum(1, keepdims=True); n = act.sum(1, keepdims=True)
        st["soc"] = np.where(n - act > 0, (tot - frac) / np.maximum(n - act, 1), st["soc"])
        st["e_last"] = np.where(act, e, st["e_last"] if st["e_last"] is not None else e)

# ---------------------------------------------------------------- engine
def play(rule, fam, G=256, T=40, P=4, R0=200.0, r=0.4, seed=0, st=None, shock=None, cap_het=None,
         return_traj=False):
    """Play G games of rule against family fam. shock: (round, fraction lost) applied to the pool.
    Returns dict of per-game metrics (and state for learners)."""
    rng = np.random.default_rng(seed)
    if st is None:
        st = fam.init(G, P, rng)
    R = np.full(G, float(R0)); alive = np.ones(G, bool)
    e_prev = np.zeros((G, P)); c_prev = np.zeros((G, P))
    mem = {}
    cum = np.zeros((G, P)); active = np.zeros(G); last_active = np.zeros(G)
    traj = [] if return_traj else None
    is_ql = isinstance(fam, QL)
    for t in range(T):
        if shock is not None and t == shock[0]:
            R = R * (1 - shock[1])
            alive &= R >= 1.0
        if t == 0:
            e = np.repeat((R / P)[:, None], P, 1)
        else:
            try:
                e = rule(R.copy(), e_prev.copy(), c_prev.copy(), t, mem, R0, r)
            except Exception:
                e = np.zeros((G, P))
            e = sanitize(np.broadcast_to(e, (G, P)), R)
        e = np.where(alive[:, None], e, 0.0)
        if is_ql and t > 0:
            s_next = QL.state(e, R, R0)
            _ql_update(fam, st, s_next, reward_prev, ~alive)
        phi = fam.act(st, e, R, R0)
        usable = e >= 1.0
        c = np.where(usable, phi * e, 0.0)
        if cap_het is not None:
            c = np.minimum(c, e * cap_het)
        s_ = e - c
        cum += s_
        active += usable.sum(1)
        last_active = usable.sum(1)
        R_next = np.minimum(R0, R - e.sum(1) + (1 + r) * c.sum(1))
        if return_traj:
            traj.append((R.copy(), e.copy(), c.copy()))
        alive = alive & (R_next >= 1.0)
        R = np.where(alive, R_next, 0.0)
        if is_ql:
            reward_prev = s_
        else:
            fam.observe(st, e, c, phi, s_, R, ~alive)
        e_prev, c_prev = e, c
    if is_ql:
        _ql_update(fam, st, None, reward_prev, np.ones(G, bool))
    out = dict(surplus=cum.sum(1), gini=gini_rows(cum), active=active / T,
               sustained=alive, all_active=alive & (last_active == P), cum=cum)
    if return_traj:
        out["traj"] = traj
    return out, st

def summarize(o, T=40, R0=200.0, r=0.4, P=4):
    ceil = ceiling(T, R0, r)
    return dict(eff=float(o["surplus"].mean() / ceil), spr=float(o["surplus"].mean() / (T * P)),
                gini=float(o["gini"].mean()), active=float(o["active"].mean() / P),
                sust=float(o["sustained"].mean()), all4=float(o["all_active"].mean()))

def eval_QL(rule, G=64, T=40, P=4, R0=200.0, r=0.4, seed=0, fam=None, keep_traj=False, **kw):
    fam = fam or QL(**kw)
    rng = np.random.default_rng(seed)
    st = fam.init(G, P, rng)
    res = []
    for ep in range(fam.episodes):
        st["eps"] = fam.eps0 + (fam.eps1 - fam.eps0) * min(1.0, ep / (0.7 * fam.episodes))
        last = keep_traj and ep == fam.episodes - 1
        o, st = play(rule, fam, G=G, T=T, P=P, R0=R0, r=r, seed=seed * 1000 + ep, st=st, return_traj=last)
        if ep >= fam.episodes - fam.eval_last:
            res.append(o)
    agg = {k: np.concatenate([x[k] for x in res]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    if keep_traj:
        agg["traj"] = res[-1]["traj"]
    return agg

# ---------------------------------------------------------------- evolutionary family
def eval_EV(rule, N=400, gens=400, P=4, T=40, R0=200.0, r=0.4, zealot=0.2, exec_sd=0.05, K=5.0, mu=0.01,
            seed=0, tail=100, keep_traj=False):
    """Constant-return-fraction agents evolve by Fermi imitation under the rule (committed free riders
    fixed at phi=0, execution noise). Returns metrics pooled over the last `tail` generations."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 1, N)
    nz = int(round(zealot * N)); fixed = np.zeros(N, bool); fixed[:nz] = True; phi[fixed] = 0.0
    class _Const:
        def __init__(s): pass
        def init(s, G, P_, rng_): return dict(rng=rng_)
        def act(s, st_, e, R, R0_):
            ph = st_["phi"] if exec_sd == 0 else np.clip(st_["phi"] + st_["rng"].normal(0, exec_sd, e.shape), 0, 1)
            return np.where(st_["fixed"], 0.0, ph)
        def observe(s, *a, **k): pass
    fam = _Const(); keep = []; phis = []
    for g in range(gens):
        perm = rng.permutation(N)
        st = dict(rng=rng, phi=phi[perm].reshape(-1, P), fixed=fixed[perm].reshape(-1, P))
        o, _ = play(rule, fam, G=N // P, T=T, P=P, R0=R0, r=r, seed=int(rng.integers(1 << 30)), st=st,
                    return_traj=(keep_traj and g == gens - 1))
        pay = np.empty(N); pay[perm] = o["cum"].ravel()
        if g >= gens - tail:
            keep.append(o); phis.append(phi[~fixed].mean())
        model = rng.integers(0, N, N)
        prob = 1.0 / (1.0 + np.exp(-(pay[model] - pay) / K))
        adopt = rng.random(N) < prob
        phi = np.where(adopt, phi[model], phi)
        phi = np.clip(phi + rng.normal(0, mu, N), 0, 1)
        phi[fixed] = 0.0
    agg = {k: np.concatenate([x[k] for x in keep]) for k in ["surplus", "gini", "active", "sustained", "all_active"]}
    agg["phi_bar"] = float(np.mean(phis))
    if keep_traj:
        agg["traj"] = keep[-1]["traj"]
    return agg


def traj_arrays(o):
    E = np.stack([x[1] for x in o["traj"]], 1); C = np.stack([x[2] for x in o["traj"]], 1)
    return E, C

def beh_moments(E, C):
    use_ = E >= 1.0
    phi = np.where(use_, C / np.maximum(E, 1e-9), np.nan)
    with np.errstate(all="ignore"):
        early = np.nanmean(phi[:, 1:6]); late = np.nanmean(phi[:, 6:])
        low = float(np.mean(phi[use_] < 0.1)) if use_.any() else 0.0
        a = phi[:, 1:-1].ravel(); b = phi[:, 2:].ravel(); m = np.isfinite(a) & np.isfinite(b)
        ac = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 10 else 0.0
    return dict(phi_early=float(np.nan_to_num(early)), phi_late=float(np.nan_to_num(late)), phi_low=low,
                phi_ac=float(np.nan_to_num(ac)))

def full_moments(rule, fam, G=400, seed=11, r=0.4):
    o, _ = play(rule, fam, G=G, seed=seed, r=r, return_traj=True)
    m = summarize(o, r=r)
    m.update(beh_moments(*traj_arrays(o)))
    return m


class EMP:
    """Case-based behavioural clone: each player's return fraction is drawn from the k nearest
    human player-rounds (standardised state features: relative offer, pool, own and others'
    previous return fractions, offer change, round). Non-parametric; no fitted parameters."""
    name = "EMP"
    def __init__(self, train_path, k=25, fr_share=None):
        from scipy.spatial import cKDTree
        d = np.load(train_path)
        self.X, self.Y = d["X"].astype(float), d["Y"].astype(float)
        self.mu, self.sd = self.X.mean(0), self.X.std(0) + 1e-9
        self.w = np.array([1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 0.5])
        self.tree = cKDTree((self.X - self.mu) / self.sd * self.w)
        self.k, self.fr_share = k, fr_share
    def init(self, G, P, rng):
        committed = np.zeros((G, P), bool)
        if self.fr_share:
            committed = rng.random((G, P)) < self.fr_share
        return dict(rng=rng, Ep=np.zeros((G, P)), Cp=np.zeros((G, P)), t=0, committed=committed)
    def act(self, st, e, R, R0):
        G, P = e.shape
        F = emp_features(e, st["Cp"], st["Ep"], R, st["t"], P, R0)
        q = ((F.reshape(-1, F.shape[-1]) - self.mu) / self.sd) * self.w
        _, idx = self.tree.query(q, k=self.k)
        pick = idx[np.arange(len(idx)), st["rng"].integers(0, self.k, len(idx))]
        phi = self.Y[pick].reshape(G, P)
        return np.where(st["committed"], 0.0, phi)
    def observe(self, st, e, c, phi, surplus, R_next, done):
        st["Ep"], st["Cp"], st["t"] = e, c, st["t"] + 1

def emp_features(E, Cprev, Eprev, R, t, P=4, R0=200.0):
    rel = E / np.maximum(R[:, None] / P, 1e-9)
    f1 = np.log(np.clip(rel, 0.05, 20))
    f2 = np.repeat((R / R0)[:, None], P, 1)
    usable_prev = Eprev >= 1.0
    phi_prev = np.where(usable_prev, Cprev / np.maximum(Eprev, 1e-9), 0.5)
    ind = usable_prev.astype(float)
    frac = np.where(usable_prev, phi_prev, 0.0); n = usable_prev.sum(1, keepdims=True)
    oth = np.where(n - usable_prev > 0, (frac.sum(1, keepdims=True) - frac) / np.maximum(n - usable_prev, 1), 0.6)
    chg = np.where(usable_prev, np.log(np.clip(E / np.maximum(Eprev, 1e-9), 0.05, 20)), 0.0)
    f7 = np.full_like(f1, min(t, 40) / 40.0)
    return np.stack([f1, f2, phi_prev, ind, oth, chg, f7], -1)
