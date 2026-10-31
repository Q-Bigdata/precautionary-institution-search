
"""Black-box reference planner: permutation-equivariant DeepSets policy (705 parameters) with an
exclusion-counter memory, trained by antithetic OpenAI-ES on the same pessimistic objective as the
program search (worst efficiency over SR, AL and EV at screening precision)."""
import json, sys, time, os
import numpy as np
from multiprocessing import Pool
import sim, evaluate

P0 = json.load(open("calib_params.json"))
D_IN, H = 8, 16
SHAPES = [("W1", (D_IN, H)), ("b1", (H,)), ("W2", (2 * H, H)), ("b2", (H,)), ("v", (H,)), ("u", (H,)), ("bret", (1,))]
NPAR = sum(int(np.prod(s)) for _, s in SHAPES)

def unpack(theta):
    out = {}; i = 0
    for n, s in SHAPES:
        k = int(np.prod(s)); out[n] = theta[i:i + k].reshape(s); i += k
    return out

def make_rule(theta):
    W = unpack(np.asarray(theta, float))
    def allocate(pool, prev_offer, prev_contrib, t, mem, cap, growth):
        G, P = prev_contrib.shape
        usable = prev_offer >= 1.0
        tot = prev_contrib.sum(1, keepdims=True)
        share = np.where(tot > 0, prev_contrib / np.maximum(tot, 1e-12), 1.0 / P)
        phi = np.where(usable, prev_contrib / np.maximum(prev_offer, 1e-9), 0.0)
        if "excl" not in mem:
            mem["excl"] = np.zeros((G, P))
        mem["excl"] = np.where(usable, 0.0, mem["excl"] + 1.0)
        X = np.stack([share, phi, usable.astype(float),
                      np.repeat((pool / cap)[:, None], P, 1),
                      np.repeat((tot[:, 0] * (1 + growth) / cap)[:, None], P, 1),
                      np.repeat(phi.mean(1, keepdims=True), P, 1),
                      np.minimum(mem["excl"], 10) / 5.0,
                      np.full((G, P), min(t, 40) / 40.0)], -1)
        h = np.tanh(X @ W["W1"] + W["b1"])
        g = h.mean(1, keepdims=True)
        z = np.tanh(np.concatenate([h, np.repeat(g, P, 1)], -1) @ W["W2"] + W["b2"])
        sc = z @ W["v"]
        ret = (g[:, 0] @ W["u"]) + W["bret"][0]
        S = np.concatenate([sc, ret[:, None]], 1)
        S = S - S.max(1, keepdims=True); E = np.exp(S); E = E / E.sum(1, keepdims=True)
        return pool[:, None] * E[:, :P]
    return allocate

def fit_one(args):
    theta, seed = args
    card = evaluate.screen(make_rule(theta), P0, seed=seed)
    return evaluate.fitness(card), card

if __name__ == "__main__":
    iters = int(sys.argv[1]); pop = int(sys.argv[2]); seed = int(sys.argv[3]); nproc = int(sys.argv[4])
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.3, NPAR)
    if len(sys.argv) > 5:
        theta = np.array(json.load(open(sys.argv[5]))["theta"], float)
    sigma, lr = 0.08, 0.03
    m = np.zeros(NPAR); v = np.zeros(NPAR); b1, b2 = 0.9, 0.999
    log = []; best = (-1, None, None)
    with Pool(nproc) as pl:
        for it in range(iters):
            t0 = time.time()
            eps = rng.normal(0, 1, (pop // 2, NPAR)); eps = np.concatenate([eps, -eps])
            cand = theta[None, :] + sigma * eps
            sd = int(rng.integers(1 << 20))
            res = pl.map(fit_one, [(c, sd) for c in cand] + [(theta, sd)])
            f = np.array([r[0] for r in res[:-1]]); fc = res[-1][0]
            if fc > best[0]:
                best = (fc, theta.copy().tolist(), res[-1][1])
            rk = np.argsort(np.argsort(f)) / (len(f) - 1) - 0.5
            grad = (rk[:, None] * eps).mean(0) / sigma
            m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
            theta = theta + lr * (m / (1 - b1 ** (it + 1))) / (np.sqrt(v / (1 - b2 ** (it + 1))) + 1e-8)
            log.append(dict(it=it, center=fc, pop_mean=float(f.mean()), pop_max=float(f.max()), secs=time.time() - t0))
            json.dump(dict(log=log, best_fit=best[0], best_theta=best[1], best_card=best[2], theta=theta.tolist(), npar=NPAR),
                      open(f"es_{seed}.json", "w"))
            print(it, round(fc, 4), round(f.max(), 4), round(time.time() - t0, 1), flush=True)
