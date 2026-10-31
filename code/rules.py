
"""Reference allocation rules with the program signature used throughout:
allocate(pool, prev_offer, prev_contrib, t, mem, cap, growth) -> offers (G, p)."""
import numpy as np

def _prop(pool, c):
    s = c.sum(1, keepdims=True); P = c.shape[1]
    return np.where(s > 0, pool[:, None] * c / np.maximum(s, 1e-12), pool[:, None] / P)

def mixture(w_fun):
    def allocate(pool, prev_offer, prev_contrib, t, mem, cap, growth):
        P = prev_contrib.shape[1]
        w = w_fun(pool, cap)
        return w[:, None] * (pool[:, None] / P) + (1 - w[:, None]) * _prop(pool, prev_contrib)
    return allocate

equal = mixture(lambda R, cap: np.ones_like(R))
proportional = mixture(lambda R, cap: np.zeros_like(R))
mixed = mixture(lambda R, cap: np.full_like(R, 0.5))
def power(k):
    return mixture(lambda R, cap: (R / cap) ** k)
heuristic22 = power(22.0)
def step(theta):
    return mixture(lambda R, cap: (R >= theta * cap).astype(float))
