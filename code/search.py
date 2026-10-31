
"""LLM-guided evolutionary program search (runs in the analysis kernel, which has model access).
Candidates are scored by parallel worker subprocesses (evaluate.py)."""
import json, os, re, subprocess, sys, time, hashlib, random
import numpy as np

ENV_TEXT = """A planner manages a shared reservoir that serves {p} agents over repeated rounds.
- The reservoir holds R units, with capacity C = {cap:g}; it starts full (R = C).
- Each round the planner chooses offers e_i >= 0 for every agent with sum_i e_i <= R. Anything not offered stays in the reservoir.
- Each agent then returns an amount c_i with 0 <= c_i <= e_i and keeps e_i - c_i as its payoff for the round. An offer below 1 unit cannot be returned at all (returns come in whole units).
- The reservoir then becomes min(C, R - sum_i e_i + (1 + g) * sum_i c_i) with growth rate g = {g:g}. If it falls below 1 unit it is exhausted permanently and nobody receives anything again.
- Round 0 offers are fixed at equal shares R/{p}. Your program chooses offers from round 1 onward. Play lasts {T} rounds, but agents are not told when it ends.
- The agents are unknown and heterogeneous. The program will be evaluated against several quite different agent populations: (i) people-like agents with persistent, partly reciprocal return habits, some of whom rarely return anything; (ii) agents that learn by trial and error over many repeated plays with the same planner; (iii) a population whose return fractions evolve by imitating more successful agents, including a fixed minority that never returns anything. You cannot observe which population you face."""

OBJ_TEXT = """Objective. Performance in a population is efficiency = total payoff kept by all agents over the {T} rounds divided by the maximum sustainable total, {ceil:.0f} units. The score to maximise is the WORST efficiency across the three populations. Secondary goals: an equal distribution of payoffs across agents (low Gini), keeping agents included (offers >= 1), and a short, readable program."""

API_TEXT = """Write a Python function with exactly this signature:

def allocate(pool, prev_offer, prev_contrib, t, mem, cap, growth):
    # pool:         float array (G,)   current reservoir R in each of G independent games
    # prev_offer:   float array (G, p) offers made last round
    # prev_contrib: float array (G, p) amounts each agent returned last round
    # t:            int, current round (>= 1)
    # mem:          dict that persists across rounds within one batch of games; store arrays of shape (G,) or (G, p) if you need memory. It is empty at t == 1.
    # cap:          float, reservoir capacity C;  growth: float, growth rate g
    # return:       float array (G, p) of offers (non-negative; if their sum exceeds pool they are rescaled down)

Rules: numpy is available as np; no imports, no randomness, no while loops, no file or network access, no names starting with underscore. The code must be vectorised over the G games (G can be 1 to 1000). Keep it under 60 lines. Helper functions defined at top level are allowed."""

API_SHARES = API_TEXT.replace("# return:       float array (G, p) of offers (non-negative; if their sum exceeds pool they are rescaled down)",
    "# return:       float array (G, p) of non-negative WEIGHTS; the whole reservoir is always offered, split in proportion to the weights (equal split if all weights are zero)")
ENV_SHARES_LINE = "- Each round the planner splits the WHOLE reservoir among the agents: offers e_i >= 0 with sum_i e_i = R."

STYLES = {
 "improve": "Propose a modified program that should raise the WORST-population efficiency above the parents'. Think about why the weakest population does badly under the parents and fix that.",
 "improve_single": "Propose a modified program that should raise the efficiency above the parents'. Think about why the parents fall short and fix that.",
 "combine": "Combine the most useful ideas of the parents into one new program that should beat all of them on the worst-population efficiency.",
 "simplify": "Propose a SHORTER and more readable program that keeps (or improves) the worst-population efficiency of the best parent. Remove anything that does not matter.",
 "explore": "Propose a program built on a clearly DIFFERENT allocation idea from the parents, one that might do well against all three populations. Be creative but principled.",
}

TELEMETRY = {"on": False}
API_MODE = {"mode": "amounts"}
HINT = {"text": ""}
def fmt_card(rec):
    c = rec["card"]
    fams = [f for f in ("SR", "AL", "EV") if f in c]
    lab = {"SR": "people-like", "AL": "learners", "EV": "evolving"}
    parts = []
    for f in fams:
        line = f"{lab[f]}: eff {c[f]['eff']:.3f}, Gini {c[f]['gini']:.2f}, included {c[f]['active']:.2f}, sustained {c[f]['sust']:.2f}"
        if TELEMETRY["on"] and "tele" in c[f]:
            tl = c[f]["tele"]
            line += f"\n    by 8-round block: reservoir {tl['pool']}, return fraction {tl['ret']}, agents included {tl['inc']}"
        parts.append(line)
    head = "worst-population efficiency" if len(fams) > 1 else "efficiency"
    return f"{head} {rec['fit']:.3f}; length {rec['length']} AST nodes\n  " + "\n  ".join(parts)

ENV_SINGLE = """- The agents are unknown and heterogeneous: people-like agents with persistent, partly reciprocal return habits, some of whom rarely return anything."""
OBJ_SINGLE = """Objective. Efficiency = total payoff kept by all agents over the {T} rounds divided by the maximum sustainable total, {ceil:.0f} units. Maximise efficiency. Secondary goals: an equal distribution of payoffs across agents (low Gini), keeping agents included (offers >= 1), and a short, readable program."""
def build_prompt(parents, style, p=4, cap=200.0, g=0.4, T=40, single=False):
    ceil = T * cap * g / (1 + g)
    env = ENV_TEXT.format(p=p, cap=cap, g=g, T=T)
    obj = OBJ_TEXT.format(T=T, ceil=ceil)
    if single:
        env = env.split("- The agents are unknown")[0] + ENV_SINGLE
        obj = OBJ_SINGLE.format(T=T, ceil=ceil)
    api = API_TEXT
    if API_MODE["mode"] == "shares":
        api = API_SHARES
        env = env.replace("- Each round the planner chooses offers e_i >= 0 for every agent with sum_i e_i <= R. Anything not offered stays in the reservoir.", ENV_SHARES_LINE)
    txt = [env, "", obj, "", api, ""] + ([HINT["text"], ""] if HINT["text"] else []) + [
           "Here are previously evaluated programs and their scores:"]
    for i, pr in enumerate(parents):
        txt += [f"\n### Program {chr(65 + i)} ({fmt_card(pr)})", "```python", pr["code"].strip(), "```"]
    txt += ["", STYLES[style], "Reply with a one-sentence rationale, then the complete program in a single ```python block."]
    return "\n".join(txt)

SYSTEM = "You are an expert in mechanism design, economics and scientific Python programming. You write correct, vectorised numpy code."

def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not m:
        return None
    code = max(m, key=len)
    code = re.sub(r"^\s*import numpy as np\s*$", "", code, flags=re.M)
    return code.strip() + "\n"

def code_hash(code):
    return hashlib.sha1(re.sub(r"\s+", " ", code).encode()).hexdigest()[:12]

# ------------------------------------------------------------------ parallel scoring
def score_parallel(items, params, workdir, n_workers=16, r=0.4, families=("SR", "AL", "EV"), cfg=None, seed=0,
                   timeout=600, py=sys.executable, srcdir=None):
    os.makedirs(workdir, exist_ok=True)
    shards = [items[i::n_workers] for i in range(n_workers)]
    procs = []
    env = dict(os.environ, PYTHONPATH=srcdir, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stamp = f"{time.time():.0f}_{random.randrange(10**6)}"
    for k, sh in enumerate(shards):
        if not sh:
            continue
        inp = os.path.join(workdir, f"in_{stamp}_{k}.json"); outp = os.path.join(workdir, f"out_{stamp}_{k}.json")
        json.dump(dict(items=sh, params=params, r=r, families=list(families), cfg=cfg or {}, seed=seed,
                       wrap=("shares" if API_MODE["mode"] == "shares" else None)), open(inp, "w"))
        pr = subprocess.Popen([py, os.path.join(srcdir, "evaluate.py"), inp, outp], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((pr, outp, sh))
    t0 = time.time(); res = []
    for pr, outp, sh in procs:
        try:
            pr.wait(timeout=max(1, timeout - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            pr.kill()
        got = json.load(open(outp)) if os.path.exists(outp) else []
        done = {g["id"] for g in got}
        res += got + [dict(id=it["id"], ok=False, error="timeout") for it in sh if it["id"] not in done]
    return res

# ------------------------------------------------------------------ archive
LEN_BINS = [0, 60, 100, 160, 260, 10**9]
GINI_BINS = [0, 0.15, 0.25, 0.35, 1.01]

def niche(rec):
    g = np.mean([rec["card"][f]["gini"] for f in rec["card"]])
    li = np.searchsorted(LEN_BINS, rec["length"], side="right") - 1
    gi = np.searchsorted(GINI_BINS, g, side="right") - 1
    return f"{li}-{gi}"
