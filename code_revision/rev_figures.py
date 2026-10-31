"""Figures 1-5 and Extended Data Figs 1-3 of the revised manuscript, from the files in SRC (source data).
Usage: python rev_figures.py SRC_DIR OUT_DIR"""
import json, os, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
SRC = sys.argv[1] if len(sys.argv) > 1 else "source_data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures_out"; os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
                     "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
def rd(f): return os.path.join(SRC, f)
def letter(ax, l, dx=-0.18, dy=1.03): ax.text(dx, dy, l, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")
C = {"prop": "#999999", "k22": "#E69F00", "share": "#0072B2", "seeded": "#009E73", "es": "#CC79A7", "amount": "#D55E00", "minimal": "#56B4E9", "equal": "#666666"}
VAL = pd.read_csv(rd("frozen_validation.csv")).set_index("mech")
META = json.load(open(rd("figure_meta_rev.json")))
LAB = META["labels"]

# ------------------------------------------------------------------ Fig. 1 (design and retrospective validity)
def fig1():
    VT = pd.read_csv(rd("validity_table.csv"))
    FAMC = {"SR": "#0072B2", "SR2": "#56B4E9", "EMP_ni": "#D55E00", "BC1": "#7f7f7f"}
    FAML = {"SR": "structural reciprocity (SR)", "SR2": "conditional cooperation (SR2)", "EMP_ni": "case-based model (EMP, 560 games)", "BC1": "published neural clones"}
    MK = {"equal": "o", "mixed": "s", "prop": "^", "interp": "D"}
    ML = {"equal": "equal", "mixed": "mixed", "prop": "proportional", "interp": "interpolating (excluded from fitting)"}
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.25], hspace=0.30, wspace=0.32, left=0.08, right=0.98, bottom=0.09, top=0.95)
    ax0 = fig.add_subplot(gs[0, :]); ax0.set_xlim(0, 20); ax0.set_ylim(0, 6.5); ax0.axis("off")
    def box(x, y, w, h, txt, fc="#f2f2f2"):
        ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.2", fc=fc, ec="#555555", lw=0.7))
        ax0.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.3, linespacing=1.2)
    def arr(x0, y0, x1, y1):
        ax0.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#555555"))
    box(0.2, 3.4, 4.3, 2.2, "language model writes\na child program from\nparents and score cards", fc="#e8f1fa")
    box(5.3, 3.4, 4.3, 2.2, "sandbox and interface\namount: may withhold\nshare: whole pool split", fc="#fdf0e6")
    box(10.4, 3.4, 4.5, 2.2, "worst-case efficiency over\nSR (calibrated), AL (learners)\nand EV (evolving population)")
    box(15.7, 3.4, 4.1, 2.2, "islands and archive\nover length x Gini;\noptional telemetry")
    arr(4.6, 4.5, 5.2, 4.5); arr(9.7, 4.5, 10.3, 4.5); arr(15.0, 4.5, 15.6, 4.5)
    ax0.annotate("", xy=(2.35, 5.75), xytext=(17.75, 5.75), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#555555", connectionstyle="arc3,rad=0.12", mutation_scale=8))
    box(0.2, 0.2, 9.4, 2.3, "re-scoring with fresh seeds; frozen validation on untouched seeds\nheld-out: SR2, EMP (case-based, human player-rounds),\nAL with lower discount and optimism, EV without free riders or noise\nstress tests: free riders, growth, group size, shock, 80 rounds", fc="#ffffff")
    box(10.4, 0.2, 9.4, 2.3, "reference mechanisms\nhand-designed: equal, mixed, proportional, interpolating\noptimised parametric: power, step, logistic weights\nblack-box DeepSets planner (705 parameters) trained by ES", fc="#ffffff")
    arr(12.65, 3.3, 12.65, 2.6)
    for j, met in enumerate(["spr", "gini"]):
        ax = fig.add_subplot(gs[1, j]); d_ = VT[VT.metric == met]
        lim = (0, 9.4) if met == "spr" else (0, 0.46)
        ax.plot(lim, lim, color="#bbbbbb", lw=0.7, zorder=0)
        for fam in ["BC1", "SR", "SR2", "EMP_ni"]:
            for _, rw in d_.iterrows():
                held = rw.mech == "interp"
                ax.errorbar(rw.human, rw[fam], xerr=[[rw.human - rw.lo], [rw.hi - rw.human]], fmt=MK[rw.mech], ms=5.5 if held else 4.2,
                            mfc=FAMC[fam] if held else "white", mec=FAMC[fam], mew=0.9, ecolor="#cccccc", elinewidth=0.6, capsize=0, zorder=3 if held else 2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("observed in human games" + (" (surplus per player-round)" if met == "spr" else " (Gini)"))
        ax.set_ylabel("predicted by model" + (" (surplus)" if met == "spr" else " (Gini)"))
        ax.set_title("Models bracket the excluded rule's surplus" if met == "spr" else "SR, EMP and clones match its lower inequality; SR2 overstates it", loc="left")
        hs = ([Line2D([], [], marker="o", ls="", mfc=FAMC[f], mec=FAMC[f], ms=4, label=FAML[f]) for f in ["SR", "SR2", "EMP_ni", "BC1"]] if met == "spr" else
              [Line2D([], [], marker=MK[m], ls="", mfc="white" if m != "interp" else "#444444", mec="#444444", ms=4, label=ML[m]) for m in MK])
        ax.legend(handles=hs, loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0.2)
    letter(ax0, "a", dx=0.0, dy=0.97); letter(fig.axes[1], "b", dx=-0.14); letter(fig.axes[2], "c", dx=-0.14)
    return fig

# ------------------------------------------------------------------ Fig. 2 (search outcomes)
def fig2():
    RS = pd.read_csv(rd("running_best.csv"))
    REF = json.load(open(rd("reference_rescored.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.75, left=0.08, right=0.98, bottom=0.15, top=0.9))
    ax = axs[0]
    for run, d_ in RS.groupby("run", sort=False):
        g = META["run_group"][run]
        ax.plot(d_.k, d_.best, color=C[g], lw=1.5 if g == "share" else 1.0, ls=":" if run == "SRonly_tele" else "-", alpha=0.95)
    for k_, lab, col in [("ref:k22", "published interpolating rule", C["k22"]), ("ref:proportional", "proportional (seed)", C["prop"])]:
        ax.axhline(REF[k_], color=col, lw=0.8, ls="--"); ax.text(855 if k_ == "ref:proportional" else 8, REF[k_] + 0.003, lab, color=col, fontsize=6, va="bottom", ha="right" if k_ == "ref:proportional" else "left")
    ax.text(600, 0.215, "seven blind runs,\namount interface", color=C["amount"], fontsize=6, va="center")
    ax.text(470, 0.372, "three blind runs,\nshare interface", color=C["share"], fontsize=6, va="center")
    ax.text(470, 0.305, "seeded with the\npublished rule", color=C["seeded"], fontsize=6, va="center")
    ax.set_xlim(0, 860); ax.set_ylim(0.1, 0.39)
    ax.set_xlabel("generated programs evaluated"); ax.set_ylabel("best worst-case efficiency so far\n(independent re-scoring)")
    ax.set_title("Gains only under the share interface", loc="left")
    ax = axs[1]
    items = META["fig2b_items"]
    for j, (lab, i, g) in enumerate(items):
        r = VAL.loc[i]
        ax.plot([r.lo, r.hi], [j, j], color=C[g], lw=1.2, zorder=2); ax.plot(r.worst, j, "o", color=C[g], ms=4, zorder=3)
    ax.axvline(VAL.loc["ref:k22", "worst"], color=C["k22"], lw=0.7, ls="--", zorder=0)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l, _, _ in items]); ax.invert_yaxis()
    ax.set_xlim(0.05, 0.66); ax.set_xlabel("worst-case efficiency, frozen validation\n(95% interval over seeds)")
    ax.set_title("Readable rules below the black box", loc="left")
    letter(axs[0], "a", dx=-0.13); letter(axs[1], "b", dx=-1.25)
    return fig

# ------------------------------------------------------------------ Fig. 3 (the proposal bias)
def fig3():
    PB = pd.read_csv(rd("prior_by_run_rev.csv")).set_index("run")
    LIN = pd.read_csv(rd("lineage.csv")); FRc = pd.read_csv(rd("framing_controls.csv")).set_index("cond")
    PC = pd.read_csv(rd("program_classification.csv")); NR = pd.read_csv(rd("normalisation_results.csv")); ND = json.load(open(rd("norm_design.json")))
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.62, wspace=1.6, left=0.17, right=0.985, bottom=0.08, top=0.95)
    ax = fig.add_subplot(gs[0, 0:2]); order = list(PB.index); yy = np.arange(len(order))
    cols = [C[META["run_group"][r]] for r in order]
    ax.barh(yy - 0.18, PB.withholds, height=0.34, color=cols)
    ax.barh(yy + 0.18, PB.wording, height=0.34, color="white", edgecolor=cols, hatch="////", lw=0.6)
    for j, r in enumerate(order):
        if META["run_group"][r] == "share": ax.text(0.02, j - 0.18, "excluded by interface", fontsize=5.5, va="center", color="#666666")
    ax.set_yticks(yy); ax.set_yticklabels([LAB[r] for r in order], fontsize=6); ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of generated programs"); ax.set_title("Withholding in every amount run", loc="left")
    ax.legend(handles=[Patch(facecolor="#777777", label="withholds >5% of pool"), Patch(facecolor="white", edgecolor="#777777", hatch="////", label="precautionary wording")],
              loc="upper center", bbox_to_anchor=(0.3, -0.16), ncol=1, handlelength=1.2, fontsize=5.8)
    # b lineage
    ax = fig.add_subplot(gs[0, 2:4])
    A = LIN[LIN.iface == "amount"]; S_ = LIN[LIN.iface == "seeded"]
    cats = [("first\ngen.", lambda d: d.gen == 0), ("no\nparent", lambda d: ~d.par_any & (d.gen > 0)), ("some\nparents", lambda d: d.par_any & ~d.par_all), ("all\nparents", lambda d: d.par_all)]
    x = np.arange(len(cats))
    for k, (d_, g, off) in enumerate([(A, "amount", -0.19), (S_, "seeded", 0.19)]):
        v = [d_[f(d_)].withholds.mean() for _, f in cats]; n = [int(f(d_).sum()) for _, f in cats]
        ax.bar(x + off, v, width=0.36, color=C[g], label="blind amount" if g == "amount" else "seeded amount")
        for xi, vi, ni in zip(x + off, v, n): ax.text(xi, vi + 0.02, f"{ni}", ha="center", fontsize=5, color="#555555")
    ax.set_xticks(x); ax.set_xticklabels([c for c, _ in cats], fontsize=5.6); ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("share that withholds"); ax.set_xlabel("parents that withhold", fontsize=6)
    ax.set_title("Introduced, not inherited", loc="left"); ax.legend(loc="upper center", fontsize=5.6, ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8, frameon=False)
    # c allocation logic
    ax = fig.add_subplot(gs[0, 4:6])
    rng = np.random.default_rng(0)
    for j, g in enumerate(["amount", "seeded", "share"]):
        d_ = PC[PC.iface == g]; v = np.clip(d_.w50.to_numpy(), -0.6, 1.4)
        col = np.where(d_.EV_fresh.to_numpy() < 0.09, "#bbbbbb", C[g])
        ax.scatter(j + rng.uniform(-0.32, 0.32, len(v)), v, s=1.6, c=col, lw=0, alpha=0.5)
        ax.plot([j - 0.35, j + 0.35], [np.median(d_.w50)] * 2, color="k", lw=1.0)
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0); ax.axhline(1, color="#dddddd", lw=0.6, zorder=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["blind\namount", "seeded\namount", "blind\nshare"], fontsize=6)
    ax.set_ylabel("effective equality weight\nat a pool of 50 (1 = equal)"); ax.set_title("Near-equal splits under scarcity", loc="left")
    ax.text(2.45, 1.33, "grey: evolving\npopulation collapses", fontsize=5.5, ha="right", va="top", color="#777777")
    # d normalisation
    ax = fig.add_subplot(gs[1, 0:3])
    rb = NR[NR.id.isin(ND["random_blind"])]; tp = NR[NR.id.isin(ND["top10"]) & ~NR.run.eq("seeded_tele")]
    ax.scatter(rb.fresh, rb.norm, s=4, color=C["amount"], alpha=0.35, lw=0, label=f"random blind programs (n = {len(rb)})")
    ax.scatter(tp.fresh, tp.norm, s=9, facecolor="white", edgecolor=C["amount"], lw=0.7, label=f"best 10 of each blind run (n = {len(tp)})")
    lim = (0.06, 0.40); ax.plot(lim, lim, color="#bbbbbb", lw=0.7, zorder=0)
    ax.axhline(META["k22_fresh"], color=C["k22"], ls="--", lw=0.8); ax.text(0.07, META["k22_fresh"] + 0.004, "published rule", fontsize=5.8, color=C["k22"], ha="left")
    ax.text(0.07, META["prop_fresh"] + 0.004, "proportional seed", fontsize=5.8, color=C["prop"], ha="left")
    ax.axhline(META["prop_fresh"], color=C["prop"], ls="--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("worst-case efficiency, original program"); ax.set_ylabel("after rescaling offers\nto the whole pool")
    ax.set_title("Rescaling does not rescue them", loc="left"); ax.legend(loc="lower right", fontsize=5.6, markerscale=1.6)
    ax.set_position([0.13, 0.1, 0.33, 0.33])
    # e framing and model controls
    ax = fig.add_subplot(gs[1, 3:6])
    conds = META["framing_order"]
    for j, (cname, lab) in enumerate(conds):
        r = FRc.loc[cname]; col = "#444444" if "model2" not in cname else C["es"]
        ax.plot([r.lo, r.hi], [j, j], color=col, lw=1.2); ax.plot(r.p, j, "o", color=col, ms=4)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([f"{l}\nn = {int(FRc.loc[c_].n)}, median withheld {abs(FRc.loc[c_].med_withheld):.2f}" for c_, l in conds], fontsize=5.6); ax.invert_yaxis(); ax.set_xlim(0, 1.02); ax.set_ylim(len(conds) - 0.5, -0.5)
    ax.set_xlabel("share of zero-shot programs that withhold\n(95% interval)"); ax.set_title("Framing and model change the habit", loc="left")
    ax.set_position([0.735, 0.1, 0.25, 0.33])
    for a_, l, dx in zip(fig.axes, "abcde", [-0.75, -0.32, -0.42, -0.25, -1.05]): letter(a_, l, dx=dx)
    return fig

# ------------------------------------------------------------------ Fig. 4 (mechanism)
def fig4():
    PW = json.load(open(rd("probe_weights_rev.json"))); IP = json.load(open(rd("inplay_offer_curves.json"))); FX = pd.read_csv(rd("forgiveness_extended.csv"))
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.62, wspace=0.55, left=0.08, right=0.93, bottom=0.09, top=0.95)
    ax = fig.add_subplot(gs[0, 0])
    for k, lab in [("prop", "proportional"), ("k22", "interpolating rule (published)"), ("share", "best blind share program"), ("minimal", "three-term rule")]:
        pw = PW[k]; ax.plot(pw["R"], np.clip(pw["w"], -0.3, 1.1), color=C[k], lw=1.4 if k == "share" else 1.0, ls="--" if k == "minimal" else "-", label=lab)
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0); ax.set_ylim(-0.35, 1.12)
    ax.set_xlabel("pool (held fixed in probe)"); ax.set_ylabel("effective equality weight\n(0 proportional, 1 equal)"); ax.set_title("Meritocratic when scarce", loc="left")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.6)
    ax = fig.add_subplot(gs[0, 1])
    bc = np.array(IP["bin_centres"]); q = np.array([IP["amount_q25"], IP["amount_median"], IP["amount_q75"]])
    ax.fill_between(bc, q[0], q[2], color=C["amount"], alpha=0.18, lw=0); ax.plot(bc, q[1], color=C["amount"], lw=1.3, marker="o", ms=2)
    ax.plot(bc, IP["k22"], color=C["k22"], lw=1.0, marker="o", ms=2); ax.plot(bc, IP["blackbox_seed22"], color=C["es"], lw=1.2, marker="o", ms=2)
    ax.set_ylim(0, 1.08); ax.set_xlabel("pool during play against SR"); ax.set_ylabel("fraction of pool offered")
    ax.set_title("Reserves: fixed vs conditional", loc="left")
    ax.text(85, 0.3, "amount-interface programs\n(median, IQR; n = 300)", color=C["amount"], fontsize=5.8, ha="center", va="top")
    ax.text(12, 0.70, "black box", color=C["es"], fontsize=5.8); ax.text(30, 1.02, "published rule", color=C["k22"], fontsize=5.8)
    ax = fig.add_subplot(gs[0, 2])
    items = META["fig4c_items"]
    for j, (lab, i) in enumerate(items):
        r = VAL.loc[i]; col = C["share"] if j == 0 else ("#9ecae1" if "skew" in i or "sharpen" in i else C["minimal"])
        ax.barh(j, r.worst, color=col, height=0.62); ax.plot([r.lo, r.hi], [j, j], color="k", lw=0.7)
        ax.text(r.hi + 0.006, j, f"{r.worst:.2f}", va="center", fontsize=5.8)
    ax.axvline(VAL.loc["ref:k22", "worst"], color=C["k22"], lw=0.9, ls="--")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l, _ in items], fontsize=5.5); ax.invert_yaxis()
    ax.set_xlim(0.1, 0.45); ax.set_xlabel("worst-case efficiency, frozen validation"); ax.set_title("Exponent must fall with the pool", loc="left")
    # d exclusion maps
    rules = [("ref:k22", "interpolating rule"), ("shares_scoreonly:g24_i2_10", "best blind share program"), ("es:seed22", "black-box planner")]
    vmax = 24
    for k, (m, lab) in enumerate(rules):
        ax = fig.add_subplot(gs[1, k]) if k == 0 else fig.add_subplot(gs[1, k], sharey=fig.axes[3])
        d_ = FX[FX.mech == m].pivot(index="R_start", columns="n_defect", values="rounds_below1").sort_index(ascending=False)
        im = ax.imshow(d_.values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        for a in range(d_.shape[0]):
            for b in range(d_.shape[1]):
                v = d_.values[a, b]; ax.text(b, a, f"{int(v)}", ha="center", va="center", fontsize=6.5, color="white" if v > 12 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels([1, 2, 3]); ax.set_xlabel("consecutive zero returns")
        ax.set_yticks(range(3)); ax.set_yticklabels([f"{int(x)}" for x in d_.index]) if k == 0 else (plt.setp(ax.get_yticklabels(), visible=False), ax.tick_params(axis="y", length=0))
        if k == 0: ax.set_ylabel("pool at start")
        ax.set_title(lab, loc="left", fontsize=6.5)
        for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.08, 0.455, "Rounds in which the defector's offer stays below one unit (no longer reciprocable)", fontsize=7)
    cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.02, pad=0.02); cb.set_label("rounds below one unit", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    for a_, l, dx in zip([fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]], "abcd", [-0.3, -0.27, -0.55, -0.3]): letter(a_, l, dx=dx)
    return fig

# ------------------------------------------------------------------ Fig. 5 (black-box reference and model dependence)
def fig5():
    ALLT = pd.read_csv(rd("all_programs_rescored_rev.csv")); REF = json.load(open(rd("reference_rescored.json")))
    CAL = pd.read_csv(rd("calibration_sensitivity_eff.csv")).set_index("mech"); COV = pd.read_csv(rd("emp_coverage.csv"))
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42, left=0.09, right=0.985, bottom=0.09, top=0.95)
    ax = fig.add_subplot(gs[0, 0]); LB = [0, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 600]
    def env(d_):
        xs, ys = [], []
        for lo, hi in zip(LB[:-1], LB[1:]):
            s_ = d_[(d_.length >= lo) & (d_.length < hi)]
            if len(s_): xs.append(np.sqrt(lo * hi) if lo > 0 else 30); ys.append(s_.fresh.max())
        return np.array(xs), np.maximum.accumulate(np.array(ys))
    for lab, g in [("blind, amount interface", "amount"), ("blind, share interface", "share"), ("seeded, amount interface", "seeded")]:
        d_ = ALLT[ALLT.group == g].dropna(subset=["fresh"]); ax.scatter(d_.length, d_.fresh, s=2, color=C[g], alpha=0.10, lw=0)
        x_, y_ = env(d_); ax.step(x_, y_, where="post", color=C[g], lw=1.3, label=lab)
    ax.plot(80, REF["ref:k22"], "D", color=C["k22"], ms=5, zorder=5); ax.text(60, REF["ref:k22"] + 0.025, "published rule", fontsize=5.8, color=C["k22"])
    ax.plot(42, META["minimal_fresh"], "s", color=C["minimal"], ms=4.5, zorder=5); ax.text(36, META["minimal_fresh"] + 0.03, "three-term rule", fontsize=5.8, color=C["minimal"])
    ax.axhline(REF["es:seed22"], color=C["es"], lw=1.0, ls="--"); ax.text(700, REF["es:seed22"] + 0.012, "best black box (705 parameters, not syntax nodes)", fontsize=5.6, color=C["es"], ha="right")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xticks([30, 100, 300]); ax.set_xticklabels(["30", "100", "300"]); ax.set_xlim(25, 720)
    ax.set_ylim(0.05, 0.68); ax.set_xlabel("syntactic length (syntax-tree nodes)"); ax.set_ylabel("worst-case efficiency (re-scored)")
    ax.set_title("Explored programs stay below the black box", loc="left"); ax.legend(loc="upper left", fontsize=5.6, bbox_to_anchor=(0.0, 0.9))
    # b EMP library sensitivity
    ax = fig.add_subplot(gs[0, 1]); libs = [("EMP", "all 680\ngames"), ("EMP_noint", "560 games\n(no interpolating)"), ("EMP_base", "280 games\n(baselines only)")]
    for m, g in META["fig5b_items"]:
        r = VAL.loc[m]; ys = [r[k] for k, _ in libs]
        ax.plot(range(3), ys, marker="o", ms=3 if g != "es" else 2.5, color=C[g], lw=1.2 if g != "es" else 0.7, alpha=1 if g != "es" else 0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in libs], fontsize=6); ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel("efficiency on EMP (held out)"); ax.set_title("Black-box lead on EMP depends on the library", loc="left")
    ax.legend(handles=[Line2D([], [], color=C[g], marker="o", ms=3, label=l) for g, l in [("k22", "published rule"), ("seeded", "best seeded"), ("share", "best blind share"), ("es", "black-box trajectories")]], loc="lower left", fontsize=5.6)
    # c calibration sensitivity
    ax = fig.add_subplot(gs[1, 0]); sets = META["fig5c_sets"]
    for m, g in META["fig5c_items"]:
        ys = [CAL.loc[m, s] for s, _ in sets]
        ax.plot(range(len(sets)), ys, marker="o", ms=3, color=C[g], lw=1.2 if g != "es" else 0.8, alpha=1 if g != "es" else 0.8)
    ax.axvspan(2.5, 4.5, color="#f0f0f0", zorder=0); ax.text(3.5, 0.62, "27-moment fits\n(less persistent)", ha="center", fontsize=5.8, color="#666666")
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([l for _, l in sets], fontsize=5.8); ax.set_ylabel("efficiency against SR")
    ax.set_title("Black-box lead depends on SR persistence", loc="left"); ax.set_ylim(0.2, 0.68)
    # d EMP support
    ax = fig.add_subplot(gs[1, 1]); cv = COV[COV.lib == "emp_train_all.npz"]
    for _, r in cv.iterrows():
        g = META["cov_group"].get(r.mech, "amount"); ax.plot(r.frac_gt95, r.eff, "o", color=C[g], ms=4 if g != "es" else 3.5)
    ax.axvline(0.05, color="#bbbbbb", lw=0.7, ls=":"); ax.text(0.055, 0.25, "human data\n(by definition)", fontsize=5.6, color="#777777")
    ax.legend(handles=[Line2D([], [], color=C[g], marker="o", ls="none", ms=3.5, label=l) for g, l in [("equal", "equal"), ("prop", "proportional, power-law"), ("k22", "published rule"), ("seeded", "best seeded"), ("share", "blind share programs"), ("amount", "best blind amount"), ("es", "black-box trajectories")]], loc="lower right", fontsize=5.4, ncol=2, handletextpad=0.3, columnspacing=0.8)
    ax.set_xlabel("share of EMP decisions outside human support\n(beyond 95th percentile of human-to-human distance)"); ax.set_ylabel("efficiency on EMP")
    ax.set_title("EMP extrapolates more for the black box", loc="left"); ax.set_xlim(0, 0.45)
    for a_, l, dx in zip(fig.axes, "abcd", [-0.14, -0.15, -0.14, -0.15]): letter(a_, l, dx=dx)
    return fig

# ------------------------------------------------------------------ Extended Data Fig. 1 (adaptive learners)
def edfig1():
    DC = json.load(open(rd("diagnostic_curves.json"))); TD = pd.read_csv(rd("diagnostics_AL_EV.csv")).set_index("index")
    G = META["diag_groups"]; fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.55, left=0.07, right=0.98, bottom=0.36, top=0.88, width_ratios=[1.2, 1, 1.3]))
    ax = axs[0]
    for m, g in G:
        c = np.array(DC["AL600"][m]); mu = c.mean(0); k = 20; sm = np.convolve(mu, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k, len(mu) + 1), sm, color=C[g], lw=1.1)
    ax.axvline(200, color="#bbbbbb", lw=0.7, ls=":"); ax.text(205, 0.05, "200 games\n(scoring)", fontsize=5.6, color="#777777")
    ax.set_xlabel("training games (600-game schedule)"); ax.set_ylabel("efficiency against AL\n(20-game moving mean)"); ax.set_title("Learners still improving at 200 games", loc="left"); ax.set_ylim(0, 0.9)
    ax = axs[1]; x = np.arange(len(G)); w = 0.2
    for k, (col, lab, ls) in enumerate([("AL_200", "200 games, online", None), ("AL200_frozen", "200 games, frozen", None), ("AL_600", "600 games, online", None), ("AL600_frozen", "600 games, frozen", None)]):
        ax.bar(x + (k - 1.5) * w, [TD.loc[m, col] for m, _ in G], width=w, color=[C[g] for _, g in G], alpha=[0.35, 0.55, 0.8, 1.0][k], edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([META["short"][m] for m, _ in G], rotation=40, ha="right", fontsize=5.8); ax.set_ylabel("efficiency against AL")
    ax.set_title("Longer training, frozen policies", loc="left")
    ax.text(0.0, 1.0, "bars per rule: 200 online, 200 frozen,\n600 online, 600 frozen (light to dark)", transform=ax.transAxes, fontsize=5.4, color="#555555", va="top")
    ax.set_ylim(0, 1.0)
    ax = axs[2]; cols = [("AL_200", "baseline\n(0.95, 150)"), ("AL_g095_q60", "optimism\n60"), ("AL_g08_q150", "discount\n0.8"), ("AL_g08_q60", "both"), ("AL_nact21", "21\nactions")]
    for m, g in G:
        ax.plot(range(len(cols)), [TD.loc[m, c] for c, _ in cols], marker="o", ms=3, color=C[g], lw=1.0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=5.8); ax.set_ylabel("efficiency against AL")
    ax.set_title("Lead needs far-sighted, optimistic learners", loc="left")
    for a_, l in zip(axs, "abc"): letter(a_, l, dx=-0.22)
    fig.legend(handles=[Line2D([], [], color=C[g], lw=1.3, label=META["short"][m]) for m, g in G], loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, 0.0), handlelength=1.6)
    return fig

# ------------------------------------------------------------------ Extended Data Fig. 2 (evolving population)
def edfig2():
    DC = json.load(open(rd("diagnostic_curves.json"))); TD = pd.read_csv(rd("diagnostics_AL_EV.csv")).set_index("index"); G = META["diag_groups"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.5, left=0.07, right=0.99, bottom=0.27, top=0.88, width_ratios=[1, 1, 1.3]))
    for m, g in G:
        tr = DC["EV"][m][0]; gg = [t["g"] for t in tr]
        axs[0].plot(gg, [t["phi"] for t in tr], color=C[g], lw=1.0); axs[1].plot(gg, [t["eff"] for t in tr], color=C[g], lw=1.0)
    axs[0].set_xlabel("generation"); axs[0].set_ylabel("mean return fraction\n(non-committed agents)"); axs[0].set_title("Return fractions settle early", loc="left")
    axs[1].set_xlabel("generation"); axs[1].set_ylabel("efficiency against EV"); axs[1].set_title("Collapse or stable cooperation", loc="left")
    ax = axs[2]; cols = [("EV_traj", "baseline"), ("EV_z0_sd05", "no free\nriders"), ("EV_z02_sd0", "no noise"), ("EV_z0_sd0", "neither"), ("EV_K2", "K = 2"), ("EV_K20", "K = 20"), ("EV_p6", "6 players"), ("EV_p8", "8 players"), ("EV_g800", "800 gen.")]
    for m, g in G:
        ax.plot(range(len(cols)), [TD.loc[m, c] for c, _ in cols], marker="o", ms=2.8, color=C[g], lw=0.9)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=5.4, rotation=45, ha="right"); ax.set_ylabel("efficiency against EV")
    ax.set_title("Rankings robust to EV settings", loc="left")
    hs = [Line2D([], [], color=C[g], lw=1.2, label=META["short"][m]) for m, g in G]
    fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=5.8, bbox_to_anchor=(0.5, 0.0))
    for a_, l in zip(axs, "abc"): letter(a_, l, dx=-0.22)
    return fig

# ------------------------------------------------------------------ Extended Data Fig. 3 (stress tests, original final battery)
def edfig3():
    RM = json.load(open(rd("robustness_matrix.json"))); ROWS = [tuple(r) for r in RM["rows"]]; COLS = [tuple(c) for c in RM["cols"]]; H = np.array(RM["H"])
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); fig.subplots_adjust(left=0.24, right=0.93, bottom=0.36, top=0.86)
    im = ax.imshow(H, cmap="viridis", vmin=0.1, vmax=0.95, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]): ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center", fontsize=5.4, color="white" if H[i, j] < 0.55 else "black")
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r for r, _ in ROWS], fontsize=6)
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c for c, _, _ in COLS], rotation=55, ha="right", fontsize=5.8)
    for b_ in [2.5, 6.5]: ax.axvline(b_, color="white", lw=2.5)
    for x0, x1, lab in [(0, 2, "search families"), (3, 6, "held-out families"), (7, len(COLS) - 1, "stress tests (SR and EMP)")]:
        ax.text((x0 + x1) / 2, -0.75, lab, ha="center", va="bottom", fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("efficiency", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    return fig

if __name__ == "__main__":
    for name, fn in [("Fig1", fig1), ("Fig2", fig2), ("Fig3", fig3), ("Fig4", fig4), ("Fig5", fig5), ("EDFig1", edfig1), ("EDFig2", edfig2), ("EDFig3", edfig3)]:
        if len(sys.argv) > 3 and name not in sys.argv[3].split(","): continue
        f = fn(); f.savefig(os.path.join(OUT, name + ".pdf")); f.savefig(os.path.join(OUT, name + ".png"), dpi=300); plt.close(f)
    print("figures written to", OUT)
