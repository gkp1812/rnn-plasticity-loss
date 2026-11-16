#!/usr/bin/env python3
"""Build every data-dependent asset of the paper (tables, figures, numbers)
from the result JSON files.  Re-run after new result zips are unpacked.

usage: python3 -I make_assets.py <paper_dir> <results_root> [<results_root> ...]
Later roots override earlier ones when a run name is duplicated.
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = sys.argv[1]
ROOTS = sys.argv[2:]
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

ACTS = ["tanh", "relu", "leaky_relu", "elu", "silu", "gelu"]
ACT_LABEL = {"tanh": "tanh", "relu": "ReLU", "leaky_relu": "Leaky ReLU", "elu": "ELU", "silu": "SiLU", "gelu": "GELU"}
ARCHS = ["rnn", "lstm", "gru", "mlp"]
ARCH_LABEL = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "mlp": "MLP"}
LRS = [3e-4, 1e-3, 3e-3]
LR0 = 1e-3
KEY = {"perm": "online_acc", "rand": "final_acc"}
BENCH_LABEL = {"perm": "input-permutation", "rand": "label-reshuffling"}
REMS = ["l2init_all", "l2init_gates", "l2init_cand", "snp", "layernorm"]
REM_LABEL = {"none": "None", "l2init_all": "L2-Init", "l2init_gates": "L2-Init (gates)",
             "l2init_cand": "L2-Init (cand.)", "snp": "S\\&P", "layernorm": "LayerNorm"}
# categorical palette (fixed order = activation identity) + redundant line styles for print
COL = {"tanh": "#2a78d6", "relu": "#eb6834", "leaky_relu": "#1baf7a", "elu": "#eda100", "silu": "#e87ba4", "gelu": "#008300"}
LS = {"tanh": "-", "relu": "--", "leaky_relu": ":", "elu": "-.", "silu": (0, (5, 1)), "gelu": (0, (3, 1, 1, 1))}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.linewidth": 0.3, "grid.color": "#d9d9d6", "pdf.fonttype": 42})
W = 4.92  # 125 mm text width

# ------------------------------------------------------------------ load
RUNS = {}
for root in ROOTS:
    for f in sorted(glob.glob(os.path.join(root, "*", "results", "*.json"))):
        r = json.load(open(f))
        r.setdefault("remedy", "none")
        r.setdefault("dataset", "mnist")
        r.setdefault("tag", "")
        RUNS[r["name"]] = r
RUNS = list(RUNS.values())
G = collections.defaultdict(list)
for r in RUNS:
    G[(r["dataset"], r["tag"], r["bench"], r["mode"], r["arch"], r["act"], r["lr"], r["remedy"])].append(r)
for k in G:
    G[k].sort(key=lambda r: r["seed"])


def cell(bench, arch, act, lr=LR0, remedy="none", mode="cont", dataset="mnist", tag=""):
    return G.get((dataset, tag, bench, mode, arch, act, lr, remedy), [])


def ser(r, key):
    return np.array([t.get(key, np.nan) for t in r["log"]], float)


def lastk(r, key, k=10):
    return float(np.nanmean(ser(r, key)[-k:]))


def fresh(bench, arch, act, lr=LR0, key=None, dataset="mnist"):
    rs = cell(bench, arch, act, lr, mode="reset", dataset=dataset)
    if not rs:
        return float("nan")
    return float(np.concatenate([ser(r, key or KEY[bench]) for r in rs]).mean())


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m, run = len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def fp(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else "%.3f" % p


VALS = {}


def setv(key, val, fmt="%.3f"):
    VALS[key] = (fmt % val) if not isinstance(val, str) else val


def write(name, text):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def pending(what):
    return ("\\begin{center}\\fbox{\\parbox{0.9\\linewidth}{\\centering\\small\\textsc{Pending result.} "
            + what + "}}\\end{center}\n")


import re
_sess = []
for root in ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        hrs = []
        for lg in glob.glob(os.path.join(d, "logs", "*.log")):
            t = open(lg).read()
            v = [float(x) for x in re.findall(r"\(elapsed ([0-9.]+) h\)", t)] + [float(x) for x in re.findall(r"budget of ([0-9.]+) h reached", t)]
            if v:
                hrs.append(max(v))
        if hrs:
            _sess.append(max(hrs))
setv("n-sessions", str(len(_sess)))
setv("session-hours", "%.0f" % sum(_sess))
SEEDS_MAIN = sorted({r["seed"] for r in cell("perm", "rnn", "tanh")})
setv("n-seeds", str(len(SEEDS_MAIN)))
setv("n-runs", str(len(RUNS)))
setv("gpu-hours", "%.0f" % (sum(r.get("seconds", 0) for r in RUNS) / 3600.0))
setv("n-seeds-grid-other-lr", str(len(cell("perm", "rnn", "tanh", 3e-4))))
setv("n-seeds-remedy", str(len(cell("perm", "rnn", "tanh", remedy="l2init_all"))))
setv("n-reset-seeds", str(len(cell("perm", "rnn", "tanh", mode="reset"))))

# ------------------------------------------------------------------ Table: main results per benchmark
MAIN = {}
for bench in ["perm", "rand"]:
    key = KEY[bench]
    rows, p_loss, p_ni = [], [], []
    for arch in ARCHS:
        for act in ACTS:
            rs = cell(bench, arch, act)
            L = np.array([lastk(r, key) for r in rs])
            f0 = fresh(bench, arch, act)
            D = L - f0
            n = len(L)
            se = L.std(ddof=1) / math.sqrt(n)
            t_loss = D.mean() / se
            pl = stats.t.cdf(t_loss, n - 1)                       # H1: last < fresh
            pn = 1.0 - stats.t.cdf((D.mean() + 0.02) / se, n - 1)  # H1: last > fresh - 0.02
            rows.append((arch, act, f0, L.mean(), L.std(ddof=1), L.mean() / f0, D.mean()))
            p_loss.append(pl)
            p_ni.append(pn)
            MAIN[(bench, arch, act)] = {"fresh": f0, "last": L.mean(), "sd": L.std(ddof=1), "R": L.mean() / f0, "D": D.mean()}
            setv("%s-%s-%s-R" % (bench, arch, act), L.mean() / f0, "%.2f")
            setv("%s-%s-%s-last" % (bench, arch, act), L.mean())
            setv("%s-%s-%s-fresh" % (bench, arch, act), f0)
    p_loss, p_ni = holm(p_loss), holm(p_ni)
    out = ["\\begin{tabular}{llcccrcc}", "\\hline",
           "Arch. & Activation & Fresh & Last-10 (mean$\\pm$sd) & $R$ & $\\Delta$ & $p_{\\mathrm{loss}}$ & $p_{\\mathrm{NI}}$\\\\", "\\hline"]
    for i, (arch, act, f0, m, s, R, D) in enumerate(rows):
        first = ARCH_LABEL[arch] if act == ACTS[0] else ""
        out.append("%s & %s & %.3f & %.3f$\\pm$%.3f & %.2f & %+.3f & %s & %s\\\\" % (
            first, ACT_LABEL[act], f0, m, s, R, D, fp(p_loss[i]), fp(p_ni[i])))
        if act == ACTS[-1] and arch != ARCHS[-1]:
            out.append("\\hline")
        MAIN[(bench, arch, act)]["p_loss"] = p_loss[i]
        MAIN[(bench, arch, act)]["p_ni"] = p_ni[i]
    out += ["\\hline", "\\end{tabular}"]
    write("main_%s.tex" % bench, "\n".join(out) + "\n")
for bench in ["perm", "rand"]:
    for arch in ARCHS:
        Rs = [MAIN[(bench, arch, a)]["R"] for a in ACTS]
        setv("%s-%s-Rmin" % (bench, arch), min(Rs), "%.2f")
        setv("%s-%s-Rmax" % (bench, arch), max(Rs), "%.2f")
        setv("%s-%s-nloss" % (bench, arch), str(sum(MAIN[(bench, arch, a)]["p_loss"] < 0.05 for a in ACTS)))
        setv("%s-%s-nni" % (bench, arch), str(sum(MAIN[(bench, arch, a)]["p_ni"] < 0.05 for a in ACTS)))

# extra values used in the prose
setv("perm-rnn-tanh-losspct", "%.0f" % (100 * (1 - MAIN[("perm", "rnn", "tanh")]["R"])))
setv("perm-rnn-silu-losspct", "%.0f" % (100 * (1 - MAIN[("perm", "rnn", "silu")]["R"])))
rec = [MAIN[("rand", a, c)]["R"] for a in ["rnn", "lstm", "gru"] for c in ACTS]
setv("rand-rec-Rmin", min(rec), "%.2f"); setv("rand-rec-Rmax", max(rec), "%.2f")
setv("rand-rec-lossmin", "%.0f" % (100 * (1 - max(rec)))); setv("rand-rec-lossmax", "%.0f" % (100 * (1 - min(rec))))
gated = [MAIN[("perm", a, c)]["R"] for a in ["lstm", "gru"] for c in ACTS]
setv("perm-gated-Rmin", min(gated), "%.2f"); setv("perm-gated-Rmax", max(gated), "%.2f")
setv("perm-mlp-Rmin", min(MAIN[("perm", "mlp", c)]["R"] for c in ACTS), "%.2f")
setv("perm-mlp-Rmax", max(MAIN[("perm", "mlp", c)]["R"] for c in ACTS), "%.2f")
for bench in ["perm", "rand"]:
    for arch in ARCHS:
        for act in ACTS:
            rs = cell(bench, arch, act)
            for nm, kk in [("usat", "unit_sat_10"), ("gsat", "gate_sat_02"), ("dorm", "act_dormant"), ("erank", "erank"), ("wnorm", "wnorm")]:
                if kk == "gate_sat_02" and arch in ("rnn", "mlp"):
                    continue
                setv("mech-%s-%s-%s-%s" % (bench, arch, act, nm), np.mean([lastk(r, kk) for r in rs]), "%.2f" if nm not in ("erank", "wnorm") else "%.0f")
            setv("mech-%s-%s-%s-erank0" % (bench, arch, act), np.mean([ser(r, "erank")[0] for r in rs]), "%.0f")
            setv("mech-%s-%s-%s-deriv0" % (bench, arch, act), np.mean([ser(r, "deriv_mean")[0] for r in rs]), "%.2f")
            setv("mech-%s-%s-%s-deriv" % (bench, arch, act), np.mean([lastk(r, "deriv_mean") for r in rs]), "%.2f")
for arch in ARCHS:
    for act in ACTS:
        setv("perm-%s-%s-test-last" % (arch, act), np.mean([lastk(r, "test_acc") for r in cell("perm", arch, act)]))
        setv("perm-%s-%s-test-fresh" % (arch, act), fresh("perm", arch, act, key="test_acc"))
lasts = {a_: [MAIN[("perm", a_, c)]["last"] for c in ACTS] for a_ in ARCHS}
for a_ in ARCHS:
    setv("perm-%s-lastmin" % a_, min(lasts[a_])); setv("perm-%s-lastmax" % a_, max(lasts[a_]))
    setv("perm-%s-bestlast" % a_, ACT_LABEL[ACTS[int(np.argmax(lasts[a_]))]])
dvals = [v["d"] for k, v in REMD.items()] if "REMD" in globals() else []
for bench in ["perm", "rand"]:
    for arch in ARCHS:
        for act in ACTS:
            setv("%s-%s-%s-D" % (bench, arch, act), MAIN[(bench, arch, act)]["D"], "%+.3f")
setv("rand-rec-lastmax", max(MAIN[("rand", a_, c)]["last"] for a_ in ["rnn", "lstm", "gru"] for c in ACTS))
for arch, gates in [("lstm", ["i", "f", "o"]), ("gru", ["r", "z"])]:
    gs = [float(VALS["mech-rand-%s-%s-gsat" % (arch, c)]) for c in ACTS]
    setv("mech-rand-%s-gsat-min" % arch, min(gs), "%.2f"); setv("mech-rand-%s-gsat-max" % arch, max(gs), "%.2f")
    for g in gates:
        setv("mech-rand-%s-gate-%s" % (arch, g), np.mean([lastk(r, "gate_sat_02_" + g) for c in ACTS for r in cell("rand", arch, c)]), "%.2f")
setv("n-grid", str(sum(1 for r in RUNS if r["mode"] == "cont" and r["remedy"] == "none" and r["dataset"] == "mnist" and r["tag"] == "")))
setv("n-ref", str(sum(1 for r in RUNS if r["mode"] == "reset" and r["dataset"] == "mnist")))
setv("n-rem", str(sum(1 for r in RUNS if r["remedy"] != "none")))

# ------------------------------------------------------------------ Table: learning-rate sensitivity (retention)
out = ["\\begin{tabular}{ll|ccc|ccc}", "\\hline",
       " & & \\multicolumn{3}{c|}{Input-permutation} & \\multicolumn{3}{c}{Label-reshuffling}\\\\",
       "Arch. & Activation & $3{\\times}10^{-4}$ & $10^{-3}$ & $3{\\times}10^{-3}$ & $3{\\times}10^{-4}$ & $10^{-3}$ & $3{\\times}10^{-3}$\\\\", "\\hline"]
LRR = {}
for arch in ARCHS:
    for act in ACTS:
        vals = []
        for bench in ["perm", "rand"]:
            for lr in LRS:
                rs = cell(bench, arch, act, lr)
                R = np.mean([lastk(r, KEY[bench]) for r in rs]) / fresh(bench, arch, act, lr)
                LRR[(bench, arch, act, lr)] = R
                vals.append("%.2f" % R)
        out.append("%s & %s & %s\\\\" % (ARCH_LABEL[arch] if act == ACTS[0] else "", ACT_LABEL[act], " & ".join(vals)))
    if arch != ARCHS[-1]:
        out.append("\\hline")
out += ["\\hline", "\\end{tabular}"]
write("lr_retention.tex", "\n".join(out) + "\n")
for arch in ["lstm", "gru"]:
    v = [LRR[("perm", arch, a, 3e-3)] for a in ACTS]
    setv("perm-%s-lr3e3-Rmin" % arch, min(v), "%.2f")
    setv("perm-%s-lr3e3-Rmax" % arch, max(v), "%.2f")
setv("perm-rnn-tanh-lr3e3-R", LRR[("perm", "rnn", "tanh", 3e-3)], "%.2f")
setv("perm-rnn-tanh-lr3e4-R", LRR[("perm", "rnn", "tanh", 3e-4)], "%.2f")

# ------------------------------------------------------------------ Friedman / Nemenyi / Wilcoxon-Holm
QA = {6: 2.850}  # Nemenyi critical value q_0.05 for k = 6 (Demsar 2006)
fr_rows, wil_rows = [], []
for bench in ["perm", "rand"]:
    for arch in ARCHS:
        blocks = []
        for lr in LRS:
            seeds = sorted(set.intersection(*[{r["seed"] for r in cell(bench, arch, a, lr)} for a in ACTS]))
            for s in seeds:
                row = []
                for a in ACTS:
                    r = [x for x in cell(bench, arch, a, lr) if x["seed"] == s][0]
                    row.append(lastk(r, KEY[bench]) / fresh(bench, arch, a, lr))
                blocks.append(row)
        M = np.array(blocks)
        N, k = M.shape
        chi2, p = stats.friedmanchisquare(*M.T)
        ranks = stats.rankdata(-M, axis=1).mean(0)
        Wk = chi2 / (N * (k - 1))
        cd = QA[k] * math.sqrt(k * (k + 1) / (6.0 * N))
        fr_rows.append((bench, arch, N, chi2, p, Wk, cd, ranks))
        best = ACTS[int(np.argmin(ranks))]
        worst = ACTS[int(np.argmax(ranks))]
        setv("fried-%s-%s-best" % (bench, arch), ACT_LABEL[best])
        setv("fried-%s-%s-worst" % (bench, arch), ACT_LABEL[worst])
        setv("fried-%s-%s-W" % (bench, arch), Wk, "%.2f")
        ps = []
        for a in ACTS[1:]:
            d = M[:, ACTS.index(a)] - M[:, 0]
            ps.append(stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
        adj = holm(ps)
        med = [float(np.median(M[:, ACTS.index(a)] - M[:, 0])) for a in ACTS[1:]]
        wil_rows.append((bench, arch, adj, med))
        setv("fried-N", str(N))
        setv("fried-CD", cd, "%.2f")
setv("fried-nsig", {8: "all"}.get(sum(r[4] < 0.05 for r in fr_rows), str(sum(r[4] < 0.05 for r in fr_rows))))
setv("fried-Wmin", min(r[5] for r in fr_rows), "%.2f"); setv("fried-Wmax", max(r[5] for r in fr_rows), "%.2f")
FR = {(r[0], r[1]): r for r in fr_rows}
out = ["\\begin{tabular}{llrcc|cccccc}", "\\hline",
       "Benchmark & Arch. & $\\chi^2_F$ & $p$ & $W$ & tanh & ReLU & L-ReLU & ELU & SiLU & GELU\\\\", "\\hline"]
for bench, arch, N, chi2, p, Wk, cd, ranks in fr_rows:
    out.append("%s & %s & %.1f & %s & %.2f & %s\\\\" % (
        ("Permutation" if bench == "perm" else "Reshuffling") if arch == ARCHS[0] else "", ARCH_LABEL[arch], chi2, fp(p), Wk,
        " & ".join("%.2f" % x for x in ranks)))
    if arch == ARCHS[-1] and bench == "perm":
        out.append("\\hline")
out += ["\\hline", "\\end{tabular}"]
write("friedman.tex", "\n".join(out) + "\n")
out = ["\\begin{tabular}{ll|ccccc}", "\\hline",
       "Benchmark & Arch. & ReLU & Leaky ReLU & ELU & SiLU & GELU\\\\", "\\hline"]
for bench, arch, adj, med in wil_rows:
    out.append("%s & %s & %s\\\\" % (("Permutation" if bench == "perm" else "Reshuffling") if arch == ARCHS[0] else "", ARCH_LABEL[arch],
                                  " & ".join("%+.2f (%s)" % (m, fp(p)) for m, p in zip(med, adj))))
    if arch == ARCHS[-1] and bench == "perm":
        out.append("\\hline")
out += ["\\hline", "\\end{tabular}"]
write("wilcoxon.tex", "\n".join(out) + "\n")

# ------------------------------------------------------------------ Remedies
REMD = {}
for bench in ["perm", "rand"]:
    key = KEY[bench]
    out = ["\\begin{tabular}{ll|c|ccccc}", "\\hline",
           "Arch. & Act. & None & L2-Init & L2-Init (gates) & L2-Init (cand.) & S\\&P & LayerNorm\\\\", "\\hline"]
    for arch in ["rnn", "lstm", "gru"]:
        for act in ["tanh", "relu"]:
            seeds = sorted({r["seed"] for r in cell(bench, arch, act, remedy="l2init_all")})
            base = {r["seed"]: lastk(r, key) for r in cell(bench, arch, act) if r["seed"] in seeds}
            cols = ["%.3f" % np.mean(list(base.values()))]
            for rem in REMS:
                rs = cell(bench, arch, act, remedy=rem)
                if not rs:
                    cols.append("--")
                    continue
                v = {r["seed"]: lastk(r, key) for r in rs}
                d = np.array([v[s] - base[s] for s in sorted(v)])
                tt = stats.ttest_1samp(d, 0.0)
                REMD[(bench, arch, act, rem)] = {"acc": float(np.mean(list(v.values()))), "d": float(d.mean()),
                                                 "p": float(tt.pvalue), "min": float(d.min()), "n": len(d)}
                cols.append("%.3f (%+.3f)" % (np.mean(list(v.values())), d.mean()))
                setv("rem-%s-%s-%s-%s-d" % (bench, arch, act, rem), d.mean(), "%+.3f")
                setv("rem-%s-%s-%s-%s-acc" % (bench, arch, act, rem), float(np.mean(list(v.values()))))
            setv("rem-%s-%s-%s-none-acc" % (bench, arch, act), float(np.mean(list(base.values()))))
            out.append("%s & %s & %s\\\\" % (ARCH_LABEL[arch] if act == "tanh" else "", ACT_LABEL[act], " & ".join(cols)))
        if arch != "gru":
            out.append("\\hline")
    out += ["\\hline", "\\end{tabular}"]
    write("remedies_%s.tex" % bench, "\n".join(out) + "\n")
allp = [(k, v["p"]) for k, v in REMD.items()]
adj = holm([p for _, p in allp])
for (k, _), a in zip(allp, adj):
    REMD[k]["p_holm"] = float(a)
setv("rem-n-tests", str(len(allp)))
for bench in ["perm", "rand"]:
    for rem in ["l2init_all", "snp", "layernorm"]:
        ds = [v["d"] for k, v in REMD.items() if k[0] == bench and k[3] == rem]
        setv("rem-%s-%s-dmin" % (bench, rem), min(ds), "%+.3f"); setv("rem-%s-%s-dmax" % (bench, rem), max(ds), "%+.3f")
        setv("rem-%s-%s-npos" % (bench, rem), str(sum(d > 0 for d in ds)))
setv("rem-n-sig-raw", str(sum(v["p"] < 0.05 for v in REMD.values())))
setv("rem-n-sig-holm", str(sum(v["p_holm"] < 0.05 for v in REMD.values())))
setv("rem-n-positive", str(sum(v["min"] > 0 for v in REMD.values())))
setv("rem-n-negative", str(sum(v["d"] < 0 for v in REMD.values())))

# selective L2-Init diagnostics (label-reshuffling)
out = ["\\begin{tabular}{lll|cccc}", "\\hline",
       "Arch. & Act. & Regularised group & Gate sat. & Unit sat. & Eff. rank & Accuracy\\\\", "\\hline"]
for arch in ["lstm", "gru"]:
    for act in ["tanh", "relu"]:
        for rem in ["none", "l2init_gates", "l2init_cand", "l2init_all"]:
            seeds = sorted({r["seed"] for r in cell("rand", arch, act, remedy="l2init_all")})
            rs = [r for r in cell("rand", arch, act, remedy=rem) if r["seed"] in seeds]
            vals = [np.mean([lastk(r, kk) for r in rs]) for kk in ["gate_sat_02", "unit_sat_10", "erank", "final_acc"]]
            lab = {"none": "none", "l2init_gates": "gates", "l2init_cand": "candidate", "l2init_all": "all"}[rem]
            out.append("%s & %s & %s & %.2f & %.2f & %.1f & %.3f\\\\" % (
                ARCH_LABEL[arch] if (act == "tanh" and rem == "none") else "", ACT_LABEL[act] if rem == "none" else "", lab, *vals))
            for nm, v in zip(["gsat", "usat", "erank", "acc"], vals):
                setv("sel-%s-%s-%s-%s" % (arch, act, lab, nm), v, "%.2f" if nm != "erank" else "%.1f")
        out.append("\\hline")
out += ["\\end{tabular}"]
write("selective.tex", "\n".join(out) + "\n")

# ------------------------------------------------------------------ Correlations diagnostics vs retention (recurrent runs)
DIAG = [("unit_sat_10", "Unit saturation"), ("gate_sat_02", "Gate saturation"), ("act_dormant", "Dormant units"),
        ("erank", "Effective rank"), ("wnorm", "Weight norm")]
out = ["\\begin{tabular}{l|cc|cc}", "\\hline",
       " & \\multicolumn{2}{c|}{Input-permutation} & \\multicolumn{2}{c}{Label-reshuffling}\\\\",
       "Diagnostic (last 10 tasks) & $\\rho$ & $n$ & $\\rho$ & $n$\\\\", "\\hline"]
for dk, dl in DIAG:
    cols = []
    for bench in ["perm", "rand"]:
        xs, ys = [], []
        for arch in ["rnn", "lstm", "gru"]:
            if dk == "gate_sat_02" and arch == "rnn":
                continue
            for act in ACTS:
                for lr in LRS:
                    f0 = fresh(bench, arch, act, lr)
                    for r in cell(bench, arch, act, lr):
                        xs.append(lastk(r, dk))
                        ys.append(lastk(r, KEY[bench]) / f0)
        rho, p = stats.spearmanr(xs, ys)
        cols.append("%+.2f%s & %d" % (rho, "$^{*}$" if p < 0.001 else "", len(xs)))
        setv("rho-%s-%s" % (bench, dk), rho, "%+.2f")
    out.append("%s & %s\\\\" % (dl, " & ".join(cols)))
out += ["\\hline", "\\end{tabular}"]
write("correlations.tex", "\n".join(out) + "\n")

# ------------------------------------------------------------------ Model / compute table
out = ["\\begin{tabular}{lcccc}", "\\hline", " & RNN & LSTM & GRU & MLP\\\\", "\\hline"]
npar, mins = {}, {}
for arch in ARCHS:
    cand = [r for r in RUNS if r["arch"] == arch and r["remedy"] == "none" and "n_params" in r.get("config", {})]
    npar[arch] = cand[0]["config"]["n_params"] if cand else None
    tt = [r["seconds"] / 60.0 for r in RUNS if r["arch"] == arch and r["mode"] == "cont" and r["remedy"] == "none"
          and r["tag"] == "" and r["bench"] == "perm"]
    mins[arch] = np.mean(tt)
out.append("Hidden units & 128 & 128 & 128 & $3\\times256$\\\\")
out.append("Input per step & 28 & 28 & 28 & 784\\\\")
out.append("Sequence length & 28 & 28 & 28 & 1\\\\")
out.append("Trainable parameters & " + " & ".join("%s" % ("{:,}".format(npar[a]).replace(",", "\\,") if npar[a] else "--") for a in ARCHS) + "\\\\")
out.append("Minutes per run (200 tasks) & " + " & ".join("%.1f" % mins[a] for a in ARCHS) + "\\\\")
out += ["\\hline", "\\end{tabular}"]
write("models.tex", "\n".join(out) + "\n")

# ------------------------------------------------------------------ pending / optional assets (filled when the data exist)
fash = [r for r in RUNS if r["dataset"] == "fashion"]
if fash:
    fseeds = sorted({r["seed"] for r in fash if r["mode"] == "cont"})
    facts = [a for a in ACTS if cell("perm", "rnn", a, dataset="fashion")]
    out = ["\\begin{tabular}{ll|ccc|ccc}", "\\hline",
           " & & \\multicolumn{3}{c|}{Input-permutation} & \\multicolumn{3}{c}{Label-reshuffling}\\\\",
           "Arch. & Activation & Fresh & Last-10 & $R$ & Fresh & Last-10 & $R$\\\\", "\\hline"]
    for arch in ARCHS:
        for act in facts:
            cols = []
            for bench in ["perm", "rand"]:
                rs = cell(bench, arch, act, dataset="fashion")
                f0 = fresh(bench, arch, act, dataset="fashion")
                L = np.mean([lastk(r, KEY[bench]) for r in rs]) if rs else float("nan")
                cols += ["%.3f" % f0, "%.3f" % L, "%.2f" % (L / f0)]
                setv("fashion-%s-%s-%s-R" % (bench, arch, act), L / f0, "%.2f")
            out.append("%s & %s & %s\\\\" % (ARCH_LABEL[arch] if act == facts[0] else "", ACT_LABEL[act], " & ".join(cols)))
        if arch != ARCHS[-1]:
            out.append("\\hline")
    out += ["\\hline", "\\end{tabular}"]
    write("fashion.tex", "\n".join(out) + "\n")
    setv("fashion-seeds", str(len(fseeds)))
else:
    write("fashion.tex", pending("Fashion-MNIST replication (retention per architecture and activation) is produced by the final run."))

# ------------------------------------------------------------------ Figures
def smooth(a, w=5):
    if len(a) < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="valid")


def mean_curve(rs, key):
    n = min(len(r["log"]) for r in rs)
    return np.nanmean(np.array([ser(r, key)[:n] for r in rs]), 0)


def legend_bottom(fig, handles, ncol=6):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, handlelength=2.6, columnspacing=1.2)


def act_handles(acts=ACTS):
    return [plt.Line2D([0], [0], color=COL[a], ls=LS[a], label=ACT_LABEL[a]) for a in acts]


# Fig: activations and derivatives
x = np.linspace(-4, 4, 801)
def f_act(a, x):
    if a == "tanh": return np.tanh(x), 1 - np.tanh(x) ** 2
    if a == "relu": return np.maximum(x, 0), (x > 0).astype(float)
    if a == "leaky_relu": return np.where(x > 0, x, 0.01 * x), np.where(x > 0, 1.0, 0.01)
    if a == "elu": return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1), np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))
    s = 1 / (1 + np.exp(-x))
    if a == "silu": return x * s, s + x * s * (1 - s)
    from scipy.special import erf
    cdf = 0.5 * (1 + erf(x / np.sqrt(2))); pdf = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    return x * cdf, cdf + x * pdf
fig, ax = plt.subplots(1, 2, figsize=(W, 1.95))
for a in ACTS:
    y, d = f_act(a, x)
    ax[0].plot(x, y, color=COL[a], ls=LS[a]); ax[1].plot(x, d, color=COL[a], ls=LS[a])
ax[0].set_ylim(-1.3, 3); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$\\phi(x)$"); ax[0].set_title("(a) activation")
ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$\\phi'(x)$"); ax[1].set_title("(b) derivative")
fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.36, wspace=0.28)
legend_bottom(fig, act_handles())
fig.savefig(os.path.join(FIG, "activations.pdf")); plt.close(fig)

# Fig: learning curves per benchmark (4 architectures)
for bench, ylab in [("perm", "Online accuracy"), ("rand", "Final accuracy")]:
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.75), sharey=True)
    for ax, arch in zip(axs, ARCHS):
        for a in ACTS:
            c = smooth(mean_curve(cell(bench, arch, a), KEY[bench]))
            ax.plot(np.arange(len(c)) + 3, c, color=COL[a], ls=LS[a])
        ax.set_title(ARCH_LABEL[arch]); ax.set_xlabel("Task")
    axs[0].set_ylabel(ylab)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.4, wspace=0.08)
    legend_bottom(fig, act_handles())
    fig.savefig(os.path.join(FIG, "curves_%s.pdf" % bench)); plt.close(fig)

# Fig: retention vs learning rate
fig, axs = plt.subplots(2, 4, figsize=(W, 2.9), sharex=True, sharey="row")
for i, bench in enumerate(["perm", "rand"]):
    for j, arch in enumerate(ARCHS):
        ax = axs[i, j]
        for a in ACTS:
            ax.plot(range(3), [LRR[(bench, arch, a, lr)] for lr in LRS], color=COL[a], ls=LS[a], marker="o", ms=2.5)
        ax.axhline(1.0, color="#52514c", lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels(["0.3", "1", "3"]); ax.set_xlim(-0.25, 2.25)
        if i == 0: ax.set_title(ARCH_LABEL[arch])
        if i == 1: ax.set_xlabel("Learning rate ($\\times10^{-3}$)")
    axs[i, 0].set_ylabel("$R$, %s" % ("permutation" if bench == "perm" else "reshuffling"))
fig.subplots_adjust(left=0.1, right=0.995, top=0.93, bottom=0.25, wspace=0.08, hspace=0.12)
legend_bottom(fig, act_handles())
fig.savefig(os.path.join(FIG, "lr_sensitivity.pdf")); plt.close(fig)

# Fig: mechanism time courses (label-reshuffling)
METS = [("unit_sat_10", "Unit saturation"), ("gate_sat_02", "Gate saturation"), ("act_dormant", "Dormant units"), ("erank", "Effective rank")]
fig, axs = plt.subplots(4, 3, figsize=(W, 4.9), sharex=True, sharey="row")
for i, (mk, ml) in enumerate(METS):
    for j, arch in enumerate(["rnn", "lstm", "gru"]):
        ax = axs[i, j]
        if mk == "gate_sat_02" and arch == "rnn":
            ax.text(0.5, 0.5, "no gates", transform=ax.transAxes, ha="center", va="center", color="#52514c")
        else:
            for a in ACTS:
                ax.plot(np.arange(1, 101), mean_curve(cell("rand", arch, a), mk), color=COL[a], ls=LS[a])
        if i == 0: ax.set_title(ARCH_LABEL[arch])
        if i == 3: ax.set_xlabel("Task")
    axs[i, 0].set_ylabel(ml)
fig.subplots_adjust(left=0.105, right=0.995, top=0.96, bottom=0.145, wspace=0.08, hspace=0.14)
legend_bottom(fig, act_handles())
fig.savefig(os.path.join(FIG, "mechanisms.pdf")); plt.close(fig)

# Fig: weight norm growth on the permutation benchmark
fig, axs = plt.subplots(1, 4, figsize=(W, 1.75), sharey=False)
for ax, arch in zip(axs, ARCHS):
    for a in ACTS:
        ax.plot(np.arange(1, 201), mean_curve(cell("perm", arch, a), "wnorm"), color=COL[a], ls=LS[a])
    ax.set_title(ARCH_LABEL[arch]); ax.set_xlabel("Task")
axs[0].set_ylabel("Weight norm")
fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.4, wspace=0.3)
legend_bottom(fig, act_handles())
fig.savefig(os.path.join(FIG, "wnorm.pdf")); plt.close(fig)
for arch in ["lstm", "gru"]:
    c = mean_curve(cell("perm", arch, "tanh"), "wnorm")
    setv("wnorm-%s-tanh-first" % arch, c[0], "%.0f"); setv("wnorm-%s-tanh-last" % arch, c[-1], "%.0f")

# Fig: remedies (dot plot, change in accuracy vs no remedy)
RCOL = {"l2init_all": "#2a78d6", "l2init_gates": "#eb6834", "l2init_cand": "#1baf7a", "snp": "#eda100", "layernorm": "#e87ba4"}
RMK = {"l2init_all": "o", "l2init_gates": "s", "l2init_cand": "^", "snp": "D", "layernorm": "v"}
fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True)
cfgs = [(a, c) for a in ["rnn", "lstm", "gru"] for c in ["tanh", "relu"]]
for ax, bench in zip(axs, ["perm", "rand"]):
    for yi, (arch, act) in enumerate(cfgs):
        for rem in REMS:
            if (bench, arch, act, rem) in REMD:
                ax.plot(REMD[(bench, arch, act, rem)]["d"], yi, RMK[rem], color=RCOL[rem], ms=4.5, mec="white", mew=0.5)
    ax.axvline(0, color="#52514c", lw=0.6)
    ax.set_yticks(range(len(cfgs))); ax.set_yticklabels(["%s, %s" % (ARCH_LABEL[a], ACT_LABEL[c]) for a, c in cfgs])
    ax.set_xlabel("Change in last-10 accuracy")
    ax.set_title("(a) input-permutation" if bench == "perm" else "(b) label-reshuffling")
axs[0].invert_yaxis()
fig.subplots_adjust(left=0.17, right=0.99, top=0.9, bottom=0.31, wspace=0.08)
fig.legend(handles=[plt.Line2D([0], [0], color=RCOL[r], marker=RMK[r], ls="", label=REM_LABEL[r].replace("\\&", "&")) for r in REMS],
           loc="lower center", ncol=5, frameon=False, columnspacing=1.0, handletextpad=0.2)
fig.savefig(os.path.join(FIG, "remedies.pdf")); plt.close(fig)

# Fig: long horizon (only when the data exist)
longruns = [r for r in RUNS if r["tag"] == "long"]
if longruns:
    cfg = sorted({(r["arch"], r["act"]) for r in longruns}, key=lambda t: (ARCHS.index(t[0]), ACTS.index(t[1])))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0))
    lsty = {"rnn": ":", "lstm": "-", "gru": "--"}
    for arch, act in cfg:
        rs = cell("perm", arch, act, tag="long")
        c = smooth(mean_curve(rs, "online_acc"), 20)
        axs[0].plot(np.arange(len(c)) + 10, c, color=COL[act], ls=lsty[arch], label="%s, %s" % (ARCH_LABEL[arch], ACT_LABEL[act]))
        axs[1].plot(mean_curve(rs, "wnorm"), color=COL[act], ls=lsty[arch])
        f0 = fresh("perm", arch, act)
        L = np.mean([lastk(r, "online_acc", 20) for r in rs])
        setv("long-%s-%s-R" % (arch, act), L / f0, "%.2f")
        setv("long-%s-%s-last" % (arch, act), L)
    setv("long-tasks", str(min(len(r["log"]) for r in longruns)))
    setv("long-seeds", str(len({r["seed"] for r in longruns})))
    axs[0].set_xlabel("Task"); axs[0].set_ylabel("Online accuracy"); axs[0].set_title("(a) accuracy")
    axs[1].set_xlabel("Task"); axs[1].set_ylabel("Weight norm"); axs[1].set_title("(b) weight norm")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.89, bottom=0.36, wspace=0.28)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, "long_horizon.pdf")); plt.close(fig)
    write("long_horizon.tex", "\\includegraphics[width=\\linewidth]{figures/long_horizon.pdf}\n")
else:
    write("long_horizon.tex", pending("Accuracy and weight norm over the extended task sequence (LSTM, GRU and SiLU RNN) are produced by the final run."))

# Fig: dataset samples (only when the sample dump exists)
samp = None
for root in ROOTS:
    for f in glob.glob(os.path.join(root, "*", "samples.npz")):
        samp = f
if samp:
    z = np.load(samp)
    names = [n for n in ["mnist", "fashion"] if n + "_x" in z]
    fig, axs = plt.subplots(len(names) * 2, 8, figsize=(W, 0.62 * 2 * len(names) + 0.2))
    for i, n in enumerate(names):
        X, perm = z[n + "_x"], z["perm"]
        for j in range(8):
            a0, a1 = axs[2 * i, j], axs[2 * i + 1, j]
            a0.imshow(X[j].reshape(28, 28), cmap="gray_r"); a1.imshow(X[j].reshape(-1)[perm].reshape(28, 28), cmap="gray_r")
            for a in (a0, a1):
                a.set_xticks([]); a.set_yticks([]); a.grid(False)
                for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.4)
        axs[2 * i, 0].set_ylabel(n[:6], fontsize=6); axs[2 * i + 1, 0].set_ylabel("perm.", fontsize=6)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.99, bottom=0.01, wspace=0.05, hspace=0.08)
    fig.savefig(os.path.join(FIG, "samples.pdf")); plt.close(fig)
    write("samples.tex", "\\includegraphics[width=\\linewidth]{figures/samples.pdf}\n")
else:
    write("samples.tex", pending("Sample images (original rows and one pixel permutation) are exported by the final run."))

# leakage audit
audit = None
for root in ROOTS:
    for f in glob.glob(os.path.join(root, "*", "audit.json")):
        audit = json.load(open(f))
if audit:
    out = ["\\begin{tabular}{lrrrrr}", "\\hline",
           "Dataset & Train & Test & Dup. in train & Dup. in test & Train--test dup.\\\\", "\\hline"]
    for n in ["mnist", "fashion"]:
        if n in audit:
            a = audit[n]
            out.append("%s & %d & %d & %d & %d & %d\\\\" % ({"mnist": "MNIST", "fashion": "Fashion-MNIST"}[n], a["n_train"], a["n_test"],
                                                           a["dup_within_train"], a["dup_within_test"], a["dup_train_test"]))
            setv("audit-%s-cross" % n, str(a["dup_train_test"]))
    out += ["\\hline", "\\end{tabular}"]
    write("audit.tex", "\n".join(out) + "\n")
else:
    write("audit.tex", pending("Exact-duplicate counts within and between the training and test splits are computed by the final run."))

# ------------------------------------------------------------------ numbers file
with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
    for k in sorted(VALS):
        f.write("\\expandafter\\def\\csname val@%s\\endcsname{%s}\n" % (k, VALS[k]))
print("runs:", len(RUNS), " values:", len(VALS), " fashion:", len(fash), " long:", len(longruns), " samples:", bool(samp), " audit:", bool(audit))
for k in ["fried-N", "fried-CD", "rem-n-tests", "rem-n-sig-raw", "rem-n-sig-holm",
          "rem-n-positive", "rem-n-negative", "gpu-hours"]:
    print(" ", k, VALS[k])

# ------------------------------------------------------------------ qualitative claims made in the prose (must all be True)
def chk(name, cond):
    print("  CLAIM %-70s %s" % (name, "ok" if cond else "*** FAILS ***"))
print("claims check:")
for arch in ARCHS:
    rk = FR[("perm", arch)][7]
    chk("perm %s: tanh has worst mean rank" % arch, int(np.argmax(rk)) == 0)
for arch in ["rnn", "lstm", "gru"]:
    rk = FR[("perm", arch)][7]
    chk("perm %s: SiLU best rank and tanh-SiLU gap > CD" % arch, int(np.argmin(rk)) == 4 and rk[0] - rk[4] > FR[("perm", arch)][6])
    chk("rand %s: ELU has worst mean rank" % arch, int(np.argmax(FR[("rand", arch)][7])) == 3)
wl = {(b, a): adj for b, a, adj, med in wil_rows}
wm = {(b, a): med for b, a, adj, med in wil_rows}
chk("perm: ReLU, leaky ReLU, GELU > tanh significantly in every arch", all(wl[("perm", a)][i] < 0.05 and wm[("perm", a)][i] > 0 for a in ARCHS for i in (0, 1, 4)))
chk("rand: some significant in RNN and GRU, none in LSTM", any(p < 0.05 for p in wl[("rand", "rnn")]) and any(p < 0.05 for p in wl[("rand", "gru")]) and not any(p < 0.05 for p in wl[("rand", "lstm")]))
chk("rand MLP: every activation below tanh", all(m < 0 for m in wm[("rand", "mlp")]))
chk("perm: ELU highest last-10 in LSTM and GRU", VALS["perm-lstm-bestlast"] == "ELU" and VALS["perm-gru-bestlast"] == "ELU")
chk("perm: tanh lowest last-10 in RNN and MLP", all(MAIN[("perm", a, "tanh")]["last"] == min(MAIN[("perm", a, c)]["last"] for c in ACTS) for a in ["rnn", "mlp"]))
chk("rand: all 18 recurrent cells lose significantly", all(MAIN[("rand", a, c)]["p_loss"] < 0.05 for a in ["rnn", "lstm", "gru"] for c in ACTS))
chk("rand MLP: ReLU, ELU, GELU end near chance (<0.13)", all(MAIN[("rand", "mlp", c)]["last"] < 0.13 for c in ["relu", "elu", "gelu"]))
chk("rand LSTM: tanh has highest last-10", MAIN[("rand", "lstm", "tanh")]["last"] == max(MAIN[("rand", "lstm", c)]["last"] for c in ACTS))
chk("perm RNN: ELU lowest and GELU highest R among relu/leaky/elu/gelu", MAIN[("perm", "rnn", "elu")]["R"] <= min(MAIN[("perm", "rnn", c)]["R"] for c in ["relu", "leaky_relu", "gelu"]) and MAIN[("perm", "rnn", "gelu")]["R"] >= max(MAIN[("perm", "rnn", c)]["R"] for c in ["relu", "leaky_relu", "elu"]))
chk("perm LSTM tanh: significant small loss", MAIN[("perm", "lstm", "tanh")]["p_loss"] < 0.05)
chk("rand: S&P largest gain in all six configs", all(REMD[("rand", a, c, "snp")]["d"] == max(v["d"] for k, v in REMD.items() if k[:3] == ("rand", a, c)) for a in ["rnn", "lstm", "gru"] for c in ["tanh", "relu"]))
chk("rand: S&P LSTM tanh above fresh", REMD[("rand", "lstm", "tanh", "snp")]["acc"] > MAIN[("rand", "lstm", "tanh")]["fresh"])
chk("perm: L2-Init gated above fresh", all(REMD[("perm", a, c, "l2init_all")]["acc"] > MAIN[("perm", a, c)]["fresh"] for a in ["lstm", "gru"] for c in ["tanh", "relu"]))
chk("perm: LayerNorm <= ~0 for the three ReLU configs", all(REMD[("perm", a, "relu", "layernorm")]["d"] < 0.005 for a in ["rnn", "lstm", "gru"]))
chk("LSTM gate-sat minimum belongs to tanh", float(VALS["mech-rand-lstm-tanh-gsat"]) == float(VALS["mech-rand-lstm-gsat-min"]))
for a_, c_ in [("lstm", "elu"), ("lstm", "silu"), ("rnn", "elu")]:
    L = [lastk(r, "online_acc") for r in cell("perm", a_, c_, 3e-3)]
    print("  perm %s %s lr 3e-3 per-seed last-10: %s" % (a_, c_, " ".join("%.3f" % v for v in L)))
