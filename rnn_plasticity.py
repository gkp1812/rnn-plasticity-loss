#!/usr/bin/env python3
"""
rnn_plasticity.py (v2)  --  Loss of plasticity in recurrent networks.

Architectures : vanilla RNN, LSTM, GRU (hand-written single-layer cells so the
                candidate/hidden activation can be swapped) and an MLP baseline.
Activations   : tanh, relu, leaky_relu, elu, silu, gelu  (gates stay sigmoid).
Benchmarks    : perm = permuted row-wise MNIST (28 steps x 28 inputs), 1 pass per task
                rand = random-label memorisation on a fixed subset (trainability)
Modes         : cont  = one network trained on the whole task sequence
                reset = fresh network for every task (the "no plasticity loss" reference)
Per-task log  : online/final accuracy and loss, first-batch loss, test accuracy
                (perm), gradient norm, first-batch gradient norm per parameter
                group (each gate, candidate, output), saturated-unit fraction
                (mean |phi'| below threshold), per-gate saturation and mean gate
                value, activation-dormant fraction, effective rank, weight norms.
Every result file stores all hyper-parameters and the software/hardware used.

FINAL RUN (--plan final), one Kaggle session with 4 workers, jobs in priority order:
  0 Fashion-MNIST replication, seed 0 (tanh, ReLU, SiLU; all architectures; both
    benchmarks) with fresh-network references    1 extended sequence of 800 tasks, seed 0
  2 Fashion-MNIST seed 1    3 fifth seed (seed 4) of the MNIST grid at lr 1e-3
  4 extended sequence, seed 1    5 Fashion-MNIST seed 2
It also writes audit.json (exact-duplicate audit of the train/test splits) and samples.npz.
    !python rnn_plasticity.py --smoke --plan final --parallel 2
    !python rnn_plasticity.py --plan final --num_shards 4 --parallel 4 --shard 0 --hours 10.5

NIGHT 2 (--plan night2): rest of the grid (rand GRU/MLP), fresh-network references,
and remedies at lr 1e-3 (L2-init on all / gate / candidate weights, shrink-and-perturb,
layer normalisation).  Jobs are spread over the 12 worker slots by estimated run time:
    !python rnn_plasticity.py --smoke --plan night2 --parallel 2                 # code check
    !python rnn_plasticity.py --plan night2 --num_shards 12 --parallel 4 --shard 0   # account 1
    !python rnn_plasticity.py --plan night2 --num_shards 12 --parallel 4 --shard 4   # account 2
    !python rnn_plasticity.py --plan night2 --num_shards 12 --parallel 4 --shard 8   # account 3

MAIN GRID (night 1) = 3 learning rates x 4 seeds = 12 units, 4 units per Kaggle session
(--parallel 4 starts 4 processes, spread over the available GPUs):
    !python rnn_plasticity.py --smoke --parallel 2                               # code check
    !python rnn_plasticity.py --seeds 4 --seed_base 0 --num_shards 12 --parallel 4 --shard 0   # account 1
    !python rnn_plasticity.py --seeds 4 --seed_base 0 --num_shards 12 --parallel 4 --shard 4   # account 2
    !python rnn_plasticity.py --seeds 4 --seed_base 0 --num_shards 12 --parallel 4 --shard 8   # account 3
Unit u = lr_index * seeds + seed_index, so every (lr, seed) pair is run once.
A session stops by itself after --hours and writes plasticity_shards{a}-{b}of{n}.zip.
Continue an unfinished shard with  --resume_zip /kaggle/input/.../file.zip
Merge: unzip all zips into one folder, then
    python rnn_plasticity.py --summarize_only --out <folder>
"""
import argparse
import csv
import glob
import json
import math
import os
import platform
import subprocess
import sys
import time
import zipfile

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # --summarize_only works without torch
    torch = None
    nn = None
    F = None

VERSION = "v5-2026-11-17"
ALL_ACTS = ["tanh", "relu", "leaky_relu", "elu", "silu", "gelu"]
ALL_ARCHS = ["rnn", "lstm", "gru", "mlp"]
GATE_NAMES = {"rnn": [], "lstm": ["i", "f", "o"], "gru": ["r", "z"]}
GROUP_NAMES = {"rnn": ["h"], "lstm": ["i", "f", "g", "o"], "gru": ["r", "z", "n"]}


# --------------------------------------------------------------------------
# activations and their derivatives
# --------------------------------------------------------------------------
def act_fn(name, x):
    if name == "tanh":
        return torch.tanh(x)
    if name == "relu":
        return F.relu(x)
    if name == "leaky_relu":
        return F.leaky_relu(x, 0.01)
    if name == "elu":
        return F.elu(x)
    if name == "silu":
        return F.silu(x)
    if name == "gelu":
        return F.gelu(x)
    raise ValueError("unknown activation " + name)


def act_deriv(name, x):
    """Analytic derivative phi'(x); used only for diagnostics (no autograd)."""
    if name == "tanh":
        return 1.0 - torch.tanh(x) ** 2
    if name == "relu":
        return (x > 0).to(x.dtype)
    if name == "leaky_relu":
        return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, 0.01))
    if name == "elu":
        return torch.where(x > 0, torch.ones_like(x), torch.exp(torch.clamp(x, max=0.0)))
    if name == "silu":
        s = torch.sigmoid(x)
        return s + x * s * (1.0 - s)
    if name == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return cdf + x * pdf
    raise ValueError("unknown activation " + name)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
if torch is not None:

    class SeqNet(nn.Module):
        """Single-layer RNN / LSTM / GRU with a swappable candidate activation.
        LSTM: c = f*c + i*phi(g),  h = o*phi(c).   GRU: n = phi(Wx + r*(Uh))."""

        def __init__(self, arch, act, n_in=28, hid=128, n_out=10, ln=False):
            super().__init__()
            self.arch, self.act, self.hid, self.use_ln = arch, act, hid, ln
            ng = len(GROUP_NAMES[arch])
            if ln:  # layer norm on every pre-activation group (and on the LSTM cell)
                self.lns = nn.ModuleList([nn.LayerNorm(hid) for _ in range(ng)])
                if arch == "lstm":
                    self.ln_c = nn.LayerNorm(hid)
                    with torch.no_grad():
                        self.lns[1].bias.fill_(1.0)  # keep the forget-gate bias of 1
            self.wx = nn.Linear(n_in, ng * hid)
            self.wh = nn.Linear(hid, ng * hid, bias=(arch == "gru"))
            self.out = nn.Linear(hid, n_out)
            with torch.no_grad():
                nn.init.xavier_uniform_(self.wx.weight)
                self.wx.bias.zero_()
                for k in range(ng):  # orthogonal recurrent blocks
                    nn.init.orthogonal_(self.wh.weight[k * hid:(k + 1) * hid])
                if self.wh.bias is not None:
                    self.wh.bias.zero_()
                if arch == "lstm":  # chunk order: i, f, g, o -> forget bias = 1
                    self.wx.bias[hid:2 * hid].fill_(1.0)

        def forward(self, x, probe=None):
            # x: (B, T, n_in)
            B, T, _ = x.shape
            H = self.hid
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
            d_sum = a_sum = g_sum = gm_sum = None
            for t in range(T):
                a = self.wx(x[:, t])
                b = self.wh(h)
                gate_pre = None
                if self.arch == "rnn":
                    pre = a + b
                    if self.use_ln:
                        pre = self.lns[0](pre)
                    h = act_fn(self.act, pre)
                elif self.arch == "lstm":
                    zi, zf, pre, zo = (a + b).chunk(4, dim=1)
                    if self.use_ln:
                        zi, zf, pre, zo = self.lns[0](zi), self.lns[1](zf), self.lns[2](pre), self.lns[3](zo)
                    gate_pre = torch.cat([zi, zf, zo], dim=1)
                    i, f, o = torch.sigmoid(zi), torch.sigmoid(zf), torch.sigmoid(zo)
                    c = f * c + i * act_fn(self.act, pre)
                    h = o * act_fn(self.act, self.ln_c(c) if self.use_ln else c)
                else:  # gru
                    ar, az, an = a.chunk(3, dim=1)
                    br, bz, bn = b.chunk(3, dim=1)
                    zr, zz = ar + br, az + bz
                    if self.use_ln:
                        zr, zz = self.lns[0](zr), self.lns[1](zz)
                    gate_pre = torch.cat([zr, zz], dim=1)
                    r, z = torch.sigmoid(zr), torch.sigmoid(zz)
                    pre = an + r * bn
                    if self.use_ln:
                        pre = self.lns[2](pre)
                    h = (1.0 - z) * act_fn(self.act, pre) + z * h
                if probe is not None:
                    d = act_deriv(self.act, pre).abs().mean(0)
                    am = h.abs().mean(0)
                    d_sum = d if d_sum is None else d_sum + d
                    a_sum = am if a_sum is None else a_sum + am
                    if gate_pre is not None:
                        s = torch.sigmoid(gate_pre)
                        g = (s * (1.0 - s)).mean(0)
                        gm = s.mean(0)
                        g_sum = g if g_sum is None else g_sum + g
                        gm_sum = gm if gm_sum is None else gm_sum + gm
            if probe is not None:
                probe["d"] = d_sum / T
                probe["a"] = [a_sum / T]
                probe["g"] = None if g_sum is None else g_sum / T
                probe["gm"] = None if gm_sum is None else gm_sum / T
            return self.out(h), h

    class MLP(nn.Module):
        def __init__(self, act, n_in=784, hid=256, n_out=10, depth=3):
            super().__init__()
            self.act = act
            dims = [n_in] + [hid] * depth
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
            self.out = nn.Linear(hid, n_out)

        def forward(self, x, probe=None):
            h = x.reshape(x.shape[0], -1)
            ds, as_ = [], []
            for lin in self.layers:
                pre = lin(h)
                h = act_fn(self.act, pre)
                if probe is not None:
                    ds.append(act_deriv(self.act, pre).abs().mean(0))
                    as_.append(h.abs().mean(0))
            if probe is not None:
                probe["d"] = torch.cat(ds)
                probe["a"] = as_
                probe["g"] = None
                probe["gm"] = None
            return self.out(h), h


def build_model(arch, act, args, remedy="none"):
    if arch == "mlp":
        return MLP(act, hid=args.mlp_hid)
    return SeqNet(arch, act, hid=args.hid, ln=(remedy == "layernorm"))


# --------------------------------------------------------------------------
# remedies
# --------------------------------------------------------------------------
class L2Init:
    """Regenerative regularisation: lam * ||theta - theta_init||^2.
    scope 'all'   -> every parameter
    scope 'gates' -> only input/recurrent weights and biases of the sigmoid gates
    scope 'cand'  -> only input/recurrent weights and biases of the candidate"""

    def __init__(self, model, arch, scope, lam):
        self.lam, self.items = lam, []
        if scope == "all":
            for p in model.parameters():
                self.items.append((p, p.detach().clone(), None))
            return
        names = GROUP_NAMES[arch]
        cand = [k for k, nm in enumerate(names) if nm in ("g", "n", "h")]
        gates = [k for k in range(len(names)) if k not in cand]
        idxs = cand if scope == "cand" else gates
        H = model.hid
        for name, p in model.named_parameters():
            if name.split(".")[0] not in ("wx", "wh"):
                continue
            m = torch.zeros_like(p)
            for k in idxs:
                m[k * H:(k + 1) * H] = 1.0
            self.items.append((p, p.detach().clone(), m))

    def penalty(self):
        tot = 0.0
        for p, p0, m in self.items:
            d = (p - p0) ** 2
            tot = tot + (d.sum() if m is None else (d * m).sum())
        return self.lam * tot


def shrink_perturb(model, arch, act, args, remedy, p, seed, dev):
    """theta <- (1-p) * theta + p * (fresh random initialisation), at a task boundary."""
    torch.manual_seed(seed)
    fresh = build_model(arch, act, args, remedy).to(dev)
    with torch.no_grad():
        for q, f in zip(model.parameters(), fresh.parameters()):
            q.mul_(1.0 - p).add_(f, alpha=p)


def shape_input(arch, X):
    return X if arch == "mlp" else X.reshape(-1, 28, 28)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def effective_rank(h):
    try:
        h = h.float()
        if not bool(torch.isfinite(h).all()):
            return float("nan")
        s = torch.linalg.svdvals(h)
        p = s / (s.sum() + 1e-12)
        p = p[p > 0]
        return float(torch.exp(-(p * p.log()).sum()))
    except Exception:
        return float("nan")


def probe_metrics(model, arch, Xp):
    model.eval()
    P = {}
    with torch.no_grad():
        _, h = model(shape_input(arch, Xp), P)
        d = P["d"]
        out = {
            "deriv_mean": d.mean(),
            "unit_sat_02": (d < 0.02).float().mean(),   # mean |phi'| < 0.02
            "unit_sat_10": (d < 0.10).float().mean(),   # mean |phi'| < 0.10
        }
        scores = torch.cat([a / (a.mean() + 1e-12) for a in P["a"]])
        out["act_dormant"] = (scores <= 0.025).float().mean()
        if P["g"] is not None:
            g, gm = P["g"], P["gm"]                      # sigma(1-sigma), max 0.25
            out["gate_deriv_mean"] = g.mean()
            out["gate_sat_02"] = (g < 0.02).float().mean()
            H = model.hid
            for k, nm in enumerate(GATE_NAMES[arch]):
                gk, gmk = g[k * H:(k + 1) * H], gm[k * H:(k + 1) * H]
                out["gate_sat_02_" + nm] = (gk < 0.02).float().mean()
                out["gate_deriv_" + nm] = gk.mean()
                out["gate_mean_" + nm] = gmk.mean()
        out["wnorm"] = torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters()))
        if arch != "mlp":
            out["wnorm_rec"] = model.wh.weight.detach().norm()
        out = {k: float(v) for k, v in out.items()}
        out["erank"] = effective_rank(h)
    model.train()
    return out


def group_grad_norms(model, arch):
    """Gradient norm of each parameter group on the first batch of a task."""
    out = {}
    if arch == "mlp":
        for i, lin in enumerate(model.layers):
            out["g0_l%d" % i] = lin.weight.grad.norm()
    else:
        H = model.hid
        for k, nm in enumerate(GROUP_NAMES[arch]):
            gx = model.wx.weight.grad[k * H:(k + 1) * H]
            gh = model.wh.weight.grad[k * H:(k + 1) * H]
            out["g0_" + nm] = torch.sqrt((gx ** 2).sum() + (gh ** 2).sum())
    out["g0_out"] = model.out.weight.grad.norm()
    return {k: float(v) for k, v in out.items()}


def eval_acc(model, arch, Xe, ye, bs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, Xe.shape[0], bs):
            logits, _ = model(shape_input(arch, Xe[s:s + bs]))
            correct += int((logits.argmax(1) == ye[s:s + bs]).sum())
    model.train()
    return correct / max(1, Xe.shape[0])


# --------------------------------------------------------------------------
# data: MNIST (downloaded by torchvision; needs Internet ON in Kaggle)
# --------------------------------------------------------------------------
DATASETS = {"mnist": ("MNIST", 0.1307, 0.3081), "fashion": ("FashionMNIST", 0.2860, 0.3530)}


def raw_data(name, args):
    """uint8 pixel tensors (train_x, train_y, test_x, test_y); test_x may be None."""
    if args.smoke:
        g = torch.Generator().manual_seed(1 + list(DATASETS).index(name))
        X = torch.randint(0, 256, (2500, 784), generator=g).to(torch.uint8)
        y = torch.randint(0, 10, (2500,), generator=g)
        Xt = torch.randint(0, 256, (500, 784), generator=g).to(torch.uint8)
        yt = torch.randint(0, 10, (500,), generator=g)
        return X, y, Xt, yt
    try:
        from torchvision import datasets
        cls = getattr(datasets, DATASETS[name][0])
        tr = cls(args.data_root, train=True, download=True)
        te = cls(args.data_root, train=False, download=True)
        return (tr.data.reshape(-1, 784), tr.targets.long(), te.data.reshape(-1, 784), te.targets.long())
    except Exception as e:
        print("[data] torchvision download of %s failed (%s)" % (name, e), flush=True)
        if name != "mnist":
            sys.exit("Dataset %s not available. Turn Internet ON in the notebook settings." % name)
    import numpy as np
    for f in glob.glob("/kaggle/input/**/mnist.npz", recursive=True):
        z = np.load(f)
        return (torch.from_numpy(z["x_train"]).reshape(-1, 784), torch.from_numpy(z["y_train"]).long(),
                torch.from_numpy(z["x_test"]).reshape(-1, 784), torch.from_numpy(z["y_test"]).long())
    for f in glob.glob("/kaggle/input/**/mnist_train.csv", recursive=True):
        import pandas as pd
        arr = pd.read_csv(f).values
        return torch.from_numpy(arr[:, 1:]).to(torch.uint8), torch.from_numpy(arr[:, 0]).long(), None, None
    sys.exit("MNIST not found. Turn Internet ON in the notebook settings, "
             "or attach a dataset containing mnist.npz / mnist_train.csv.")


def load_data(name, args, dev):
    X, y, Xte, yte = raw_data(name, args)
    X = X.float() / 255.0
    if Xte is None:  # hold out the last part of the training set as test data
        n_te = max(500, X.shape[0] // 6)
        Xte, yte = X[-n_te:], y[-n_te:]
        X, y = X[:-n_te], y[:-n_te]
        print("[data] no official test split: holding out %d training examples" % n_te, flush=True)
    else:
        Xte = Xte.float() / 255.0
    mean, std = DATASETS[name][1], DATASETS[name][2]
    X, Xte = (X - mean) / std, (Xte - mean) / std
    Xte, yte = Xte[:args.test_n], yte[:args.test_n]
    print("[data] %s train %s  test %s" % (name, tuple(X.shape), tuple(Xte.shape)), flush=True)
    return X.to(dev), y.to(dev), Xte.to(dev), yte.to(dev)


def audit_and_samples(args, names):
    """Exact-duplicate audit of the train/test splits and a few sample images."""
    import hashlib
    import numpy as np
    audit, samples = {}, {}
    for name in names:
        Xtr, ytr, Xte, yte = raw_data(name, args)
        if Xte is None:
            continue
        a, b = Xtr.numpy().astype(np.uint8), Xte.numpy().astype(np.uint8)
        ha = [hashlib.md5(r.tobytes()).hexdigest() for r in a]
        hb = [hashlib.md5(r.tobytes()).hexdigest() for r in b]
        sa, sb = set(ha), set(hb)
        audit[name] = {"n_train": len(ha), "n_test": len(hb),
                       "dup_within_train": len(ha) - len(sa), "dup_within_test": len(hb) - len(sb),
                       "dup_train_test": sum(1 for h in hb if h in sa),
                       "n_test_used": min(args.test_n, len(hb)),
                       "dup_train_test_used": sum(1 for h in hb[:args.test_n] if h in sa)}
        yy = ytr.numpy()
        idx = [int(np.argmax(yy == c)) for c in range(10) if (yy == c).any()]
        samples[name + "_x"] = a[idx]
        samples[name + "_y"] = yy[idx]
    samples["perm"] = torch.randperm(784, generator=torch.Generator().manual_seed(0)).numpy()  # seed 0, task 0
    with open(os.path.join(args.out, "audit.json"), "w") as f:
        json.dump(audit, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "samples.npz"), **samples)
    print("[audit] %s" % json.dumps(audit), flush=True)


# --------------------------------------------------------------------------
# one job = one (bench, mode, arch, act, lr, seed) run over the task sequence
# --------------------------------------------------------------------------
def job_name(j):
    base = "%s_%s_%s_%s_lr%g_s%d" % (j["bench"], j["mode"], j["arch"], j["act"], j["lr"], j["seed"])
    if j.get("dataset", "mnist") != "mnist":
        base += "_" + j["dataset"]
    if j.get("tasks"):
        base += "_T%d" % j["tasks"]
    rem = j.get("remedy", "none")
    return base if rem == "none" else base + "_" + rem


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path, dev):
    try:
        return torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=dev)


def env_info(dev):
    info = {"script": VERSION, "torch": torch.__version__, "python": platform.python_version(),
            "device": str(dev)}
    if dev.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_job(job, data, args, dev, deadline):
    X, y, Xte, yte = data
    name = job_name(job)
    res_path = os.path.join(args.out, "results", name + ".json")
    ck_path = os.path.join(args.out, "ckpt", name + ".pt")
    if os.path.exists(res_path):
        return "skip"
    bench, mode, arch, act = job["bench"], job["mode"], job["arch"], job["act"]
    lr, seed = job["lr"], job["seed"]
    remedy = job.get("remedy", "none")
    torch.manual_seed(seed)
    model = build_model(arch, act, args, remedy).to(dev)
    reg = None
    if remedy.startswith("l2init_"):  # reference point = the initial weights (before any resume)
        reg = L2Init(model, arch, remedy.split("_", 1)[1], args.l2init_lam)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    log, start, spent = [], 0, 0.0
    if os.path.exists(ck_path):
        ck = load_ckpt(ck_path, dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        log, start, spent = ck["log"], ck["task"], ck.get("spent", 0.0)
        print("  resumed %s at task %d" % (name, start), flush=True)

    N = X.shape[0]
    if mode == "reset":
        n_tasks = args.reset_tasks
    else:
        n_tasks = args.perm_tasks if bench == "perm" else args.rand_tasks
        if job.get("tasks"):
            n_tasks = int(job["tasks"])
    if bench == "rand":
        g0 = torch.Generator().manual_seed(seed * 7919 + 13)
        n_fix = min(args.rand_n, N)
        Xfix = X[torch.randperm(N, generator=g0)[:n_fix].to(dev)]
    t0 = time.time()
    diverged = False
    model.train()
    for task in range(start, n_tasks):
        if time.time() > deadline:
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log,
                         "task": task, "spent": spent + time.time() - t0}, ck_path)
            return "timeout"
        if mode == "reset":  # fresh network and optimiser for every task
            torch.manual_seed(seed * 1000 + task + 1)
            model = build_model(arch, act, args, remedy).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
        if remedy == "snp" and task > 0:
            shrink_perturb(model, arch, act, args, remedy, args.snp_p, seed * 1000 + task + 500000, dev)
        g = torch.Generator().manual_seed(seed * 100003 + task)
        perm = None
        if bench == "perm":
            perm = torch.randperm(784, generator=g).to(dev)
            idx = torch.randperm(N, generator=g)[:min(args.perm_n, N)].to(dev)
            Xt, yt, epochs = X[idx][:, perm], y[idx], 1
        else:
            Xt = Xfix
            yt = torch.randint(0, 10, (Xt.shape[0],), generator=g).to(dev)
            epochs = args.rand_epochs
        accs, losses, gns = [], [], []
        g0_stats = None
        for _ in range(epochs):
            order = torch.randperm(Xt.shape[0], generator=g).to(dev)
            for s in range(0, Xt.shape[0], args.bs):
                b = order[s:s + args.bs]
                logits, _ = model(shape_input(arch, Xt[b]))
                loss = F.cross_entropy(logits, yt[b])   # logged loss = cross-entropy only
                total = loss if reg is None else loss + reg.penalty()
                opt.zero_grad(set_to_none=True)
                total.backward()
                if g0_stats is None:
                    g0_stats = group_grad_norms(model, arch)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                accs.append((logits.argmax(1) == yt[b]).float().mean().detach())
                losses.append(loss.detach())
                gns.append(gn.detach() if torch.is_tensor(gn) else torch.tensor(float(gn)))
        A, L, G = torch.stack(accs), torch.stack(losses), torch.stack(gns)
        tail = max(1, len(A) // 10)
        rec = {"task": task,
               "online_acc": float(A.mean()), "final_acc": float(A[-tail:].mean()),
               "online_loss": float(L.mean()), "final_loss": float(L[-tail:].mean()),
               "loss0": float(L[0]), "grad_norm": float(G.mean())}
        rec.update(g0_stats)
        if not math.isfinite(rec["online_loss"]):
            diverged = True
            rec["diverged"] = True
            log.append(rec)
            print("  %s diverged at task %d" % (name, task), flush=True)
            break
        rec.update(probe_metrics(model, arch, Xt[:args.probe_n]))
        if perm is not None:
            rec["test_acc"] = eval_acc(model, arch, Xte[:, perm], yte)
        log.append(rec)
        if (task + 1) % args.print_every == 0 or task == 0:
            print("  %s task %4d  acc %.3f  sat %.3f  gate_sat %s  erank %.1f"
                  % (name, task + 1, rec["online_acc"], rec["unit_sat_02"],
                     ("%.3f" % rec["gate_sat_02"]) if "gate_sat_02" in rec else "-",
                     rec["erank"]), flush=True)
        if (task + 1) % args.ck_every == 0:
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log,
                         "task": task + 1, "spent": spent + time.time() - t0}, ck_path)
    result = dict(job)
    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"optimizer": "Adam(betas=0.9,0.999, eps=1e-8)", "loss": "cross_entropy",
                "n_params": n_params, "n_tasks": n_tasks, "remedy": remedy, "dataset": job.get("dataset", "mnist"), "seq_len": 28, "input_dim": 28,
                "init": "xavier input, orthogonal recurrent, zero bias, LSTM forget bias 1"
                        if arch != "mlp" else "PyTorch default Linear init",
                "normalisation": "mean %.4f, std %.4f" % DATASETS[job.get("dataset", "mnist")][1:]})
    result.update({"name": name, "diverged": diverged, "log": log,
                   "seconds": spent + time.time() - t0, "config": cfg, "env": env_info(dev)})
    tmp = res_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, res_path)
    if os.path.exists(ck_path):
        os.remove(ck_path)
    return "done"


# --------------------------------------------------------------------------
# summary, plots, zip  (no torch needed)
# --------------------------------------------------------------------------
def _mean(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(v) / len(v) if v else float("nan")


def summarize(out):
    files = sorted(glob.glob(os.path.join(out, "results", "*.json")))
    rows, curves = [], {}
    for fpath in files:
        with open(fpath) as f:
            r = json.load(f)
        log = [t for t in r["log"] if "erank" in t]
        if not log:
            continue
        k = min(5, len(log))

        def fl(key):
            a = [t.get(key) for t in log]
            return round(_mean(a[:k]), 4), round(_mean(a[-k:]), 4)
        row = {"bench": r["bench"], "mode": r.get("mode", "cont"), "arch": r["arch"],
               "act": r["act"], "remedy": r.get("remedy", "none"), "lr": r.get("lr"), "seed": r["seed"],
               "tasks_done": len(log), "diverged": int(bool(r.get("diverged"))),
               "acc_all": round(_mean([t["online_acc"] for t in log]), 4)}
        for key, short in [("online_acc", "acc"), ("final_acc", "final_acc"), ("test_acc", "test_acc"),
                           ("unit_sat_02", "unit_sat"), ("gate_sat_02", "gate_sat"),
                           ("erank", "erank"), ("wnorm", "wnorm"), ("loss0", "loss0")]:
            row[short + "_first5"], row[short + "_last5"] = fl(key)
        row["acc_drop"] = round(row["acc_first5"] - row["acc_last5"], 4)
        row["minutes"] = round(r.get("seconds", 0) / 60.0, 1)
        rows.append(row)
        if row["mode"] == "cont":
            label = r["act"] if row["remedy"] == "none" else r["act"] + "+" + row["remedy"]
            curves.setdefault((r["bench"], r["arch"], row["lr"]), {}).setdefault(label, []).append(
                [t["online_acc"] for t in log])
    if not rows:
        print("[summary] no finished jobs yet")
        return
    rows.sort(key=lambda d: (d["bench"], d["mode"], d["arch"], d["act"], d["remedy"], d["lr"] or 0, d["seed"]))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[summary] %d finished jobs -> summary.csv" % len(rows))
    fresh, agg = {}, {}
    for d in rows:
        key = (d["bench"], d["arch"], d["act"], d["lr"])
        if d["mode"] == "reset":
            fresh.setdefault(key, []).append(d["acc_all"])
        else:
            agg.setdefault(key + (d["remedy"],), []).append(d)
    print("%-5s %-5s %-11s %-13s %-7s %3s %9s %9s %8s %9s" % (
        "bench", "arch", "act", "remedy", "lr", "n", "acc_first", "acc_last", "drop", "fresh_net"))
    for key in sorted(agg, key=lambda t: (t[0], t[1], t[2], t[4], t[3] or 0)):
        ds = agg[key]
        print("%-5s %-5s %-11s %-13s %-7g %3d %9.3f %9.3f %8.3f %9.3f" % (
            key[0], key[1], key[2], key[4], key[3] or 0, len(ds), _mean([d["acc_first5"] for d in ds]),
            _mean([d["acc_last5"] for d in ds]), _mean([d["acc_drop"] for d in ds]),
            _mean(fresh.get(key[:4], []))))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(out, "plots"), exist_ok=True)
        for (bench, arch, lr), by_act in sorted(curves.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
            plt.figure(figsize=(7, 4))
            for act in sorted(by_act):
                runs = by_act[act]
                n = min(len(c) for c in runs)
                mean = [sum(c[i] for c in runs) / len(runs) for i in range(n)]
                plt.plot(range(1, n + 1), mean, label="%s (n=%d)" % (act, len(runs)))
            plt.xlabel("task")
            plt.ylabel("online accuracy within task")
            plt.title("%s / %s / lr %g" % (bench, arch, lr or 0))
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(out, "plots", "%s_%s_lr%g.png" % (bench, arch, lr or 0)), dpi=120)
            plt.close()
    except Exception as e:
        print("[summary] plots skipped: %s" % e)


def make_zip(out, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(out):
            for fn in fs:
                if fn.endswith(".tmp"):
                    continue
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, out))
    print("[zip] wrote %s (%.1f MB)" % (zip_path, os.path.getsize(zip_path) / 1e6), flush=True)


# --------------------------------------------------------------------------
# job list
# --------------------------------------------------------------------------
def make_jobs(args):
    """Continual runs first, fresh-network references last.  A shard owns whole
    (lr, seed) units, so every shard holds complete activation comparisons."""
    lr_map = None
    if args.lr_map:
        with open(args.lr_map) as f:
            lr_map = json.load(f)
    jobs = []
    for mode in args.modes:
        for bench in args.benches:
            for arch in args.archs:
                for act in args.acts:
                    if lr_map is not None:
                        key = "%s|%s|%s" % (bench, arch, act)
                        if key not in lr_map:
                            sys.exit("lr_map has no entry for " + key)
                        lrs = [float(lr_map[key])]
                    else:
                        lrs = args.lrs
                    for li, lr in enumerate(lrs):
                        for si in range(args.seeds):
                            unit = li * args.seeds + si
                            if unit % args.num_shards == args.shard:
                                jobs.append({"bench": bench, "mode": mode, "arch": arch, "act": act,
                                             "lr": lr, "seed": args.seed_base + si})
    return jobs


# measured on Kaggle T4 with 4 workers per session: minutes per continual job
EST_MIN = {"rnn": 12.0, "lstm": 25.0, "gru": 27.0, "mlp": 2.0}
REMEDIES = {"rnn": ["l2init_all", "snp", "layernorm"],
            "lstm": ["l2init_all", "l2init_gates", "l2init_cand", "snp", "layernorm"],
            "gru": ["l2init_all", "l2init_gates", "l2init_cand", "snp", "layernorm"]}


def est_minutes(j, args):
    m = EST_MIN[j["arch"]]
    if j["mode"] == "reset":
        full = args.perm_tasks if j["bench"] == "perm" else args.rand_tasks
        m = m * args.reset_tasks / float(max(1, full))
    elif j.get("tasks"):
        m = m * j["tasks"] / float(args.perm_tasks if j["bench"] == "perm" else args.rand_tasks)
    if j.get("remedy") == "layernorm":
        m *= 1.3
    return m


def plan_final(args):
    """Priority list for the single final session (see the top of the file)."""
    f_acts, f_seeds = ["tanh", "relu", "silu"], [0, 1, 2]
    long_cfg = [("lstm", "tanh"), ("lstm", "relu"), ("gru", "tanh"), ("gru", "relu"), ("rnn", "silu")]
    long_seeds, long_tasks, g_acts, g_seed, lr = [0, 1], 800, ALL_ACTS, 4, 1e-3
    if args.smoke:
        f_acts, f_seeds, long_cfg, long_seeds, long_tasks, g_acts = ["tanh"], [0], [("lstm", "tanh")], [0], 4, ["tanh"]
    items = []

    def fashion(pri, seed, mode):
        for bench in ["perm", "rand"]:
            for arch in ALL_ARCHS:
                for act in f_acts:
                    items.append((pri, {"bench": bench, "mode": mode, "arch": arch, "act": act, "lr": lr,
                                        "seed": seed, "remedy": "none", "dataset": "fashion"}))

    def long_runs(pri, seed):
        for arch, act in long_cfg:
            items.append((pri, {"bench": "perm", "mode": "cont", "arch": arch, "act": act, "lr": lr,
                                "seed": seed, "remedy": "none", "tasks": long_tasks, "tag": "long"}))

    fashion(0, f_seeds[0], "cont")
    fashion(0, 0, "reset")
    long_runs(1, long_seeds[0])
    if len(f_seeds) > 1:
        fashion(2, f_seeds[1], "cont")
    for bench in ["perm", "rand"]:
        for arch in ALL_ARCHS:
            for act in g_acts:
                items.append((3, {"bench": bench, "mode": "cont", "arch": arch, "act": act, "lr": lr,
                                  "seed": g_seed, "remedy": "none"}))
    if len(long_seeds) > 1:
        long_runs(4, long_seeds[1])
    if len(f_seeds) > 2:
        fashion(5, f_seeds[2], "cont")
    return items


def plan_night2(args):
    """(priority, job) list.  0 = fresh-network references and the rand GRU/MLP runs
    at lr 1e-3; 1-3 = remedies, seed by seed; 4 = rand GRU/MLP at the other lrs."""
    lrs, acts, grid_seeds, reset_seeds = [3e-4, 1e-3, 3e-3], ALL_ACTS, [0, 1, 2, 3], [0, 1]
    b_acts, b_seeds, b_lr = ["tanh", "relu"], [0, 1, 2], 1e-3
    if args.smoke:
        lrs, acts, grid_seeds, reset_seeds, b_acts, b_seeds = [1e-3], ["tanh"], [0], [0], ["tanh"], [0]
    items = []
    for lr in lrs:
        for seed in grid_seeds:
            for act in acts:
                for arch in ["gru", "mlp"]:
                    items.append((0 if lr == b_lr else 4,
                                  {"bench": "rand", "mode": "cont", "arch": arch, "act": act,
                                   "lr": lr, "seed": seed, "remedy": "none"}))
        for seed in reset_seeds:
            for bench in ["perm", "rand"]:
                for arch in ALL_ARCHS:
                    for act in acts:
                        items.append((0, {"bench": bench, "mode": "reset", "arch": arch, "act": act,
                                          "lr": lr, "seed": seed, "remedy": "none"}))
    for si, seed in enumerate(b_seeds):
        for bench in ["perm", "rand"]:
            for arch in ["rnn", "lstm", "gru"]:
                for act in b_acts:
                    for rem in REMEDIES[arch]:
                        items.append((1 + si, {"bench": bench, "mode": "cont", "arch": arch, "act": act,
                                               "lr": b_lr, "seed": seed, "remedy": rem}))
    return items


def plan_jobs(args, shard):
    """Greedy balancing by estimated minutes, priority class by priority class."""
    items = plan_final(args) if args.plan == "final" else plan_night2(args)
    loads = [0.0] * args.num_shards
    out = []
    for pri in sorted({p for p, _ in items}):
        cls = sorted([j for p, j in items if p == pri], key=lambda j: (-est_minutes(j, args), job_name(j)))
        for j in cls:
            k = min(range(args.num_shards), key=lambda i: (loads[i], i))
            loads[k] += est_minutes(j, args)
            if k == shard:
                out.append(j)
    return out


def jobs_for_shard(args, shard):
    if args.plan in ("night2", "final"):
        return plan_jobs(args, shard)
    keep = args.shard
    args.shard = shard
    jobs = make_jobs(args)
    args.shard = keep
    return jobs


def needed_datasets(args, shards):
    return sorted({j.get("dataset", "mnist") for sh in shards for j in jobs_for_shard(args, sh)})


def prefetch(args, names):
    """Download the data once in the parent so parallel workers do not race."""
    for name in names:
        try:
            raw_data(name, args)
        except SystemExit as e:
            print("[data] prefetch of %s failed (%s)" % (name, e), flush=True)


def launch_parallel(args, argv, script, n_gpus, poll=30, report=900):
    """Start one worker per shard (argparse keeps the last value of a repeated
    option, so the appended --shard/--parallel override the parent's)."""
    logdir = os.path.join(args.out, "logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    for k in range(args.parallel):
        shard = args.shard + k
        env = dict(os.environ)
        if n_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(k % n_gpus)
        env["OMP_NUM_THREADS"] = "1"
        logpath = os.path.join(logdir, "shard%d.log" % shard)
        logf = open(logpath, "a")
        cmd = [sys.executable, script] + list(argv) + ["--shard", str(shard), "--parallel", "1", "--child"]
        procs.append((shard, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env), logf, logpath))
        print("[launch] shard %d on GPU %s -> %s" % (shard, env.get("CUDA_VISIBLE_DEVICES", "-"), logpath), flush=True)
    t0, last = time.time(), time.time()
    while any(p.poll() is None for _, p, _, _ in procs):
        time.sleep(poll)
        if time.time() - last >= report:
            last = time.time()
            n = len(glob.glob(os.path.join(args.out, "results", "*.json")))
            print("[progress] %.2f h  %d result files" % ((time.time() - t0) / 3600.0, n), flush=True)
    for shard, p, logf, logpath in procs:
        logf.close()
        with open(logpath) as f:
            tail = f.readlines()[-(25 if p.returncode else 3):]
        print("[worker] shard %d exit code %s; log tail:" % (shard, p.returncode), flush=True)
        print("    " + "    ".join(tail), flush=True)


def finalize(args, shards, zip_path):
    summarize(args.out)
    make_zip(args.out, zip_path)
    total, left = 0, 0
    for s in shards:
        for j in jobs_for_shard(args, s):
            total += 1
            left += not os.path.exists(os.path.join(args.out, "results", job_name(j) + ".json"))
    print("[end] %d of %d jobs finished, %d still open" % (total - left, total, left), flush=True)
    if args.smoke:
        print("SMOKE TEST OK" if left == 0 else "SMOKE TEST INCOMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default=",".join(ALL_ARCHS))
    ap.add_argument("--acts", default=",".join(ALL_ACTS))
    ap.add_argument("--benches", default="perm,rand")
    ap.add_argument("--modes", default="cont,reset")
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3")
    ap.add_argument("--lr_map", default="", help='JSON {"bench|arch|act": lr}; overrides --lrs')
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed_base", type=int, default=100, help="100 = tuning seed; use 0 for final runs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1,
                    help="run shards shard..shard+parallel-1 as parallel processes in this session")
    ap.add_argument("--child", action="store_true", help="internal: worker started by --parallel")
    ap.add_argument("--plan", default="", help="'final' or 'night2' = predefined job list (see top of file)")
    ap.add_argument("--l2init_lam", type=float, default=1e-3)
    ap.add_argument("--snp_p", type=float, default=0.1)
    ap.add_argument("--hours", type=float, default=11.0, help="stop cleanly after this many hours")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--mlp_hid", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--perm_tasks", type=int, default=200)
    ap.add_argument("--perm_n", type=int, default=12800)
    ap.add_argument("--rand_tasks", type=int, default=100)
    ap.add_argument("--rand_n", type=int, default=1280)
    ap.add_argument("--rand_epochs", type=int, default=20)
    ap.add_argument("--reset_tasks", type=int, default=20)
    ap.add_argument("--probe_n", type=int, default=512)
    ap.add_argument("--test_n", type=int, default=2000)
    ap.add_argument("--ck_every", type=int, default=10)
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--out", default="plasticity_out")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume_zip", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to check the code")
    ap.add_argument("--summarize_only", action="store_true")
    args, _ = ap.parse_known_args()
    args.archs = [s for s in args.archs.split(",") if s]
    args.acts = [s for s in args.acts.split(",") if s]
    args.benches = [s for s in args.benches.split(",") if s]
    args.modes = [s for s in args.modes.split(",") if s]
    args.lrs = [float(s) for s in args.lrs.split(",") if s]

    if args.summarize_only:
        summarize(args.out)
        return
    if torch is None:
        sys.exit("PyTorch is required for training.")
    if args.smoke:
        args.out = "smoke_out"
        args.acts = ["tanh", "relu", "gelu"]
        args.lrs, args.lr_map = [1e-3], ""
        if not args.child:
            P = max(1, min(args.parallel, 2))
            args.parallel, args.seeds, args.num_shards, args.shard = P, P, P, 0
        args.perm_tasks, args.perm_n = 3, 640
        args.rand_tasks, args.rand_n, args.rand_epochs = 3, 320, 2
        args.reset_tasks, args.test_n = 2, 300
        args.ck_every, args.print_every, args.probe_n = 1, 1, 256
    if args.shard < 0 or args.parallel < 1 or args.shard + args.parallel > args.num_shards:
        sys.exit("shards --shard .. --shard+parallel-1 must lie in [0, num_shards)")

    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    if args.resume_zip and not args.child:
        with zipfile.ZipFile(args.resume_zip) as z:
            z.extractall(args.out)
        print("[resume] extracted %s" % args.resume_zip, flush=True)
    shards = list(range(args.shard, args.shard + args.parallel))
    tag = "smoke_" if args.smoke else ""
    if len(shards) == 1:
        zip_path = "plasticity_%sshard%dof%d.zip" % (tag, args.shard, args.num_shards)
    else:
        zip_path = "plasticity_%sshards%d-%dof%d.zip" % (tag, shards[0], shards[-1], args.num_shards)

    names = needed_datasets(args, shards)
    if not args.child:
        prefetch(args, names)
        if args.plan == "final":
            try:
                audit_and_samples(args, names)
            except Exception as e:
                print("[audit] skipped: %s" % e, flush=True)
    if args.parallel > 1:
        extra = ["--seeds", str(args.seeds), "--num_shards", str(args.num_shards)] if args.smoke else []
        launch_parallel(args, sys.argv[1:] + extra, os.path.abspath(__file__), torch.cuda.device_count())
        finalize(args, shards, zip_path)
        return
    if args.child:
        torch.set_num_threads(1)

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    print("[env] %s" % json.dumps(env_info(dev)), flush=True)
    start = time.time()
    deadline = start + args.hours * 3600.0
    cache = {}
    jobs = jobs_for_shard(args, args.shard)
    print("[plan] shard %d/%d -> %d jobs, estimated %.1f h" % (
        args.shard, args.num_shards, len(jobs), sum(est_minutes(j, args) for j in jobs) / 60.0), flush=True)
    with open(os.path.join(args.out, "config_shard%d.json" % args.shard), "w") as f:
        json.dump(vars(args), f, indent=1)

    for k, job in enumerate(jobs):
        t0 = time.time()
        ds = job.get("dataset", "mnist")
        if ds not in cache:
            cache[ds] = load_data(ds, args, dev)
        status = run_job(job, cache[ds], args, dev, deadline)
        if status == "done":
            print("[%d/%d] %s  %.1f min  (elapsed %.2f h)"
                  % (k + 1, len(jobs), job_name(job), (time.time() - t0) / 60.0,
                     (time.time() - start) / 3600.0), flush=True)
        if status == "timeout":
            print("[time] budget of %.1f h reached; stopping cleanly" % args.hours, flush=True)
            break
    if not args.child:
        finalize(args, shards, zip_path)


if __name__ == "__main__":
    main()
