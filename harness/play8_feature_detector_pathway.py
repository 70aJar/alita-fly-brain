#!/usr/bin/env python3
"""Play #8 — vision end-to-end through the feature-detector pathway.

Play 5 drove raw luminance into the retina and found no decodable signal for three
visual tasks. Play 1, by contrast, reached AUC 0.925 on looming through the
feature-detector front end (LPLC2 = looming, LC4 = fast looming/escape, LC10a = chase).

This harness asks two questions on that same pathway:

  A. LOOMING DETECTION — is an approaching object distinguishable from a receding one?
     (logistic readout, AUC and accuracy vs 0.5)

  B. GRADED THREAT — is the *degree* of threat linearly encoded?
     (threat in {0, .25, .5, .75, 1}; ridge readout, R^2 and Spearman rho)

Stimuli enter via FeatureDetectors.inject(), which drives the identified looming/escape
neurons; everything downstream (descending neurons, MBONs) is the real connectome.
"""
import json
import sys
import time

import numpy as np
import flybrain

FRAMES = 40
TRIALS = 40
SEED = 20261030


def run_trial(brain, fd, traces, approach: bool, threat: float):
    brain.reset()
    for t in traces:
        t.reset()
    dx = 240.0 if approach else 40.0
    size = 8.0 if approach else 30.0
    for _f in range(FRAMES):
        dx += -12.0 if approach else 6.0           # approach closes, recede opens
        size += 6.0 if approach else -3.0
        inj = fd.inject(opp=(dx, size), threat=threat)
        fired = brain.step(inject=inj)
        for t in traces:
            t.observe(fired)
    return np.concatenate([np.asarray(t.features(), dtype=np.float64).ravel() for t in traces])


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.time()
    brain = flybrain.FlyBrain(seed=SEED)
    fd = flybrain.FeatureDetectors(brain)
    ct = np.asarray(brain.cell_type).astype(str)
    mbon = sorted({t for t in ct if "MBON" in t})
    traces = [flybrain.Trace(brain, types=mbon)] + \
             [flybrain.Trace(brain, group=g) for g in brain.groups]
    rng = np.random.default_rng(SEED)

    # A. looming detection: approach vs recede, no threat assumed
    X, y = [], []
    for label, approach in ((1, True), (0, False)):
        for _ in range(TRIALS):
            X.append(run_trial(brain, fd, traces, approach, 0.0))
            y.append(label)
    X = np.vstack(X); y = np.asarray(y)
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    split = int(len(y) * 0.8)
    mu, sd = X[:split].mean(0), X[:split].std(0) + 1e-8
    w, b = flybrain.fit_logistic((X[:split] - mu) / sd, y[:split], lam=1.0)
    score = (X[split:] - mu) / sd @ w + b
    acc_a = float(((score > 0.5) == y[split:]).mean())
    auc_a = auc(y[split:], score)

    # B. graded threat: five levels, ridge readout, all trials
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    Xb, yb = [], []
    for lv in levels:
        for _ in range(TRIALS):
            Xb.append(run_trial(brain, fd, traces, True, lv))
            yb.append(lv)
    Xb = np.vstack(Xb); yb = np.asarray(yb)
    # 80/20 split, ridge regression
    idxb = rng.permutation(len(yb)); Xb, yb = Xb[idxb], yb[idxb]
    sb = int(len(yb) * 0.8)
    mub, sdb = Xb[:sb].mean(0), Xb[:sb].std(0) + 1e-8
    Xtr, Xte = (Xb[:sb] - mub) / sdb, (Xb[sb:] - mub) / sdb
    lam = 1.0
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    wb = np.linalg.solve(A, Xtr.T @ yb[:sb])
    pred = Xte @ wb
    ss_res = float(((yb[sb:] - pred) ** 2).sum())
    ss_tot = float(((yb[sb:] - yb[sb:].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    rho = spearman(yb[sb:], pred)

    out = {
        "play": 8,
        "brain": "MaleCNS v1.0 (166,700 neurons)",
        "pathway": "FeatureDetectors (LPLC2/LC4/LC1/LC10a) -> connectome -> MBON + descending",
        "protocol": {"frames": FRAMES, "trials_per_condition": TRIALS, "seed": SEED},
        "A_looming_detection": {"accuracy": round(acc_a, 4), "auc": round(auc_a, 4),
                                "chance": 0.5, "n_test": int(len(y) - split)},
        "B_graded_threat": {"r2": round(r2, 4), "spearman": round(rho, 4),
                            "levels": levels, "n_test": int(len(yb) - sb)},
        "seconds": round(time.time() - t0, 1),
    }
    path = sys.argv[1] if len(sys.argv) > 1 else "play8_results.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"  A looming detection : acc={acc_a:.3f}  AUC={auc_a:.3f}   (chance 0.5)")
    print(f"  B graded threat     : R2={r2:.3f}  rho={rho:.3f}   (5 levels)")
    print(f"  {out['seconds']}s -> {path}")


if __name__ == "__main__":
    main()
