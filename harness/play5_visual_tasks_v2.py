#!/usr/bin/env python3
"""Play #5 (v2) — visual feature discrimination on the MaleCNS fly brain.

v1 drove a STATIC luminance image and read 115 mean-aggregated features; every task
landed at or below chance. Two method faults were identified in that result:
  1. a static image never excites change/looming detectors, so the retina is silent
     after the first few steps;
  2. 8 descending groups + 18 MBON means collapse almost all of the connectome.
v2 fixes both: an onset transient (blank -> object, Eyes-style change encoding) and a
Kenyon-cell readout, with 5-fold cross-validation instead of a single split.

Reported metric is cross-validated accuracy against the majority-class baseline.
"""
import json
import sys
import time

import numpy as np
import flybrain

STEPS_BLANK = 15
STEPS_OBJECT = 45
TRIALS_PER_CLASS = 60
FOLDS = 5
SEED = 20261030
BACKGROUND = 1.0


def lum_for(azimuth, blobs):
    lum = np.full(len(azimuth), BACKGROUND, np.float32)
    for center, half_width, darkness in blobs:
        inside = np.abs(azimuth - center) <= half_width
        lum[inside] = np.minimum(lum[inside], BACKGROUND * (1.0 - darkness))
    return lum


def drive_sequence(azimuth, blobs):
    """Blank frames, then object onset — luminance + change, as the package's Eyes does."""
    blank = lum_for(azimuth, [])
    obj = lum_for(azimuth, blobs)
    frames = [blank] * STEPS_BLANK + [obj] * STEPS_OBJECT
    out, prev = [], blank
    for f in frames:
        change = np.abs(f - prev)
        out.append(np.clip(0.45 * f + 1.6 * change, 0.0, 1.0).astype(np.float32))
        prev = f
    return out


TASKS = {
    "numerosity": ([(0.0, 0.06, 0.9)],
                   [(-0.4, 0.06, 0.9), (0.0, 0.06, 0.9), (0.4, 0.06, 0.9)]),
    "width": ([(0.0, 0.04, 0.9)],
              [(0.0, 0.18, 0.9)]),
    "side": ([(-0.5, 0.08, 0.9)],
             [(0.5, 0.08, 0.9)]),
}


def collect(brain, traces, azimuth, name):
    b0, b1 = TASKS[name]
    X, y = [], []
    for label, blobs in ((0, b0), (1, b1)):
        drive = drive_sequence(azimuth, blobs)
        for _ in range(TRIALS_PER_CLASS):
            brain.reset()
            for t in traces:
                t.reset()
            for d in drive:
                fired = brain.step(eye_drive=d)
                for t in traces:
                    t.observe(fired)
            X.append(np.concatenate([np.asarray(t.features(), dtype=np.float64).ravel()
                                     for t in traces]))
            y.append(label)
    return np.vstack(X), np.asarray(y)


def cv_accuracy(X, y):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    folds = np.array_split(np.arange(len(y)), FOLDS)
    accs, base = [], []
    for k in range(FOLDS):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, b = flybrain.fit_logistic(Xtr, y[tr], lam=1.0)
        pred = (Xte @ w + b) > 0.5
        accs.append(float((pred == y[te]).mean()))
        base.append(max(float(y[te].mean()), 1 - float(y[te].mean())))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(base))


def main():
    t0 = time.time()
    brain = flybrain.FlyBrain(seed=SEED)
    azimuth = np.asarray(brain.azimuth, dtype=np.float32)
    ct = np.asarray(brain.cell_type).astype(str)
    kc_types = sorted({t for t in ct if t.startswith("KC")})
    mbon = sorted({t for t in ct if "MBON" in t})
    traces = [flybrain.Trace(brain, types=kc_types), flybrain.Trace(brain, types=mbon)] + \
             [flybrain.Trace(brain, group=g) for g in brain.groups]
    print(f"  readout: {len(kc_types)} KC types + {len(mbon)} MBON types + {len(brain.groups)} groups", flush=True)

    results = []
    for name in TASKS:
        X, y = collect(brain, traces, azimuth, name)
        acc, sd, base = cv_accuracy(X, y)
        results.append({"task": name, "cv_accuracy": round(acc, 4), "cv_sd": round(sd, 4),
                        "majority_baseline": round(base, 4), "features": int(X.shape[1]),
                        "trials": int(len(y)), "above_baseline": acc > base})
        print(f"  {name:12s} cv_acc={acc:.3f} ±{sd:.3f}  baseline={base:.3f}  "
              f"features={X.shape[1]}", flush=True)

    out = {"play": 5, "version": 2, "brain": "MaleCNS v1.0 (166,700 neurons)",
           "readout": {"kenyon_types": len(kc_types), "mbon_types": len(mbon),
                       "descending_groups": len(brain.groups)},
           "protocol": {"steps_blank": STEPS_BLANK, "steps_object": STEPS_OBJECT,
                        "trials_per_class": TRIALS_PER_CLASS, "folds": FOLDS},
           "seed": SEED, "seconds": round(time.time() - t0, 1),
           "note": "v1 drove static luminance and read 115 features; all tasks were at/below chance. "
                   "v2 adds an onset transient and a Kenyon-cell readout.",
           "results": results}
    path = sys.argv[1] if len(sys.argv) > 1 else "play5_results.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"  total {out['seconds']}s -> {path}")


if __name__ == "__main__":
    main()
