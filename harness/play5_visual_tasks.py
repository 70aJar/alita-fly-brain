#!/usr/bin/env python3
"""Play #5 — visual feature discrimination suite on the MaleCNS fly brain.

Three decodable tasks over the real connectome:
  1. numerosity  — one blob vs three blobs
  2. width       — narrow object vs wide object
  3. side        — object at left vs right azimuth

Method (identical for every task):
  - stimuli are rendered as luminance over the eye's azimuth axis
  - luminance is delivered through brain.step(eye_drive=...), i.e. the real photoreceptors
  - the brain runs T steps with the stimulus held
  - Trace features over descending-output groups are collected each trial
  - a logistic readout is fitted 80/20 with a fixed seed; accuracy vs chance is reported

Nothing is trained inside the brain: this measures how much task information the
fixed connectome's dynamics expose to a linear decoder.
"""
import json
import sys
import time

import numpy as np
import flybrain

T_STEPS = 80
TRIALS_PER_CLASS = 40
SEED = 20261030


def render_lum(azimuth, blobs):
    """Blob list -> luminance per photoreceptor (1.0 = bright background)."""
    lum = np.full(len(azimuth), 1.0, np.float32)
    for center, half_width, darkness in blobs:
        inside = np.abs(azimuth - center) <= half_width
        lum[inside] = np.minimum(lum[inside], 1.0 - darkness)
    return np.clip(lum, 0.0, 1.0)


TASKS = {
    # name: (class-0 blobs, class-1 blobs)
    "numerosity": ([(0.0, 0.06, 0.9)],
                   [(-0.4, 0.06, 0.9), (0.0, 0.06, 0.9), (0.4, 0.06, 0.9)]),
    "width": ([(0.0, 0.04, 0.9)],
              [(0.0, 0.18, 0.9)]),
    "side": ([(-0.5, 0.08, 0.9)],
             [(0.5, 0.08, 0.9)]),
}


def run_task(brain, traces, azimuth, name, rng):
    b0, b1 = TASKS[name]
    X, y = [], []
    for label, blobs in ((0, b0), (1, b1)):
        for _ in range(TRIALS_PER_CLASS):
            brain.reset()
            for t in traces:
                t.reset()
            drive = render_lum(azimuth, blobs)
            for _t in range(T_STEPS):
                fired = brain.step(eye_drive=drive)
                for t in traces:
                    t.observe(fired)
            X.append(np.concatenate([np.asarray(t.features(), dtype=np.float64).ravel()
                                     for t in traces]))
            y.append(label)
    X = np.vstack(X)
    y = np.asarray(y)

    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(len(y) * 0.8)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w, b = flybrain.fit_logistic(Xtr, ytr, lam=1.0)
    pred = (Xte @ w + b) > 0.5
    acc = float((pred == yte).mean())
    chance = max(float(yte.mean()), 1 - float(yte.mean()))
    return {"task": name, "accuracy": round(acc, 4), "chance": round(chance, 4),
            "n_train": int(split), "n_test": int(len(y) - split),
            "features": int(X.shape[1])}


def main():
    t0 = time.time()
    brain = flybrain.FlyBrain(seed=SEED)
    azimuth = np.asarray(brain.azimuth, dtype=np.float32)

    # readout populations: the 8 descending-output groups + every identified MBON
    cell_types = np.asarray(brain.cell_type).astype(str)
    mbon = sorted({t for t in cell_types if "MBON" in t})
    traces = [flybrain.Trace(brain, group=g) for g in brain.groups] + \
             [flybrain.Trace(brain, types=mbon)]
    rng = np.random.default_rng(SEED)

    results = []
    for name in TASKS:
        r = run_task(brain, traces, azimuth, name, rng)
        r["above_chance"] = r["accuracy"] > r["chance"]
        results.append(r)
        print(f"  {name:12s} acc={r['accuracy']:.3f}  chance={r['chance']:.3f}  "
              f"features={r['features']}", flush=True)

    out = {"play": 5, "brain": "MaleCNS v1.0 (166,700 neurons)",
           "steps_per_trial": T_STEPS, "trials_per_class": TRIALS_PER_CLASS,
           "seed": SEED, "seconds": round(time.time() - t0, 1), "results": results}
    with open(sys.argv[1] if len(sys.argv) > 1 else "play5_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"  total {out['seconds']}s -> {sys.argv[1] if len(sys.argv) > 1 else 'play5_results.json'}")


if __name__ == "__main__":
    main()
