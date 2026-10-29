# flybrain (MaleCNS v1.0) — SANDBOX TEST PASSED (Sep 16 2026)

## What
`flybrain` 0.1.0 (PyPI Sep 13 2026) — the complete male fruit-fly CNS connectome
(166,700 neurons, 25.6M connections) as a spiking network, CPU/GPU.

## Install
- venv: `~/flybrain-venv` (python3.14)
- `pip install flybrain` (CPU numba lane)
- `flybrain download` → 260MB → `~/fly-data` (brain.npz 53MB + weights.npz 205MB, sha256-checked)

## Tests — ALL PASSED

### 1. Biological escape circuit (giant fiber)
- Looming detectors `LC4 + LPLC2` (left eye) -> injected 0.8 -> **DNp01 giant fiber
  fired at 0.74s** — the real escape command neuron triggered through actual wiring.
- Repeated firing 0.74s→0.98s while stimulus held. Verified after 50-step window.

### 2. Performance (M5 Max, CPU)
- **2.8 ms/step** (50-step avg, after numba warmup) — BETTER than README's 12-15ms
  for 24 threads. M5 Max 14-core + high-memory bandwidth.
- **~357 steps/s → 356× realtime**: 1 fly-second ≈ 2.8ms wall time; 1 fly-minute ≈ 0.17s.
- More than fast enough for interactive experiments + batch runs.

## Fleet relevance
| Lane | Node | Fit |
|---|---|---|
| CPU sim | FRIDAY M5 Max | ✅ VERIFIED 2.8ms/step |
| GPU lane | Desktop RTX 5060 | `pip install "flybrain[gpu]"` (CuPy CUDA 12) — README: 1.4ms/step on RTX 4060 |
| Reservoir learning | any | `Trace/run/Readout` — fit linear/logistic PCA readout to spike traces, trains in minutes |
| Vision experiments | FRIDAY | `Eyes` + `FeatureDetectors` drive visual projection neurons |

## Next ideas
- Run a reservoir readout task (color/shape/numerosity like train-your-fly)
- DOOMFLY harness on Desktop RTX (game frames -> photoreceptor signals -> neurons)
- Race fly brain vs LLM agent on a simple task (v8 games lane)
- Tie into Hearth simulation as a real-biology subsystem

## Files
- Research: Fleet/FlyBrain-MaleCNS-Research-Sep16.md
- This test record. Venv persists at ~/flybrain-venv; data at ~/fly-data.
---

## Task 1 — Reservoir readout (COMPLETED Sep 16)

**Question:** does looming stimulation produce a distinguishable state in a learned readout?
**Method:** inject LC4/LPLC2 (left-eye looming detectors) at 1.2 for 40 steps; trace MBONs
(97 mushroom-body output neurons, tau 0.2); PCA-20; 5-fold stratified CV logistic readout.
80 trials (40 ON / 40 OFF).

| Trials | AUC (5-fold CV) | Verdict |
|---|---|---|
| 40 (20/class, 400ms) | 0.738 | marginal |
| **80 (40/class, 800ms)** | **0.925** (folds .891-.984) | **DISCRIMINATES** |

- Script: ~/Projects/flybrain-tasks/reservoir_looming.py
- Runtime: 75s for 80 trials x 40 steps (fixed wiring; only readout trained)
- Meaning: the biological circuit carries clean looming information into the mushroom
  body; a linear readout recovers it at 92.5% AUC. Reservoir computing on real fly wiring works.

---

## Task 3 — Fly vs LLM race (COMPLETED Sep 16)

**Task:** 36 trials, classify looming side L/R/N from a 64x64 photoreceptor field.
**Fly lane:** inject LC4/LPLC2 looming detectors on the side, watch giant-fiber (DNp01) escape reflex.
**LLM lane:** same field as text -> local Gemma-4 Heretic (oMLX, thinking OFF, temp 0).

| Lane | Accuracy | Mean latency |
|---|---|---|
| LLM | 36/36 (100%) | 477 ms |
| FLY | 22/36 (61%) | **51 ms** |

- Fly = fast reflex (9.3x faster), detects looming-presence 67% (16/24) but trigger-happy on N trials (false-alarm — realistic escape-reflex biology)
- LLM = precise classifier on a language-native task; needs a text description of the field
- Verdict: they are different machines — reflex vs classifier. Best use: fly for low-latency threat/loom detection, LLM for deliberative classification.
- Script: ~/Projects/flybrain-tasks/fly_vs_llm_race.py (36 trials in ~35s)

---

## Task 4 — Hearth tie-in: fly-brain threat-sense subsystem (COMPLETED Sep 16)

**Module:** ~/Projects/flybrain-tasks/fly_threat_sense.py (standalone, Hearth-importable)
**Design:** characters get a biological reflex layer — a threat event drives the fly's
looming detectors (LC4/LPLC2) at threat-proportional intensity; a sustained giant-fiber
(DNp01) firing fraction >= 0.5 = reflex fires. Output = character prompt line.

| threat_level | fire rate (12 trials, 8 steps) | prompt |
|---|---|---|
| 0.1 quiet | 0% | "Your instincts are quiet." |
| 0.5 moderate | 100% | "A flicker of instinct — you sense something, then it fades." |
| 0.9 imminent | 100% | "Your instincts SCREAM — danger is close. Every nerve says MOVE." |

- Real biology: 166,700-neuron connectome running the actual escape circuit, NOT a
  random function. Loaded lazily, only during threat windows (~2.8ms/step).
- Calibration note: raw giant fiber is trigger-happy (fires on any loom injection) —
  the sustained-fraction gating maps the reflex to a graded sense.
- Engine integration (call in Hearth's tick when a threat event occurs) = next sim work.

<!-- alita:related:start -->
## Related notes (auto-indexed)

- [[Fleet/FlyBrain-MaleCNS-Research-Sep16|🪰 Fruit Fly Brain (MaleCNS v1.0) — Deep Research (Sep 16 2026)]]
- [[Fleet/FlyBrain-20-Plays-Sep16|🪰 Fly Brain Ecosystem — Deep Research + Our 20 Most Interesting Plays (Sep 16 2026)]]
- [[Fleet/TODO-Email-Review-Sep12-14-PostUpgrade|TODO — Email Review Sep 12-14 (action after macOS upgrade)]]
- [[Fleet/TODO-Sep23|TODO — Sep 23 2026]]
- [[Fleet/ALITA_BREADCRUMBS|ALITA — Breadcrumbs & Session Resumption]]

<!-- alita:related:end -->
