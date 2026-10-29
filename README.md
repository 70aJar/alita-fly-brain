# Alita Fly Brain — plays, tests and dashboard

Working record of what we measured with **`flybrain` 0.1.0** — the *MaleCNS v1.0* fruit-fly
central nervous system (166,700 neurons, ~25.6M connections; Janelia / Google / Cambridge,
CC BY 4.0) — run as a spiking network on a single Apple Silicon machine.

This repository holds the **harness, the raw results and the dashboard**. Every number here
came out of a run in `results/`; nothing is illustrative.

- Dashboard: [`dashboard.html`](dashboard.html) — 20 plays, tiered by fleet leverage
- Play background: [`docs/20-plays.md`](docs/20-plays.md)
- Sandbox/bring-up notes: [`docs/sandbox-test.md`](docs/sandbox-test.md)

## Bring-up (verified)

| Item | Value |
|---|---|
| Package | `flybrain` 0.1.0 (CPU/numba lane) |
| Data | `brain.npz` 53 MB + `weights.npz` 205 MB (sha256-checked) |
| Host | Apple Silicon M5 Max, 128 GB |
| Step cost | **2.8 ms/step** (50-step average after numba warm-up) |
| Throughput | ~357 steps/s → **~356× realtime** |

## Plays — status

| # | Play | State |
|---|---|---|
| 1 | Reservoir readout (looming → MBON) | **DONE** — AUC 0.925 |
| 2 | Hearth fly threat-sense (graded reflex) | **DONE** |
| 3 | Fly vs LLM race (reflex vs classifier) | **DONE** |
| 4 | DOOMFLY on Desktop RTX 5060 | queued (needs the Windows/RTX node) |
| 5 | Colour/shape/numerosity benchmark suite | **DONE — null result (below)** |
| 6 | Neuroevolution (NEAT) over fixed graph | next up |
| 7–20 | olfactory, end-to-end vision, lesion studies, camera loop, swarm, graph analytics… | planned |

## Play 5 — visual feature discrimination: a measured null

**Question.** Does the fixed connectome expose simple visual features — object count, object
width, and side of presentation — to a *linear* readout of its descending output?

**Method (identical for both runs).** Stimuli are rendered as luminance over the eye's azimuth
axis and delivered through `brain.step(eye_drive=…)`, i.e. the real photoreceptor population.
The brain runs with the stimulus held, and a logistic readout is fitted on the collected
features. Nothing is trained inside the brain: this measures how much task information the
fixed dynamics expose.

### v1 — static image, coarse readout

8 descending groups + identified MBON types (115 features), 40 trials/class, 80 steps, single 80/20 split.

| Task | Accuracy | Majority baseline |
|---|---|---|
| numerosity (1 vs 3 blobs) | 0.562 | 0.625 |
| width (narrow vs wide) | 0.438 | 0.500 |
| side (left vs right) | 0.375 | 0.625 |

### v2 — onset transient, Kenyon-cell readout

12 Kenyon-cell types + 36 MBON types + 12 descending groups (**4,179 features**), 60 trials/class,
15 blank + 45 object steps with luminance **and** change encoding, 5-fold cross-validation.

| Task | CV accuracy | Baseline |
|---|---|---|
| numerosity | 0.542 ± 0.083 | 0.533 |
| width | 0.500 ± 0.053 | 0.533 |
| side | 0.558 ± 0.042 | 0.533 |

**Result: no decodable linear signal for any of the three tasks** — every score sits inside one
standard deviation of the majority-class baseline, in both protocols.

**Two method faults were found and fixed between the runs** (a static image never excites
change detectors; and 115 mean-aggregated features discard almost the whole connectome), and
the null survived the fix. That makes the negative result worth stating rather than hiding:
with this stimulation path, the connectome's linear readouts do not carry these features.

**What this does not say.** It is not evidence the fly brain cannot discriminate these stimuli.
Play 1 reached AUC 0.925 on looming **through the feature-detector pathway** (LPLC2/LC4 →
descending neurons), not through raw luminance. The likely explanation is that our drive enters
the retina at a level the model does not propagate to the readout we sampled — not that the
circuit lacks the computation. Reproducing play 1's pathway for these stimuli is the obvious
next experiment (play 8, end-to-end vision).

## Reproduce

```bash
python3 -m venv ~/flybrain-venv && . ~/flybrain-venv/bin/activate
pip install flybrain
flybrain download            # 260 MB, sha256-checked
python harness/play5_visual_tasks_v2.py results/play5_results_v2.json
```

Both harnesses are deterministic given the seed (`20261030`) and print the exact protocol they
ran; the JSON records the protocol, feature counts and per-task scores.

## Data and attribution

MaleCNS v1.0 is released under **CC BY 4.0** by its authors (Janelia, Google, Cambridge).
`flybrain` is a third-party package that loads it; we only run it. See `docs/20-plays.md` for
the ecosystem survey these plays came from.

## Licence

MIT for the harness and results in this repository.
