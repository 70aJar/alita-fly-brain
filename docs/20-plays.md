# 🪰 Fly Brain Ecosystem — Deep Research + Our 20 Most Interesting Plays (Sep 16 2026)

> Companion to FlyBrain-MaleCNS-Research-Sep16.md. Covers what the community has
> done since MaleCNS v1.0 (Jun 8 2026 release, Sep 3 2026 Cell paper) and the
> 20 most interesting things WE can do with it. HTML dashboard companion: this file.

---

## PART 1 — The Ecosystem (what exists + what people have done)

### Datasets (adult Drosophila connectomes)
| Dataset | Sex | Region | Neurons | Acquisition | Access |
|---|---|---|---|---|---|
| **hemibrain** | F | 1/3 brain | 25K | FIB-SEM | neuPrint |
| **FlyWire (FAFB)** | F | whole brain | ~140K | ssTEM | Codex |
| **FANC** | F | VNC | 14.6K | ssTEM | GitHub |
| **MANC** | M | VNC | 15.8K | FIB-SEM | neuPrint |
| **MaleCNS v1.0** | **M** | **whole CNS** | **166.7K** | FIB-SEM | neuPrint (+bulk GCS) |
| **BANC (2026)** | F | whole CNS | 142K | ssTEM | Codex — female complement to MaleCNS! |

### What the community has DONE (as of Sep 16 2026)
1. **DOOMFLY** (Alex Wormuth, Coinbase; open Sep 9): Doom frames → 4,146 signals →
   166K neurons → actions; dopamine-reinforcement on damage. Honest: hasn't survived yet.
2. **Stonkfly** (same author): K-line charts → fly brain → Bitcoin trades.
3. **Fly64 / Mario 64** (Jessica Paquette): vibe-coded harness, erratic-but-alive Mario.
4. **Beat Saber, Minecraft, Bad Apple** ports: overfit motor replay, not true learning.
5. **flybrain** (alextitonis, PyPI Sep 13): whole connectome as spiking net, CPU/CUDA,
   260MB, batch=8 flies, reservoir Trace/Readout, Eyes visual encoder. ⬅ we run this.
6. **train-your-fly** (eudald-seeslab): PyTorch Geometric, Drosophila eye model +
   message passing over FlyWire graph + Kenyon-cell readout; color/shape/numerosity
   tasks, manifold analysis.
7. **ZAPBench** (Google): neural-activity benchmark for whole-brain models.
8. **WebGPU browser sims**: draw/stimulate neurons, watch 3D fly behavior live.
9. **VNC circuit papers** (eLife Jul 2026): descending → motor pathways; local circuit
   control discovery (Harvard/Princeton).
10. **Sexual-dimorphism science** (Berg 2026): 262 sex-specific + 114 dimorphic cell
    types; the first male-vs-female whole-organism brain comparison.
11. **FlyWire ecosystem tooling**: Navis (morphology), wing/leg VAT, fafbseg (R),
    Clio (annotation), neuPrint (query), Neuromorpho integration.
12. **Effectome** (Pospisil et al): simulated perturbation effects over the connectome —
    the "function" counterpart to the "wiring diagram".

### Notable scientific findings that inform our plays
- 75% of neurons are connected by ≤3 interneurons → tiny paths, cheap computation.
- Motor control is LOCAL (leg circuits control legs) not centralized — a modularity
  template for our fleets/agents.
- Optic lobes = retina analog; VNC = spinal cord — sensory-to-motor in one graph.

---

## PART 2 — The 20 Most Interesting Things We Can Do (ranked by fleet leverage)

### Tier A — Do THIS (highest leverage, hardware we already have)
1. **Reservoir readout on FRIDAY (DONE ✅):** looming discriminates at 0.925 AUC in
   MBONs; 80 trials in 75s. Foundation for everything below.
2. **Fly threat-sense for Hearth (DONE ✅):** graded biological reflex (0% @quiet,
   100% @moderate/imminent), prompt-injectable, ~2.8ms/step.
3. **Fly vs LLM race (DONE ✅):** reflex (51ms) vs classifier (477ms) — different
   machines; both usable.
4. **DOOMFLY on Desktop RTX (LAHORE):** GPU lane, real game-reinforcement training,
   the obvious public showcase.
5. **Color/shape/numerosity benchmark suite** (train-your-fly tasks on MaleCNS, not
   FlyWire): get our own copy of the SOTA results on OUR hardware.
6. **Neuroevolution on the fly** (NEAT over the fixed graph): evolve the readout +
   a few synapse gains to maximize survival in a simple arena. CPU-cheap.
7. **Olfactory discrimination reservoir** (antennal lobe → mushroom body): the fly's
   most-studied learning circuit — perfect for a second readout benchmark.
8. **Vision pipeline end-to-end** (Eyes → medulla → lobula → DN): real R1-R6 visual
   input, not hand-injected loom — the "eyes to legs" promise Jefferis described.

### Tier B — High value, more build
9. **Fly brain as ALITA architecture inspiration:** 166K neurons, 25.6M edges,
   modular local circuits — a genuine sparse-spiking reference for a compact
   companion model (vs transformer).
10. **Training-data generator for Hearth:** extract spike-trace "sensory states" as
    structured event sequences to feed character prompts (already partially done in #2).
11. **Effectome-style in-silico perturbations:** knock out cell types (mushroom body,
    central complex) → measure behavior change → interpretable "lesion studies" on FRIDAY.
12. **Sensor fusion demo:** real camera frame → edge detection → fly visual neurons →
    threat flash (camera-vision + flybrain pipeline).
13. **Batch=8 multi-fly evolution:** run 8 flies in parallel, evolutionary selection on
    a task → 8× faster population experiments.
14. **Latency-stability audit:** realtime ratio (2.8ms/step = 357×) makes continuous
    "always-on fly" feasible for fleet ambient-sensing experiments.

### Tier C — Nice to have / research flavor
15. **Male-vs-female connectome comparison** (MaleCNS vs BANC when it lands): "what
    changes if we swap sex?" — great benchmark/novelty corpus.
16. **Fly brain on oMLX/LM Studio?** Not directly (it's not an LLM), but a readout
    trained on the fly could be exported as weights for a tiny net we serve locally.
17. **Mushroom-body associative conditioning demo:** classical conditioning
    (CS+US pairing) on real KC→MBON wiring — the insect Pavlov.
18. **Central-complex navigation** (heading integration): fly's internal compass —
    a natural "tiny navigator" for robotics experiments (we have Pi/ESP32 fleet).
19. **Swarm fly lab:** N flies on N cores, behavioral statistics (escape latency
    distributions), feed into Hearth's relationship dynamics.
20. **Connectome graph analytics:** run gortex-style graph queries on the 25.6M-edge
    graph (shortest paths, hub structure) — combine our code-intelligence tooling
    with the biggest biological graph on hand.

### What this implies for the fleet
- Compute: nothing new needed (CPU lane verified; RTX GPU lane at Lahore for DOOMFLY).
- Storage: 260MB already local; bulk MaleCNS (connectivity 1.1GB) optional later.
- Licensing: MIT code / CC BY 4.0 data — safe to run, train, and publish results.
- Delivery form: each play becomes a skill + vault doc with exact parameters
  (reproducibility rule). DOOMFLY + benchmark suite are the public-facing candidates
  for the 70aJar public-records lane (sanitized, no personal data).

<!-- alita:related:start -->
## Related notes (auto-indexed)

- [[Fleet/FlyBrain-MaleCNS-Research-Sep16|🪰 Fruit Fly Brain (MaleCNS v1.0) — Deep Research (Sep 16 2026)]]
- [[Fleet/FlyBrain-Sandbox-Test-Sep16|flybrain (MaleCNS v1.0) — SANDBOX TEST PASSED (Sep 16 2026)]]
- [[Fleet/TODO-Email-Review-Sep12-14-PostUpgrade|TODO — Email Review Sep 12-14 (action after macOS upgrade)]]
- [[Fleet/TODO-Sep23 (imported 1)|TODO — Sep 23 2026]]

<!-- alita:related:end -->
