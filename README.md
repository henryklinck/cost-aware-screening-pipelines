# Cost-Aware Molecular Screening Pipelines

**Research question.** Given a fixed computational budget, how should existing molecular
screening models be *combined and ordered* to maximize recovery of experimentally
validated high-affinity compounds? The central trade-off is

```
computational cost  <-->  hit recovery
```

The project is built in three phases (spec):

```
Phase 1  Synthetic simulation           develop & validate the methodology
   v
Phase 2  Real models + public data       BindingDB + RDKit/fingerprint (+ GNINA/Boltz-2)
   v
Phase 3  Prospective wet-lab validation  only if Phase 2 justifies it (design only)
```

### Status by phase

| Phase | State | Notes |
|---|---|---|
| **1 — synthetic** | ✅ implemented & runnable | full methodology; produces the cost/recall Pareto result |
| **2 — real data** | ✅ runnable (cheap models) | BindingDB preprocessing, leakage-aware splits, RDKit + fingerprint baselines, and correlation/subgroup/bootstrap analysis all run through the **unchanged** Phase-1 framework |
| **2 — GNINA / Boltz-2** | ⚙️ honest adapters | interface-conformant, provenance-recording, **availability-gated**: they require external installs + GPU and *refuse* rather than fabricate when absent |
| **3 — prospective** | 📄 design + freeze utility | `PROTOCOL.md` + tamper-evident candidate freezing; no wet-lab work is performed |

> **Nothing in this repo fabricates real-model performance, costs, or benchmark numbers.**
> Phase-1 numbers come from the simulation; Phase-2 numbers come from real preprocessing +
> runnable models on data you supply; GNINA/Boltz-2 produce numbers only on a properly
> configured machine, and otherwise raise a clear error (spec §31).

---

## What Phase 1 does

A synthetic world lets us develop the methodology where we know the ground truth:

1. **Continuous latent affinity** for each molecule: `y_true ~ N(mu, sigma)`
   (molecules are opaque IDs, not chemical structures — spec §4.1).
2. **Assay noise**, kept distinct from model error: `y_obs = y_true + N(0, sigma_assay)`
   (spec §4.2). Hits are defined by thresholding a *continuous* affinity — prevalence is
   an emergent consequence of where the threshold sits, never a binary coin flip.
3. **Five simulated models** with controllable accuracy, bias, variance, cost, **and
   cross-model error correlation** (spec §5). Each produces a *continuous* score
   `y_hat = gain*y_true + bias + eps`, where the errors `eps` are drawn jointly with a
   specified correlation matrix — correlated model error is central to the question.
4. **Multi-stage pipeline execution** (spec §6): each stage scores only the survivors of
   the previous stage, keeps the top fraction by *its own* score, and the expensive model
   only pays for what it actually scores.
5. **Exhaustive search** over every ordered pipeline (≤ 3 stages) × a threshold grid
   `{1,2,5,10,20,30,50}%` (spec §7, §19).
6. **Cost/recall analysis and the Pareto frontier** (spec §8, §20, §21).

### The testing-budget constraint (an important modeling choice)

Recall *alone* is trivially maximized by cheaply keeping a huge fraction of the library
(e.g. the cheapest model at 50% retention recovers ~all hits) — useless, because you
could never assay that many molecules. In real screening the **final candidates are what
you pay to test experimentally**, so their count is capped by a *testing budget* `K`.
The primary analysis therefore reports **recall at fixed computational cost, subject to a
fixed final shortlist size** (`final ≤ K`, default 1% of the library). Within that
constraint, achieving high recall in a small shortlist *requires* good — and usually more
expensive — screening, so compute genuinely buys recall. See
`constrain_to_test_budget` in [`optimization.py`](src/screening/optimization.py).

---

## Repository layout

```
src/screening/
  simulation.py      continuous affinity, assay noise, MoleculeLibrary (§4)
  models/
    base.py          ScreeningModel interface: predict() + estimate_cost() (§16)
    simulated.py     SimulatedModel + correlated-error construction (§5)
    cheap.py         RDKitPropertyFilter + FingerprintSimilarity, runnable (§14)
    external.py      shared availability-gating + provenance for external tools
    gnina.py         GNINA docking adapter, availability-gated (§12)
    boltz2.py        Boltz-2 affinity adapter, availability-gated (§13)
  pipelines.py       run_pipeline: staged thresholding & cost accrual (§6)
  metrics.py         recall/precision/enrichment/cost-per-hit; model quality (§20,§23)
  optimization.py    enumerate pipelines, grid-search thresholds, Pareto (§7,§19,§20)
  chem.py            RDKit layer: standardize/scaffold/descriptors/fingerprints (§9,11,14)
  data.py            BindingDB preprocessing + CompoundLibrary (§9-10)
  splits.py          compound / scaffold / temporal / target splits + verifier (§11)
  analysis.py        error correlation / subgroup error / grouped bootstrap (§22,23,26)
  utils.py           reproducibility manifest, timing, selection freezing (§27,§29)
experiments/
  simulation/        run_simulation.py + plots.py — Figures 1-5 (§8)
  real_data/         run_real_data.py — Phase-2 driver on BindingDB (§18-26)
  prospective/       PROTOCOL.md + freeze_candidates.py (§27-28, design only)
configs/             simulation.yaml, experiments.yaml
tests/               80 unit tests incl. leakage checks (§30); fixtures/ tiny TSV sample
results/             generated tables & figures (git-ignored)
```

The **model interface** is the key to phase portability. Everything downstream depends
only on:

```python
class ScreeningModel:
    name: str
    def predict(self, molecule_ids) -> np.ndarray:  # one score per molecule
    def estimate_cost(self, n_molecules) -> float:   # compute cost
```

Scores from different models are **never** compared as if calibrated together — the
pipeline only ranks each model against *its own* scores (spec §12, §13, §31).

---

## Install & run

```bash
pip install -e .            # core deps only: numpy, pandas, scipy, matplotlib, pyyaml
pip install -e ".[dev]"     # + pytest

pytest                       # run the test suite

python experiments/simulation/run_simulation.py --config configs/simulation.yaml
# add --skip-sweeps to skip Figures 4 & 5 (faster)
```

### Phase 2 — real data

```bash
pip install -e ".[phase2]"   # adds RDKit

# Demo on the bundled tiny fixture (plumbing only — random SA data, NOT a result):
python experiments/real_data/run_real_data.py --split scaffold

# Real run: download the BindingDB TSV yourself (record its version/date), then:
python experiments/real_data/run_real_data.py \
    --data path/to/BindingDB_All.tsv --measurement-type Kd \
    --hit-threshold 7.0 --split scaffold
```

This preprocesses to a single endpoint (e.g. pKd), builds leakage-safe splits, runs the
RDKit + fingerprint models through the same pipeline/optimization code, and reports
per-target recall with cross-target bootstrap CIs and model-error correlations. To add
GNINA/Boltz-2, install those tools and wire their receptor/sequence inputs — the adapters
already conform to the interface and record provenance.

### Phase 3 — prospective (design only)

```bash
# Freeze a tamper-evident candidate selection (models+pipeline+thresholds+SHA-256):
python experiments/prospective/freeze_candidates.py \
    --pipeline rdkit_filter fingerprint --thresholds 0.5 0.2 \
    --out results/prospective/selection_A.json
```

See [`experiments/prospective/PROTOCOL.md`](experiments/prospective/PROTOCOL.md) for the
full design (comparison arms, outcomes). No wet-lab work is done by this repo.

Outputs (all under `results/`, reproducible from config + seed):

| Path | Contents |
|---|---|
| `model_benchmarks/model_quality.csv` | per-model Spearman/AUROC/top-k recall, cost |
| `simulation/all_configurations.csv` | every pipeline × threshold evaluated |
| `pipelines/pareto_frontier.csv` | the cost/recall Pareto frontier |
| `pipelines/primary_table.csv` | the primary results table (§8) |
| `pipelines/recall_at_budget.csv` | recall at fixed compute budgets (§20) |
| `simulation/<id>_manifest.json` | reproducibility envelope (§29) |
| `figures/fig1..5_*.png` | Figures 1-5 (§8, §21) |

---

## Figures

- **Figure 1** — Recall vs computational cost; every configuration plotted, Pareto
  frontier highlighted, best single-model points overlaid. *The central result (§21).*
- **Figure 2** — Cost per recovered hit along the frontier.
- **Figure 3** — Ordering vs performance for a fixed model set: cheap-first orderings
  dominate the low-cost region; expensive-first orderings are pushed right.
- **Figure 4** — Effect of hit prevalence on the frontier.
- **Figure 5** — Effect of model error correlation on the frontier.

---

## Reproducibility (spec §29)

Every run writes an `ExperimentRecord` manifest capturing `experiment_id`, `git_commit`
(and dirty flag), full `config`, `seed`, `hardware`, `runtime_seconds`, and headline
results. Randomness flows from a single seed through `numpy`'s `SeedSequence`, with
independent sub-streams for affinity, assay noise, and model errors.

---

## What this project deliberately does **not** do (spec §31)

Invent benchmark results, model performance, or costs; assume datasets are leakage-free;
train models from scratch; scrape datasets without provenance; mix Kd/Ki/IC50
indiscriminately; compare raw scores from different models as if identically calibrated;
claim GNINA/Boltz-2 are independent; use random splits without discussing chemical
leakage; tune thresholds on a final test set; or include wet-lab work in v1.

## License

MIT — see [LICENSE](LICENSE).
