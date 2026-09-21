# Cost-Aware Molecular Screening Pipelines

**Research question.** Given a fixed computational budget, how should existing molecular
screening models be *combined and ordered* to maximize recovery of experimentally
validated high-affinity compounds? The central trade-off is

```
computational cost  <-->  hit recovery
```

The project is built in three phases (spec):

```
Phase 1  Synthetic simulation      (implemented — this repo)
   |     develop & validate the methodology
   v
Phase 2  Real models + public data  (scaffolded, not implemented)
   |     GNINA, Boltz-2, RDKit filters on BindingDB
   v
Phase 3  Prospective wet-lab validation (only if Phase 2 justifies it)
```

> **Status: Phase 1 only.** GNINA, Boltz-2, and BindingDB are deliberately *not*
> implemented — they exist as interface-conformant stubs so the pipeline/optimization
> code can be reused unchanged in Phase 2. Nothing in this repo fabricates real-model
> performance, costs, or benchmark numbers; every result is produced by the simulation.

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
    gnina.py         Phase-2 stub (§12)
    boltz2.py        Phase-2 stub (§13)
  pipelines.py       run_pipeline: staged thresholding & cost accrual (§6)
  metrics.py         recall/precision/enrichment/cost-per-hit; model quality (§20,§23)
  optimization.py    enumerate pipelines, grid-search thresholds, Pareto (§7,§19,§20)
  data.py            Phase-2 BindingDB stub (§9-11)
  utils.py           reproducibility manifest, timing (§29)
experiments/simulation/
  run_simulation.py  end-to-end driver
  plots.py           Figures 1-5 (§8)
configs/             simulation.yaml (Phase 1), experiments.yaml (Phase 2 placeholder)
results/             generated tables & figures (git-ignored)
tests/               unit tests (§30)
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
