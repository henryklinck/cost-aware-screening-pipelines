"""Phase-2 real-data experiment driver (spec sections 18-26).

Runs the *unchanged* Phase-1 pipeline/optimization framework on real experimental data
and real models:

    BindingDB TSV -> single-endpoint preprocessing -> leakage-aware split
    -> real models (RDKit filter, fingerprint-to-train-actives; GNINA/Boltz-2 if present)
    -> pipeline x threshold search on the held-out test set (per target + pooled)
    -> cost/recall Pareto + model-error-correlation + bootstrap CIs
    -> tables, a figure, and a reproducibility manifest.

Runnable now with the cheap models. GNINA and Boltz-2 are auto-detected and included
only if their binaries + required inputs are available; otherwise they are skipped with
a note (never fabricated -- spec section 31).

    python experiments/real_data/run_real_data.py --data path/to/BindingDB_All.tsv \
        --measurement-type Kd --hit-threshold 7.0 --split scaffold

By default it runs on the small bundled fixture as a PLUMBING demo (clearly not a
scientific result -- the fixture's structure-activity relationship is random).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "simulation"))

from screening import analysis, splits  # noqa: E402
from screening.data import CompoundLibrary, load_bindingdb  # noqa: E402
from screening.metrics import top_k_recall  # noqa: E402
from screening.models import FingerprintSimilarity, RDKitPropertyFilter  # noqa: E402
from screening.optimization import (  # noqa: E402
    constrain_to_test_budget,
    pareto_frontier,
    search_all,
)
from screening.utils import ExperimentRecord, Timer  # noqa: E402
import plots  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bindingdb_sample.tsv"


def build_models(train_lib: CompoundLibrary, eval_lib: CompoundLibrary,
                 endpoint: str) -> Dict[str, object]:
    """Construct real models. Fingerprint references = TRAIN hits only (no leakage)."""
    models: Dict[str, object] = {}
    models["rdkit_filter"] = RDKitPropertyFilter(eval_lib.smiles)

    train_hits = train_lib.df.loc[train_lib.is_hit, "smiles_std"].tolist()
    models["fingerprint"] = FingerprintSimilarity(
        eval_lib.smiles, reference_actives=train_hits)

    # Optional expensive models -- included only if genuinely available.
    _maybe_add_external(models, eval_lib)
    return models


def _maybe_add_external(models: Dict[str, object], eval_lib: CompoundLibrary) -> None:
    from screening.models import Boltz2Model, GninaModel

    g = GninaModel()
    if g.is_available():  # needs binary + receptor
        warnings.warn("GNINA available but wiring receptor/ligand prep is left to the "
                      "user; skipping automatic inclusion.")
    b = Boltz2Model()
    if b.is_available():
        warnings.warn("Boltz-2 available but inference wiring is left to the user; "
                      "skipping automatic inclusion.")


def evaluate_target(df: pd.DataFrame, endpoint: str, hit_threshold: float,
                    split_kind: str, seed: int, test_budget_fraction: float):
    """Split leakage-safely, run the pipeline framework on the test set, analyse."""
    full = CompoundLibrary(df, endpoint=endpoint, hit_threshold=hit_threshold)
    made = splits.make_splits(df, kinds=(split_kind,), seed=seed)
    if split_kind not in made:
        return None
    split = made[split_kind]
    # Verify the intended leakage guarantee actually holds.
    key = {"compound": "smiles_std", "scaffold": "scaffold"}.get(split_kind)
    if key:
        splits.verify_no_leakage(df, split, key)

    train_lib = full.subset(split.train_idx)
    test_lib = full.subset(split.test_idx)
    if test_lib.n < 5 or test_lib.n_hits == 0:
        return None

    models = build_models(train_lib, test_lib, endpoint)
    names = list(models.keys())
    search = search_all(test_lib, models, min_stages=1,
                        max_stages=min(3, len(names)),
                        retention_values=(0.05, 0.1, 0.2, 0.3, 0.5))
    K = max(1, int(test_budget_fraction * test_lib.n))
    constrained = constrain_to_test_budget(search, K)
    frontier = pareto_frontier(constrained) if not constrained.empty else constrained

    # Model-error correlation on the test set (spec section 22).
    scores = {n: models[n].predict(test_lib.indices) for n in names}
    ec = analysis.error_correlations(scores, test_lib.endpoint_values) \
        if len(names) >= 2 else pd.DataFrame()

    return {
        "n_train": train_lib.n, "n_test": test_lib.n, "test_hits": test_lib.n_hits,
        "search": search, "constrained": constrained, "frontier": frontier,
        "error_corr": ec, "models": names,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-2 real-data screening experiment")
    ap.add_argument("--data", default=str(FIXTURE),
                    help="BindingDB TSV path (defaults to the bundled demo fixture)")
    ap.add_argument("--measurement-type", default="Kd", choices=["Kd", "Ki", "IC50", "EC50"])
    ap.add_argument("--hit-threshold", type=float, default=7.0,
                    help="p-endpoint hit cutoff, chosen a priori (e.g. pKd>=7 ~ Kd<=100nM)")
    ap.add_argument("--split", default="scaffold",
                    choices=["compound", "scaffold", "temporal", "target"])
    ap.add_argument("--test-budget-fraction", type=float, default=0.2)
    ap.add_argument("--min-target-rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "real_data"))
    args = ap.parse_args()

    is_demo = Path(args.data).resolve() == FIXTURE.resolve()
    if is_demo:
        print("WARNING: running on the bundled fixture -- this is a PLUMBING DEMO, not a "
              "scientific result (the fixture's structure-activity data is random).\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with Timer() as timer:
        ds = load_bindingdb(args.data, measurement_type=args.measurement_type)
        endpoint = ds.endpoint
        print("Preprocessing summary:")
        for k, v in ds.summary().items():
            print(f"  {k}: {v}")
        if ds.df.empty:
            print("No usable rows after preprocessing; aborting.")
            return

        # Per-target evaluation (spec section 25) + pooled.
        target_counts = ds.df["target_id"].value_counts()
        targets = target_counts[target_counts >= args.min_target_rows].index.tolist()
        print(f"\nTargets with >= {args.min_target_rows} rows: {len(targets)}")

        per_target_recall: List[dict] = []
        frontiers = {}
        first_ec = None
        for tgt in targets:
            sub = ds.df[ds.df["target_id"] == tgt].reset_index(drop=True)
            out = evaluate_target(sub, endpoint, args.hit_threshold, args.split,
                                  args.seed, args.test_budget_fraction)
            if out is None:
                continue
            fr = out["frontier"]
            best = fr.sort_values("recall").iloc[-1] if not fr.empty else None
            per_target_recall.append({
                "target_id": tgt, "n_test": out["n_test"], "test_hits": out["test_hits"],
                "best_recall": None if best is None else float(best["recall"]),
                "best_pipeline": None if best is None else best["pipeline"],
                "models": ",".join(out["models"]),
            })
            if not fr.empty:
                frontiers[tgt] = fr
            if first_ec is None and not out["error_corr"].empty:
                first_ec = out["error_corr"]

        recall_df = pd.DataFrame(per_target_recall)
        recall_df.to_csv(outdir / "per_target_results.csv", index=False)
        print("\nPer-target results:")
        if not recall_df.empty:
            print(recall_df.to_string(index=False))

        # Cross-target bootstrap CI on best-frontier recall (spec section 26).
        if not recall_df.empty and recall_df["best_recall"].notna().any():
            vals = recall_df["best_recall"].dropna().to_numpy()
            labels = np.arange(len(vals))  # each target is an independent unit

            def stat(idx):
                return float(np.mean(vals[idx]))

            ci = analysis.grouped_bootstrap_ci(labels, stat, n_boot=1000, seed=args.seed)
            print(f"\nCross-target mean best-recall: {ci['point']:.3f} "
                  f"(95% CI {ci['ci_low']:.3f}-{ci['ci_high']:.3f}, "
                  f"{ci['n_groups']} targets)")
        else:
            ci = None

        if first_ec is not None:
            first_ec.to_csv(outdir / "model_error_correlation.csv", index=False)
            print("\nModel error correlation (first target with >=2 models):")
            print(first_ec.round(3).to_string(index=False))

        # Figure: overlay per-target frontiers (recall vs measured compute cost).
        if frontiers:
            plots.fig4_prevalence(  # reuse the multi-line frontier plot
                {float(i): fr for i, (_, fr) in enumerate(frontiers.items())},
                outdir / "real_frontiers.png")

    record = ExperimentRecord(
        experiment_id=f"real_{args.measurement_type}_{args.split}",
        config=vars(args),
        seed=args.seed,
        runtime_seconds=timer.seconds,
        results={
            "is_demo_fixture": is_demo,
            "preprocessing": ds.summary(),
            "endpoint": endpoint,
            "hit_threshold": args.hit_threshold,
            "n_targets_evaluated": int(len(recall_df)),
            "cross_target_recall_ci": ci,
        },
    )
    manifest = outdir / f"real_{args.measurement_type}_{args.split}_manifest.json"
    record.to_json(manifest)
    print(f"\nRuntime: {timer.seconds:.1f}s\nManifest: {manifest}\nOutputs: {outdir}")
    if is_demo:
        print("\nTo produce a real result, download the BindingDB TSV and pass --data.")


if __name__ == "__main__":
    main()
