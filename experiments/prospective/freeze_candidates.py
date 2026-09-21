"""Freeze a prospective candidate selection (spec sections 27-28).

Runs a chosen (already-validated) pipeline over a candidate library and writes a
tamper-evident frozen selection: the models, pipeline, thresholds, exact candidate list,
git commit, and a SHA-256 of the candidates.  This is the *only* Phase-3 code that runs
computationally -- the wet-lab assay itself is external and out of scope for the software
(spec section 28: the project must be useful without it).

Usage (with real data + a validated pipeline):

    python experiments/prospective/freeze_candidates.py --data BindingDB_All.tsv \
        --pipeline rdkit_filter fingerprint --thresholds 0.2 0.1 \
        --hit-threshold 7.0 --out results/prospective/selection_A.json

See PROTOCOL.md for the surrounding experimental design (comparison arms, outcomes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from screening.data import CompoundLibrary, load_bindingdb  # noqa: E402
from screening.models import FingerprintSimilarity, RDKitPropertyFilter  # noqa: E402
from screening.pipelines import run_pipeline  # noqa: E402
from screening.utils import freeze_selection  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "bindingdb_sample.tsv"

BUILDERS = {
    "rdkit_filter": lambda lib, refs: RDKitPropertyFilter(lib.smiles),
    "fingerprint": lambda lib, refs: FingerprintSimilarity(lib.smiles, refs),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a prospective candidate selection")
    ap.add_argument("--data", default=str(FIXTURE))
    ap.add_argument("--measurement-type", default="Kd")
    ap.add_argument("--hit-threshold", type=float, default=7.0)
    ap.add_argument("--pipeline", nargs="+", default=["rdkit_filter", "fingerprint"],
                    help=f"ordered model names from {list(BUILDERS)}")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.2])
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "prospective" / "selection.json"))
    args = ap.parse_args()

    if len(args.pipeline) != len(args.thresholds):
        ap.error("pipeline and thresholds must have equal length")
    unknown = [m for m in args.pipeline if m not in BUILDERS]
    if unknown:
        ap.error(f"unknown model(s): {unknown} (available: {list(BUILDERS)})")

    ds = load_bindingdb(args.data, measurement_type=args.measurement_type)
    lib = CompoundLibrary(ds.df, endpoint=ds.endpoint, hit_threshold=args.hit_threshold)

    # References for fingerprint = known actives in the library (in a true prospective
    # run these come from the frozen training set, never the candidates being tested).
    refs = lib.df.loc[lib.is_hit, "smiles_std"].tolist()
    models = {name: BUILDERS[name](lib, refs) for name in args.pipeline}

    result = run_pipeline(lib, models, args.pipeline, args.thresholds)
    candidate_smiles = lib.smiles[result.survivors].tolist()

    record = freeze_selection(
        candidates=candidate_smiles,
        pipeline=args.pipeline,
        thresholds=args.thresholds,
        config={
            "data": args.data, "measurement_type": args.measurement_type,
            "endpoint": ds.endpoint, "hit_threshold": args.hit_threshold,
            "preprocessing": ds.summary(),
        },
        path=Path(args.out),
        extra={"final_candidates": result.final_molecules,
               "total_cost_seconds": result.total_cost},
    )
    print(f"Froze {record['n_candidates']} candidates "
          f"(sha256 {record['candidates_sha256'][:12]}...) -> {args.out}")
    print("Selection is now immutable input to a prospective assay; see PROTOCOL.md.")


if __name__ == "__main__":
    main()
