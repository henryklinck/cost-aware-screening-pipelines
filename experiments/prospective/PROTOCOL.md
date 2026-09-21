# Phase 3 — Prospective Validation Protocol (design only)

> **This phase is optional and must not be part of Version 1** (spec §28). It is executed
> **only if** the Phase-2 retrospective results are compelling (a clear cost↔recall
> advantage for an optimized pipeline over the best single model, robust across targets
> and split types). The software deliverable is useful without it. No wet-lab work is
> performed by this repository.

## Purpose

Test, prospectively, whether the pipeline that won retrospectively actually selects more
experimentally-confirmed binders per unit cost than baselines — on molecules **outside**
any model's training data, with **no** existing measurement for the target.

## Preconditions (all must hold before freezing)

- Phase-2 shows an optimized multi-stage pipeline dominating the best single model on the
  cost/recall Pareto frontier, holding under **scaffold** and **temporal** splits and
  across multiple targets (with target-level bootstrap CIs that exclude the null).
- A target with a feasible, sensitive, affordable assay is selected.
- A hit definition (e.g. `pKd ≥ 7`) is fixed **a priori** and justified biologically.

## Freeze (what `freeze_candidates.py` records)

Everything below is frozen and hash-stamped **before** any compound is ordered:

- models + exact versions/commits, pipeline order, thresholds, candidate-selection rule;
- the exact candidate list per arm, with a SHA-256 over the candidates;
- dataset version, git commit, seeds, hardware.

After freezing, nothing may change in response to results.

## Candidate eligibility

Each candidate must be: outside all models' training data; without a known measurement
for this target; chemically appropriate and synthetically/commercially accessible; and
compatible with the chosen assay.

## Comparison arms (same assay, blinded where possible)

| Arm | Selection |
|-----|-----------|
| **A** | Optimized multi-stage pipeline (Phase-2 winner) |
| **B** | Best single model at matched compute |
| **C** | Random / property-matched baseline |

Match the number of compounds tested per arm to a fixed **testing budget** so the arms
are compared at equal experimental cost.

## Outcomes

- Primary: **experimentally confirmed hit rate** per arm (fraction of tested compounds
  meeting the a-priori hit definition).
- Secondary: **cost per experimentally confirmed hit** = (compute cost + assay cost) /
  confirmed hits; and the confirmed-hit-rate difference **A − C** and **A − B** with
  bootstrap CIs.

## Analysis

Pre-register the comparison. Report confirmed hit rates with CIs; do not reuse the assay
results to retune any pipeline (doing so would forfeit the prospective claim).

## Explicitly out of scope for the software

Ordering/synthesis, assay execution, and any interaction with a CRO. This repository
stops at producing and freezing the selection and, afterwards, ingesting returned results
for the outcome analysis above.
