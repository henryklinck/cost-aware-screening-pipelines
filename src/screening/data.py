"""Real experimental-data preprocessing for Phase 2 (spec sections 9-11).

Loads the **downloadable BindingDB TSV** (never scraped) and turns it into a tidy,
analysis-ready table for a single, consistently-defined endpoint -- without ever
blindly merging Kd/Ki/IC50 (spec section 10).

Pipeline (spec section 9):
  load -> select one measurement type -> parse values (drop censored '>' / '<')
       -> standardize SMILES -> drop invalid -> compute endpoint (pKd)
       -> deduplicate (compound x target, median of replicates)
       -> preserve assay info / publication year / target id / measurement type.

Because the full BindingDB TSV is large and must be downloaded by the user (with its
version/date recorded), this module is written to run on a real file *or* on a small
schema-compatible sample.  It fabricates nothing: if you point it at real data you get
real numbers; the tests exercise it on a tiny synthetic fixture that mimics the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import chem

# Default BindingDB_All TSV column names (override via ``columns=`` if a release differs).
BINDINGDB_COLUMNS: Dict[str, str] = {
    "smiles": "Ligand SMILES",
    "target_id": "UniProt (SwissProt) Primary ID of Target Chain",
    "target_name": "Target Name Assigned by Curator or DataSource",
    "Kd": "Kd (nM)",
    "Ki": "Ki (nM)",
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
    "ph": "pH",
    "temp": "Temp (C)",
    "doi": "Article DOI",
    "pmid": "PMID",
    "year": "Year",  # not in every release; left NaN if absent (temporal split then N/A)
}

MEASUREMENT_TYPES = ("Kd", "Ki", "IC50", "EC50")


@dataclass
class ProcessedDataset:
    """A standardized single-endpoint dataset plus provenance (spec sections 9, 29)."""

    df: pd.DataFrame
    measurement_type: str
    endpoint: str  # e.g. "pKd"
    source_path: Optional[str] = None
    n_raw: int = 0
    n_after_type_filter: int = 0
    n_after_valid_smiles: int = 0
    n_after_dedup: int = 0
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "measurement_type": self.measurement_type,
            "endpoint": self.endpoint,
            "source_path": self.source_path,
            "n_raw": self.n_raw,
            "n_after_type_filter": self.n_after_type_filter,
            "n_after_valid_smiles": self.n_after_valid_smiles,
            "n_after_dedup": self.n_after_dedup,
            "n_targets": int(self.df["target_id"].nunique()) if len(self.df) else 0,
            **self.meta,
        }


def parse_measurement(value) -> tuple[Optional[float], bool]:
    """Parse a BindingDB measurement cell to (value_nM, is_censored).

    Values may carry qualifiers, e.g. ``">10000"`` or ``"<1"`` (a bound, not a point
    estimate).  We return the numeric part and a flag; censored values are dropped by
    default downstream because converting a bound to a point pKd would be misleading.
    """
    if value is None:
        return None, False
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None, False
    censored = s[0] in "<>"
    if censored:
        s = s[1:].strip()
    # Some cells contain ranges or stray characters; keep only a leading number.
    try:
        return float(s), censored
    except ValueError:
        import re

        m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return (float(m.group()), censored) if m else (None, False)


def nm_to_p_endpoint(value_nm: float) -> float:
    """Convert a concentration in nM to a p-scale endpoint: ``9 - log10(value_nM)``.

    For Kd this is pKd = -log10(Kd[M]).  Higher = stronger binding.
    """
    return 9.0 - np.log10(value_nm)


def load_bindingdb(
    path,
    measurement_type: str = "Kd",
    columns: Optional[Dict[str, str]] = None,
    drop_censored: bool = True,
    min_value_nm: float = 1e-4,
    max_value_nm: float = 1e11,
    chunksize: int = 200_000,
) -> ProcessedDataset:
    """Load and preprocess a BindingDB TSV into a single-endpoint dataset.

    Parameters
    ----------
    path:
        Path to the downloaded BindingDB TSV.
    measurement_type:
        One of ``{"Kd","Ki","IC50","EC50"}``.  Only this type is kept (spec section 10):
        endpoints from different assay types are *not* merged.
    drop_censored:
        Drop measurements with ``<``/``>`` qualifiers (bounds, not point estimates).
    min_value_nm, max_value_nm:
        Sanity bounds; values outside are treated as errors and dropped.
    chunksize:
        Rows per read chunk (the full TSV is large).
    """
    if measurement_type not in MEASUREMENT_TYPES:
        raise ValueError(f"measurement_type must be one of {MEASUREMENT_TYPES}")
    cols = {**BINDINGDB_COLUMNS, **(columns or {})}
    endpoint = "p" + measurement_type[:2] if measurement_type in ("Kd", "Ki") else \
        "p" + measurement_type  # pKd, pKi, pIC50, pEC50

    value_col = cols[measurement_type]
    keep_src_cols = [
        cols["smiles"], cols["target_id"], cols["target_name"], value_col,
        cols["ph"], cols["temp"], cols["doi"], cols["pmid"], cols["year"],
    ]

    frames: List[pd.DataFrame] = []
    n_raw = 0
    reader = pd.read_csv(
        path, sep="\t", chunksize=chunksize, dtype=str,
        on_bad_lines="skip", low_memory=False,
    )
    for chunk in reader:
        n_raw += len(chunk)
        present = [c for c in keep_src_cols if c in chunk.columns]
        if value_col not in chunk.columns or cols["smiles"] not in chunk.columns:
            raise KeyError(
                f"expected columns not found; need at least {cols['smiles']!r} and "
                f"{value_col!r}. Pass a `columns=` mapping for this BindingDB release."
            )
        sub = chunk[present].copy()
        sub = sub[sub[value_col].notna()]  # only rows with this measurement type
        if len(sub):
            frames.append(sub)

    if not frames:
        empty = _empty_processed(endpoint)
        return ProcessedDataset(empty, measurement_type, endpoint, str(path), n_raw,
                                0, 0, 0)

    raw = pd.concat(frames, ignore_index=True)
    n_after_type_filter = len(raw)

    # Parse measurement values.
    parsed = raw[value_col].map(parse_measurement)
    raw["value_nm"] = [p[0] for p in parsed]
    raw["censored"] = [p[1] for p in parsed]
    raw = raw[raw["value_nm"].notna()]
    if drop_censored:
        raw = raw[~raw["censored"]]
    raw = raw[(raw["value_nm"] >= min_value_nm) & (raw["value_nm"] <= max_value_nm)]

    # Standardize SMILES and drop invalids.
    raw["smiles_std"] = raw[cols["smiles"]].map(chem.standardize_smiles)
    raw = raw[raw["smiles_std"].notna()]
    n_after_valid_smiles = len(raw)

    raw[endpoint] = nm_to_p_endpoint(raw["value_nm"].to_numpy(dtype=float))

    tidy = _to_tidy(raw, cols, endpoint)
    deduped = deduplicate(tidy, endpoint)
    n_after_dedup = len(deduped)

    return ProcessedDataset(
        df=deduped,
        measurement_type=measurement_type,
        endpoint=endpoint,
        source_path=str(path),
        n_raw=n_raw,
        n_after_type_filter=n_after_type_filter,
        n_after_valid_smiles=n_after_valid_smiles,
        n_after_dedup=n_after_dedup,
        meta={"drop_censored": drop_censored},
    )


def _empty_processed(endpoint: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "smiles_std", "target_id", "target_name", "measurement_type", endpoint,
        "value_nm", "year", "ph", "temp", "doi", "pmid", "n_measurements", "scaffold",
    ])


def _to_tidy(raw: pd.DataFrame, cols: Dict[str, str], endpoint: str) -> pd.DataFrame:
    def get(name):
        col = cols[name]
        return raw[col] if col in raw.columns else pd.Series([np.nan] * len(raw),
                                                             index=raw.index)

    out = pd.DataFrame({
        "smiles_std": raw["smiles_std"].values,
        "target_id": get("target_id").values,
        "target_name": get("target_name").values,
        "measurement_type": endpoint[1:],  # e.g. "Kd"
        endpoint: raw[endpoint].values,
        "value_nm": raw["value_nm"].values,
        "year": pd.to_numeric(get("year"), errors="coerce").values,
        "ph": pd.to_numeric(get("ph"), errors="coerce").values,
        "temp": pd.to_numeric(get("temp"), errors="coerce").values,
        "doi": get("doi").values,
        "pmid": get("pmid").values,
    })
    # A missing target id would collapse distinct targets together; drop such rows.
    out = out[out["target_id"].notna() & (out["target_id"].astype(str) != "")]
    return out.reset_index(drop=True)


def deduplicate(tidy: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Collapse replicate measurements of the same compound-target to one row.

    Replicates of the same (compound, target) are aggregated by **median** endpoint
    (robust to outliers); assay metadata from the first record is kept and the replicate
    count recorded.  Bemis-Murcko scaffold is attached for later scaffold splits.
    """
    if tidy.empty:
        d = tidy.copy()
        d["n_measurements"] = []
        d["scaffold"] = []
        return d

    grp = tidy.groupby(["smiles_std", "target_id"], as_index=False)
    agg = grp.agg(
        target_name=("target_name", "first"),
        measurement_type=("measurement_type", "first"),
        **{endpoint: (endpoint, "median")},
        value_nm=("value_nm", "median"),
        year=("year", "min"),          # earliest publication (temporal split lower bound)
        ph=("ph", "median"),
        temp=("temp", "median"),
        doi=("doi", "first"),
        pmid=("pmid", "first"),
        n_measurements=(endpoint, "size"),
    )
    agg["scaffold"] = agg["smiles_std"].map(chem.bemis_murcko_scaffold)
    return agg.reset_index(drop=True)


def define_hits(
    df: pd.DataFrame,
    endpoint: str,
    hit_threshold: float,
) -> pd.Series:
    """Boolean high-affinity hit label: ``endpoint >= hit_threshold`` (spec section 10).

    The threshold must be chosen on scientific grounds *before* touching the test set,
    never tuned on it (spec section 31).
    """
    return df[endpoint] >= hit_threshold


@dataclass
class CompoundLibrary:
    """A real-data screening library over which pipelines run (Phase-2 analog of
    :class:`~screening.simulation.MoleculeLibrary`).

    It exposes exactly the attributes the pipeline/optimization code depends on
    (``n``, ``indices``, ``is_hit``) plus the chemistry columns real models need
    (``smiles``, ``endpoint_values``, ``target_id``, ``scaffold``, ``year``), so the
    unchanged Phase-1 framework operates on real molecules (spec section 32).

    A "hit" is defined by a scientifically-chosen endpoint threshold fixed *before*
    evaluation (spec section 10).
    """

    df: pd.DataFrame
    endpoint: str
    hit_threshold: float

    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)

    @property
    def n(self) -> int:
        return int(len(self.df))

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.n)

    @property
    def smiles(self) -> np.ndarray:
        return self.df["smiles_std"].to_numpy()

    @property
    def endpoint_values(self) -> np.ndarray:
        return self.df[self.endpoint].to_numpy(dtype=float)

    @property
    def target_id(self) -> np.ndarray:
        return self.df["target_id"].to_numpy()

    @property
    def scaffold(self) -> np.ndarray:
        return self.df["scaffold"].to_numpy()

    @property
    def year(self) -> np.ndarray:
        return self.df["year"].to_numpy()

    @property
    def is_hit(self) -> np.ndarray:
        return self.endpoint_values >= self.hit_threshold

    @property
    def n_hits(self) -> int:
        return int(self.is_hit.sum())

    @property
    def prevalence(self) -> float:
        return self.n_hits / self.n if self.n else 0.0

    def subset(self, idx: np.ndarray) -> "CompoundLibrary":
        """Return a new library restricted to ``idx`` (e.g. a train or test split)."""
        return CompoundLibrary(self.df.iloc[idx].reset_index(drop=True),
                               self.endpoint, self.hit_threshold)
