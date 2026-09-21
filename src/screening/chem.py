"""Cheminformatics utilities for the real-data layer (Phase 2).

Thin, well-tested wrappers over RDKit for the operations the project needs:
standardization, validity, Bemis-Murcko scaffolds, descriptors, and Morgan
fingerprints.  RDKit is an *optional* dependency (Phase 2 only); importing this module
without it raises a clear, actionable error rather than failing obscurely later.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:  # RDKit is a Phase-2 optional dependency.
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without rdkit installed
    _RDKIT_AVAILABLE = False


def require_rdkit() -> None:
    """Raise an informative error if RDKit is not importable."""
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for the Phase-2 chemistry layer but is not installed. "
            "Install it with `pip install rdkit` (or `pip install -e \".[phase2]\"`)."
        )


def mol_from_smiles(smiles: str):
    """Parse a SMILES string to an RDKit Mol, or ``None`` if invalid."""
    require_rdkit()
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """True iff ``smiles`` parses to a non-empty molecule."""
    m = mol_from_smiles(smiles)
    return m is not None and m.GetNumAtoms() > 0


def standardize_smiles(smiles: str, remove_salts: bool = True) -> Optional[str]:
    """Return a canonical, standardized SMILES, or ``None`` if the input is invalid.

    Standardization (spec section 9): cleanup, keep the largest organic fragment (strip
    salts/solvents), neutralize charges, and canonicalize.  Two inputs that denote the
    same compound should map to the same output -- this is what makes deduplication and
    compound-overlap leakage checks meaningful.
    """
    require_rdkit()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        if remove_salts:
            mol = rdMolStandardize.FragmentParent(mol)  # largest fragment
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def inchikey(smiles: str) -> Optional[str]:
    """Return the InChIKey for a (standardized) SMILES, a robust dedup/identity key."""
    require_rdkit()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def bemis_murcko_scaffold(smiles: str, generic: bool = False) -> Optional[str]:
    """Return the Bemis-Murcko scaffold SMILES (framework), or ``None`` if invalid.

    Used for scaffold-based splits (spec section 11): molecules sharing a scaffold are
    chemically similar, so they must not straddle a train/test boundary.  ``generic``
    reduces the scaffold to its atom-and-bond skeleton (all atoms -> C, bonds -> single).
    """
    require_rdkit()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        smi = Chem.MolToSmiles(scaffold)
        return smi if smi else None
    except Exception:
        return None


# Descriptor names used by the RDKit property filter (spec section 14).
DESCRIPTOR_NAMES = (
    "mol_weight", "logp", "h_bond_donors", "h_bond_acceptors",
    "rotatable_bonds", "lipinski_violations",
)


def descriptors(smiles: str) -> Optional[dict]:
    """Return the physchem descriptors of spec section 14, or ``None`` if invalid."""
    require_rdkit()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return {
        "mol_weight": mw,
        "logp": logp,
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "rotatable_bonds": rot,
        "lipinski_violations": violations,
    }


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Return the Morgan (ECFP-like) fingerprint bit vector, or ``None`` if invalid."""
    require_rdkit()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(fp1, fp2) -> float:
    """Tanimoto similarity between two RDKit bit-vector fingerprints."""
    require_rdkit()
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def max_similarity_to_references(
    query_fp, reference_fps: Sequence
) -> float:
    """Max Tanimoto similarity of ``query_fp`` to any reference fingerprint."""
    require_rdkit()
    if not reference_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, list(reference_fps))
    return float(np.max(sims)) if len(sims) else 0.0
