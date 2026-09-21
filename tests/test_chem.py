"""Tests for the RDKit chemistry layer (spec sections 9, 11, 14)."""

import pytest

pytest.importorskip("rdkit")

from screening import chem


def test_valid_and_invalid_smiles():
    assert chem.is_valid_smiles("CCO")
    assert chem.is_valid_smiles("c1ccccc1")
    assert not chem.is_valid_smiles("not_a_smiles")
    assert not chem.is_valid_smiles("")


def test_standardize_is_canonical_and_salt_stripped():
    # Two encodings of ethanol canonicalize to the same string.
    a = chem.standardize_smiles("CCO")
    b = chem.standardize_smiles("OCC")
    assert a == b
    # Sodium acetate -> keep the organic fragment (acetate), drop Na.
    s = chem.standardize_smiles("CC(=O)[O-].[Na+]")
    assert s is not None and "Na" not in s


def test_standardize_invalid_returns_none():
    assert chem.standardize_smiles("not_a_smiles") is None


def test_scaffold_shared_by_analogues():
    # Two substituted benzenes share the benzene scaffold.
    s1 = chem.bemis_murcko_scaffold("Cc1ccccc1")
    s2 = chem.bemis_murcko_scaffold("CCc1ccccc1")
    assert s1 == s2 == chem.bemis_murcko_scaffold("c1ccccc1")


def test_descriptors_present_and_reasonable():
    d = chem.descriptors("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    for key in chem.DESCRIPTOR_NAMES:
        assert key in d
    assert 150 < d["mol_weight"] < 200
    assert d["lipinski_violations"] == 0


def test_fingerprint_and_tanimoto():
    fp1 = chem.morgan_fingerprint("CCO")
    fp2 = chem.morgan_fingerprint("CCO")
    fp3 = chem.morgan_fingerprint("c1ccccc1")
    assert chem.tanimoto(fp1, fp2) == pytest.approx(1.0)
    assert chem.tanimoto(fp1, fp3) < 1.0
    assert chem.max_similarity_to_references(fp1, [fp3, fp2]) == pytest.approx(1.0)
