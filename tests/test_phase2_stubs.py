"""Phase-1 guardrail tests: real-data / real-model components must not fabricate results.

Leakage tests proper (train ∩ test = ∅ on compounds and scaffolds, spec section 11/30)
belong to Phase 2 and will live here once ``screening.data`` is implemented. Until then
these tests assert the honest contract: the real components are *not* implemented and say
so, rather than returning invented numbers (spec section 31).
"""

import pytest

from screening import data
from screening.models.boltz2 import Boltz2Model
from screening.models.gnina import GninaModel


def test_bindingdb_loader_not_implemented():
    with pytest.raises(NotImplementedError):
        data.load_bindingdb("some/path.tsv")


def test_splits_not_implemented():
    with pytest.raises(NotImplementedError):
        data.make_splits()


def test_real_model_adapters_not_implemented():
    with pytest.raises(NotImplementedError):
        GninaModel()
    with pytest.raises(NotImplementedError):
        Boltz2Model()
