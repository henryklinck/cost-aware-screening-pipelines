"""GNINA/Boltz-2 adapters must refuse cleanly, never fabricate (spec sections 12,13,31)."""

import numpy as np
import pytest

from screening.models import Boltz2Model, GninaModel
from screening.models.base import ScreeningModel
from screening.models.external import ToolNotAvailableError, find_binary


def test_adapters_construct_and_expose_interface():
    g = GninaModel()
    b = Boltz2Model()
    assert isinstance(g, ScreeningModel)
    assert isinstance(b, ScreeningModel)


def test_predict_refuses_without_tool_or_inputs():
    g = GninaModel()
    b = Boltz2Model()
    with pytest.raises(ToolNotAvailableError):
        g.predict(np.array([0]), smiles=["CCO"])
    with pytest.raises(ToolNotAvailableError):
        b.predict(np.array([0]), smiles=["CCO"])


def test_estimate_cost_refuses_without_measurement():
    with pytest.raises(ToolNotAvailableError):
        GninaModel().estimate_cost(100)
    with pytest.raises(ToolNotAvailableError):
        Boltz2Model().estimate_cost(100)


def test_boltz_score_choice_validated():
    with pytest.raises(ValueError):
        Boltz2Model(score="not_a_real_score")
    # Defaults to the binder-detection score for hit discovery (spec section 13).
    assert Boltz2Model().score == "affinity_probability_binary"


def test_find_binary_missing_returns_none():
    assert find_binary("definitely_not_a_real_binary_xyz") is None
