"""Shared test fixtures/paths."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def bindingdb_sample_path():
    return FIXTURES / "bindingdb_sample.tsv"
