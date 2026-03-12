"""Shared fixtures for the test suite."""

import os
import sys

import pytest

# Ensure the project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


ISABELLA_CONFIG = os.path.join(PROJECT_ROOT, "configs", "isabella.yaml")


@pytest.fixture
def isabella_cfg():
    """Load the Isabella reference config."""
    from src.core.config import load_config
    return load_config(ISABELLA_CONFIG)
