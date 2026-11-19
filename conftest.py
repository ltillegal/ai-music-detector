"""Shared test setup."""

import os
from pathlib import Path

import pytest

import check_ai_music
import sonics_detector

# Tests use relative paths (model, fixtures) - pin cwd to the repository
# root so `pytest` works from any directory.
os.chdir(Path(__file__).resolve().parent)


@pytest.fixture(autouse=True)
def _sonics_offline(monkeypatch):
    """Prevent tests from downloading ~138 MB of SONICS ONNX models.

    SONICS runs on onnxruntime, but the ONNX models are downloaded lazily.
    CI and clean machines do not have them, so SONICS is reported unavailable
    to keep tests offline. When a cache exists (or ``SONICS_ONNX_DIR`` is set),
    the model integration tests run in full.
    """
    if not sonics_detector.models_cached():
        monkeypatch.setattr(check_ai_music.sonics_detector, "available", lambda: False)
    yield
