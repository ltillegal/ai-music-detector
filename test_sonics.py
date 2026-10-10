import numpy as np
import pytest
import soundfile as sf

import sonics_detector

pytestmark = pytest.mark.skipif(
    not sonics_detector.available(),
    reason="SONICS не установлен (опциональная группа зависимостей 'sonics')",
)


def _load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def test_sonics_reports_both_models():
    audio, sr = _load("fixtures/known_human.wav")
    result = sonics_detector.score_audio(audio, sr)
    assert "p_ai" in result
    assert any(name.startswith("sonics-spectttra") for name in result)


def test_sonics_human_scores_low():
    audio, sr = _load("fixtures/known_human.wav")
    assert sonics_detector.score_audio(audio, sr)["p_ai"] < 0.5


def test_sonics_ai_scores_high():
    audio, sr = _load("fixtures/known_suno.wav")
    assert sonics_detector.score_audio(audio, sr)["p_ai"] > 0.8
