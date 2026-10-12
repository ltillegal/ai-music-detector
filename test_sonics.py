import numpy as np
import pytest
import soundfile as sf

import sonics_detector

needs_models = pytest.mark.skipif(
    not sonics_detector.models_cached(),
    reason="нет локального кэша ONNX-моделей SONICS (см. SONICS_ONNX_DIR)",
)


def _load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def test_logmel_shape_and_finite():
    seg = np.random.randn(sonics_detector.SONICS_WIN).astype(np.float32)
    fb = np.ones((sonics_detector.N_STFT, sonics_detector.N_MELS), np.float32)
    mel = sonics_detector._logmel(seg, fb)
    assert mel.shape == (sonics_detector.N_MELS, sonics_detector.N_FRAMES)
    assert np.isfinite(mel).all()


def test_windows_normalized():
    y = np.random.randn(sonics_detector.SONICS_WIN * 2).astype(np.float32)
    wins = sonics_detector._windows(y)
    assert wins.shape[1] == sonics_detector.SONICS_WIN
    assert np.allclose(np.std(wins, axis=1), 1.0, atol=1e-4)


@needs_models
def test_sonics_reports_both_models():
    audio, sr = _load("fixtures/known_human.wav")
    result = sonics_detector.score_audio(audio, sr)
    assert "p_ai" in result
    assert any(name.startswith("sonics-spectttra") for name in result)


@needs_models
def test_sonics_human_scores_low():
    audio, sr = _load("fixtures/known_human.wav")
    assert sonics_detector.score_audio(audio, sr)["p_ai"] < 0.5


@needs_models
def test_sonics_ai_scores_high():
    audio, sr = _load("fixtures/known_suno.wav")
    assert sonics_detector.score_audio(audio, sr)["p_ai"] > 0.8
