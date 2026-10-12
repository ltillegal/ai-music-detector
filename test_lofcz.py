import pytest
import soundfile as sf

import lofcz_detector

pytestmark = pytest.mark.skipif(
    not lofcz_detector.available(),
    reason="lofcz недоступен (нет onnx-модели или scipy)",
)


def _load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype("float32"), sr


def test_lofcz_returns_probability():
    audio, sr = _load("fixtures/known_suno.wav")
    result = lofcz_detector.score_audio(audio, sr)
    assert 0.0 <= result["p_ai"] <= 1.0


def test_lofcz_human_scores_low():
    audio, sr = _load("fixtures/known_human.wav")
    assert lofcz_detector.score_audio(audio, sr)["p_ai"] < 0.5


def test_lofcz_ai_scores_high():
    audio, sr = _load("fixtures/known_suno.wav")
    assert lofcz_detector.score_audio(audio, sr)["p_ai"] >= 0.5
