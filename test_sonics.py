import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

import sonics_detector

needs_models = pytest.mark.skipif(
    not sonics_detector.models_cached(),
    reason="no local SONICS ONNX cache (see SONICS_ONNX_DIR)",
)


class FakeResponse:
    """Minimal urllib response for downloader tests."""

    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if size is None or size < 0 or size >= len(self._payload):
            data, self._payload = self._payload, b""
            return data
        data, self._payload = self._payload[:size], self._payload[size:]
        return data


def _fake_download_source(monkeypatch, tmp_path, payloads):
    """Point SONICS at a fake release with a manifest and the given payloads."""
    monkeypatch.setattr(sonics_detector, "SONICS_ONNX_BASE", "https://example.invalid/assets")
    monkeypatch.setattr(sonics_detector, "SONICS_MODELS", ["gamma"])
    monkeypatch.setenv("SONICS_ONNX_DIR", str(tmp_path / "cache"))

    manifest = [
        {
            "file": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in payloads.items()
    ]

    def fake_urlopen(req, timeout):
        if req.full_url.endswith("/manifest.json"):
            return FakeResponse(json.dumps(manifest).encode("utf-8"))
        name = req.full_url.rsplit("/", 1)[-1]
        data = payloads[name]
        return FakeResponse(data, headers={"Content-Length": str(len(data))})

    monkeypatch.setattr(sonics_detector.urllib.request, "urlopen", fake_urlopen)
    return manifest


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


def test_download_rejects_bad_checksum_and_cleans_temp(monkeypatch, tmp_path):
    _fake_download_source(monkeypatch, tmp_path, {"gamma.onnx": b"payload"})
    dest = tmp_path / "cache" / "gamma.onnx"
    expected = {"bytes": 999, "sha256": "0" * 64}

    with pytest.raises(ValueError, match="checksum mismatch"):
        sonics_detector._download(
            "https://example.invalid/assets/gamma.onnx", dest, expected
        )

    assert not dest.exists()
    assert list(dest.parent.glob(".sonics_*")) == []


def test_ensure_assets_redownloads_corrupt_cache(monkeypatch, tmp_path):
    payloads = {"gamma.onnx": b"good-onnx", "mel_fb.npy": b"good-mel"}
    _fake_download_source(monkeypatch, tmp_path, payloads)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "gamma.onnx").write_bytes(b"bad")
    (cache / "mel_fb.npy").write_bytes(b"bad")

    sonics_detector._ensure_assets()

    assert (cache / "gamma.onnx").read_bytes() == b"good-onnx"
    assert (cache / "mel_fb.npy").read_bytes() == b"good-mel"
    assert list(cache.glob(".sonics_*")) == []


def test_fetch_manifest_rejects_file_scheme(monkeypatch):
    monkeypatch.setattr(sonics_detector, "SONICS_ONNX_BASE", "file:///tmp/sonics")
    with pytest.raises(ValueError, match="http"):
        sonics_detector._fetch_manifest()


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
