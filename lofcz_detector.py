#!/usr/bin/env python3
"""Lofcz-детектор AI-музыки — «fakeprint» + маленький ONNX-классификатор.

Идея из lofcz/ai-music-detector (MIT): усреднённый спектр в полосе 1–8 кГц
сглаживается (min-фильтр), из него вычитается «корпус» (hull) и остаётся
residue — периодические артефакты вокодеров Suno/Udio. Residue подаётся в
компактную ONNX-модель, которая отдаёт вероятность AI.

Это отдельный вокодерный сигнал, дополняющий ArtifactNet; к диффузионным
генераторам (Lyria 3 и т.п.) он, как и ArtifactNet, слеп.

Зависимости: numpy, scipy, onnxruntime, soundfile, soxr (все уже в проекте).
Модель и конфиг лежат рядом с репозиторием:
    lofcz_ai_music_detector.onnx, lofcz_preprocessing.json

Служит опорой для `check_ai_music.py <file>`.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    import scipy.ndimage
except ImportError:  # pragma: no cover - опциональные зависимости
    ort = None
    scipy = None

# Порог «уверенного» срабатывания lofcz (когда другие сигналы молчат).
LOFCZ_STRONG = 0.5

_MODEL_FILENAME = "lofcz_ai_music_detector.onnx"
_CONFIG_FILENAME = "lofcz_preprocessing.json"
_SESSION = None
_CFG = None


def _candidate_dirs():
    return [
        Path(__file__).resolve().parent,
        Path(sys.executable).resolve().parent,
        Path.cwd(),
    ]


def _find(name):
    for d in _candidate_dirs():
        cand = d / name
        if cand.is_file():
            return cand
    return None


def available():
    """True, если зависимости импортируются и файлы модели на месте."""
    if ort is None or scipy is None:
        return False
    return _find(_MODEL_FILENAME) is not None and _find(_CONFIG_FILENAME) is not None


def _load_config():
    global _CFG
    if _CFG is None:
        with open(_find(_CONFIG_FILENAME), encoding="utf-8") as fh:
            _CFG = json.load(fh)
    return _CFG


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = ort.InferenceSession(
            str(_find(_MODEL_FILENAME)), providers=["CPUExecutionProvider"]
        )
    return _SESSION


def _resample(audio, src_sr, dst_sr):
    if len(audio) == 0 or src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    try:
        import soxr

        return soxr.resample(audio, src_sr, dst_sr).astype(np.float32)
    except ImportError:
        src_t = np.arange(len(audio)) / src_sr
        dst_t = np.arange(int(len(audio) * dst_sr / src_sr)) / dst_sr
        return np.interp(dst_t, src_t, audio).astype(np.float32)


def _stft_power(a, n_fft):
    hop = n_fft // 2
    pad = n_fft // 2
    a = np.pad(a, (pad, pad), mode="reflect")
    n_frames = 1 + (len(a) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(a, n_fft)[::hop][:n_frames]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = frames * win
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (np.abs(spec) ** 2).T  # (freq, time)


def _fakeprint(audio, cfg, expected_len):
    power = _stft_power(audio, cfg["n_fft"])
    spec_db = 10 * np.log10(np.clip(power, 1e-10, 1e6))
    mean_spectrum = spec_db.mean(axis=1)

    n_bins = cfg["n_fft"] // 2 + 1
    freqs = np.linspace(0, cfg["sample_rate"] / 2, num=n_bins)
    mask = (freqs >= cfg["freq_min"]) & (freqs <= cfg["freq_max"])
    freq_spectrum = mean_spectrum[mask]

    hull = scipy.ndimage.minimum_filter1d(freq_spectrum, size=cfg["hull_area"], mode="nearest")
    hull = np.clip(hull, cfg["min_db"], None)
    residue = np.clip(freq_spectrum - hull, 0, None)
    residue = np.clip(residue, 0, cfg["max_db"])

    if len(residue) != expected_len:
        residue = np.interp(
            np.linspace(0, 1, expected_len), np.linspace(0, 1, len(residue)), residue
        )
    return (residue / (np.max(residue) + 1e-6)).astype(np.float32)


def score_audio(audio, src_sr, max_duration=300):
    """audio: моно float. Возвращает {'p_ai': prob, 'model': name}."""
    cfg = _load_config()
    y = _resample(np.asarray(audio, dtype=np.float32), src_sr, cfg["sample_rate"])
    y = y[: int(max_duration * cfg["sample_rate"])]

    sess = _get_session()
    expected_len = int(sess.get_inputs()[0].shape[1])
    fp = _fakeprint(y, cfg, expected_len)
    prob = float(sess.run(None, {sess.get_inputs()[0].name: fp.reshape(1, -1)})[0][0, 0])
    return {"p_ai": round(prob, 6), "model": _MODEL_FILENAME}
