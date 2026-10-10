#!/usr/bin/env python3
"""SONICS (SpecTTTra) — спектрограммный детектор AI-музыки.

В отличие от ArtifactNet/lofcz, которые ищут «физический» отпечаток вокодеров
(периодические пики от транспонированных свёрток), SONICS — стиль-ориентированный
spectrogram-transformer. Он ловит современные диффузионные генераторы (например
Google Lyria 3 в producer.ai), которые не оставляют вокодерного отпечатка.

Модуль опциональный: если torch/sonics не установлены, ``available()`` вернёт
False и вызывающий код просто пропустит этот сигнал.

Зависимости (опциональная группа ``sonics`` в pyproject.toml):
    torch, torchaudio, timm, fvcore, sonics (git+https://github.com/awsaf49/sonics.git)

Модели скачиваются с HuggingFace при первом запуске и кэшируются в HF_HOME.
"""

import os

import numpy as np

SONICS_SR = 16000
SONICS_WIN = SONICS_SR * 5  # 5 секунд = 80000 сэмплов
SONICS_STRIDE = SONICS_WIN // 2  # 2.5 c
SONICS_TOP_K = int(os.environ.get("SONICS_TOP_K", "3"))
SONICS_MODELS = os.environ.get(
    "SONICS_MODELS",
    "awsaf49/sonics-spectttra-gamma-5s,awsaf49/sonics-spectttra-beta-5s",
).split(",")

# порог «уверенного» срабатывания SONICS (когда ArtifactNet молчит)
SONICS_STRONG = 0.8
# порог «человек» для SONICS в паре с низким ArtifactNet
SONICS_HUMAN_MAX = 0.5

_MODELS = None


def available():
    """True, если torch и sonics импортируются."""
    try:
        import torch  # noqa: F401
        import sonics  # noqa: F401
    except Exception:
        return False
    return True


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


def _load_models():
    global _MODELS
    if _MODELS is None:
        from sonics import HFAudioClassifier

        _MODELS = []
        for model_id in SONICS_MODELS:
            model_id = model_id.strip()
            if not model_id:
                continue
            model = HFAudioClassifier.from_pretrained(model_id)
            _MODELS.append((model_id.split("/")[-1], model))
    return _MODELS


def _windows(y):
    """Все 5-секундные окна @16кГц, каждое нормировано на единичную дисперсию."""
    import torch

    if len(y) < SONICS_WIN:
        y = np.pad(y, (0, SONICS_WIN - len(y)))
    starts = list(range(0, max(len(y) - SONICS_WIN, 0) + 1, SONICS_STRIDE))
    if not starts:
        starts = [0]
    if starts[-1] != len(y) - SONICS_WIN:
        starts.append(len(y) - SONICS_WIN)
    wins = []
    for s in starts:
        seg = y[s : s + SONICS_WIN]
        wins.append(seg / max(float(np.std(seg)), 1e-6))
    stacked = np.ascontiguousarray(np.stack(wins))
    return torch.from_numpy(stacked).float()


def score_audio(audio, src_sr):
    """audio: моно float. Возвращает dict со срабатываниями каждой модели."""
    import torch

    y = _resample(np.asarray(audio, dtype=np.float32), src_sr, SONICS_SR)
    x = _windows(y)
    models = _load_models()

    out = {}
    tops = []
    with torch.no_grad():
        for name, model in models:
            probs = torch.sigmoid(model(x)).flatten()
            k = min(SONICS_TOP_K, probs.numel())
            top_k = float(probs.topk(k).values.mean())
            out[name] = {
                "top_k": round(top_k, 6),
                "max": round(float(probs.max()), 6),
                "mean": round(float(probs.mean()), 6),
                "win_frac": round(float((probs >= 0.5).float().mean()), 6),
                "n_windows": int(probs.numel()),
            }
            tops.append(top_k)
    out["p_ai"] = round(float(np.mean(tops)) if tops else 0.0, 6)
    return out
