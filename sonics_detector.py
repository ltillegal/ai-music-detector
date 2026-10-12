"""SONICS (SpecTTTra) — спектрограммный детектор AI-музыки.

В отличие от ArtifactNet/lofcz, которые ищут «физический» отпечаток вокодеров
(периодические пики от транспонированных свёрток), SONICS — стиль-ориентированный
spectrogram-transformer. Он ловит современные диффузионные генераторы (например
Google Lyria 3 в producer.ai), которые не оставляют вокодерного отпечатка.

Детектор работает на ``onnxruntime`` (уже ядровая зависимость для ArtifactNet),
поэтому torch/timm/fvcore НЕ нужны: модель SpecTTTra экспортирована в ONNX, а
log-mel препроцессинг считается в numpy. Готовые ``.onnx`` (~69 МБ каждая) и
мел-фильтр скачиваются при первом запуске и кэшируются.

Публичный API: ``available()``, ``models_cached()``, ``score_audio()`.

Переменные окружения:
    SONICS_MODELS      список базовых имён моделей через запятую
    SONICS_ONNX_DIR    каталог кэша (по умолчанию ~/.cache/ai-music-detector/sonics-onnx)
    SONICS_ONNX_BASE   базовый URL для скачивания артефактов
    SONICS_TOP_K       сколько верхних окон усреднять для оценки
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

SONICS_SR = 16000
SONICS_WIN = SONICS_SR * 5  # 5 секунд = 80000 сэмплов
SONICS_STRIDE = SONICS_WIN // 2  # 2.5 c
SONICS_TOP_K = int(os.environ.get("SONICS_TOP_K", "3"))
SONICS_MODELS = [
    m.strip()
    for m in os.environ.get(
        "SONICS_MODELS",
        "sonics-spectttra-gamma-5s,sonics-spectttra-beta-5s",
    ).split(",")
    if m.strip()
]

# порог «уверенного» срабатывания SONICS (когда ArtifactNet молчит)
SONICS_STRONG = 0.8
# порог «человек» для SONICS в паре с низким ArtifactNet
SONICS_HUMAN_MAX = 0.5

# Препроцессинг (совпадает с tools/export_sonics_onnx.py и config.json моделей).
N_FFT = 2048
HOP = 512
WIN_LEN = 2048
N_MELS = 128
TOP_DB = 80.0
N_FRAMES = 1 + SONICS_WIN // HOP  # 157
MEL_FILTER = "mel_fb.npy"
N_STFT = N_FFT // 2 + 1  # 1025

DEFAULT_BASE = "https://github.com/ltillegal/ai-music-detector/releases/download/sonics-onnx-v1"
SONICS_ONNX_BASE = os.environ.get("SONICS_ONNX_BASE", DEFAULT_BASE).rstrip("/")

_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN_LEN) / WIN_LEN)).astype(np.float32)
_SESSIONS = None
_FB = None


def _cache_dir():
    override = os.environ.get("SONICS_ONNX_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ai-music-detector" / "sonics-onnx"


def _assets():
    return [*[f"{name.split('/')[-1]}.onnx" for name in SONICS_MODELS], MEL_FILTER]


def available():
    """True, если onnxruntime установлен (ядровая зависимость)."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def models_cached():
    """True, если все артефакты (ONNX + мел-фильтр) уже лежат в кэше."""
    cache = _cache_dir()
    return all((cache / f).is_file() for f in _assets())


def _download(url, dest, progress=None, frac=(0.0, 1.0)):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-music-detector"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (доверенный источник)
        total = int(resp.headers.get("Content-Length") or 0)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            got = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                tmp.write(block)
                got += len(block)
                if progress is not None and total:
                    lo, hi = frac
                    progress(lo + (hi - lo) * got / total, "sonics_model")
            tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def _ensure_assets(progress=None):
    assets = _assets()
    cache = _cache_dir()
    missing = [f for f in assets if not (cache / f).is_file()]
    if not missing:
        return
    n = len(assets)
    for i, name in enumerate(assets):
        if (cache / name).is_file():
            continue
        if progress is not None:
            progress(i / n, "sonics_model")
        _download(f"{SONICS_ONNX_BASE}/{name}", cache / name, progress=progress,
                  frac=(i / n, (i + 1) / n))


def _load_sessions(progress=None):
    global _SESSIONS, _FB
    if _SESSIONS is None:
        import onnxruntime as ort

        _ensure_assets(progress=progress)
        cache = _cache_dir()
        _FB = np.load(cache / MEL_FILTER).astype(np.float32)
        if _FB.ndim != 2 or _FB.shape[1] != N_MELS:
            raise RuntimeError(f"некорректный мел-фильтр: {_FB.shape}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _SESSIONS = []
        for name in SONICS_MODELS:
            path = cache / f"{name.split('/')[-1]}.onnx"
            sess = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            _SESSIONS.append((name.split("/")[-1], sess))
    return _SESSIONS


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


def _windows(y):
    """Все 5-секундные окна @16кГц, каждое нормировано на единичную дисперсию."""
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
    return np.ascontiguousarray(np.stack(wins), dtype=np.float32)


def _logmel(seg, fb):
    """log-mel спектрограмма (128, 157) ровно как torchaudio в исходном SONICS."""
    pad = N_FFT // 2
    yp = np.pad(seg, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(yp, WIN_LEN)[::HOP][:N_FRAMES]
    spec = np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=-1)
    power = (spec.real**2 + spec.imag**2).T  # (1025, n_frames)
    mel = fb.T @ power  # (128, n_frames)
    x_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    x_db = np.maximum(x_db, x_db.max() - TOP_DB)
    std = x_db.reshape(-1).std(ddof=1)
    return ((x_db - x_db.mean()) / (std + 1e-6)).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def score_audio(audio, src_sr, progress=None):
    """audio: моно float. Возвращает dict со срабатываниями каждой модели.

    ``progress(frac, stage)`` вызывается по мере скачивания/загрузки моделей и
    обработки окон.
    """
    y = _resample(np.asarray(audio, dtype=np.float32), src_sr, SONICS_SR)
    wins = _windows(y)
    sessions = _load_sessions(progress=progress)
    fb = _FB

    mels = np.stack([_logmel(seg, fb) for seg in wins]) if len(wins) else np.zeros((0, N_MELS, N_FRAMES), np.float32)

    out = {}
    tops = []
    per_model_windows = []
    n_models = max(len(sessions), 1)
    for i, (name, sess) in enumerate(sessions):
        if progress is not None:
            progress(i / n_models, "sonics")
        logits = sess.run(["logits"], {"melspec": mels})[0]
        probs = _sigmoid(np.asarray(logits).reshape(-1))
        k = min(SONICS_TOP_K, probs.size)
        top_k = float(np.sort(probs)[-k:].mean()) if k else 0.0
        out[name] = {
            "top_k": round(top_k, 6),
            "max": round(float(probs.max()), 6) if probs.size else 0.0,
            "mean": round(float(probs.mean()), 6) if probs.size else 0.0,
            "win_frac": round(float((probs >= 0.5).mean()), 6) if probs.size else 0.0,
            "n_windows": int(probs.size),
        }
        tops.append(top_k)
        per_model_windows.append(probs)
    if progress is not None:
        progress(1.0, "sonics")
    out["p_ai"] = round(float(np.mean(tops)) if tops else 0.0, 6)
    if per_model_windows:
        n = min(len(w) for w in per_model_windows)
        combined = np.mean([w[:n] for w in per_model_windows], axis=0)
        out["windows"] = [round(float(v), 4) for v in combined]
    return out


def _clear_cache():
    """Удалить скачанные артефакты (для тестов/диагностики)."""
    global _SESSIONS, _FB
    _SESSIONS = None
    _FB = None
    cache = _cache_dir()
    if cache.is_dir():
        for f in _assets():
            (cache / f).unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - ручная проверка
    print("cache:", _cache_dir())
    print("base:", SONICS_ONNX_BASE)
    print("available:", available(), "cached:", models_cached())
    print("assets:", _assets(), file=sys.stderr)
