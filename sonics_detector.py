"""SONICS (SpecTTTra) spectrogram detector for AI-generated music.

Unlike ArtifactNet/lofcz, which look for a physical vocoder fingerprint
(periodic peaks from transposed convolutions), SONICS is a style-based
spectrogram transformer. It catches modern diffusion generators (for example
Google Lyria 3 in producer.ai) that leave no vocoder fingerprint.

The detector runs on ``onnxruntime`` (already a core dependency for
ArtifactNet), so torch/timm/fvcore are NOT needed: the SpecTTTra model is
exported to ONNX and the log-mel preprocessing is computed in numpy. The
ready-made ``.onnx`` files (~69 MB each) and the mel filter are downloaded on
first run, cached, and verified against the release ``manifest.json``.

Public API: ``available()``, ``models_cached()``, ``score_audio()``.

Environment variables:
    SONICS_MODELS      comma-separated model base names
    SONICS_ONNX_DIR    cache directory (default ~/.cache/ai-music-detector/sonics-onnx)
    SONICS_ONNX_BASE   base URL for release artifacts
    SONICS_TOP_K       how many top windows to average for the score
"""

import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SONICS_SR = 16000
SONICS_WIN = SONICS_SR * 5  # 5 seconds = 80000 samples
SONICS_STRIDE = SONICS_WIN // 2  # 2.5 s
SONICS_TOP_K = int(os.environ.get("SONICS_TOP_K", "3"))
SONICS_MODELS = [
    m.strip()
    for m in os.environ.get(
        "SONICS_MODELS",
        "sonics-spectttra-gamma-5s,sonics-spectttra-beta-5s",
    ).split(",")
    if m.strip()
]

# Confidence threshold for SONICS when ArtifactNet is silent.
SONICS_STRONG = 0.8
# Human threshold for SONICS paired with a low ArtifactNet score.
SONICS_HUMAN_MAX = 0.5

# Preprocessing (matches tools/export_sonics_onnx.py and the model config.json).
N_FFT = 2048
HOP = 512
WIN_LEN = 2048
N_MELS = 128
TOP_DB = 80.0
N_FRAMES = 1 + SONICS_WIN // HOP  # 157
MEL_FILTER = "mel_fb.npy"
N_STFT = N_FFT // 2 + 1  # 1025

MANIFEST = "manifest.json"
DOWNLOAD_TIMEOUT = 60
_CHUNK = 1 << 20

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
    """True when onnxruntime is installed (a core dependency)."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def models_cached():
    """True when all ONNX/mel artifacts exist (existence only, works offline)."""
    cache = _cache_dir()
    return all((cache / name).is_file() for name in _assets())


def _validate_base():
    parsed = urllib.parse.urlparse(SONICS_ONNX_BASE)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("SONICS_ONNX_BASE must be an http(s) URL")


def _fetch_manifest():
    """Return {filename: {'bytes': int, 'sha256': str}} from the release."""
    _validate_base()
    url = f"{SONICS_ONNX_BASE}/{MANIFEST}"
    req = urllib.request.Request(url, headers={"User-Agent": "ai-music-detector"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
        raw = resp.read()

    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid SONICS manifest") from exc
    if not isinstance(data, list):
        raise ValueError("invalid SONICS manifest")

    entries = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("invalid SONICS manifest entry")
        name = item.get("file")
        digest = item.get("sha256")
        size = item.get("bytes")
        if not isinstance(name, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise ValueError(f"incomplete SONICS manifest entry: {name!r}")
        entries[name] = {"bytes": size, "sha256": digest.lower()}
    return entries


def _file_ok(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest() == expected["sha256"]


def _download(url, dest, expected, progress=None, frac=(0.0, 1.0)):
    """Download one artifact and verify its size and sha256 before replacing dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-music-detector"})
    tmp_path = None
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            with tempfile.NamedTemporaryFile(
                dir=dest.parent, prefix=".sonics_", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                got = 0
                digest = hashlib.sha256()
                while True:
                    block = resp.read(_CHUNK)
                    if not block:
                        break
                    tmp.write(block)
                    digest.update(block)
                    got += len(block)
                    if progress is not None and total:
                        lo, hi = frac
                        progress(lo + (hi - lo) * got / total, "sonics_model")

        actual = digest.hexdigest()
        if got != expected["bytes"] or actual != expected["sha256"]:
            raise ValueError(
                f"checksum mismatch for {dest.name}: "
                f"expected {expected['bytes']} bytes/{expected['sha256']}, "
                f"got {got} bytes/{actual}"
            )
        tmp_path.replace(dest)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _ensure_assets(progress=None):
    names = _assets()
    cache = _cache_dir()
    all_present = all((cache / name).is_file() for name in names)

    try:
        entries = _fetch_manifest()
    except Exception as exc:
        if all_present:
            # A complete cache still works offline; verification resumes on
            # the next run that has network access.
            print(f"SONICS cache verification skipped: {exc}", file=sys.stderr)
            return
        raise

    missing_entries = [name for name in names if name not in entries]
    if missing_entries:
        raise ValueError(f"SONICS manifest is missing: {', '.join(missing_entries)}")

    cache.mkdir(parents=True, exist_ok=True)
    n = len(names)
    for i, name in enumerate(names):
        expected = entries[name]
        dest = cache / name
        if _file_ok(dest, expected):
            if progress is not None:
                progress((i + 1) / n, "sonics_model")
            continue
        if progress is not None:
            progress(i / n, "sonics_model")
        _download(
            f"{SONICS_ONNX_BASE}/{name}",
            dest,
            expected,
            progress=progress,
            frac=(i / n, (i + 1) / n),
        )


def _load_sessions(progress=None):
    global _SESSIONS, _FB
    if _SESSIONS is None:
        import onnxruntime as ort

        _ensure_assets(progress=progress)
        cache = _cache_dir()
        _FB = np.load(cache / MEL_FILTER).astype(np.float32)
        if _FB.ndim != 2 or _FB.shape[1] != N_MELS:
            raise RuntimeError(f"invalid mel filter: {_FB.shape}")
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
    """All 5-second windows at 16 kHz, each normalised to unit variance."""
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
    """Log-mel spectrogram (128, 157), matching torchaudio in upstream SONICS."""
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
    """audio: mono float. Returns per-model detections.

    ``progress(frac, stage)`` is called while downloading/loading models and
    processing windows.
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
    """Delete downloaded artifacts (for tests/diagnostics)."""
    global _SESSIONS, _FB
    _SESSIONS = None
    _FB = None
    cache = _cache_dir()
    if cache.is_dir():
        for name in _assets():
            (cache / name).unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - manual check
    print("cache:", _cache_dir())
    print("base:", SONICS_ONNX_BASE)
    print("available:", available(), "cached:", models_cached())
    print("assets:", _assets(), file=sys.stderr)
