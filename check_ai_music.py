#!/usr/bin/env python3
"""CLI-утилита для детекции AI-музыки.

Два сигнала:
- ArtifactNet (ONNX) — отпечаток вокодеров Suno/Udio;
- SONICS (SpecTTTra, опционально) — стиль-ориентированный детектор для
  современных диффузионных моделей (Lyria 3 и т.п.).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

import sonics_detector

try:
    import soxr
except ImportError:  # pragma: no cover - опциональная зависимость
    soxr = None

SR = 44100
CHUNK_LEN = 4 * SR  # 176400
MODEL_FILENAME = "artifactnet_v94_full.onnx"
THRESHOLD_AI = 0.6
THRESHOLD_HUMAN = 0.4

_SESSION_CACHE = {}


def _default_model_path():
    candidates = [
        Path(__file__).resolve().parent / MODEL_FILENAME,
        Path(sys.executable).resolve().parent / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return str(candidates[0])


MODEL_PATH = _default_model_path()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="check_ai_music.py",
        description="Детекция AI-музыки через ONNX-модель ArtifactNet",
    )
    parser.add_argument("file", help="путь к аудиофайлу (wav/mp3/flac)")
    parser.add_argument("--model", default=MODEL_PATH, help="путь к ONNX-модели")
    parser.add_argument(
        "--no-sonics",
        action="store_true",
        help="не запускать SONICS-детектор (только ArtifactNet)",
    )
    args = parser.parse_args(argv)
    return args


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # стерео → моно
    if sr != SR:
        audio = _resample(audio, sr, SR)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def _resample(audio, src_sr, dst_sr):
    if len(audio) == 0 or src_sr == dst_sr:
        return audio
    if soxr is not None:
        return soxr.resample(audio, src_sr, dst_sr).astype(np.float32)
    # запасной вариант, если soxr не установлен: линейная интерполяция
    src_t = np.arange(len(audio)) / src_sr
    dst_t = np.arange(int(len(audio) * dst_sr / src_sr)) / dst_sr
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def make_chunks(audio):
    if len(audio) == 0:
        return np.empty((0, CHUNK_LEN), dtype=np.float32)
    n_chunks = math.ceil(len(audio) / CHUNK_LEN)
    total = n_chunks * CHUNK_LEN
    if len(audio) < total:
        # добиваем повтором ("wrap"), а не нулями: тишина заставляет модель выдавать NaN
        audio = np.pad(audio, (0, total - len(audio)), mode="wrap")
    return audio.reshape(n_chunks, CHUNK_LEN).astype(np.float32)


def get_session(model_path):
    key = str(Path(model_path).resolve())
    sess = _SESSION_CACHE.get(key)
    if sess is None:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _SESSION_CACHE[key] = sess
    return sess


def predict(chunks, model_path):
    sess = get_session(model_path)
    probs = []
    for chunk in chunks:
        out = sess.run(None, {"audio": chunk[np.newaxis, ...]})[0]
        probs.append(float(np.asarray(out).ravel()[0]))
    return np.asarray(probs, dtype=np.float32)


def _apply_verdict(result, artifact, sonics):
    """Совместить ArtifactNet (физика вокодера) и SONICS (стиль).

    ArtifactNet ловит Suno/Udio, но слеп к диффузионным моделям (Lyria 3).
    SONICS наоборот — стиль-ориентированный, поэтому «человек» ставим только
    когда оба сигнала низкие.
    """
    sonics_p = sonics["p_ai"] if sonics else None
    signals = [p for p in (artifact, sonics_p) if p is not None]
    if not signals:
        result["verdict"] = "Uncertain"
        return
    result["p_ai"] = round(max(signals), 6)

    if artifact is not None and artifact >= THRESHOLD_AI:
        result["verdict"] = "AI"
        return
    if sonics_p is not None and sonics_p >= sonics_detector.SONICS_STRONG:
        result["verdict"] = "AI"
        return
    human_artifact = artifact is None or artifact <= THRESHOLD_HUMAN
    human_sonics = sonics_p is None or sonics_p <= sonics_detector.SONICS_HUMAN_MAX
    result["verdict"] = "Human" if (human_artifact and human_sonics) else "Uncertain"


def detect(file, model_path=MODEL_PATH, use_sonics=True):
    audio = load_audio(file)
    duration_sec = round(len(audio) / SR, 3)
    chunks = make_chunks(audio)
    probs = predict(chunks, model_path)
    finite = probs[np.isfinite(probs)]
    result = {
        "file": file,
        "p_ai": None,
        "verdict": "Uncertain",
        "duration_sec": duration_sec,
        "n_chunks": int(len(chunks)),
        "n_valid": int(finite.size),
    }
    warnings = []
    if duration_sec < CHUNK_LEN / SR:
        warnings.append("аудио короче 4 c — результат ненадёжен")
    if finite.size == 0:
        warnings.append("модель не вернула финитных вероятностей")
    if warnings:
        result["warning"] = "; ".join(warnings)

    artifact = float(np.median(finite)) if finite.size else None
    if artifact is not None:
        result["p_ai_artifactnet"] = round(artifact, 6)

    sonics = None
    if use_sonics and sonics_detector.available():
        try:
            sonics = sonics_detector.score_audio(audio, SR)
            result["sonics"] = sonics
            result["p_ai_sonics"] = sonics["p_ai"]
        except Exception as exc:  # сам детектор не должен ломать основной результат
            result["sonics_error"] = str(exc)

    _apply_verdict(result, artifact, sonics)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = detect(args.file, args.model, use_sonics=not args.no_sonics)
    except Exception as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
