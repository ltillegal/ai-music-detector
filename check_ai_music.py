#!/usr/bin/env python3
"""CLI-утилита для детекции AI-музыки.

Три сигнала:
- ArtifactNet (ONNX) — отпечаток вокодеров Suno/Udio;
- SONICS (SpecTTTra, по умолчанию) — стиль-ориентированный детектор для
  современных диффузионных моделей (Lyria 3 и т.п.);
- lofcz (fakeprint, по умолчанию) — компактный вокодерный классификатор.

Сигналы совмещаются в вердикт AI / Human / Uncertain (см. `_apply_verdict`).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

import i18n
import lofcz_detector
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

# Бюджет прогресса по стадиям (доли 0..1).
_P_READ = 0.05
_P_ARTIFACT = 0.50
_P_SONICS = 0.90

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


def _emit(progress, frac, stage):
    """Безопасно вызвать progress-колбэк (не должен ломать детекцию)."""
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, float(frac))), stage)
    except Exception:
        pass


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
        help="не запускать SONICS-детектор",
    )
    parser.add_argument(
        "--no-lofcz",
        action="store_true",
        help="не запускать lofcz-детектор",
    )
    parser.add_argument("--lang", default=i18n.DEFAULT_LANG, choices=i18n.LANGUAGES,
                        help="язык предупреждений в выводе")
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


def predict(chunks, model_path, progress=None):
    sess = get_session(model_path)
    probs = []
    n = max(len(chunks), 1)
    for i, chunk in enumerate(chunks):
        out = sess.run(None, {"audio": chunk[np.newaxis, ...]})[0]
        probs.append(float(np.asarray(out).ravel()[0]))
        _emit(progress, (i + 1) / n, "artifactnet")
    return np.asarray(probs, dtype=np.float32)


def _apply_verdict(result, artifact, sonics, lofcz, degraded=False):
    """Совместить ArtifactNet, SONICS и lofcz в вердикт.

    ArtifactNet и lofcz ловят вокодеры Suno/Udio, но слепы к диффузионным
    моделям (Lyria 3). SONICS — стиль-ориентированный, поэтому «человек»
    ставим только когда все доступные сигналы низкие.

    degraded=True — SONICS запрошен, но недоступен: без стиль-детектора нельзя
    уверенно подтвердить «человека», низкие сигналы дают Uncertain.
    """
    sonics_p = sonics["p_ai"] if sonics else None
    lofcz_p = lofcz["p_ai"] if lofcz else None
    available = {
        "artifactnet": artifact,
        "sonics": sonics_p,
        "lofcz": lofcz_p,
    }
    signals = {k: v for k, v in available.items() if v is not None}
    if not signals:
        result["verdict"] = "Uncertain"
        result["reasons"] = ["no_signals"]
        return

    result["p_ai"] = round(max(signals.values()), 6)
    result["signals"] = {k: round(v, 6) for k, v in signals.items()}

    ai_reasons = []
    if artifact is not None and artifact >= THRESHOLD_AI:
        ai_reasons.append("artifactnet")
    if sonics_p is not None and sonics_p >= sonics_detector.SONICS_STRONG:
        ai_reasons.append("sonics")
    if lofcz_p is not None and lofcz_p >= lofcz_detector.LOFCZ_STRONG:
        ai_reasons.append("lofcz")
    if ai_reasons:
        result["verdict"] = "AI"
        result["reasons"] = [f"{r}_ai" for r in ai_reasons]
        return

    human = (
        (artifact is None or artifact <= THRESHOLD_HUMAN)
        and (sonics_p is None or sonics_p <= sonics_detector.SONICS_HUMAN_MAX)
        and (lofcz_p is None or lofcz_p < lofcz_detector.LOFCZ_STRONG)
    )
    if human and not degraded:
        result["verdict"] = "Human"
        result["reasons"] = ["all_low"]
    else:
        result["verdict"] = "Uncertain"
        result["reasons"] = ["sonics_missing"] if degraded else ["mixed_signals"]


def detect(file, model_path=MODEL_PATH, use_sonics=True, use_lofcz=True,
           progress=None, lang=i18n.DEFAULT_LANG):
    _emit(progress, 0.0, "reading")
    audio = load_audio(file)
    duration_sec = round(len(audio) / SR, 3)
    _emit(progress, _P_READ, "reading")

    chunks = make_chunks(audio)
    probs = predict(
        chunks, model_path,
        progress=lambda f, s: _emit(progress, _P_READ + (_P_ARTIFACT - _P_READ) * f, s),
    )
    finite = probs[np.isfinite(probs)]
    result = {
        "file": file,
        "p_ai": None,
        "verdict": "Uncertain",
        "duration_sec": duration_sec,
        "n_chunks": int(len(chunks)),
        "n_valid": int(finite.size),
        "windows": {
            "artifactnet": [None if not np.isfinite(p) else round(float(p), 4) for p in probs],
        },
    }

    warnings = []
    if duration_sec < CHUNK_LEN / SR:
        warnings.append("short_audio")
    if finite.size == 0:
        warnings.append("no_finite")

    artifact = float(np.median(finite)) if finite.size else None
    if artifact is not None:
        result["p_ai_artifactnet"] = round(artifact, 6)

    sonics = None
    sonics_available = sonics_detector.available()
    result["sonics_available"] = sonics_available
    if use_sonics and sonics_available:
        if not sonics_detector.models_cached():
            _emit(progress, _P_ARTIFACT, "downloading")
        try:
            sonics = sonics_detector.score_audio(
                audio, SR,
                progress=lambda f, s: _emit(
                    progress, _P_ARTIFACT + (_P_SONICS - _P_ARTIFACT) * f, s
                ),
            )
            result["sonics"] = sonics
            result["p_ai_sonics"] = sonics["p_ai"]
            if "windows" in sonics:
                result["windows"]["sonics"] = sonics["windows"]
        except Exception as exc:  # детектор не должен ломать основной результат
            result["sonics_error"] = str(exc)
            warnings.append("sonics_error")
    elif use_sonics:
        warnings.append("sonics_unavailable")

    lofcz = None
    lofcz_available = lofcz_detector.available()
    result["lofcz_available"] = lofcz_available
    if use_lofcz and lofcz_available:
        _emit(progress, _P_SONICS, "lofcz")
        try:
            lofcz = lofcz_detector.score_audio(audio, SR)
            result["lofcz"] = lofcz
            result["p_ai_lofcz"] = lofcz["p_ai"]
        except Exception as exc:
            result["lofcz_error"] = str(exc)
            warnings.append("lofcz_error")
    elif use_lofcz:
        warnings.append("lofcz_unavailable")

    _emit(progress, 0.99, "finalizing")

    result["warnings"] = warnings
    if warnings:
        result["warning"] = "; ".join(i18n.tr(lang, f"warn_{w}") for w in warnings)

    degraded = use_sonics and sonics is None
    _apply_verdict(result, artifact, sonics, lofcz, degraded=degraded)
    _emit(progress, 1.0, "done")
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = detect(
            args.file, args.model,
            use_sonics=not args.no_sonics,
            use_lofcz=not args.no_lofcz,
            lang=args.lang,
        )
    except Exception as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
