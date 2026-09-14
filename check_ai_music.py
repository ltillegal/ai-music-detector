#!/usr/bin/env python3
"""CLI-утилита для детекции AI-музыки через ONNX-модель ArtifactNet."""

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 44100
CHUNK_LEN = 4 * SR  # 176400
MODEL_PATH = "./artifactnet_v94_full.onnx"
THRESHOLD_AI = 0.6
THRESHOLD_HUMAN = 0.4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="check_ai_music.py",
        description="Детекция AI-музыки через ONNX-модель ArtifactNet",
    )
    parser.add_argument("file", help="путь к аудиофайлу (wav/mp3/flac)")
    parser.add_argument("--model", default=MODEL_PATH, help="путь к ONNX-модели")
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
    if len(audio) == 0:
        return audio
    if src_sr == dst_sr:
        return audio
    src_t = np.arange(len(audio)) / src_sr
    dst_t = np.arange(int(len(audio) * dst_sr / src_sr)) / dst_sr
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def make_chunks(audio):
    if len(audio) < CHUNK_LEN:
        audio = np.pad(audio, (0, CHUNK_LEN - len(audio)))
    n_chunks = len(audio) // CHUNK_LEN
    audio = audio[: n_chunks * CHUNK_LEN]
    return audio.reshape(n_chunks, CHUNK_LEN).astype(np.float32)


def predict(chunks, model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    probs = []
    for chunk in chunks:
        out = sess.run(None, {"audio": chunk[np.newaxis, ...]})[0]
        probs.append(float(np.asarray(out).ravel()[0]))
    return np.asarray(probs, dtype=np.float32)


def make_verdict(p_ai):
    if p_ai > THRESHOLD_AI:
        return "AI"
    if p_ai < THRESHOLD_HUMAN:
        return "Human"
    return "Uncertain"


def detect(file, model_path=MODEL_PATH):
    audio = load_audio(file)
    chunks = make_chunks(audio)
    probs = predict(chunks, model_path)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return {"file": file, "p_ai": None, "verdict": "Uncertain"}
    p_ai = float(np.median(probs))
    return {"file": file, "p_ai": round(p_ai, 6), "verdict": make_verdict(p_ai)}


def main(argv=None):
    args = parse_args(argv)
    result = detect(args.file, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())