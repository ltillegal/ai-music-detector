#!/usr/bin/env python3
"""Запись 10 секунд с микрофона и детекция AI-музыки."""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf

import i18n
from check_ai_music import detect

DURATION = 10
SAMPLE_RATE = 44100


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="record_and_check.py",
        description="Запись 10с аудио и детекция AI-музыки через ArtifactNet",
    )
    parser.add_argument("--seconds", type=float, default=DURATION, help="длительность записи, c")
    parser.add_argument("--model", default=None, help="путь к ONNX-модели (по умолчанию из check_ai_music)")
    parser.add_argument("--output", default=None, help="сохранить wav в указанный файл (иначе temp)")
    args = parser.parse_args(argv)
    return args


def record(seconds, samplerate=SAMPLE_RATE):
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def record_and_detect(seconds=DURATION, model=None, output=None, progress=None, lang=i18n.DEFAULT_LANG):
    audio = record(seconds)
    created_tmp = output is None
    if created_tmp:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
    else:
        wav_path = output
    try:
        sf.write(wav_path, audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        kwargs = {"progress": progress, "lang": lang}
        if model:
            kwargs["model_path"] = model
        return detect(wav_path, **kwargs)
    finally:
        if created_tmp:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def main(argv=None):
    args = parse_args(argv)
    try:
        result = record_and_detect(args.seconds, args.model, args.output)
    except Exception as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())