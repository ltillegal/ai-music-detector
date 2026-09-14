#!/usr/bin/env python3
"""Запись 10 секунд с микрофона и детекция AI-музыки."""

import argparse
import json
import sys
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf

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


def record_and_detect(seconds=DURATION, model=None, output=None):
    audio = record(seconds)
    if output is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name
    else:
        wav_path = output
    sf.write(wav_path, audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    return detect(wav_path, model) if model else detect(wav_path)


def main(argv=None):
    args = parse_args(argv)
    result = record_and_detect(args.seconds, args.model, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())