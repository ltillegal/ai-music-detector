"""Общая настройка тестов."""

import os
from pathlib import Path

import pytest

import check_ai_music
import sonics_detector

# Тесты используют относительные пути (модель, fixtures) — фиксируем cwd на корень проекта,
# чтобы `pytest` работал из любого каталога.
os.chdir(Path(__file__).resolve().parent)


@pytest.fixture(autouse=True)
def _sonics_offline(monkeypatch):
    """Не даём тестам скачивать ~138 МБ ONNX-моделей SONICS, если их нет в кэше.

    SONICS теперь работает на onnxruntime, но ONNX-модели скачиваются лениво.
    В CI/на чистой машине их нет, поэтому считаем SONICS недоступным, чтобы
    тесты не лезли в сеть. Когда кэш есть (или задан ``SONICS_ONNX_DIR``),
    интеграционные тесты с моделями идут полностью.
    """
    if not sonics_detector.models_cached():
        monkeypatch.setattr(check_ai_music.sonics_detector, "available", lambda: False)
    yield
