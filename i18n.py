#!/usr/bin/env python3
"""Строки интерфейса GUI на английском и русском.

Ключи общие для всех языков; ``tr()`` отдаёт строку для выбранного языка и
падает обратно на английский, если перевода нет.
"""

DEFAULT_LANG = "ru"
LANGUAGES = ("ru", "en")

STRINGS = {
    "ru": {
        "subtitle": "ArtifactNet + SONICS — детекция AI-музыки",
        "choose_file": "Выбрать файл",
        "record_mic": "Запись с микрофона",
        "drop_hint": "Перетащите аудиофайл сюда",
        "formats_hint": "wav · mp3 · flac — или выберите файл / запишите с микрофона",
        "analyzing": "Анализирую…",
        "bar_artifactnet": "ArtifactNet (вокодеры Suno/Udio)",
        "bar_sonics": "SONICS (стиль, диффузионные модели)",
        "show_details": "Показать детали",
        "hide_details": "Скрыть детали",
        "sonics_warn": "SONICS не активен — детекция ограничена ArtifactNet, "
                       "вердикт Human не выдаётся. Установите: uv sync",
    },
    "en": {
        "subtitle": "ArtifactNet + SONICS — AI music detection",
        "choose_file": "Choose file",
        "record_mic": "Record from microphone",
        "drop_hint": "Drop an audio file here",
        "formats_hint": "wav · mp3 · flac — or choose a file / record from mic",
        "analyzing": "Analyzing…",
        "bar_artifactnet": "ArtifactNet (Suno/Udio vocoders)",
        "bar_sonics": "SONICS (style, diffusion models)",
        "show_details": "Show details",
        "hide_details": "Hide details",
        "sonics_warn": "SONICS is not active — detection is limited to ArtifactNet "
                       "and the Human verdict is suppressed. Install with: uv sync",
    },
}


def tr(lang, key):
    """Строка ``key`` для языка ``lang`` (fallback — английский, затем ключ)."""
    table = STRINGS.get(lang) or STRINGS[DEFAULT_LANG]
    if key in table:
        return table[key]
    return STRINGS[DEFAULT_LANG].get(key, key)
