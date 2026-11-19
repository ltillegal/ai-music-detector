"""CLI/GUI interface strings in English and Russian.

Keys are shared across languages; ``tr()`` returns the string for the selected
language and falls back to the default language when a translation is missing.
"""

DEFAULT_LANG = "en"
LANGUAGES = ("en", "ru")

STRINGS = {
    "en": {
        # Title and common labels
        "subtitle": "ArtifactNet + SONICS + lofcz — AI music detection",
        "choose_file": "Choose file",
        "record_mic": "Record from microphone",
        "drop_hint": "Drop an audio file here",
        "formats_hint": "wav · mp3 · flac — or choose a file / record from mic",
        "analyzing": "Analyzing…",
        "show_details": "Show details",
        "hide_details": "Hide details",
        "theme_dark": "Dark theme",
        "theme_light": "Light theme",
        # Signal bars
        "bar_artifactnet": "ArtifactNet (Suno/Udio vocoders)",
        "bar_sonics": "SONICS (style, diffusion models)",
        "bar_lofcz": "lofcz (vocoder fakeprint)",
        # Verdicts
        "verdict_ai": "AI",
        "verdict_human": "Human",
        "verdict_uncertain": "Uncertain",
        "verdict_error": "Error",
        # Analysis stages
        "stage_reading": "Reading file…",
        "stage_artifactnet": "ArtifactNet…",
        "stage_downloading": "Downloading SONICS models (~140 MB)…",
        "stage_sonics_model": "Loading SONICS model…",
        "stage_sonics": "SONICS…",
        "stage_lofcz": "lofcz…",
        "stage_finalizing": "Finalizing…",
        "stage_done": "Done",
        # Verdict reasons
        "reason_artifactnet_ai": "ArtifactNet: vocoder fingerprint ≥ 0.6",
        "reason_sonics_ai": "SONICS: style detector ≥ 0.8",
        "reason_lofcz_ai": "lofcz: vocoder fakeprint ≥ 0.5",
        "reason_all_low": "all signals below thresholds",
        "reason_mixed_signals": "signals disagree",
        "reason_sonics_missing": "SONICS unavailable — Human not confirmed",
        "reason_no_signals": "no available signals",
        "reason_silent_audio": "audio is too quiet (silence)",
        # Warnings
        "warn_short_audio": "audio shorter than 4 s — unreliable result",
        "warn_silent_audio": "audio is nearly silent — unreliable result",
        "warn_no_finite": "the model returned no finite probabilities",
        "warn_sonics_unavailable": "SONICS unavailable — limited to ArtifactNet/lofcz",
        "warn_sonics_error": "SONICS failed",
        "warn_lofcz_unavailable": "lofcz unavailable",
        "warn_lofcz_error": "lofcz failed",
        # Timeline
        "timeline_title": "Over time",
        "timeline_artifactnet": "ArtifactNet (4 s)",
        "timeline_sonics": "SONICS (5 s)",
        # SONICS banner
        "sonics_warn": "SONICS is not active — detection is limited to ArtifactNet/lofcz "
                       "and the Human verdict is suppressed. Install with: uv sync",
    },
    "ru": {
        # Title and common labels
        "subtitle": "ArtifactNet + SONICS + lofcz — детекция AI-музыки",
        "choose_file": "Выбрать файл",
        "record_mic": "Запись с микрофона",
        "drop_hint": "Перетащите аудиофайл сюда",
        "formats_hint": "wav · mp3 · flac — или выберите файл / запишите с микрофона",
        "analyzing": "Анализирую…",
        "show_details": "Показать детали",
        "hide_details": "Скрыть детали",
        "theme_dark": "Тёмная тема",
        "theme_light": "Светлая тема",
        # Signal bars
        "bar_artifactnet": "ArtifactNet (вокодеры Suno/Udio)",
        "bar_sonics": "SONICS (стиль, диффузионные модели)",
        "bar_lofcz": "lofcz (fakeprint вокодера)",
        # Verdicts
        "verdict_ai": "AI",
        "verdict_human": "Человек",
        "verdict_uncertain": "Неясно",
        "verdict_error": "Ошибка",
        # Analysis stages
        "stage_reading": "Чтение файла…",
        "stage_artifactnet": "ArtifactNet…",
        "stage_downloading": "Скачивание моделей SONICS (~140 МБ)…",
        "stage_sonics_model": "Загрузка модели SONICS…",
        "stage_sonics": "SONICS…",
        "stage_lofcz": "lofcz…",
        "stage_finalizing": "Завершение…",
        "stage_done": "Готово",
        # Verdict reasons
        "reason_artifactnet_ai": "ArtifactNet: отпечаток вокодера ≥ 0.6",
        "reason_sonics_ai": "SONICS: стиль-детектор ≥ 0.8",
        "reason_lofcz_ai": "lofcz: fakeprint вокодера ≥ 0.5",
        "reason_all_low": "все сигналы ниже порогов",
        "reason_mixed_signals": "сигналы противоречивы",
        "reason_sonics_missing": "SONICS недоступен — «человек» не подтверждён",
        "reason_no_signals": "нет доступных сигналов",
        "reason_silent_audio": "аудио слишком тихое (тишина)",
        # Warnings
        "warn_short_audio": "аудио короче 4 c — результат ненадёжен",
        "warn_silent_audio": "аудио почти бесшумно — результат ненадёжен",
        "warn_no_finite": "модель не вернула финитных вероятностей",
        "warn_sonics_unavailable": "SONICS недоступен — детекция ограничена ArtifactNet/lofcz",
        "warn_sonics_error": "SONICS завершился с ошибкой",
        "warn_lofcz_unavailable": "lofcz недоступен",
        "warn_lofcz_error": "lofcz завершился с ошибкой",
        # Timeline
        "timeline_title": "По времени",
        "timeline_artifactnet": "ArtifactNet (4 с)",
        "timeline_sonics": "SONICS (5 с)",
        # SONICS banner
        "sonics_warn": "SONICS не активен — детекция ограничена ArtifactNet/lofcz, "
                       "вердикт «человек» не выдаётся. Установите: uv sync",
    },
}


def tr(lang, key):
    """Return ``key`` for ``lang`` (fallback: default language, then the key)."""
    table = STRINGS.get(lang) or STRINGS[DEFAULT_LANG]
    if key in table:
        return table[key]
    return STRINGS[DEFAULT_LANG].get(key, key)
