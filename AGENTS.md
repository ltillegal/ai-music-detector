# AI Music Detector — инструкции по упаковке

## Структура проекта
- app.py — главный вход (GUI)
- check_ai_music.py — логика детекции (ArtifactNet + объединение с SONICS)
- sonics_detector.py — опциональный SONICS-детектор (SpecTTTra)
- ui/main.qml — определение интерфейса
- artifactnet_v94_full.onnx — файл модели
- fixtures/ — тестовые аудио

## Команда упаковки
Использовать `uv run --no-group sonics pyside6-deploy app.py` для упаковки
(иначе `uv run` синхронизирует default-groups и потащит torch в bundle).

## Ключевые настройки (pysidedeploy.spec)
- title: "AI Music Detector"
- project_dir: "." (не указывать на .venv, иначе упадёт)
- qml_files: "ui/main.qml" (обязательно явно указать, иначе QML не попадёт в bundle)
- icon: если есть .icns — прописать, иначе пропустить

## Зависимости
При упаковке должны быть в окружении:
nuitka, ordered_set, zstandard, onnxruntime, soundfile, sounddevice, PySide6

SONICS (torch/torchaudio/timm/fvcore/sonics) — входит в `default-groups`
(`sonics`), поэтому обычный `uv sync`/`uv run` его ставит. Для лёгкой упаковки
`.app` используйте `uv sync --no-group sonics` и `uv run --no-group sonics ...`:
без torch SONICS сам отключается (`available()` → False). Если включать его в
bundle, размер .app сильно вырастет (десятки-сотни МБ из-за torch); тогда
дополнительно нужен `--include-data-files` для кэшированных весов или их
скачивание при первом запуске.

## Способ проверки
После упаковки выполнить `open "release/AI Music Detector.app"`.

## Важное напоминание
- Логика детекции — в `check_ai_music.py` (+ `sonics_detector.py`).
- SONICS активируется автоматически, если import torch/sonics проходит.
- Без SONICS вердикт `Human` подавляется (`Uncertain` + `warning`), чтобы не
  пропустить AI-треки (см. `_apply_verdict(..., degraded=True)`).
