# AI Music Detector — инструкции по упаковке

## Структура проекта
- app.py — главный вход (GUI)
- check_ai_music.py — логика детекции (ArtifactNet + SONICS + lofcz и вердикт)
- sonics_detector.py — SONICS-детектор (SpecTTTra)
- lofcz_detector.py — lofcz-детектор (fakeprint + ONNX)
- i18n.py — строки интерфейса (en/ru)
- ui/main.qml — определение интерфейса, ui/icon.png — иконка
- artifactnet_v94_full.onnx[.data], lofcz_ai_music_detector.onnx,
  lofcz_preprocessing.json — файлы моделей
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
nuitka, ordered_set, zstandard, onnxruntime, scipy, soundfile, sounddevice, PySide6

Артефакты нужно включать в bundle через `--include-data-files` (см. spec):
ArtifactNet (.onnx + .onnx.data), lofcz (.onnx + конфиг), ui/icon.png.

SONICS (torch/torchaudio/timm/fvcore/sonics) — входит в `default-groups`
(`sonics`), поэтому обычный `uv sync`/`uv run` его ставит. Для лёгкой упаковки
`.app` используйте `uv sync --no-group sonics` и `uv run --no-group sonics ...`:
без torch SONICS сам отключается (`available()` → False), а ArtifactNet/lofcz
продолжают работать. Если включать SONICS в bundle, размер .app сильно вырастет
(десятки-сотни МБ из-за torch); тогда дополнительно нужен `--include-data-files`
для кэшированных весов или их скачивание при первом запуске.

## Способ проверки
После упаковки выполнить `open "release/AI Music Detector.app"`.

## Важное напоминание
- Логика детекции — в `check_ai_music.py` (+ `sonics_detector.py`, `lofcz_detector.py`).
- SONICS активируется автоматически, если import torch/sonics проходит.
- Без SONICS вердикт `Human` подавляется (`Uncertain` + `warning`), чтобы не
  пропустить AI-треки (см. `_apply_verdict(..., degraded=True)`).
- GUI показывает прогресс/стадию, причины вердикта, таймлайн окон; язык и тему
  можно переключать (хранится в QSettings).
