# AI Music Detector — инструкции по упаковке

## Структура проекта
- app.py — главный вход (GUI)
- check_ai_music.py — логика детекции (ArtifactNet + SONICS + lofcz и вердикт)
- sonics_detector.py — SONICS-детектор (SpecTTTra на onnxruntime, mel в numpy)
- tools/export_sonics_onnx.py — пересборка ONNX-моделей SONICS (группа `export`)
- lofcz_detector.py — lofcz-детектор (fakeprint + ONNX)
- i18n.py — строки интерфейса (en/ru)
- ui/main.qml — определение интерфейса, ui/icon.png — иконка
- artifactnet_v94_full.onnx[.data], lofcz_ai_music_detector.onnx,
  lofcz_preprocessing.json — файлы моделей в репозитории
- fixtures/ — тестовые аудио

## Команда упаковки
Использовать `uv run pyside6-deploy app.py` (torch в окружении больше нет).

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

SONICS больше НЕ требует torch: детектор работает на onnxruntime (ядровая
зависимость), а log-mel считается в numpy. ONNX-модели SONICS (~69 МБ каждая,
собраны из awsaf49/sonics-spectttra-*) и мел-фильтр скачиваются при первом
запуске в `~/.cache/ai-music-detector/sonics-onnx` (переопределяется
`SONICS_ONNX_DIR`; источник — `SONICS_ONNX_BASE`, по умолчанию релиз
`sonics-onnx-v1`). Тяжёлые torch/timm/sonics остались ТОЛЬКО в группе `export`
(не в default-groups) — для пересборки ONNX.

## Способ проверки
После упаковки выполнить `open "release/AI Music Detector.app"`.

## Важное напоминание
- Логика детекции — в `check_ai_music.py` (+ `sonics_detector.py`, `lofcz_detector.py`).
- SONICS активен всегда (onnxruntime в ядре); нужны лишь скачанные ONNX-модели.
- Если SONICS запрошен, но не отработал (нет моделей/сеть) — вердикт `Human`
  подавляется (`Uncertain` + `warning`), см. `_apply_verdict(..., degraded=True)`.
- GUI показывает прогресс/стадию, причины вердикта, таймлайн окон; язык и тему
  можно переключать (хранится в QSettings).
- Пересборка ONNX: `uv run --group export python tools/export_sonics_onnx.py --out dist/sonics-onnx`,
  затем `gh release upload sonics-onnx-v1 dist/sonics-onnx/<файлы>`.
- GUI показывает прогресс/стадию, причины вердикта, таймлайн окон; язык и тему
  можно переключать (хранится в QSettings).
