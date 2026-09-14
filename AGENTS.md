# AI Music Detector — инструкции по упаковке

## Структура проекта
- app.py — главный вход (GUI)
- check_ai_music.py — логика детекции
- ui/main.qml — определение интерфейса
- artifactnet_v94_full.onnx — файл модели
- fixtures/ — тестовые аудио

## Команда упаковки
Использовать `uv run pyside6-deploy app.py` для упаковки.

## Ключевые настройки (pysidedeploy.spec)
- title: "AI Music Detector"
- project_dir: "." (не указывать на .venv, иначе упадёт)
- qml_files: "ui/main.qml" (обязательно явно указать, иначе QML не попадёт в bundle)
- icon: если есть .icns — прописать, иначе пропустить

## Зависимости
При упаковке должны быть в окружении:
nuitka, ordered_set, zstandard, onnxruntime, soundfile, sounddevice, PySide6

## Способ проверки
После упаковки выполнить `open "release/AI Music Detector.app"`.

## Важное напоминание
- Не изменять логику в check_ai_music.py, record_and_check.py, test_detector.py
- Работать только с конфигами и командами упаковки
