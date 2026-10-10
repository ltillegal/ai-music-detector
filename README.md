# AI Music Detector

Детектор AI-сгенерированной музыки. Использует два независимых сигнала:

- **ArtifactNet** (ONNX) — «физический» отпечаток вокодеров Suno/Udio
  (периодические артефакты транспонированных свёрток в 1–8 кГц).
- **SONICS** (SpecTTTra, опционально) — стиль-ориентированный
  spectrogram-transformer; ловит современные диффузионные генераторы
  (например Google Lyria 3 в producer.ai), к которым ArtifactNet слеп.

Приложение анализирует аудио и возвращает вероятность того, что трек создан ИИ.

## Возможности

- Детекция по аудиофайлу (`wav` / `mp3` / `flac`) — CLI.
- Запись 10 секунд с микрофона и детекция — CLI и GUI.
- Графический интерфейс на PySide6 + QML.
- Упаковка в macOS `.app` через `pyside6-deploy` (Nuitka).

## Как работает

Аудио приводится к моно, 44.1 кГц и нарезается на окна по 4 секунды
(`176400` сэмплов), затем каждое окно прогоняется через ONNX-модель
**ArtifactNet**. Итоговая вероятность — медиана по окнам.

Если установлена опциональная группа `sonics`, параллельно запускается
**SONICS**: аудио ресемплируется в 16 кГц, нарезается на окна по 5 с
(шаг 2.5 с), каждое окно нормируется на единичную дисперсию и прогоняется
через две модели SpecTTTra (`gamma-5s`, `beta-5s`). По каждой модели берётся
среднее top-3 окон.

Итоговый вердикт `verdict`:

- `AI` — если ArtifactNet `≥ 0.6` **или** SONICS `≥ 0.8`;
- `Human` — если оба сигнала низкие (ArtifactNet `≤ 0.4` и SONICS `≤ 0.5`);
- иначе `Uncertain`.

В JSON также выводятся поля `p_ai_artifactnet`, `p_ai_sonics` и подробности
`sonics` по каждой модели (`top_k`, `max`, `mean`, `win_frac`).
`win_frac` — доля «горячих» окон: средние значения указывают на
гибридный контент (AI-фрагмент внутри человеческого трека).

Артефакты перекодирования в MP3 ниже 48 kbps и «физический» отпечаток
вокодера не являются надёжным признаком для диффузионных моделей — именно
для таких случаев нужен SONICS.

## Требования

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- Микрофон (только для записи)
- Для SONICS: `torch`, `torchaudio` (на CPU), `timm`, `fvcore`, `sonics`

## Установка

```bash
uv sync
```

### SONICS (опционально)

SONICS включается автоматически, если импортируются `torch` и `sonics`.
Для установки:

```bash
# CPU-сборка torch (меньше и быстрее на ноутбуках)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# остальные зависимости + сам пакет SONICS
uv pip install "sonics @ git+https://github.com/awsaf49/sonics.git" timm fvcore
```

Веса SONICS (`gamma-5s`, `beta-5s`, ~69 МБ каждая) скачиваются с HuggingFace
при первом запуске и кэшируются в `HF_HOME`. Инференс на CPU — около
2 c на минуту аудио на модель.

Отключить SONICS можно флагом `--no-sonics` у CLI (используется только
ArtifactNet).

> Внимание: обычный `uv sync` (без `--extra sonics`) удаляет вручную
> установленные `torch`/`sonics` из окружения. После `uv sync` повторите
> команды установки SONICS выше.

## Использование

### CLI: файл

```bash
uv run python check_ai_music.py track.wav
```

### CLI: микрофон

```bash
uv run python record_and_check.py --seconds 10
# сохранить запись: --output rec.wav
```

### GUI

```bash
uv run python app.py
# или
./run.sh app.py
```

## Тесты

```bash
uv run pytest
```

Тесты используют файлы из `fixtures/` (`known_suno.wav`, `known_human.wav`).
Тесты SONICS автоматически пропускаются, если `torch`/`sonics` не установлены.

## Модели

- **ArtifactNet**: `artifactnet_v94_full.onnx` (+ внешние веса
  `artifactnet_v94_full.onnx.data`). Путь по умолчанию определяется
  относительно `check_ai_music.py`, переопределяется флагом `--model`.
- **SONICS**: HuggingFace `awsaf49/sonics-spectttra-gamma-5s` и
  `awsaf49/sonics-spectttra-beta-5s` (id можно переопределить переменной
  окружения `SONICS_MODELS`, число top-окон — `SONICS_TOP_K`).

## Упаковка в macOS `.app`

Команда и настройки описаны в [`AGENTS.md`](AGENTS.md).

## Благодарности

- [SONICS](https://github.com/awsaf49/sonics) (huggingface: `awsaf49/sonics-*`) — MIT.
- [lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector) — MIT
  (идея «fakeprint» вокодеров).
- [Prototypr/ai-music-classifier](https://github.com/Prototypr/ai-music-classifier) —
  ансамбль и правила корроборации сигналов.

## Лицензия

[MIT](LICENSE)
