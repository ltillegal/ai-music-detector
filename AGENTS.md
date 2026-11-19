# AI Music Detector — packaging notes

## Project layout

- `app.py` — main GUI entry point
- `check_ai_music.py` — detection logic (ArtifactNet + SONICS + lofcz and the verdict)
- `sonics_detector.py` — SONICS detector (SpecTTTra on ONNX Runtime, mel in numpy)
- `tools/export_sonics_onnx.py` — SONICS ONNX re-export (the `export` dependency group)
- `lofcz_detector.py` — lofcz detector (vocoder fakeprint + ONNX)
- `i18n.py` — interface strings (English default, Russian translation)
- `ui/main.qml` — interface definition; `ui/icon.png` — icon
- `artifactnet_v94_full.onnx[.data]`, `lofcz_ai_music_detector.onnx`,
  `lofcz_preprocessing.json` — model files shipped in the repository
- `fixtures/` — test audio files

## Packaging command

Run `uv run --group dev pyside6-deploy app.py` (torch is no longer needed in the
runtime environment). The `dev` group must be installed explicitly: `uv sync`
installs only the core dependencies.

## Key settings in `pysidedeploy.spec`

- `title: "AI Music Detector"`
- `project_dir: "."` (do not point this at `.venv`, the build will fail)
- `qml_files: "ui/main.qml"` (must be explicit, otherwise QML is not bundled)
- `icon`: set it if an `.icns` file is available, otherwise leave it empty

## Dependencies

The packaging environment needs:
nuitka, ordered_set, zstandard, onnxruntime, scipy, soundfile, sounddevice, PySide6.

Artifacts must be added to the bundle with `--include-data-files` (see the spec):
ArtifactNet (`.onnx` + `.onnx.data`), lofcz (model + config), `ui/icon.png`.

SONICS no longer requires torch: the detector runs on ONNX Runtime (a core
dependency) and computes log-mel in numpy. The ONNX models (~69 MB each, built
from `awsaf49/sonics-spectttra-*`) and the mel filter are downloaded on first
run into `~/.cache/ai-music-detector/sonics-onnx` (override with
`SONICS_ONNX_DIR`; source override `SONICS_ONNX_BASE`, default release
`sonics-onnx-v1`). Downloads are verified against the `sha256` values in the
release `manifest.json`. The heavy torch/timm/sonics/onnx stack remains only in
the `export` group (not in the default groups) for model re-export.

## How to verify

After packaging, run `open "release/AI Music Detector.app"`.

## Important reminders

- Detection logic lives in `check_ai_music.py` (+ `sonics_detector.py`,
  `lofcz_detector.py`).
- SONICS is always active when onnxruntime is available; only the downloaded
  ONNX models are needed.
- If SONICS is requested but cannot run (no models/no network), the `Human`
  verdict is suppressed (`Uncertain` + `warning`), see
  `_apply_verdict(..., degraded=True)`.
- Nearly silent input (RMS < `SILENCE_RMS`, about -80 dBFS) yields `Uncertain`
  (`reasons: ["silent_audio"]`) because models produce arbitrary scores on
  silence; see `_apply_verdict(..., silent=True)`.
- The GUI shows progress/stage, verdict reasons and per-window timelines; the
  language and theme can be switched and are stored in QSettings. English is
  the default language.
- ONNX re-export: `uv run --group export python tools/export_sonics_onnx.py --out dist/sonics-onnx`,
  then `gh release upload sonics-onnx-v1 dist/sonics-onnx/<files>` (afterwards
  run `uv sync` to restore the lean environment). The exporter writes `sha256`
  values into `manifest.json` for the models and the mel filter.
