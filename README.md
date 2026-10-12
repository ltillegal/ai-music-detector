# AI Music Detector

A desktop app and CLI that detect AI-generated music. It combines three
complementary detectors:

- **ArtifactNet** (ONNX) — hears the *vocoder fingerprint* of Suno/Udio
  (periodic deconvolution artefacts in the 1–8 kHz band).
- **lofcz** (fakeprint + a tiny ONNX classifier) — a second, independent
  vocoder-fingerprint detector; corroborates ArtifactNet.
- **SONICS** (SpecTTTra) — a style-based spectrogram transformer
  that catches modern diffusion-based generators (e.g. Google Lyria 3 in
  producer.ai) which leave no vocoder fingerprint.

Each detector alone has blind spots; together they cover both older and
newer generators.

## How it works

Audio is converted to mono, resampled to 44.1 kHz and split into 4-second
windows (`176400` samples) for **ArtifactNet**. The final score is the median
over windows.

**lofcz** reuses the same 4-second pipeline: the averaged spectrum (1–8 kHz) is
smoothed into a hull, the hull is subtracted, and the remaining residue
("fakeprint") is fed to a 15 KB ONNX model that returns an AI probability.

**SONICS** runs by default on **ONNX Runtime** (no torch): audio is resampled
to 16 kHz, split into 5-second windows (2.5 s hop), each window is
unit-variance normalised and its log-mel spectrogram (computed in numpy) is
passed through two SpecTTTra models (`gamma-5s`, `beta-5s`) exported to ONNX.
Per model, the score is the mean of the top-3 windows.

Verdict logic:

| Condition | Verdict |
| --------- | ------- |
| ArtifactNet `>= 0.6` **or** SONICS `>= 0.8` **or** lofcz `>= 0.5` | `AI` |
| all available signals below their human thresholds | `Human` |
| otherwise | `Uncertain` |

If SONICS is requested but does not run (lean install), a low score from the
vocoder detectors yields `Uncertain` rather than a confident `Human`, and a
`warning` is added — without the style detector a human verdict cannot be
corroborated.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- A microphone (only for mic recording)
- All three detectors share the core dependencies (onnxruntime, numpy, scipy).
  There is no heavy ML stack: no torch. The SONICS ONNX models (~138 MB) are
  downloaded on first run.

## Installation

```bash
git clone https://github.com/ltillegal/ai-music-detector.git
cd ai-music-detector
uv sync
```

`uv sync` installs the core dependencies (PySide6, onnxruntime, soundfile,
sounddevice, scipy, soxr) — no torch. The model files
(`artifactnet_v94_full.onnx`[.data], `lofcz_ai_music_detector.onnx`,
`lofcz_preprocessing.json`) are shipped in the repository.

The SONICS ONNX models (`gamma-5s`, `beta-5s`, ~69 MB each) and the mel filter
are downloaded on first run from the project's
[`sonics-onnx-v1`](https://github.com/ltillegal/ai-music-detector/releases/tag/sonics-onnx-v1)
release and cached in `~/.cache/ai-music-detector/sonics-onnx` (the GUI/CLI
shows a "loading model" stage). Override the cache with `SONICS_ONNX_DIR` and
the download base with `SONICS_ONNX_BASE`. CPU inference is roughly 2 seconds
per minute of audio per model.

The heavy stack (torch, torchaudio, timm, fvcore, sonics, onnx) is only needed
to **re-export** the models — see `tools/export_sonics_onnx.py` and the `export`
dependency group (`uv run --group export ...`).

## Usage

### GUI

```bash
uv run python app.py
```

Drag & drop an audio file onto the window, use **Choose file**, or record
10 seconds from the microphone. The window shows:

- a **progress bar with the current stage** (reading → ArtifactNet →
  downloading/loading SONICS → SONICS → lofcz);
- a big colour verdict plus the **reason** ("which signal decided");
- **signal bars** for ArtifactNet / SONICS / lofcz;
- a **timeline** that colours each 4 s (ArtifactNet) and 5 s (SONICS) window —
  mid values reveal an AI section inside a human track;
- a collapsible raw-JSON panel.

The interface language (**RU / EN** switcher, top-right) and the theme
(dark / light toggle) are remembered between runs.

### CLI: analyze a file

```bash
uv run python check_ai_music.py track.wav
uv run python check_ai_music.py track.mp3 --no-sonics --no-lofcz   # ArtifactNet only
uv run python check_ai_music.py track.wav --lang en                # warnings in English
```

The result is printed as JSON:

```json
{
  "file": "track.wav",
  "verdict": "AI",
  "p_ai": 0.999565,
  "p_ai_artifactnet": 0.999565,
  "p_ai_sonics": 0.98546,
  "p_ai_lofcz": 0.960958,
  "sonics_available": true,
  "lofcz_available": true,
  "signals": { "artifactnet": 0.999565, "sonics": 0.98546, "lofcz": 0.960958 },
  "reasons": ["artifactnet_ai", "sonics_ai", "lofcz_ai"],
  "warnings": [],
  "windows": { "artifactnet": [0.999, 0.999, 0.999],
               "sonics": [0.99, 0.98, 0.97] },
  "sonics": { "sonics-spectttra-gamma-5s": { "top_k": 0.984, "max": 0.992,
              "mean": 0.984, "win_frac": 1.0, "n_windows": 3 } },
  "lofcz": { "p_ai": 0.960958, "model": "lofcz_ai_music_detector.onnx" },
  "duration_sec": 10.0,
  "n_chunks": 3,
  "n_valid": 3
}
```

Fields:

- `p_ai` — combined probability (max of the available signals).
- `p_ai_artifactnet` / `p_ai_sonics` / `p_ai_lofcz` — per-detector scores.
- `signals` — the signals that were actually available.
- `reasons` — why the verdict was chosen (machine-readable codes; the GUI
  localises them).
- `sonics_available` / `lofcz_available` — whether each detector could run
  (SONICS only needs the onnxruntime core dependency, so it is virtually always
  `true`; the models may still be downloaded on first use).
- `warnings` — list of codes (`short_audio`, `no_finite`, `silent_audio`,
  `sonics_unavailable`, `sonics_error`, `lofcz_unavailable`, `lofcz_error`); the
  CLI also renders a human `warning` string in the selected `--lang`. Without
  SONICS the `Human` verdict is suppressed. Nearly-silent input (RMS below
  `SILENCE_RMS`, ~ -80 dBFS) never gets an `AI` verdict — the models return
  arbitrary scores on pure zeros — so the verdict is `Uncertain`
  (`reasons: ["silent_audio"]`).
- `windows` — per-window probabilities for the timeline.
- `sonics.<model>.win_frac` — share of 5-second windows flagged as AI;
  mid values indicate hybrid content.

### CLI: microphone

```bash
uv run python record_and_check.py --seconds 10
uv run python record_and_check.py --seconds 10 --output rec.wav  # keep the recording
```

## Models

- **ArtifactNet**: `artifactnet_v94_full.onnx` (+ external weights
  `artifactnet_v94_full.onnx.data`). The default path is resolved relative to
  `check_ai_music.py`; override with `--model`.
- **lofcz**: `lofcz_ai_music_detector.onnx` + `lofcz_preprocessing.json`.
- **SONICS**: ONNX exports of `awsaf49/sonics-spectttra-gamma-5s` and
  `awsaf49/sonics-spectttra-beta-5s`, hosted in the
  [`sonics-onnx-v1`](https://github.com/ltillegal/ai-music-detector/releases/tag/sonics-onnx-v1)
  release and downloaded on demand. Override the base names / cache / source
  with `SONICS_MODELS`, `SONICS_ONNX_DIR`, `SONICS_ONNX_BASE`, and the number
  of top windows with `SONICS_TOP_K`.

## Testing

```bash
uv run pytest
```

Tests use files from `fixtures/` (`known_suno.wav`, `known_human.wav`).
Integration tests for SONICS are skipped automatically when the ONNX models are
not yet cached (so CI does not download them); the numpy preprocessing is
covered by unit tests either way.

## Packaging into a macOS `.app`

A `.app` can only be built **on macOS** (Nuitka compiles for the host
platform). You need Xcode Command Line Tools (`xcode-select --install`) and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ltillegal/ai-music-detector.git
cd ai-music-detector
uv sync                       # PySide6 + dev group (no torch)
uv run pyside6-deploy app.py  # reads pysidedeploy.spec
open "release/AI Music Detector.app"
```

What is configured in [`pysidedeploy.spec`](pysidedeploy.spec):

- `title = AI Music Detector`, `exec_directory = release`, `mode = onefile`;
- `qml_files = ui/main.qml` — required, otherwise QML is not bundled;
- `extra_args ... --include-data-files=...` — the ArtifactNet, lofcz (model +
  config) and icon files are embedded into the bundle;
- `macos.permissions` — microphone usage description.

See [`AGENTS.md`](AGENTS.md) for more details.

### SONICS in the packaged app

SONICS runs on ONNX Runtime, which is already bundled, so there is nothing extra
to exclude — and no torch to accidentally drag in. The `.app` stays small; the
SONICS ONNX models (~138 MB) are downloaded to
`~/.cache/ai-music-detector/sonics-onnx` on first use. To make the app work
offline, add the ONNX files and `mel_fb.npy` to `extra_args` with
`--include-data-files` (and point `SONICS_ONNX_DIR` at the bundled folder).

### Apple Silicon

The build virtual environment must use a native arm64 Python (not Rosetta),
otherwise you get an x86_64 app.

### Signing and running

For local use, just `open` the app. If macOS blocks it:

```bash
xattr -dr com.apple.quarantine "release/AI Music Detector.app"
```

For distribution, sign and notarize:

```bash
codesign --deep --force --sign "<Developer ID>" "release/AI Music Detector.app"
# notarytool submit ... && xcrun stapler staple ...
```

An icon is optional for macOS: convert `ui/icon.png` to `.icns` and set its
path in `icon = ` in the spec. The app already sets `ui/icon.png` as the
window icon at runtime.

## Credits

- [SONICS](https://github.com/awsaf49/sonics) (`awsaf49/sonics-*`) — MIT.
- [lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector) — MIT
  (the vocoder "fakeprint" idea).
- [Prototypr/ai-music-classifier](https://github.com/Prototypr/ai-music-classifier) —
  detector ensemble and signal-corroboration rules.

## License

[MIT](LICENSE)
