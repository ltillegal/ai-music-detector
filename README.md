# AI Music Detector

A desktop app and CLI that detect AI-generated music. It combines two
complementary detectors:

- **ArtifactNet** (ONNX) — hears the *vocoder fingerprint* of Suno/Udio
  (periodic deconvolution artefacts in the 1–8 kHz band).
- **SONICS** (SpecTTTra) — a style-based spectrogram transformer
  that catches modern diffusion-based generators (e.g. Google Lyria 3 in
  producer.ai) which leave no vocoder fingerprint.

Each detector alone has blind spots; together they cover both older and
newer generators.

## How it works

Audio is converted to mono, resampled to 44.1 kHz and split into 4-second
windows (`176400` samples) for **ArtifactNet**. The final score is the median
over windows.

**SONICS** runs by default (a CPU build of torch is part of the default
install): audio is resampled to 16 kHz, split into 5-second windows (2.5 s
hop), each window is unit-variance normalised and passed through two SpecTTTra
models (`gamma-5s`, `beta-5s`). Per model, the score is the mean of the top-3
windows.

Verdict logic:

| Condition | Verdict |
| --------- | ------- |
| ArtifactNet `>= 0.6` **or** SONICS `>= 0.8` | `AI` |
| ArtifactNet `<= 0.4` **and** SONICS `<= 0.5` | `Human` |
| otherwise | `Uncertain` |

If SONICS is requested but does not run (lean install), a low ArtifactNet score
yields `Uncertain` rather than a confident `Human`, and a `warning` is added to
the result — without the style detector a human verdict cannot be corroborated.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)
- A microphone (only for mic recording)
- SONICS is installed by default (`torch`, `torchaudio`, `timm`, `fvcore`,
  `sonics`); it can be skipped with `--no-group sonics`

## Installation

```bash
git clone https://github.com/ltillegal/ai-music-detector.git
cd ai-music-detector
uv sync
```

`uv sync` installs the core dependencies (PySide6, onnxruntime, soundfile,
sounddevice) **and** the SONICS stack — the `sonics` dependency group is part
of `default-groups`, so the CPU build of torch is pulled from the CPU index
automatically. The ONNX model files (`artifactnet_v94_full.onnx` and
`artifactnet_v94_full.onnx.data`) are shipped in the repository.

The SONICS weights (`gamma-5s`, `beta-5s`, ~69 MB each) are downloaded from
Hugging Face on first run and cached in `HF_HOME`. CPU inference is roughly
2 seconds per minute of audio per model.

### Lean install (without SONICS)

To skip torch (~200 MB) — for a minimal CLI or when packaging a `.app`:

```bash
uv sync --no-group sonics
```

SONICS then disables itself (`available()` returns `False`), the CLI adds a
`warning` and the GUI shows a banner; `Human` verdicts become `Uncertain`
because the style detector cannot corroborate them.

## Usage

### GUI

```bash
uv run python app.py
```

Drag & drop an audio file onto the window, use **Choose file**, or record
10 seconds from the microphone.

### CLI: analyze a file

```bash
uv run python check_ai_music.py track.wav
uv run python check_ai_music.py track.mp3 --no-sonics   # ArtifactNet only
```

The result is printed as JSON:

```json
{
  "file": "track.wav",
  "verdict": "AI",
  "p_ai": 0.999565,
  "p_ai_artifactnet": 0.999565,
  "p_ai_sonics": 0.98546,
  "sonics_available": true,
  "sonics": { "sonics-spectttra-gamma-5s": { "top_k": 0.984, "max": 0.992,
              "mean": 0.984, "win_frac": 1.0, "n_windows": 3 } },
  "duration_sec": 10.0,
  "n_chunks": 3,
  "n_valid": 3
}
```

Fields:

- `p_ai` — combined probability (max of the available signals).
- `p_ai_artifactnet` — ArtifactNet median over 4-second windows.
- `p_ai_sonics` — SONICS mean of the two models (omitted without SONICS).
- `sonics_available` — whether the SONICS detector could be imported.
- `warning` — present when SONICS is missing (lean install) or the audio is
  too short; without SONICS the `Human` verdict is suppressed.
- `sonics.<model>.win_frac` — share of 5-second windows flagged as AI;
  mid values indicate hybrid content (an AI section inside a human track).

### CLI: microphone

```bash
uv run python record_and_check.py --seconds 10
uv run python record_and_check.py --seconds 10 --output rec.wav  # keep the recording
```

## Models

- **ArtifactNet**: `artifactnet_v94_full.onnx` (+ external weights
  `artifactnet_v94_full.onnx.data`). The default path is resolved relative to
  `check_ai_music.py`; override with `--model`.
- **SONICS**: Hugging Face `awsaf49/sonics-spectttra-gamma-5s` and
  `awsaf49/sonics-spectttra-beta-5s`. Override the model ids with the
  `SONICS_MODELS` environment variable and the number of top windows with
  `SONICS_TOP_K`.

## Testing

```bash
uv run pytest
```

Tests use files from `fixtures/` (`known_suno.wav`, `known_human.wav`).
SONICS tests are skipped automatically when `torch`/`sonics` are not installed.

## Packaging into a macOS `.app`

A `.app` can only be built **on macOS** (Nuitka compiles for the host
platform). You need Xcode Command Line Tools (`xcode-select --install`) and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ltillegal/ai-music-detector.git
cd ai-music-detector
uv sync --no-group sonics   # PySide6 + dev group, without the ~200 MB torch
uv run pyside6-deploy app.py  # reads pysidedeploy.spec
open "release/AI Music Detector.app"
```

What is configured in [`pysidedeploy.spec`](pysidedeploy.spec):

- `title = AI Music Detector`, `exec_directory = release`, `mode = onefile`;
- `qml_files = ui/main.qml` — required, otherwise QML is not bundled;
- `extra_args ... --include-data-files=artifactnet_v94_full.onnx[.data]` —
  the ONNX model is embedded into the bundle;
- `macos.permissions` — microphone usage description.

See [`AGENTS.md`](AGENTS.md) for more details.

### SONICS in the packaged app

Build with `uv sync --no-group sonics` so torch is not in the build
environment. Nuitka then cannot find it, SONICS disables itself (`available()`
returns `False`) and the `.app` stays a reasonable size — it runs on
ArtifactNet only and shows a warning banner. Installing torch makes Nuitka
bundle it (hundreds of MB and a very slow build).

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

An icon is optional: add an `.icns` file and set its path in `icon = ` in the
spec.

## Credits

- [SONICS](https://github.com/awsaf49/sonics) (`awsaf49/sonics-*`) — MIT.
- [lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector) — MIT
  (the vocoder "fakeprint" idea).
- [Prototypr/ai-music-classifier](https://github.com/Prototypr/ai-music-classifier) —
  detector ensemble and signal-corroboration rules.

## License

[MIT](LICENSE)
