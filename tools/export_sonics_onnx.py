"""Экспорт моделей SONICS (SpecTTTra) в ONNX, чтобы запускать детектор без torch.

Скрипт нужен только для подготовки артефактов и требует тяжёлую группу
зависимостей `export` (torch, torchaudio, timm, sonics). Обычному пользователю
и CI он не нужен: готовые .onnx скачиваются приложением автоматически.

Тонкость: `torch.stft` (внутри torchaudio MelSpectrogram) НЕ экспортируется в
ONNX, поэтому в граф уходит только «голова» модели — interpolate → SpecTTTra
→ mean → classifier, — а log-mel спектрограмма считается в numpy тем же
способом, что и torchaudio (проверяется здесь же).

Пример:
    uv run --group export python tools/export_sonics_onnx.py --out dist/sonics-onnx

(Группа `export` ставит torch в .venv — после экспорта верните обычное окружение
командой `uv sync`.)

Артефакты на выход:
    <model>.onnx   — вход ``melspec`` (float32, [batch, 128, 157]), выход ``logits``
    mel_fb.npy     — мел-фильтр (128, 1025), float32
    preproc.json   — константы препроцессинга (раундтрип-проверка)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

SONICS_SR = 16000
SONICS_WIN = SONICS_SR * 5
N_FFT = 2048
HOP = 512
WIN = 2048
N_MELS = 128
F_MIN = 20.0
F_MAX = 8000.0
TOP_DB = 80.0
N_FRAMES = 1 + SONICS_WIN // HOP  # 157
DEFAULT_MODELS = "awsaf49/sonics-spectttra-gamma-5s,awsaf49/sonics-spectttra-beta-5s"


def make_head(clf):
    import torch.nn as nn
    import torch.nn.functional as F

    class EncoderHead(nn.Module):
        def __init__(self, clf):
            super().__init__()
            self.input_shape = tuple(clf.input_shape)
            self.encoder = clf.encoder
            self.classifier = clf.classifier

        def forward(self, melspec):
            spec = melspec.unsqueeze(1)
            spec = F.interpolate(spec, size=self.input_shape, mode="bilinear")
            feats = self.encoder(spec)
            return self.classifier(feats.mean(dim=1))

    return EncoderHead(clf).eval()


def np_logmel(y):
    """log-mel спектрограмма ровно как torchaudio (power=2, top_db=80, mean_std)."""
    pad = N_FFT // 2
    yp = np.pad(np.asarray(y, dtype=np.float32), (pad, pad), mode="reflect")
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(yp, WIN)[::HOP][:N_FRAMES]
    spec = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    power = (np.abs(spec) ** 2).T  # (1025, n_frames)
    return power


def mel_from_power(power, fb):
    mel = fb.T @ power  # fb (freq, n_mels) -> (128, n_frames)
    x_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    x_db = np.maximum(x_db, x_db.max() - TOP_DB)
    mean = x_db.mean()
    std = x_db.reshape(-1).std(ddof=1)
    return ((x_db - mean) / (std + 1e-6)).astype(np.float32)


def export_one(model_id, out_dir, opset=17):
    import torch
    from sonics import HFAudioClassifier

    name = model_id.split("/")[-1]
    out_path = out_dir / f"{name}.onnx"

    clf = HFAudioClassifier.from_pretrained(model_id)
    clf.eval()
    head = make_head(clf)

    # мел-фильтр: берём готовый буфер torchaudio, чтобы гарантировать совпадение
    fb = clf.ft_extractor.audio2melspec.mel_scale.fb.detach().cpu().numpy().astype(np.float32)

    dummy = torch.randn(2, N_MELS, N_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        torch_ref = torch.sigmoid(head(dummy)).numpy()

    torch.onnx.export(
        head,
        (dummy,),
        str(out_path),
        input_names=["melspec"],
        output_names=["logits"],
        dynamic_axes={"melspec": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = 1.0 / (1.0 + np.exp(-sess.run(["logits"], {"melspec": dummy.numpy()})[0]))
    head_diff = float(np.abs(torch_ref - ort_out).max())

    # сквозная проверка препроцессинга на случайном «аудио»
    audio = np.random.randn(SONICS_WIN).astype(np.float32) / 3
    seg = audio / max(float(np.std(audio)), 1e-6)
    with torch.no_grad():
        full_torch = torch.sigmoid(clf(torch.from_numpy(seg[None]).float())).item()
    mel = mel_from_power(np_logmel(seg), fb)[None]
    full_np = 1.0 / (1.0 + np.exp(-sess.run(["logits"], {"melspec": mel})[0].item()))
    full_diff = abs(full_torch - full_np)

    info = {
        "name": name,
        "file": out_path.name,
        "bytes": out_path.stat().st_size,
        "opset": opset,
        "max_abs_diff_head": head_diff,
        "max_abs_diff_end_to_end": full_diff,
    }
    print(
        f"[{name}] -> {out_path.name} ({info['bytes'] / 1e6:.1f} MB) | "
        f"head Δ={head_diff:.2e} | end-to-end Δ={full_diff:.2e}"
    )
    return info, fb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dist/sonics-onnx", help="каталог для .onnx")
    ap.add_argument("--models", default=os.environ.get("SONICS_MODELS", DEFAULT_MODELS))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    fb = None
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        info, fb = export_one(model_id, out_dir, opset=args.opset)
        manifest.append(info)

    np.save(out_dir / "mel_fb.npy", fb)
    preproc = {
        "sample_rate": SONICS_SR,
        "win_samples": SONICS_WIN,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "win_length": WIN,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "top_db": TOP_DB,
        "n_frames": N_FRAMES,
        "power": 2.0,
        "norm": "mean_std",
        "mel_filter": "mel_fb.npy",
    }
    (out_dir / "preproc.json").write_text(json.dumps(preproc, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nГотово: {len(manifest)} модел(и) + mel_fb.npy + preproc.json в {out_dir}")


if __name__ == "__main__":
    main()
