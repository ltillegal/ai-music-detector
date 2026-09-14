import onnxruntime as ort
import numpy as np
import soundfile as sf

MODEL_PATH = "artifactnet_v94_full.onnx"
EXPECTED_LEN = 4 * 44100  # 176400

def _load_chunk(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 44100, f"{path}: ожидалось 44100, получено {sr}"
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # стерео → моно
    if len(audio) < EXPECTED_LEN:
        audio = np.pad(audio, (0, EXPECTED_LEN - len(audio)))
    else:
        audio = audio[:EXPECTED_LEN]
    return audio.reshape(1, EXPECTED_LEN).astype(np.float32)

def test_model_loads():
    sess = ort.InferenceSession(MODEL_PATH)
    assert sess is not None
    # Проверяем, что модель действительно ждёт фиксированную длину
    inp = sess.get_inputs()[0]
    assert inp.shape == [1, EXPECTED_LEN], f"Неожиданная форма входа: {inp.shape}"

def test_ai_audio_scores_high():
    chunk = _load_chunk("fixtures/known_suno.wav")
    sess = ort.InferenceSession(MODEL_PATH)
    prob = float(sess.run(None, {"audio": chunk})[0][0])
    assert prob > 0.5, f"Ожидалось P(AI) > 0.5, получено {prob}"

def test_human_audio_scores_low():
    chunk = _load_chunk("fixtures/known_human.wav")
    sess = ort.InferenceSession(MODEL_PATH)
    prob = float(sess.run(None, {"audio": chunk})[0][0])
    assert prob < 0.3, f"Ожидалось P(AI) < 0.3, получено {prob}"
