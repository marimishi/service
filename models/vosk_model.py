import os
from vosk import Model

MODEL_PATH = "vosk-model-small-ru-0.22"
SAMPLERATE = 16000

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Модель Vosk не найдена в {MODEL_PATH}")

_model = Model(MODEL_PATH)

def get_vosk_model():
    return _model