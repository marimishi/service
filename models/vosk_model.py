import os
from vosk import Model

MODEL_PATH = "models/vosk-model-ru-0.42"
SAMPLERATE = 16000

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Модель Vosk не найдена в {MODEL_PATH}")

_model = Model(MODEL_PATH)

def get_vosk_model():
    return _model