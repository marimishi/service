import io
import wave
import numpy as np
from transformers import pipeline
from models.vosk_model import SAMPLERATE

_whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

def run_whisper(audio_float32: np.ndarray) -> str:
    audio_int16 = (audio_float32 * 32767).astype(np.int16)

    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLERATE)
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = wav_io.getvalue()

    result = _whisper_pipeline(wav_bytes)
    return result["text"]
