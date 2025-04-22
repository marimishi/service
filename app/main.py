import os
import logging
import json
import io
import wave
from collections import deque

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from vosk import Model, KaldiRecognizer
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="service/app/static/"), name="static")

@app.get("/")
async def root():
    return {"message": "WebSocket работает!"}

@app.get("/voice")
async def get_voice_html():
    file_path = "voice.html"
    if not os.path.exists(file_path):
        return {"error": "Файл voice.html не найден"}
    with open(file_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(f.read())

MODEL_PATH = "vosk-model-small-ru-0.22"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Модель Vosk не найдена в {MODEL_PATH}")
vosk_model = Model(MODEL_PATH)
samplerate = 16000

whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

client_states = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket соединение установлено")
    await websocket.accept()

    recognizer = KaldiRecognizer(vosk_model, samplerate)
    audio_buffer = deque()
    full_audio = np.array([], dtype=np.float32)

    partial_result = ""
    last_sent_text = ""

    try:
        while True:
            data = await websocket.receive_bytes()
            logger.info(f"Получено {len(data)} байт аудио")

            audio_np = np.frombuffer(data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0

            full_audio = np.concatenate((full_audio, audio_float))
            audio_buffer.extend(audio_np)

            if recognizer.AcceptWaveform(audio_np.tobytes()):
                result = json.loads(recognizer.Result())["text"]
                logger.info(f"Распознанный текст (Vosk): {result}")
                if result != last_sent_text:
                    await websocket.send_text(json.dumps({"type": "final", "text": result}))
                    last_sent_text = result
                partial_result = ""
            else:
                new_partial_result = json.loads(recognizer.PartialResult())["partial"]
                if new_partial_result != partial_result and new_partial_result != last_sent_text:
                    logger.info(f"Частичный текст (Vosk): {new_partial_result}")
                    await websocket.send_text(json.dumps({"type": "partial", "text": new_partial_result}))
                    partial_result = new_partial_result

    except Exception as e:
        logger.warning(f"Соединение закрыто. Финальное распознавание Whisper: {e}")
        try:
            logger.info("Запуск Whisper для финального распознавания...")
            result_text = await run_whisper(full_audio)
            await websocket.close()
            logger.info("Whisper текст: %s", result_text)
        except Exception as whisper_err:
            logger.error(f"Ошибка при распознавании Whisper: {whisper_err}")

async def run_whisper(audio_float32: np.ndarray) -> str:
    audio_int16 = (audio_float32 * 32767).astype(np.int16)

    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = wav_io.getvalue()

    result = whisper_pipeline(wav_bytes)
    return result["text"]

if __name__ == "__main__":
    logger.info("Запуск FastAPI сервера")
    uvicorn.run(app, host="127.0.0.1", port=8000)
