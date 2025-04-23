import json
import logging
import numpy as np
import io
import wave
from fastapi import WebSocket
from vosk import KaldiRecognizer
from models.vosk_model import get_vosk_model, SAMPLERATE
from models.whisper_model import _whisper_pipeline 

logger = logging.getLogger(__name__)

async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket соединение установлено")
    await websocket.accept()

    vosk_model = get_vosk_model()
    recognizer = KaldiRecognizer(vosk_model, SAMPLERATE)
    recognizer.SetWords(True)

    full_audio = np.array([], dtype=np.float32)
    vosk_buffer = np.array([], dtype=np.int16)

    CHUNK_SIZE = 4000
    partial_result = ""
    last_sent_text = ""

    try:
        while True:
            data = await websocket.receive_bytes()
            logger.info(f"Получено {len(data)} байт аудио")

            audio_np = np.frombuffer(data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0

            full_audio = np.concatenate((full_audio, audio_float))
            vosk_buffer = np.concatenate((vosk_buffer, audio_np))

            while len(vosk_buffer) >= CHUNK_SIZE:
                chunk = vosk_buffer[:CHUNK_SIZE]
                vosk_buffer = vosk_buffer[CHUNK_SIZE:]

                if recognizer.AcceptWaveform(chunk.tobytes()):
                    result = json.loads(recognizer.Result())["text"]
                    logger.info(f"Распознанный текст (Vosk): {result}")
                    if result and result != last_sent_text:
                        await websocket.send_text(json.dumps({"type": "final", "text": result}))
                        last_sent_text = result
                    partial_result = ""
                else:
                    new_partial_result = json.loads(recognizer.PartialResult())["partial"]
                    if new_partial_result and new_partial_result != partial_result and new_partial_result != last_sent_text:
                        logger.info(f"Частичный текст (Vosk): {new_partial_result}")
                        await websocket.send_text(json.dumps({"type": "partial", "text": new_partial_result}))
                        partial_result = new_partial_result

    except Exception as e:
        logger.warning(f"Соединение закрыто. Финальное распознавание Whisper: {e}")
        try:
            logger.info("Запуск Whisper для финального распознавания...")

            audio_int16 = (full_audio * 32767).astype(np.int16)
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLERATE)
                    wf.writeframes(audio_int16.tobytes())
                wav_bytes = wav_io.getvalue()

            result = _whisper_pipeline(wav_bytes)
            result_text = result["text"]

            logger.info("Whisper текст: %s", result_text)
            await websocket.send_text(json.dumps({"type": "whisper", "text": result_text}))

        except Exception as whisper_err:
            logger.error(f"Ошибка при распознавании Whisper: {whisper_err}")

        finally:
            await websocket.close()
