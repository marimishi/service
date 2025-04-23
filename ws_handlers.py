import json, logging, io, wave, numpy as np
from fastapi import WebSocket
from vosk import KaldiRecognizer
from models.vosk_model import get_vosk_model, SAMPLERATE
from models.whisper_model import _whisper_pipeline   

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4000        

async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket соединение установлено")
    await websocket.accept()

    recognizer = KaldiRecognizer(get_vosk_model(), SAMPLERATE)
    recognizer.SetWords(True)

    full_audio_int16: list[bytes] = []   
    vosk_buf = bytearray()
    partial_cache = ""

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()         
            full_audio_int16.append(pcm_bytes)
            vosk_buf.extend(pcm_bytes)

            while len(vosk_buf) >= CHUNK_SIZE:
                chunk = bytes(vosk_buf[:CHUNK_SIZE])
                del vosk_buf[:CHUNK_SIZE]

                if recognizer.AcceptWaveform(chunk):
                    txt = json.loads(recognizer.Result())["text"]
                    if txt:
                        await websocket.send_text(json.dumps({"type": "final", "text": txt}))
                    partial_cache = ""
                else:
                    partial = json.loads(recognizer.PartialResult())["partial"]
                    if partial and partial != partial_cache:
                        await websocket.send_text(json.dumps({"type": "partial", "text": partial}))
                        partial_cache = partial

    except Exception as e:
        logger.info("Соединение закрыто: %s. Запускаем Whisper…", e)

        pcm_np = (np.frombuffer(b"".join(full_audio_int16), dtype=np.int16)
                    .astype(np.float32) / 32768.0)

        result_text = _whisper_pipeline(pcm_np)["text"]
        logger.info("Whisper текст: %s", result_text)
        await websocket.send_text(json.dumps({"type": "whisper", "text": result_text}))
        await websocket.close()
