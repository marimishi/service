import logging
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ws_handlers import websocket_endpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "WebSocket работает!"}

@app.get("/voice")
async def get_voice_html():
    file_path = "static/voice.html"
    if not os.path.exists(file_path):
        return {"error": "Файл voice.html не найден"}
    with open(file_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(f.read())

app.add_api_websocket_route("/ws", websocket_endpoint)

if __name__ == "__main__":
    logger.info("Запуск FastAPI сервера")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)