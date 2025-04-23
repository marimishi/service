let audioContext, stream, processor, source, ws;

const mic = document.querySelector(".mic");
const container = document.querySelector(".container");

mic.addEventListener("click", async () => {
  mic.classList.toggle("active");

  if (mic.classList.contains("active")) {
    console.log("Начало распознавания");
    await startRecognition();
  } else {
    console.log("Остановка распознавания");
    stopRecognition();
  }
});

async function startRecognition() {
  ws = new WebSocket("ws://localhost:8000/ws");

  ws.onopen = () => {
    console.log("WebSocket соединение установлено.");
  };

  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "partial") {
        console.log("Частично: ", message.text);
      } else if (message.type === "final") {
        console.log("Финально: ", message.text);
      }
      container.textContent = message.text; 
    } catch (err) {
      console.error("Ошибка при парсинге сообщения: ", err);
    }
  };
  

  ws.onclose = () => {
    console.log("WebSocket соединение закрыто.");
    cleanUp();
  };

  ws.onerror = (error) => {
    console.error("Ошибка WebSocket: ", error);
  };

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  await audioContext.resume();

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  console.log("Получен доступ к микрофону.");

  source = audioContext.createMediaStreamSource(stream);

  try {
    await audioContext.audioWorklet.addModule('/static/audioProcessor.js');
    console.log("AudioWorkletModule загружен.");
  } catch (error) {
    console.error("Ошибка загрузки AudioWorklet:", error);
    return;
  }

  processor = new AudioWorkletNode(audioContext, 'audio-processor');

  source.connect(processor);
  processor.connect(audioContext.destination);

  processor.port.onmessage = (event) => {
    const floatData = event.data;
    const int16Data = new Int16Array(floatData.length);

    for (let i = 0; i < floatData.length; i++) {
      int16Data[i] = Math.max(-32768, Math.min(32767, floatData[i] * 32768));
    }

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(int16Data.buffer);
    } else {
      console.error("WebSocket не открыт, данные не отправлены.");
    }
  };
}

function stopRecognition() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  if (container) {
    container.textContent = '';
    console.log("Текст очищен.");
  }
}

function cleanUp() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    console.log("Остановлен захват аудио.");
  }

  if (audioContext) {
    audioContext.close();
    console.log("AudioContext закрыт.");
  }

  audioContext = null;
  stream = null;
  source = null;
  processor = null;
  ws = null;
}
