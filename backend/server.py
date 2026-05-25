"""
server.py
FastAPI backend.

Endpoint:  ws://localhost:8000/ws/audio
Protocol:
  - Browser sends raw binary PCM frames (Int16, 16kHz, mono)
  - Server forwards to Sarvam STT via stt.py
  - Server sends JSON events back to browser:
      {"type": "speech_start"}
      {"type": "speech_end"}
      {"type": "transcript", "text": "..."}
"""

import asyncio
import json
import logging
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stt import run_streaming_stt

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("server")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Event Assistant - STT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("[CLIENT CONNECTED]")

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    stop_event = asyncio.Event()

    # Callback: STT events → browser
    async def on_event(event_type: str, text: str):
        payload = {"type": event_type}
        if text:
            payload["text"] = text
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass  # client may have disconnected

    # Start STT in background task
    stt_task = asyncio.create_task(
        run_streaming_stt(audio_queue, on_event, stop_event)
    )

    try:
        while True:
            # Receive raw binary PCM from browser
            data = await websocket.receive_bytes()
            if data:
                try:
                    audio_queue.put_nowait(data)
                except asyncio.QueueFull:
                    # Drop oldest frame to prevent buildup
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put_nowait(data)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info("[CLIENT DISCONNECTED]")
    except Exception as e:
        logger.error(f"[ERROR] {e}")
    finally:
        stop_event.set()
        await audio_queue.put(None)   # unblock sender coroutine
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except asyncio.TimeoutError:
            stt_task.cancel()
        logger.info("[SESSION CLOSED]")