"""
server.py
FastAPI backend.

Endpoint:  ws://localhost:8000/ws/audio
Protocol:
  - Browser sends raw binary PCM frames (Int16, 16kHz, mono)
  - Server forwards to Sarvam STT via stt.py
  - LLM fires ONCE per speech turn: triggered by speech_end + last transcript
  - Server sends JSON events back to browser:
      {"type": "speech_start"}
      {"type": "speech_end"}
      {"type": "transcript", "text": "..."}
      {"type": "llm_start"}
      {"type": "llm_token", "token": "..."}
      {"type": "llm_end", "text": "..."}
      {"type": "llm_error", "text": "..."}

FIX: Sarvam sends speech_end ~200ms BEFORE the transcript arrives.
     _wait_and_run_llm() waits 500ms for the transcript to land before giving up.
"""

import asyncio
import json
import logging
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stt import run_streaming_stt
import llm as llm_module
import tts as tts_module

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("server")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Event Assistant - STT + LLM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown():
    await llm_module.close()
    await tts_module.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("[CLIENT CONNECTED]")

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    stop_event = asyncio.Event()

    # Per-session state
    conversation_history: list = []
    llm_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()

    # Buffer: accumulate transcript chunks during a speech turn.
    # Reset on speech_start, finalized on speech_end.
    _transcript_buffer: list = []

    # ── Event callback ────────────────────────────────────────────────────
    async def on_event(event_type: str, text: str):
        nonlocal _transcript_buffer

        if event_type == "speech_start":
            # New turn — clear buffer
            _transcript_buffer = []
            await _send({"type": "speech_start"})

        elif event_type == "transcript" and text:
            # Accumulate partial/final transcripts; send to browser live
            _transcript_buffer.append(text)
            await _send({"type": "transcript", "text": text})

        elif event_type == "speech_end":
            await _send({"type": "speech_end"})
            # Sarvam sends speech_end ~200ms BEFORE the transcript arrives.
            # If buffer is already populated, fire immediately.
            # Otherwise wait briefly for the transcript to land.
            if _transcript_buffer:
                final_transcript = _transcript_buffer[-1]
                logger.info(f"[TURN END] Final transcript: {final_transcript!r}")
                asyncio.create_task(_run_llm(final_transcript))
            else:
                asyncio.create_task(_wait_and_run_llm())

        else:
            # llm_start / llm_token / llm_end / llm_error — forward as-is
            payload = {"type": event_type}
            if text:
                payload["token" if event_type == "llm_token" else "text"] = text
            await _send(payload)

            # Fire TTS after LLM finishes a complete response
            if event_type == "llm_end" and text:
                asyncio.create_task(_run_tts(text))

    async def _send(payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass

    async def _wait_and_run_llm():
        """
        Sarvam VAD sends speech_end before the transcript message arrives.
        Wait up to 500ms for the transcript to land, then fire LLM.
        500ms > observed ~200ms gap, with headroom for slow network.
        """
        await asyncio.sleep(0.5)
        if _transcript_buffer:
            final_transcript = _transcript_buffer[-1]
            logger.info(f"[TURN END] Transcript arrived after wait: {final_transcript!r}")
            await _run_llm(final_transcript)
        else:
            logger.info("[TURN END] Still no transcript after 500ms wait, skipping LLM")

    async def _run_llm(transcript: str):
        """Stream LLM response for one completed speech turn."""
        async with llm_lock:
            if stop_event.is_set():
                return
            try:
                await llm_module.stream_llm_response(
                    transcript=transcript,
                    conversation_history=conversation_history,
                    event_callback=on_event,
                )
            except Exception as e:
                logger.error(f"[LLM TASK ERROR] {e}")

    async def _run_tts(text: str):
        """Stream TTS audio for one LLM response turn."""
        async with tts_lock:
            if stop_event.is_set():
                return

            await _send({"type": "tts_start"})

            async def on_audio_chunk(chunk: bytes):
                if stop_event.is_set():
                    return
                try:
                    await websocket.send_bytes(chunk)
                except Exception:
                    pass

            async def on_tts_done():
                await _send({"type": "tts_end"})

            try:
                await tts_module.stream_tts_audio(
                    text=text,
                    audio_chunk_callback=on_audio_chunk,
                    done_callback=on_tts_done,
                )
            except Exception as e:
                logger.error(f"[TTS TASK ERROR] {e}")
                await _send({"type": "tts_end"})

    # Start STT background task
    stt_task = asyncio.create_task(
        run_streaming_stt(audio_queue, on_event, stop_event)
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            if data:
                try:
                    audio_queue.put_nowait(data)
                except asyncio.QueueFull:
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
        await audio_queue.put(None)
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except asyncio.TimeoutError:
            stt_task.cancel()
        logger.info("[SESSION CLOSED]")