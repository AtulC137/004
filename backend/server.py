"""
server.py — greeting + farewell + reactive transcript fix + interrupt handling.

CHANGES:
  - Greeting TTS plays immediately on connect (interruptible — user can speak over it)
  - Farewell detection: if user says bye/thanks/done, AI responds then closes WebSocket
  - _is_farewell flag: protects bye TTS from being cancelled mid-sentence
  - Reactive _speech_ended flag: no timer-based wait for transcript (zero latency fix)

INTERRUPT BEHAVIOUR:
  - During greeting TTS   → interrupted immediately, user starts speaking
  - During normal AI TTS  → interrupted immediately, user starts speaking
  - During farewell TTS   → NOT interrupted (_is_farewell=True), let it finish cleanly

IMPORTANT: _active_tasks is a SET, not a list. Tasks self-remove via done_callback.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("server")

app = FastAPI(title="Voice Event Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GREETING = "Hi! I'm Susha, your Adobe event assistant. Feel free to ask me anything about the event!"

FAREWELL_WORDS = {
    "bye", "goodbye", "thank you", "thanks", "that's all", "thats all", "done", "ok bye", "okay bye",
    "chalo bye", "chalo", "alvida", "see you",
    "बाय", "बाइ", "धन्यवाद", "अलविदा", "ठीक है बाय",
}


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
    conversation_history: list = []
    _transcript_buffer: list = []

    # SET — tasks add themselves on spawn, auto-remove on completion
    _active_tasks: set = set()
    _turn_cancel: asyncio.Event = asyncio.Event()

    # Flag: speech_end arrived but transcript hasn't yet — wait for it reactively
    _speech_ended: bool = False

    # Flag: farewell TTS is playing — do NOT cancel it on interrupt
    _is_farewell: bool = False

    _turn_stt_lang: str | None = None
    _last_lang: str | None = None

    def _spawn(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)
        return task

    def _interrupt():
        if _is_farewell:
            return  # Never cancel farewell TTS — let it finish
        _turn_cancel.set()
        for t in list(_active_tasks):
            t.cancel()

    def _start_new_turn():
        nonlocal _turn_cancel, _speech_ended
        _turn_cancel = asyncio.Event()
        _speech_ended = False

    async def _send(payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass

    def _is_farewell_transcript(text: str) -> bool:
        text_norm = text.strip().rstrip(".,!?")
        text_lower = text_norm.lower()
        return any(fw in text_lower or fw in text_norm for fw in FAREWELL_WORDS)

    async def on_event(event_type: str, text: str, stt_lang: str | None = None, response_lang: str | None = None):
        nonlocal _transcript_buffer, _speech_ended, _is_farewell, _turn_stt_lang

        if event_type == "speech_start":
            if _is_farewell:
                logger.info("[VAD] Farewell in progress — ignoring speech_start")
                return
            # Sarvam VAD fired — cancel active tasks
            if _active_tasks:
                logger.info(f"[VAD INTERRUPT] Cancelling {len(_active_tasks)} task(s)")
                _interrupt()
                await _send({"type": "interrupted"})
            _start_new_turn()
            _transcript_buffer = []
            _turn_stt_lang = None
            await _send({"type": "speech_start"})

        elif event_type == "transcript" and text:
            if _is_farewell:
                return
            if stt_lang:
                _turn_stt_lang = stt_lang
            _transcript_buffer.append(text)
            await _send({"type": "transcript", "text": text})

            # speech_end already fired but transcript arrived late — trigger LLM now
            if _speech_ended and not _active_tasks:
                _speech_ended = False
                logger.info(f"[TURN END] late transcript (reactive): {text!r}")
                _spawn(_run_llm(text, _turn_cancel, _turn_stt_lang))

        elif event_type == "speech_end":
            if _is_farewell:
                return
            await _send({"type": "speech_end"})
            if _transcript_buffer:
                final_transcript = _transcript_buffer[-1]
                logger.info(f"[TURN END] transcript: {final_transcript!r}")
                _spawn(_run_llm(final_transcript, _turn_cancel, _turn_stt_lang))
            else:
                logger.info("[TURN END] Transcript not yet received, waiting reactively...")
                _speech_ended = True

        else:
            payload = {"type": event_type}
            if text:
                payload["token" if event_type == "llm_token" else "text"] = text
            await _send(payload)

            if event_type == "llm_end" and text:
                # Check if the last user utterance was a farewell
                last_user_text = _transcript_buffer[-1] if _transcript_buffer else ""
                tts_lang = llm_module.tts_code_for_language(response_lang or _last_lang or "english")
                if _is_farewell_transcript(last_user_text):
                    logger.info("[FAREWELL] Detected — playing bye TTS then closing")
                    _is_farewell = True
                    await _send({"type": "farewell_start"})
                    _spawn(_run_farewell_tts(text, _turn_cancel, tts_lang))
                else:
                    _spawn(_run_tts(text, _turn_cancel, tts_lang))

    async def _run_llm(transcript: str, cancel: asyncio.Event, stt_lang: str | None = None):
        nonlocal _last_lang
        if cancel.is_set() or stop_event.is_set():
            return

        lang = llm_module.resolve_response_language(stt_lang, transcript)
        if _last_lang is not None and lang != _last_lang:
            logger.info(f"[LLM] Language switch {_last_lang} → {lang}, clearing history")
            conversation_history.clear()
        _last_lang = lang

        try:
            await llm_module.stream_llm_response(
                transcript=transcript,
                conversation_history=conversation_history,
                event_callback=on_event,
                cancel_event=cancel,
                stt_lang=stt_lang,
            )
        except asyncio.CancelledError:
            logger.info("[LLM] Task cancelled")
        except Exception as e:
            logger.error(f"[LLM TASK ERROR] {e}")

    async def _run_tts(text: str, cancel: asyncio.Event, tts_lang: str = "en-IN"):
        if cancel.is_set() or stop_event.is_set():
            return
        await _send({"type": "tts_start"})

        async def on_audio_chunk(chunk: bytes):
            if cancel.is_set() or stop_event.is_set():
                raise asyncio.CancelledError
            try:
                await websocket.send_bytes(chunk)
            except Exception:
                pass

        async def on_tts_done():
            if not cancel.is_set():
                await _send({"type": "tts_end"})

        try:
            await tts_module.stream_tts_audio(
                text=text,
                audio_chunk_callback=on_audio_chunk,
                done_callback=on_tts_done,
                target_language_code=tts_lang,
            )
        except asyncio.CancelledError:
            logger.info("[TTS] Task cancelled")
            await _send({"type": "tts_end"})
        except Exception as e:
            logger.error(f"[TTS TASK ERROR] {e}")
            await _send({"type": "tts_end"})

    async def _run_farewell_tts(text: str, cancel: asyncio.Event, tts_lang: str = "en-IN"):
        """TTS for farewell — not cancellable. Closes WebSocket after playing."""
        if stop_event.is_set():
            return
        await _send({"type": "tts_start"})

        async def on_audio_chunk(chunk: bytes):
            # No cancel check — farewell plays to completion
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
                target_language_code=tts_lang,
            )
        except Exception as e:
            logger.error(f"[FAREWELL TTS ERROR] {e}")
            await _send({"type": "tts_end"})

        logger.info("[FAREWELL] TTS complete — closing WebSocket")
        await _send({"type": "session_end"})
        stop_event.set()
        await asyncio.sleep(0.5)
        try:
            await websocket.close()
        except Exception:
            pass

    # ── client_interrupt handler ───────────────────────────────────────────
    async def handle_client_interrupt():
        if _is_farewell:
            logger.info("[CLIENT INTERRUPT] Farewell in progress — ignoring")
            return
        if _active_tasks:
            logger.info(f"[CLIENT INTERRUPT] Energy detected — cancelling {len(_active_tasks)} task(s)")
            _interrupt()
            await _send({"type": "interrupted"})
            _start_new_turn()
            _transcript_buffer.clear()

    stt_task = asyncio.create_task(run_streaming_stt(audio_queue, on_event, stop_event))

    # ── Play greeting immediately on connect ──────────────────────────────
    logger.info("[GREETING] Playing opening greeting")
    _spawn(_run_tts(GREETING, _turn_cancel, "en-IN"))

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                if _is_farewell:
                    continue
                data = message["bytes"]
                try:
                    audio_queue.put_nowait(data)
                except asyncio.QueueFull:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put_nowait(data)
                    except Exception:
                        pass

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "client_interrupt":
                        await handle_client_interrupt()
                except Exception as e:
                    logger.debug(f"[TEXT MSG PARSE ERROR] {e}")

    except WebSocketDisconnect:
        logger.info("[CLIENT DISCONNECTED]")
    except Exception as e:
        logger.error(f"[ERROR] {e}")
    finally:
        stop_event.set()
        _interrupt()
        await audio_queue.put(None)
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except asyncio.TimeoutError:
            stt_task.cancel()
        logger.info("[SESSION CLOSED]")