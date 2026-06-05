"""
server.py — greeting + farewell + reactive transcript + interrupt + TTS WebSocket pipeline.

TTS: Sarvam text_to_speech_streaming WebSocket — LLM tokens → convert() → PCM to client.
INTERRUPT: abort TTS WS + cancel LLM tasks (Sarvam barge-in recipe).
"""

import asyncio
import json
import logging
import sys
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stt import run_streaming_stt
import llm as llm_module
import tts as tts_module
from tts import TtsWsSession

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

PCM_TTS_START = {"type": "tts_start", "format": "pcm_s16le", "sampleRate": 16000}


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

    _active_tasks: set = set()
    _turn_cancel: asyncio.Event = asyncio.Event()

    _speech_ended: bool = False
    _is_farewell: bool = False
    _greeting_active: bool = True
    _llm_spawned_this_turn: bool = False

    _turn_stt_lang: str | None = None
    _last_lang: str | None = None
    _current_tts_lang: str = "en-IN"
    _farewell_turn: bool = False
    _tts_streaming: bool = False
    _tts_aborted_for_retry: bool = False

    _llm_start_at: float | None = None
    _first_pcm_logged: bool = False
    _prewarm_task: asyncio.Task | None = None

    tts_session = TtsWsSession()

    def _spawn(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)
        return task

    def _schedule_tts_prewarm(lang: str = "en-IN", *, after_abort: bool = False):
        nonlocal _prewarm_task
        if _prewarm_task and not _prewarm_task.done():
            return
        _prewarm_task = asyncio.create_task(
            tts_session.prewarm(lang, after_abort=after_abort)
        )

    async def _cancel_prewarm_task():
        nonlocal _prewarm_task
        if _prewarm_task and not _prewarm_task.done():
            _prewarm_task.cancel()
            try:
                await _prewarm_task
            except asyncio.CancelledError:
                pass
        _prewarm_task = None

    async def _await_tts_prewarm(lang: str, *, after_abort: bool = False):
        await _cancel_prewarm_task()
        await tts_session.prewarm(lang, after_abort=after_abort)

    def _interrupt():
        if _is_farewell:
            return
        _turn_cancel.set()
        for t in list(_active_tasks):
            t.cancel()

    def _start_new_turn():
        nonlocal _turn_cancel, _speech_ended, _llm_start_at, _first_pcm_logged, _llm_spawned_this_turn
        _turn_cancel = asyncio.Event()
        _speech_ended = False
        _llm_spawned_this_turn = False
        _llm_start_at = None
        _first_pcm_logged = False

    async def _send(payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass

    def _is_farewell_transcript(text: str) -> bool:
        text_norm = text.strip().rstrip(".,!?")
        text_lower = text_norm.lower()
        return any(fw in text_lower or fw in text_norm for fw in FAREWELL_WORDS)

    def _maybe_start_llm(transcript: str):
        nonlocal _llm_spawned_this_turn, _speech_ended
        if _llm_spawned_this_turn or _is_farewell or _greeting_active or _turn_cancel.is_set():
            return
        _llm_spawned_this_turn = True
        _speech_ended = False
        logger.info(f"[TURN END] starting LLM: {transcript!r}")
        _spawn(_run_llm(transcript, _turn_cancel, _turn_stt_lang))

    async def _abort_tts():
        if _is_farewell:
            return
        await tts_session.abort()

    async def _send_pcm_chunk(chunk: bytes, cancel: asyncio.Event):
        nonlocal _first_pcm_logged
        if cancel.is_set() or stop_event.is_set():
            return
        if not _first_pcm_logged and _llm_start_at is not None:
            elapsed_ms = (time.monotonic() - _llm_start_at) * 1000
            logger.info(f"[TTS TIMING] llm_start → first_pcm_to_client={elapsed_ms:.0f}ms")
            _first_pcm_logged = True
        try:
            await websocket.send_bytes(chunk)
        except Exception:
            pass

    async def _begin_tts_turn(cancel: asyncio.Event, tts_lang: str):
        if cancel.is_set() or stop_event.is_set():
            return

        async def on_chunk(chunk: bytes):
            await _send_pcm_chunk(chunk, cancel)

        await _send(PCM_TTS_START)
        await tts_session.begin_turn(tts_lang, on_chunk)

    async def _finish_tts_turn(cancel: asyncio.Event, fallback_text: str, tts_lang: str):
        if cancel.is_set() or stop_event.is_set():
            return
        try:
            ok = await tts_session.end_turn()
            if ok and not cancel.is_set():
                await _send({"type": "tts_end"})
            else:
                logger.warning("[TTS] Stream failed — falling back to speak_full")
                await tts_session.abort()
                await _await_tts_prewarm(tts_lang, after_abort=True)
                await _speak_full_turn(fallback_text, tts_lang, cancel)
        except Exception as e:
            logger.error(f"[TTS] end_turn error: {e}")
            await tts_session.abort()
            await _await_tts_prewarm(tts_lang, after_abort=True)
            await _speak_full_turn(fallback_text, tts_lang, cancel)

    async def _speak_full_turn(text: str, tts_lang: str, cancel: asyncio.Event, cancellable: bool = True):
        if stop_event.is_set() or (cancellable and cancel.is_set()):
            return

        await _send(PCM_TTS_START)

        async def on_chunk(chunk: bytes):
            if cancellable and (cancel.is_set() or stop_event.is_set()):
                return
            try:
                await websocket.send_bytes(chunk)
            except Exception:
                pass

        async def on_done():
            if cancellable and cancel.is_set():
                await _send({"type": "tts_end"})
            elif not cancellable or not cancel.is_set():
                await _send({"type": "tts_end"})

        try:
            await tts_session.speak_full(text, tts_lang, on_chunk, on_done)
        except asyncio.CancelledError:
            logger.info("[TTS] speak_full cancelled")
            await _send({"type": "tts_end"})
        except Exception as e:
            logger.error(f"[TTS] speak_full error: {e}")
            await _send({"type": "tts_end"})

    async def on_event(
        event_type: str,
        text: str,
        stt_lang: str | None = None,
        response_lang: str | None = None,
        is_final: bool | None = None,
    ):
        nonlocal _transcript_buffer, _speech_ended, _is_farewell, _turn_stt_lang
        nonlocal _tts_streaming, _tts_aborted_for_retry, _farewell_turn, _llm_start_at

        if event_type == "speech_start":
            if _is_farewell:
                logger.info("[VAD] Farewell in progress — ignoring speech_start")
                return
            if _greeting_active:
                logger.info("[VAD] Greeting in progress — ignoring speech_start interrupt")
                return

            _tts_streaming = False
            await _abort_tts()
            _schedule_tts_prewarm(_current_tts_lang, after_abort=True)

            if _active_tasks:
                logger.info(f"[VAD INTERRUPT] Cancelling {len(_active_tasks)} task(s)")
                _interrupt()
                await _send({"type": "interrupted"})
            _start_new_turn()
            _transcript_buffer = []
            _turn_stt_lang = None
            await _send({"type": "speech_start"})

        elif event_type == "transcript" and text:
            if _is_farewell or _greeting_active:
                return

            if stt_lang:
                _turn_stt_lang = stt_lang
            _transcript_buffer.append(text)
            await _send({"type": "transcript", "text": text})

            if is_final is True:
                logger.info(f"[TURN END] final transcript (early): {text!r}")
                _maybe_start_llm(text)
            elif _speech_ended:
                logger.info(f"[TURN END] late transcript (reactive): {text!r}")
                _maybe_start_llm(text)

        elif event_type == "speech_end":
            if _is_farewell or _greeting_active:
                return

            await _send({"type": "speech_end"})
            if _transcript_buffer:
                final_transcript = _transcript_buffer[-1]
                _maybe_start_llm(final_transcript)
            else:
                logger.info("[TURN END] Transcript not yet received, waiting reactively...")
                _speech_ended = True
                _schedule_tts_prewarm(_current_tts_lang)

        else:
            payload = {"type": event_type}
            if text:
                payload["token" if event_type == "llm_token" else "text"] = text
            await _send(payload)

            if event_type == "llm_start" and not _farewell_turn:
                if _turn_cancel.is_set() or stop_event.is_set():
                    return
                _llm_start_at = time.monotonic()
                _tts_streaming = True
                await _begin_tts_turn(_turn_cancel, _current_tts_lang)

            elif event_type == "llm_token" and text and _tts_streaming and not _farewell_turn:
                if _turn_cancel.is_set() or stop_event.is_set():
                    return
                await tts_session.feed_text(text)

            elif event_type == "llm_end" and text:
                if _turn_cancel.is_set() or stop_event.is_set():
                    _tts_streaming = False
                    return

                tts_lang = llm_module.tts_code_for_language(response_lang or _last_lang or "english")
                last_user_text = _transcript_buffer[-1] if _transcript_buffer else ""

                if _is_farewell_transcript(last_user_text):
                    logger.info("[FAREWELL] Detected — playing bye TTS then closing")
                    _is_farewell = True
                    _tts_streaming = False
                    await _send({"type": "farewell_start"})
                    _spawn(_run_farewell_tts(text, tts_lang))
                elif _tts_aborted_for_retry:
                    _tts_streaming = False
                    _tts_aborted_for_retry = False
                    await _await_tts_prewarm(tts_lang, after_abort=True)
                    await _speak_full_turn(text, tts_lang, _turn_cancel)
                elif _tts_streaming:
                    _tts_streaming = False
                    await _finish_tts_turn(_turn_cancel, text, tts_lang)
                else:
                    await _speak_full_turn(text, tts_lang, _turn_cancel)

    async def _run_llm(transcript: str, cancel: asyncio.Event, stt_lang: str | None = None):
        nonlocal _last_lang, _current_tts_lang, _farewell_turn, _tts_aborted_for_retry, _tts_streaming
        if cancel.is_set() or stop_event.is_set():
            return

        lang = llm_module.resolve_response_language(stt_lang, transcript)
        if _last_lang is not None and lang != _last_lang:
            logger.info(f"[LLM] Language switch {_last_lang} → {lang}, clearing history")
            conversation_history.clear()
        _last_lang = lang
        _current_tts_lang = llm_module.tts_code_for_language(lang)
        _farewell_turn = _is_farewell_transcript(transcript)
        _tts_aborted_for_retry = False
        _tts_streaming = False

        async def on_language_retry():
            nonlocal _tts_aborted_for_retry, _tts_streaming
            logger.info("[TTS] Language retry — aborting streamed TTS")
            _tts_aborted_for_retry = True
            _tts_streaming = False
            await tts_session.abort()
            _schedule_tts_prewarm(_current_tts_lang, after_abort=True)

        try:
            await llm_module.stream_llm_response(
                transcript=transcript,
                conversation_history=conversation_history,
                event_callback=on_event,
                cancel_event=cancel,
                stt_lang=stt_lang,
                on_language_retry=on_language_retry,
            )
        except asyncio.CancelledError:
            logger.info("[LLM] Task cancelled")
            _tts_streaming = False
        except Exception as e:
            logger.error(f"[LLM TASK ERROR] {e}")

    async def _run_greeting():
        nonlocal _greeting_active
        try:
            logger.info("[GREETING] Playing opening greeting")
            await _send({"type": "greeting_start", "text": GREETING})
            await _speak_full_turn(GREETING, "en-IN", _turn_cancel, cancellable=False)
        except Exception as e:
            logger.error(f"[GREETING] Failed: {e}")
        finally:
            _greeting_active = False
            await _send({"type": "greeting_end"})
            _schedule_tts_prewarm("en-IN")

    async def _session_startup():
        try:
            await tts_session.prewarm("en-IN")
            await _run_greeting()
        except Exception as e:
            logger.error(f"[SESSION STARTUP] {e}")
            _greeting_active = False
            await _send({"type": "greeting_end"})

    async def _run_farewell_tts(text: str, tts_lang: str):
        if stop_event.is_set():
            return
        await _await_tts_prewarm(tts_lang, after_abort=True)
        await _speak_full_turn(text, tts_lang, _turn_cancel, cancellable=False)
        logger.info("[FAREWELL] TTS complete — closing WebSocket")
        await _send({"type": "session_end"})
        stop_event.set()
        await asyncio.sleep(0.5)
        try:
            await websocket.close()
        except Exception:
            pass

    async def handle_client_interrupt():
        if _is_farewell:
            logger.info("[CLIENT INTERRUPT] Farewell in progress — ignoring")
            return
        if _greeting_active:
            logger.info("[CLIENT INTERRUPT] Greeting in progress — ignoring")
            return

        _tts_streaming = False
        await _abort_tts()
        _schedule_tts_prewarm(_current_tts_lang, after_abort=True)

        if _active_tasks:
            logger.info("[CLIENT INTERRUPT] Energy detected — aborting TTS + cancelling tasks")
            _interrupt()
            await _send({"type": "interrupted"})
            _start_new_turn()
            _transcript_buffer.clear()

    stt_task = asyncio.create_task(run_streaming_stt(audio_queue, on_event, stop_event))
    asyncio.create_task(_session_startup())

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
        await tts_session.close()
        await audio_queue.put(None)
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except asyncio.TimeoutError:
            stt_task.cancel()
        logger.info("[SESSION CLOSED]")
