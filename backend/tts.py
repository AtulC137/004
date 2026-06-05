"""
tts.py — Sarvam AI WebSocket TTS for voice agent pipeline.

Uses text_to_speech_streaming WebSocket: LLM tokens → buffer → convert() → flush().
Client-side buffering ensures each convert() has valid language characters (Sarvam 422).
Model: bulbul:v3, speaker: ritu (voice only; agent name is Susha).
"""

import asyncio
import base64
import logging
import os
import re
import time

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse, ErrorResponse

logger = logging.getLogger("tts")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
TTS_MODEL = "bulbul:v3"
TTS_SPEAKER = "ritu"
MIN_BUFFER_SIZE = int(os.environ.get("TTS_MIN_BUFFER_SIZE", "50"))

_LANG_CHAR = re.compile(r"[a-zA-Z\u0900-\u097F]")
_SENTENCE_END = re.compile(r"[.?!।]\s*$")
_LONG_DIGITS = re.compile(r"\d{5,}")


def tts_language_for_text(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"
    if re.search(r"\b(kya|kahan|kab|hai|hain|mein|hoga|hogi|nahi|haan|subah|shaam)\b", text, re.I):
        return "hi-IN"
    return "en-IN"


def _has_language_chars(text: str) -> bool:
    return bool(_LANG_CHAR.search(text))


def _format_numbers_for_tts(text: str) -> str:
    """Comma-format digit runs >4 chars for better Sarvam pronunciation."""

    def _add_commas(num: str) -> str:
        if len(num) <= 4:
            return num
        parts: list[str] = []
        s = num
        while len(s) > 3:
            parts.insert(0, s[-3:])
            s = s[:-3]
        if s:
            parts.insert(0, s)
        return ",".join(parts)

    return _LONG_DIGITS.sub(lambda m: _add_commas(m.group(0)), text)


def _should_flush_buffer(buf: str) -> bool:
    if not _has_language_chars(buf):
        return False
    if len(buf) >= MIN_BUFFER_SIZE:
        return True
    return bool(_SENTENCE_END.search(buf))


class TtsWsSession:
    """Persistent Sarvam TTS WebSocket — feeds LLM tokens via convert() + flush()."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._client: AsyncSarvamAI | None = None
        self._ws = None
        self._ctx = None
        self._receiver_task: asyncio.Task | None = None
        self._audio_callback = None
        self._final_event = asyncio.Event()
        self._aborted = False
        self._configured_lang: str | None = None
        self._ws_configured = False
        self._turn_failed = False
        self._turn_started_at: float | None = None
        self._first_convert_logged = False
        self._first_audio_logged = False
        self._pending_text = ""

    @property
    def turn_failed(self) -> bool:
        return self._turn_failed

    async def _open(self):
        if self._ws is not None:
            return
        if not SARVAM_API_KEY:
            raise RuntimeError("SARVAM_API_KEY not set")
        self._client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        self._ctx = self._client.text_to_speech_streaming.connect(
            model=TTS_MODEL,
            send_completion_event=True,
        )
        self._ws = await self._ctx.__aenter__()
        self._aborted = False
        self._ws_configured = False
        logger.info("[TTS WS] Connection established")

    async def configure(self, target_language_code: str):
        async with self._lock:
            await self._open()
            if self._ws_configured and self._configured_lang == target_language_code:
                return
            await self._ws.configure(
                target_language_code=target_language_code,
                speaker=TTS_SPEAKER,
                output_audio_codec="linear16",
                speech_sample_rate=16000,
                min_buffer_size=MIN_BUFFER_SIZE,
                max_chunk_length=200,
                pace=1.0,
            )
            self._configured_lang = target_language_code
            self._ws_configured = True
            logger.info(f"[TTS WS] Configured lang={target_language_code} min_buffer={MIN_BUFFER_SIZE}")

    async def prewarm(self, target_language_code: str = "en-IN", *, after_abort: bool = False):
        """Open WS + configure — idempotent if already warm for same lang."""
        async with self._lock:
            try:
                if (
                    self._ws is not None
                    and self._ws_configured
                    and self._configured_lang == target_language_code
                    and not self._aborted
                ):
                    logger.info(f"[TTS WS] Already warm lang={target_language_code}")
                    return
                await self._open()
                if self._ws_configured and self._configured_lang == target_language_code:
                    return
                await self._ws.configure(
                    target_language_code=target_language_code,
                    speaker=TTS_SPEAKER,
                    output_audio_codec="linear16",
                    speech_sample_rate=16000,
                    min_buffer_size=MIN_BUFFER_SIZE,
                    max_chunk_length=200,
                    pace=1.0,
                )
                self._configured_lang = target_language_code
                self._ws_configured = True
                label = "Reprewarmed" if after_abort else "Prewarmed"
                logger.info(f"[TTS WS] {label} lang={target_language_code}")
            except Exception as e:
                logger.warning(f"[TTS WS] Prewarm failed: {e}")

    def _reset_turn_state(self):
        self._turn_failed = False
        self._final_event.clear()
        self._turn_started_at = time.monotonic()
        self._first_convert_logged = False
        self._first_audio_logged = False
        self._pending_text = ""

    def _start_receiver(self, audio_chunk_callback):
        self._audio_callback = audio_chunk_callback
        self._final_event.clear()
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
        self._receiver_task = asyncio.create_task(self._receiver_loop())

    async def _receiver_loop(self):
        try:
            async for message in self._ws:
                if self._aborted:
                    break
                if isinstance(message, AudioOutput):
                    audio_b64 = getattr(getattr(message, "data", None), "audio", None)
                    if audio_b64 and self._audio_callback:
                        if not self._first_audio_logged and self._turn_started_at is not None:
                            elapsed_ms = (time.monotonic() - self._turn_started_at) * 1000
                            logger.info(f"[TTS TIMING] turn_start → first_pcm={elapsed_ms:.0f}ms")
                            self._first_audio_logged = True
                        pcm = base64.b64decode(audio_b64)
                        await self._audio_callback(pcm)
                elif isinstance(message, EventResponse):
                    event_type = getattr(getattr(message, "data", None), "event_type", "")
                    if event_type == "final":
                        self._final_event.set()
                        break
                elif isinstance(message, ErrorResponse):
                    err = getattr(message, "data", None)
                    logger.error(f"[TTS WS] Error: {err}")
                    self._turn_failed = True
                    self._final_event.set()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[TTS WS] Receiver error: {e}")
            self._turn_failed = True
            self._final_event.set()

    async def _safe_convert(self, text: str):
        if self._aborted or self._ws is None or self._turn_failed:
            return
        if not text or not _has_language_chars(text):
            return
        text = _format_numbers_for_tts(text)
        if not self._first_convert_logged and self._turn_started_at is not None:
            elapsed_ms = (time.monotonic() - self._turn_started_at) * 1000
            logger.info(f"[TTS TIMING] turn_start → first_convert={elapsed_ms:.0f}ms")
            self._first_convert_logged = True
        try:
            await self._ws.convert(text)
        except Exception as e:
            logger.error(f"[TTS WS] convert error: {e}")
            self._turn_failed = True

    async def _flush_pending_buffer(self):
        if self._pending_text.strip() and _has_language_chars(self._pending_text):
            await self._safe_convert(self._pending_text)
        self._pending_text = ""

    async def begin_turn(self, target_language_code: str, audio_chunk_callback):
        """Open/configure WS and start receiving audio for a new utterance."""
        async with self._lock:
            await self._open()
            if not (self._ws_configured and self._configured_lang == target_language_code):
                await self._ws.configure(
                    target_language_code=target_language_code,
                    speaker=TTS_SPEAKER,
                    output_audio_codec="linear16",
                    speech_sample_rate=16000,
                    min_buffer_size=MIN_BUFFER_SIZE,
                    max_chunk_length=200,
                    pace=1.0,
                )
                self._configured_lang = target_language_code
                self._ws_configured = True
            self._reset_turn_state()
            self._start_receiver(audio_chunk_callback)

    async def feed_text(self, text: str):
        """Buffer LLM tokens; convert() only when chunk has language chars."""
        if not text or self._aborted or self._ws is None or self._turn_failed:
            return
        async with self._lock:
            if self._aborted or self._ws is None or self._turn_failed:
                return
            self._pending_text += text
            if _should_flush_buffer(self._pending_text):
                await self._safe_convert(self._pending_text)
                self._pending_text = ""

    async def flush_and_wait(self, timeout: float = 30.0):
        if self._aborted or self._ws is None or self._turn_failed:
            return
        async with self._lock:
            if self._aborted or self._ws is None or self._turn_failed:
                return
            try:
                await self._flush_pending_buffer()
                await self._ws.flush()
                await asyncio.wait_for(self._final_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("[TTS WS] Timed out waiting for final event")
                self._turn_failed = True
            except Exception as e:
                logger.error(f"[TTS WS] flush error: {e}")
                self._turn_failed = True
            finally:
                if self._receiver_task and not self._receiver_task.done():
                    self._receiver_task.cancel()
                    try:
                        await self._receiver_task
                    except asyncio.CancelledError:
                        pass
                    self._receiver_task = None

    async def end_turn(self) -> bool:
        """flush() at utterance end. Returns False if caller should speak_full fallback."""
        if self._turn_failed or self._aborted:
            return False
        await self.flush_and_wait()
        return not self._turn_failed

    async def speak_full(
        self,
        text: str,
        target_language_code: str,
        audio_chunk_callback,
        done_callback=None,
    ):
        """Single utterance: convert full text + flush (greeting, farewell, retry)."""
        if not text or not text.strip():
            if done_callback:
                await done_callback()
            return
        try:
            async with self._lock:
                await self._open()
                if not (self._ws_configured and self._configured_lang == target_language_code):
                    await self._ws.configure(
                        target_language_code=target_language_code,
                        speaker=TTS_SPEAKER,
                        output_audio_codec="linear16",
                        speech_sample_rate=16000,
                        min_buffer_size=MIN_BUFFER_SIZE,
                        max_chunk_length=200,
                        pace=1.0,
                    )
                    self._configured_lang = target_language_code
                    self._ws_configured = True
                self._reset_turn_state()
                self._start_receiver(audio_chunk_callback)
                prepared = _format_numbers_for_tts(text.strip())
                logger.info(f"[TTS WS] speak_full [{target_language_code}]: {prepared[:60]}...")
                await self._safe_convert(prepared)
                if not self._turn_failed and not self._aborted:
                    try:
                        await self._ws.flush()
                        await asyncio.wait_for(self._final_event.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning("[TTS WS] speak_full timed out waiting for final event")
                        self._turn_failed = True
                    except Exception as e:
                        logger.error(f"[TTS WS] speak_full flush error: {e}")
                        self._turn_failed = True
                if self._receiver_task and not self._receiver_task.done():
                    self._receiver_task.cancel()
                    try:
                        await self._receiver_task
                    except asyncio.CancelledError:
                        pass
                    self._receiver_task = None
        except Exception as e:
            logger.error(f"[TTS WS] speak_full error: {e}")
            self._turn_failed = True
        finally:
            if done_callback:
                await done_callback()

    async def abort(self):
        """Close TTS socket on interrupt (Sarvam barge-in recipe)."""
        async with self._lock:
            self._aborted = True
            self._pending_text = ""
            if self._receiver_task and not self._receiver_task.done():
                self._receiver_task.cancel()
                try:
                    await self._receiver_task
                except asyncio.CancelledError:
                    pass
            self._receiver_task = None
            self._audio_callback = None
            if self._ctx is not None:
                try:
                    await self._ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"[TTS WS] Close: {e}")
            self._ws = None
            self._ctx = None
            self._client = None
            self._configured_lang = None
            self._ws_configured = False
            logger.info("[TTS WS] Aborted")

    async def close(self):
        await self.abort()


async def stream_tts_audio(
    text: str,
    audio_chunk_callback,
    done_callback,
    target_language_code: str | None = None,
    session: TtsWsSession | None = None,
):
    """Backward-compatible wrapper — delegates to speak_full on a session."""
    lang = target_language_code or tts_language_for_text(text)
    own_session = session is None
    if own_session:
        session = TtsWsSession()
    await session.speak_full(text, lang, audio_chunk_callback, done_callback)
    if own_session:
        await session.close()


async def close():
    """App shutdown — per-connection sessions close themselves."""
    pass
