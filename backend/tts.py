"""

tts.py — Sarvam AI WebSocket TTS for voice agent pipeline.



Uses text_to_speech_streaming WebSocket: LLM tokens → convert() → flush().

Sarvam min_buffer_size handles synthesis gating — no client-side threshold.

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





def tts_language_for_text(text: str) -> str:

    if re.search(r"[\u0900-\u097F]", text):

        return "hi-IN"

    if re.search(r"\b(kya|kahan|kab|hai|hain|mein|hoga|hogi|nahi|haan|subah|shaam)\b", text, re.I):

        return "hi-IN"

    return "en-IN"





def _has_allowed_chars(text: str, lang: str) -> bool:

    if lang == "hi-IN":

        return bool(re.search(r"[\u0900-\u097F]|[a-zA-Z]", text))

    return bool(re.search(r"[a-zA-Z]", text))





class TtsWsSession:

    """Persistent Sarvam TTS WebSocket — feeds LLM tokens via convert() + flush()."""



    def __init__(self):

        self._client: AsyncSarvamAI | None = None

        self._ws = None

        self._ctx = None

        self._receiver_task: asyncio.Task | None = None

        self._audio_callback = None

        self._final_event = asyncio.Event()

        self._aborted = False

        self._configured_lang: str | None = None

        self._turn_failed: bool = False

        self._turn_started_at: float | None = None

        self._first_convert_logged: bool = False

        self._first_audio_logged: bool = False



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

        logger.info("[TTS WS] Connection established")



    async def configure(self, target_language_code: str):

        await self._open()

        if self._configured_lang == target_language_code:

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

        logger.info(f"[TTS WS] Configured lang={target_language_code} min_buffer={MIN_BUFFER_SIZE}")



    async def prewarm(self, target_language_code: str = "en-IN", *, after_abort: bool = False):

        """Open WS + configure — idempotent if already warm for same lang."""

        try:

            if (

                self._ws is not None

                and self._configured_lang == target_language_code

                and not self._aborted

            ):

                logger.info(f"[TTS WS] Already warm lang={target_language_code}")

                return

            await self.configure(target_language_code)

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

        if not text or not _has_allowed_chars(text, self._configured_lang or "en-IN"):

            return

        if not self._first_convert_logged and self._turn_started_at is not None:

            elapsed_ms = (time.monotonic() - self._turn_started_at) * 1000

            logger.info(f"[TTS TIMING] turn_start → first_convert={elapsed_ms:.0f}ms")

            self._first_convert_logged = True

        try:

            await self._ws.convert(text)

        except Exception as e:

            logger.error(f"[TTS WS] convert error: {e}")

            self._turn_failed = True



    async def begin_turn(self, target_language_code: str, audio_chunk_callback):

        """Open/configure WS and start receiving audio for a new utterance."""

        await self.configure(target_language_code)

        self._reset_turn_state()

        self._start_receiver(audio_chunk_callback)



    async def feed_text(self, text: str):

        """Pass LLM tokens straight to convert(); Sarvam min_buffer_size gates synthesis."""

        if not text or self._aborted or self._ws is None or self._turn_failed:

            return

        await self._safe_convert(text)



    async def flush_and_wait(self, timeout: float = 30.0):

        if self._aborted or self._ws is None or self._turn_failed:

            return

        try:

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

            await self.begin_turn(target_language_code, audio_chunk_callback)

            logger.info(f"[TTS WS] speak_full [{target_language_code}]: {text[:60]}...")

            await self._safe_convert(text.strip())

            await self.flush_and_wait()

        except Exception as e:

            logger.error(f"[TTS WS] speak_full error: {e}")

            self._turn_failed = True

        finally:

            if done_callback:

                await done_callback()



    async def abort(self):

        """Close TTS socket on interrupt (Sarvam barge-in recipe)."""

        self._aborted = True

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


