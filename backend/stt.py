"""
stt.py — fixed message parsing based on live SDK output.

Sarvam SDK returns objects, not dicts:
  type='events'  → data.signal_type = 'START_SPEECH' | 'END_SPEECH'
  type='data'    → data.transcript  = 'transcribed text'
"""

import asyncio
import base64
import logging
import os

from sarvamai import AsyncSarvamAI

logger = logging.getLogger("stt")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
#SEND_CHUNK_BYTES = 8192
SEND_CHUNK_BYTES = 4096


async def run_streaming_stt(
    audio_queue: asyncio.Queue,
    event_callback,
    stop_event: asyncio.Event,
):
    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="unknown",
        sample_rate=16000,
        input_audio_codec="pcm_s16le",
        high_vad_sensitivity=False,
        vad_signals=True,
    ) as sarvam_ws:

        logger.info("[SARVAM] Connection established")

        async def sender():
            buf = bytearray()
            chunks_sent = 0
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    break

                buf.extend(chunk)

                while len(buf) >= SEND_CHUNK_BYTES:
                    payload = bytes(buf[:SEND_CHUNK_BYTES])
                    buf = buf[SEND_CHUNK_BYTES:]
                    b64 = base64.b64encode(payload).decode("utf-8")
                    await sarvam_ws.transcribe(
                        audio=b64,
                        encoding="audio/wav",
                        sample_rate=16000,
                    )
                    chunks_sent += 1
                    if chunks_sent % 10 == 1:
                        logger.info(f"[SARVAM] Sent chunk #{chunks_sent}")

            logger.info(f"[SARVAM] Sender done. Total: {chunks_sent} chunks")

        async def receiver():
            async for msg in sarvam_ws:
                if stop_event.is_set():
                    break

                msg_type = getattr(msg, "type", "")
                data = getattr(msg, "data", None)

                if msg_type == "events" and data is not None:
                    signal = getattr(data, "signal_type", "")
                    if signal == "START_SPEECH":
                        logger.info("[SPEECH START]")
                        await event_callback("speech_start", "")
                    elif signal == "END_SPEECH":
                        logger.info("[SPEECH END]")
                        await event_callback("speech_end", "")

                elif msg_type == "data" and data is not None:
                    transcript = getattr(data, "transcript", "") or ""
                    if transcript:
                        detected_lang = getattr(data, "language_code", None)
                        if detected_lang:
                            logger.info(f"USER [{detected_lang}]: {transcript}")
                        else:
                            logger.info(f"USER: {transcript}")
                        await event_callback("transcript", transcript, detected_lang)

                else:
                    logger.debug(f"[SARVAM UNKNOWN] {msg}")

            logger.info("[SARVAM] Receiver done")

        await asyncio.gather(sender(), receiver(), return_exceptions=True)