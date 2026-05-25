"""
tts.py — Sarvam AI HTTP Streaming TTS for Hinglish voice responses.

Uses Sarvam's /text-to-speech/stream endpoint (HTTP POST, binary MP3 stream).
No SDK — raw httpx, same as before. Client pre-warmed once at module load.
Trigger: call stream_tts_audio() after llm_end.

Model: bulbul:v3 — 30+ voices, supports Hindi/Hinglish natively, low latency.
Voice: "ananya" — warm female voice, excellent for Hinglish. 
Other good options: "shubh" (male), "vidya" (female), "arjun" (male).
Full list: https://dashboard.sarvam.ai/text-to-speech
"""

import logging
import os

import httpx

logger = logging.getLogger("tts")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

# ── Voice & model config ──────────────────────────────────────────────────────
# bulbul:v3 — best quality, 30+ voices, Hinglish native
TTS_MODEL   = "bulbul:v3"
TTS_SPEAKER = "ritu"        # Warm female voice, natural for Hinglish
                               # Alternatives: "shubh", "vidya", "arjun", "meera"

TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"

# ── Pre-warm: HTTP client initialized once at module load ─────────────────────
_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    headers={
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
        # Sarvam streams raw MP3 binary — same as what frontend already handles
        "Accept": "audio/mpeg",
    },
)


async def stream_tts_audio(
    text: str,
    audio_chunk_callback,  # async fn(chunk: bytes) — called for each audio chunk
    done_callback,         # async fn() — called when stream is complete
):
    """
    Stream TTS audio chunk by chunk from Sarvam AI.

    Calls audio_chunk_callback(bytes) for each MP3 chunk received.
    Calls done_callback() when stream ends.

    Frontend receives binary WebSocket frames and plays via Web Audio API.
    No changes needed on the frontend — Sarvam returns the same raw MP3 stream
    that ElevenLabs did.
    """
    if not text or not text.strip():
        await done_callback()
        return

    if not SARVAM_API_KEY:
        logger.error("[TTS] SARVAM_API_KEY not set")
        await done_callback()
        return

    payload = {
        "text": text,
        "model": TTS_MODEL,
        "speaker": TTS_SPEAKER,
        "target_language_code": "hi-IN",   # Hinglish — Hindi script + English words
        "output_audio_codec": "mp3",        # Raw MP3 stream (frontend decodes it)
        "output_audio_bitrate": "128k",     # Good quality, matches previous ElevenLabs output
        "pace": 1.0,                        # 0.5–2.0; 1.0 = natural speed
        "enable_preprocessing": True,       # Normalises English words/numbers in Hinglish
    }

    try:
        async with _http_client.stream(
            "POST",
            TTS_STREAM_URL,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(f"[TTS] HTTP {response.status_code}: {body.decode()}")
                await done_callback()
                return

            logger.info(f"[TTS] Streaming audio for: {text[:60]}...")
            chunk_count = 0

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    await audio_chunk_callback(chunk)
                    chunk_count += 1

            logger.info(f"[TTS] Done. Sent {chunk_count} audio chunks")

    except httpx.TimeoutException:
        logger.error("[TTS] Request timed out")
    except Exception as e:
        logger.error(f"[TTS] Unexpected error: {e}")
    finally:
        await done_callback()


async def close():
    """Call on shutdown to cleanly close HTTP client."""
    await _http_client.aclose()