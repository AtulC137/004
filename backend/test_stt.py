"""
test_stt.py
Run: python test_stt.py
Expects test.wav in the same folder.
"""

import asyncio
import base64
import os
import wave
from sarvamai import AsyncSarvamAI

CHUNK_BYTES = 8192


def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        print(f"[WAV] {path}: {channels}ch, {rate}Hz, {len(frames)} bytes, "
              f"{len(frames)/rate/channels/2:.2f}s")
        return frames, rate


async def test():
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        print("[ERROR] SARVAM_API_KEY not set")
        return

    pcm, rate = load_wav("test.wav")

    client = AsyncSarvamAI(api_subscription_key=api_key)

    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="en-IN",
        sample_rate=rate,
        input_audio_codec="pcm_s16le",
        high_vad_sensitivity=True,
        vad_signals=True,
    ) as ws:
        print("[OK] Connected to Sarvam")

        # Send all audio chunks
        offset = 0
        chunk_num = 0
        while offset < len(pcm):
            chunk = pcm[offset:offset + CHUNK_BYTES]
            offset += CHUNK_BYTES
            chunk_num += 1
            await ws.transcribe(
                audio=base64.b64encode(chunk).decode(),
                encoding="audio/wav",
                sample_rate=rate,
            )
            print(f"[SEND] chunk {chunk_num} ({len(chunk)} bytes)")
            await asyncio.sleep(0.05)

        # Send silence to trigger VAD end
        print("[SEND] silence padding to trigger VAD...")
        silence = bytes(CHUNK_BYTES)
        for _ in range(10):
            await ws.transcribe(
                audio=base64.b64encode(silence).decode(),
                encoding="audio/wav",
                sample_rate=rate,
            )

        # Read all responses
        print("[WAIT] waiting for Sarvam response...")
        async for msg in ws:
            print(f"[RAW] {msg}")
            if isinstance(msg, dict):
                t = msg.get("type", "")
            else:
                t = getattr(msg, "type", "")
            if t == "speech_end":
                print("[DONE] Got speech_end, exiting")
                break


asyncio.run(test())