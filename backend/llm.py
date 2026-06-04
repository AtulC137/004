"""
llm.py
- Model: sarvam-m
- Key fix: if model runs out of tokens mid-think, we still return whatever
  answer came after </think>. If NO </think> ever arrives within the stream,
  we fall back to a canned "I can only answer short questions" response.
- cancel_event for interrupt support
"""

import asyncio
import json
import logging
import os
import re

import httpx

logger = logging.getLogger("llm")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
LLM_MODEL = "sarvam-30b"

_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
    headers={
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
)

SYSTEM_PROMPT = """You are a voice receptionist for an Adobe event. Answer immediately and briefly.
ONE sentence only. Maximum 20 words. No thinking. No explanation. Just the answer. if user questions in english reply in english, if in hindi reply in hindi. if he questions in Hinglish then reply in Hinglish. If you don't know the answer, say "Sorry, I don't have that information." Here is the event info:
User: "Where is the event?" → "The Pride Hotel, 5 University Road, Shivajinagar, Pune."
User: "What time?" → "10:00 AM on 8 May 2026."
User: "What is this about?" → "Adobe roundtable on PDF innovation, GenAI, and creative workflows."
User: "Tell me in detail." → "Adobe roundtable on 8 May at The Pride Hotel, Pune, covering PDF, GenAI, and creative workflows."
User: "Who can attend?" → "CMOs, CIOs, CTOs, Heads of Design and Legal."
User: "kya event free hai?" → "Haan, free hai lekin invite-only hai."
User: "contact?" → "+91 9850362300."
User: "bye" / "thank you" / "thanks" / "that's all" → "Thank you for your interest! Hope to see you at the event. Goodbye!"

EVENT: Adobe Exclusive Roundtable | 8 May 2026 10:00 AM | The Pride Hotel, 5 University Road, Shivajinagar, Pune | +91 9850362300 | Eligible: CMOs CIOs CTOs Heads of Design/Legal | Topics: PDF innovation GenAI Creative workflows Networking"""

FALLBACK_RESPONSE = "Sorry, please ask a specific question about the event."


async def stream_llm_response(
    transcript: str,
    conversation_history: list,
    event_callback,
    cancel_event: asyncio.Event = None,
):
    if cancel_event is None:
        cancel_event = asyncio.Event()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": transcript},
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 120,   # enough for think block + answer
        "temperature": 0.2,
        "stream": True,
        "reasoning_effort": None
    }

    await event_callback("llm_start", "")

    full_text = ""   # raw including <think>
    emitted = ""     # cleaned text already sent

    try:
        async with _http_client.stream("POST", SARVAM_CHAT_URL, json=payload) as response:

            if response.status_code != 200:
                body = await response.aread()
                logger.error(f"[LLM] HTTP {response.status_code}: {body.decode()}")
                await event_callback("llm_error", f"LLM error {response.status_code}")
                return

            async for raw_line in response.aiter_lines():

                if cancel_event.is_set():
                    logger.info("[LLM] Interrupted mid-stream")
                    return

                if not raw_line or not raw_line.startswith("data:"):
                    continue

                data_str = raw_line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    token = chunk["choices"][0].get("delta", {}).get("content", "") or ""
                    if not token:
                        continue

                    full_text += token

                    # Suppress tokens while inside an open <think> block
                    in_think = full_text.count("<think>") > full_text.count("</think>")
                    if in_think:
                        continue

                    cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
                    new_part = cleaned[len(emitted):]
                    if new_part:
                        await event_callback("llm_token", new_part)
                        emitted = cleaned

                except Exception as e:
                    logger.debug(f"[LLM PARSE ERROR] {e}")
                    continue

    except asyncio.CancelledError:
        logger.info("[LLM] CancelledError — interrupted")
        return
    except Exception as e:
        logger.error(f"[LLM] {e}")
        await event_callback("llm_error", str(e))
        return

    if cancel_event.is_set():
        return

    # Strip think blocks from final text
    final_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
    final_text = re.sub(r"<think>.*", "", final_text, flags=re.DOTALL).strip()

    # If model ran out of tokens mid-think and produced no answer, use fallback
    if not final_text:
        logger.warning(f"[LLM] No answer after think strip — using fallback. think_len={len(full_text)}")
        final_text = FALLBACK_RESPONSE
        await event_callback("llm_token", final_text)

    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": final_text})
    conversation_history[:] = conversation_history[-20:]

    logger.info(f"AI: {final_text}")
    await event_callback("llm_end", final_text)


async def close():
    await _http_client.aclose()