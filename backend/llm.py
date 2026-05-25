"""
llm.py
- sarvam-m is a reasoning model: thinking goes in reasoning_content, NOT in content
- So <think> stripping was wrong approach — content arrives clean already
- Root cause of empty response: max_tokens:200 was still too low for reasoning budget
- Fix: raise max_tokens to 1024, use thinking_tokens param if available, log raw chunks
"""

import json
import logging
import os
import re

import httpx

logger = logging.getLogger("llm")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
LLM_MODEL = "sarvam-m"

_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
    headers={
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
)

SYSTEM_PROMPT = """You are a voice receptionist for an Adobe event.
Respond with ONE short sentence, maximum 15 words. No greetings, no filler. Direct answer only.
English question → English answer. Hindi/Hinglish question → Hinglish answer.

EXAMPLES:
User: "Where is the event?" → "The event is at The Pride Hotel, Pune."
User: "What time?" → "It starts at 10:00 AM on 8 May 2026."
User: "What is this event about?" → "It's an Adobe roundtable on PDF innovation, GenAI, and creative workflows."
User: "Who can attend?" → "It's for CMOs, CIOs, CTOs, Heads of Design, and Legal."
User: "kya event free hai?" → "Haan, free hai lekin invite-only hai."
User: "contact number?" → "Contact number hai +91 9850362300."

EVENT DETAILS:
- Name: Adobe Exclusive Roundtable
- Date: 8 May 2026, 10:00 AM
- Venue: The Pride Hotel, 5 University Road, Shivajinagar, Pune
- Contact: +91 9850362300
- Eligible: CMOs, CIOs, CTOs, Heads of Design, Heads of Legal
- Topics: PDF innovation, GenAI, Creative workflows, Networking"""


async def stream_llm_response(
    transcript: str,
    conversation_history: list,
    event_callback
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": transcript},
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 1024,   # sarvam-m needs budget for hidden reasoning + answer
        "temperature": 0.2,
        "stream": True,
    }

    await event_callback("llm_start", "")

    full_text = ""
    last_visible = ""

    try:
        async with _http_client.stream(
            "POST",
            SARVAM_CHAT_URL,
            json=payload
        ) as response:

            if response.status_code != 200:
                body = await response.aread()
                logger.error(f"[LLM] HTTP {response.status_code}: {body.decode()}")
                await event_callback("llm_error", f"LLM error {response.status_code}")
                return

            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue

                data_str = raw_line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})

                    # --- DEBUG: log the raw delta so we can see what the model sends ---
                    if delta:
                        logger.debug(f"[LLM DELTA] {json.dumps(delta)}")

                    # sarvam-m puts visible answer in delta.content
                    # reasoning/thinking goes in delta.reasoning_content — we ignore it
                    token = delta.get("content", "") or ""

                    # Safety net: strip any inline <think> blocks just in case
                    if token:
                        full_text += token
                        cleaned = re.sub(
                            r"<think>.*?</think>", "", full_text, flags=re.DOTALL
                        )
                        # Also drop any unclosed <think> tail still streaming
                        if "<think>" in cleaned and "</think>" not in cleaned.split("<think>")[-1]:
                            cleaned = cleaned.rsplit("<think>", 1)[0]
                        cleaned = cleaned.strip()

                        visible = cleaned[len(last_visible):]
                        if visible:
                            await event_callback("llm_token", visible)
                        last_visible = cleaned

                except Exception as e:
                    logger.debug(f"[LLM PARSE ERROR] {e} | raw: {data_str[:120]}")
                    continue

    except Exception as e:
        logger.error(f"[LLM] {e}")
        await event_callback("llm_error", str(e))
        return

    # Final strip pass
    final_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
    final_text = re.sub(r"<think>.*", "", final_text, flags=re.DOTALL).strip()

    # Fallback: if content was empty but last_visible has something, use that
    if not final_text and last_visible:
        final_text = last_visible

    if not final_text:
        logger.warning(f"[LLM] Empty response. Raw full_text was: {repr(full_text[:300])}")
        await event_callback("llm_error", "No response from model")
        return

    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": final_text})
    conversation_history[:] = conversation_history[-20:]

    logger.info(f"AI: {final_text}")
    await event_callback("llm_end", final_text)


async def close():
    await _http_client.aclose()