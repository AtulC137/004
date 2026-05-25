"""
llm.py — Pre-warmed Sarvam LLM streaming module.

Uses sarvam-30b via OpenAI-compatible API with SSE streaming.
Client and system prompt initialized ONCE at import time (pre-warmed).
Trigger: call stream_llm_response() on every transcript.
"""

import asyncio
import json
import logging
import os

import httpx

logger = logging.getLogger("llm")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
LLM_MODEL = "sarvam-30b"  # Low-latency, Hinglish-native, 2.4B active params

# ── Pre-warm: HTTP client initialized once at module load ────────────────────
_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
    headers={
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    },
    http2=False,
)

# ── System prompt — loaded once ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI voice assistant handling incoming calls for an exclusive Adobe event.

EVENT DETAILS:
- Event: Adobe Exclusive Roundtable
- Description: Exclusive roundtable followed by lunch for senior business and technology leaders
- Topics: PDF innovation, future creative workflows, collaboration & asset ownership, Gen AI for business, industry use cases, networking with experts
- Date & Time: 8 May 2026, 10:00 AM onwards
- Venue: The Pride Hotel, 5 University Rd, Narveer Tanaji Wadi, Shivajinagar, Pune – 411005
- Registration: Sujata India Event Registration link
- Contact: +91 9850362300
- Website: Sujata India Official Website
- Eligibility: ONLY open for CMOs, CIOs, CTOs, Heads of Design, Heads of Legal. NOT open for channel partners.

INSTRUCTIONS:
- Respond in Hinglish (mix of Hindi and English, just like Indians naturally speak)
- Keep answers SHORT — 1 to 3 sentences max. This is a voice call, not an essay.
- Be warm, professional, helpful
- If someone asks about eligibility, clearly tell them who can and cannot attend
- If someone wants to register, give them the contact number: +91 9850362300
- Do NOT use markdown, bullet points, or asterisks — speak naturally
- Do NOT say "I" too much — be conversational
- Match the language style of the caller (more Hindi = reply more Hindi, more English = reply more English)

Examples of good responses:
- "Haan bilkul! Event 8 May ko hai, The Pride Hotel Pune mein, 10 baje se. Aap CMO hain toh eligible hain."
- "Registration ke liye +91 9850362300 pe call karein ya Sujata India ki website visit karein."
- "Sorry, yeh event channel partners ke liye open nahi hai. Sirf CXO level leaders ke liye hai."
"""


async def stream_llm_response(
    transcript: str,
    conversation_history: list,
    event_callback,  # async fn(event_type: str, text: str)
):
    """
    Stream LLM response token by token.
    Fires event_callback with:
      ("llm_start", "")
      ("llm_token", "<token>")  — for each streamed token
      ("llm_end", "<full_text>")

    Appends assistant reply to conversation_history in-place.
    """
    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": transcript},
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
        "stream": True,
    }

    await event_callback("llm_start", "")
    full_text = ""

    try:
        async with _http_client.stream(
            "POST",
            SARVAM_CHAT_URL,
            json=payload,
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

                data_str = raw_line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        full_text += token
                        await event_callback("llm_token", token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except httpx.TimeoutException:
        logger.error("[LLM] Request timed out")
        await event_callback("llm_error", "Request timed out")
        return
    except Exception as e:
        logger.error(f"[LLM] Unexpected error: {e}")
        await event_callback("llm_error", str(e))
        return

    if full_text:
        logger.info(f"[LLM] Response: {full_text}")
        # Append to conversation history for multi-turn
        conversation_history.append({"role": "user", "content": transcript})
        conversation_history.append({"role": "assistant", "content": full_text})
        # Keep history bounded (last 10 turns = 20 messages)
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]

    await event_callback("llm_end", full_text)


async def close():
    """Call on shutdown to cleanly close HTTP client."""
    await _http_client.aclose()