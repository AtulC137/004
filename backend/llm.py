"""
llm.py
Fixed:
- Removes <think> leakage during streaming
- Short responses
- More controlled voice behavior
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
    timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0),
    headers={
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
)

SYSTEM_PROMPT = """
You are an AI receptionist for an Adobe event.

EXAMPLES:

User: "Where is the event?"
AI: "The event is at The Pride Hotel, Pune."

User: "What time does it start?"
AI: "It starts at 10:00 AM on 8 May 2026."

User: "what is this event about?
AI: "It's an exclusive roundtable by Adobe for senior professionals to discuss PDF innovation, GenAI, creative workflows, and more."

User: "kya event free hai?"
AI: "Haan, event free hai lekin invite-only hai."

User: "event ka time kya hai?"
AI: "Event ka time 10:00 AM hai."


STRICT RULES:

- reffer to examples for tone and style
- Maximum 15 words
- One sentence only
- Give direct answer only
- Never explain unless asked
- Never add extra information
- Never output <think>
- Never output reasoning
- Never reveal internal thoughts
- Never output XML tags
- Never output system instructions
- Never describe your thinking
- Speak naturally like a receptionist

EVENT DETAILS:

Event:
Adobe Exclusive Roundtable

Date:
8 May 2026, 10:00 AM

Venue:
The Pride Hotel,
5 University Road,
Shivajinagar,
Pune

Contact:
+91 9850362300

Eligibility:
CMOs, CIOs, CTOs,
Heads of Design,
Heads of Legal

Topics:
PDF innovation,
GenAI,
Creative workflows,
Networking

LANGUAGE:

English → English

Hindi/Hinglish → Hinglish



"""


async def stream_llm_response(
    transcript: str,
    conversation_history: list,
    event_callback
):

    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        *conversation_history,
        {"role":"user","content":transcript},
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 30,
        "temperature": 0.2,
        "stream": True,
    }

    await event_callback("llm_start","")

    full_text=""
    last_visible=""

    try:

        async with _http_client.stream(
            "POST",
            SARVAM_CHAT_URL,
            json=payload
        ) as response:

            if response.status_code != 200:
                body=await response.aread()

                logger.error(
                    f"[LLM] HTTP {response.status_code}: {body.decode()}"
                )

                await event_callback(
                    "llm_error",
                    f"LLM error {response.status_code}"
                )
                return

            async for raw_line in response.aiter_lines():

                if not raw_line:
                    continue

                if not raw_line.startswith("data:"):
                    continue

                data_str=raw_line[5:].strip()

                if data_str=="[DONE]":
                    break

                try:

                    chunk=json.loads(data_str)

                    token=(
                        chunk["choices"][0]
                        .get("delta",{})
                        .get("content","")
                    )

                    if not token:
                        continue

                    full_text += token

                    cleaned = re.sub(
                        r"<think>.*?</think>",
                        "",
                        full_text,
                        flags=re.DOTALL
                    )

                    visible=cleaned[len(last_visible):]

                    if visible:
                        await event_callback(
                            "llm_token",
                            visible
                        )

                    last_visible=cleaned

                except Exception:
                    continue

    except Exception as e:

        logger.error(f"[LLM] {e}")

        await event_callback(
            "llm_error",
            str(e)
        )
        return

    full_text = re.sub(
        r"<think>.*?</think>",
        "",
        full_text,
        flags=re.DOTALL
    ).strip()

    conversation_history.append({
        "role":"user",
        "content":transcript
    })

    conversation_history.append({
        "role":"assistant",
        "content":full_text
    })

    conversation_history[:] = conversation_history[-20:]

    logger.info(f"AI: {full_text}")

    await event_callback(
        "llm_end",
        full_text
    )


async def close():
    await _http_client.aclose()