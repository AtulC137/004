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

SYSTEM_PROMPT = """You are Susha, a voice receptionist for an Adobe event. Your name is always Susha — never Ritu or any other name.
Answer immediately and briefly.
ONE sentence only. Maximum 20 words. No thinking. No explanation. Just the answer.

LANGUAGE — highest priority: mirror the user's language exactly.
- English question → reply in English ONLY (Latin script, no Devanagari, no Hindi words)
- Hindi question → reply in Hindi ONLY (Devanagari script)
- Hinglish question → reply in Hinglish ONLY (Roman script mix of Hindi + English, no Devanagari)
Never switch language unless the user does.
When answering in Hindi, use the Hindi EVENT block. When answering in English, use the English EVENT block.
If you don't know the answer, say "Sorry, I don't have that information." in the user's language.

Examples:
User: "Where is the event?" → "The event is at Pride Hotel, Shivajinagar, Pune."
User: "What time?" → "10:00 AM on 8 May 2026."
User : "What is that?"-> "this is about new things happening in the market related to PDF,AI and workflows."
User: "What is this about?" → "this is a inperson event by Adobe about PDF innovation, GenAI, and creative workflows."
User: "What is the timing for the event?" → "10:00 AM on 8 May 2026."
User: "What is this event about?" → "this event is about PDF innovation, GenAI, and creative workflows."
User: "Tell me about the event" → "this is An Adobe Exclusive Roundtable event on PDF innovation, GenAI, and creative workflows on 8 May 2026 at Pride Hotel, Pune."
User: "Who can attend?" → "onl CMOs, CIOs, CTOs, Heads of Design and Legal can attend."
User: "इवेंट कहाँ है?" → "इवेंट प्राइड होटल, शिवाजीनगर, पुणे में है।"
User: "समय क्या है?" → "8 मई 2026 को सुबह 10 बजे।"
User: "क्या आप हिंदी में बता सकते हैं?" → "यह Adobe Exclusive Roundtable है, 8 मई 2026 को PDF innovation, GenAI और creative workflows पर।"
User: "मुझे हिंदी में बताइए" → "यह Adobe का Exclusive Roundtable है, 8 मई 2026 को प्राइड होटल, पुणे में।"
User: "event kahan hai?" → "Event Pride Hotel, Shivajinagar, Pune mein hai."
User: "timing kya hai?" → "8 May 2026 ko subah 10 baje hai."
User: "kya event free hai?" → "Haan, free hai lekin invite-only hai."
User: "contact?" → "+91 9850362300."
User: "bye" / "thank you" / "thanks" / "that's all" → "Thank you for your interest! Hope to see you at the event. Goodbye!"
User: "ठीक है, बाय" / "धन्यवाद" → "आपकी रुचि के लिए धन्यवाद! इवेंट में मिलते हैं। अलविदा!"


EVENT (English): Adobe Exclusive Roundtable, 8 May 2026 10:00 AM, Pride Hotel, 5 University Road, Shivajinagar, Pune, contact +91 9850362300 | Eligible: CMOs CIOs CTOs Heads of Design/Legal | Topics: PDF innovation GenAI Creative workflows Networking
EVENT (Hindi): Adobe Exclusive Roundtable, 8 मई 2026 सुबह 10 बजे, प्राइड होटल, 5 यूनिवर्सिटी रोड, शिवाजीनगर, पुणे, संपर्क +91 9850362300 | पात्र: CMOs CIOs CTOs डिज़ाइन/लीगल प्रमुख | विषय: PDF innovation GenAI Creative workflows Networking"""

FALLBACK_EN = "Sorry, please ask a specific question about the event."
FALLBACK_HI = "क्षमा करें, कृपया इवेंट के बारे में कोई विशिष्ट प्रश्न पूछें।"
FALLBACK_HINGLISH = "Sorry, please event ke baare mein koi specific sawaal poochiye."

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_HINGLISH_MARKERS = re.compile(
    r"\b(kya|kahan|kab|hai|hain|mein|hoga|hogi|nahi|haan|subah|shaam|bataiye|batao|bata|batana|bataye|batao|baj|baje|ko|ka|ki|ke)\b",
    re.I,
)

LANG_INSTRUCTIONS = {
    "english": (
        "[Reply ONLY in English. Use Latin script only. "
        "No Devanagari. No Hindi words like ko, subah, baje, hai, mein.]"
    ),
    "hindi": (
        "[Reply ONLY in Hindi Devanagari. ONE sentence, max 20 words. "
        "Translate event details to Hindi. No English words except proper nouns like Adobe, PDF, GenAI.]"
    ),
    "hinglish": (
        "[Reply ONLY in Hinglish — Roman script mix of Hindi and English. "
        "No Devanagari characters. Example: 'Event 8 May ko subah 10 baje hai.']"
    ),
}

RETRY_INSTRUCTIONS = {
    "english": (
        "[CRITICAL: Your answer MUST be 100% English in Latin script. "
        "Zero Devanagari. Zero Hindi words. Example: '10:00 AM on 8 May 2026.']"
    ),
    "hindi": (
        "[CRITICAL: Your answer MUST be 100% Hindi in Devanagari script only. "
        "ONE sentence, max 20 words.]"
    ),
    "hinglish": (
        "[CRITICAL: Your answer MUST be Hinglish in Roman script only. "
        "Mix Hindi and English naturally. No Devanagari.]"
    ),
}


def resolve_response_language(stt_lang: str | None, transcript: str) -> str:
    if _DEVANAGARI.search(transcript):
        return "hindi"
    if _HINGLISH_MARKERS.search(transcript):
        return "hinglish"
    if stt_lang == "hi-IN":
        return "hindi"
    return "english"


def tts_code_for_language(lang: str) -> str:
    return "en-IN" if lang == "english" else "hi-IN"


def output_matches_language(text: str, lang: str) -> bool:
    has_dev = bool(_DEVANAGARI.search(text))
    has_roman_hindi = bool(_HINGLISH_MARKERS.search(text))
    if lang == "english":
        return not has_dev and not has_roman_hindi
    if lang == "hindi":
        return has_dev and not has_roman_hindi
    if lang == "hinglish":
        return not has_dev
    return True


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


async def _complete_once(
    messages: list,
    cancel_event: asyncio.Event,
    event_callback,
    stream_tokens: bool,
) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 120,
        "temperature": 0.2,
        "stream": stream_tokens,
        "reasoning_effort": None,
    }

    if not stream_tokens:
        response = await _http_client.post(SARVAM_CHAT_URL, json=payload)
        if response.status_code != 200:
            logger.error(f"[LLM] HTTP {response.status_code}: {response.text}")
            return ""
        data = response.json()
        raw = data["choices"][0]["message"].get("content", "") or ""
        return _strip_think(raw)

    full_text = ""
    emitted = ""

    async with _http_client.stream("POST", SARVAM_CHAT_URL, json=payload) as response:
        if response.status_code != 200:
            body = await response.aread()
            logger.error(f"[LLM] HTTP {response.status_code}: {body.decode()}")
            return ""

        async for raw_line in response.aiter_lines():
            if cancel_event.is_set():
                return ""

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
                in_think = full_text.count("<think>") > full_text.count("</think>")
                if in_think:
                    continue

                cleaned = _strip_think(full_text)
                new_part = cleaned[len(emitted):]
                if new_part:
                    await event_callback("llm_token", new_part)
                    emitted = cleaned
            except Exception as e:
                logger.debug(f"[LLM PARSE ERROR] {e}")

    return _strip_think(full_text)


async def stream_llm_response(
    transcript: str,
    conversation_history: list,
    event_callback,
    cancel_event: asyncio.Event = None,
    stt_lang: str | None = None,
):
    if cancel_event is None:
        cancel_event = asyncio.Event()

    response_language = resolve_response_language(stt_lang, transcript)
    logger.info(f"[LLM] response_language={response_language} stt_lang={stt_lang!r}")

    instruction = LANG_INSTRUCTIONS[response_language]
    user_content = f"{instruction}\n{transcript}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_content},
    ]

    await event_callback("llm_start", "")

    try:
        final_text = await _complete_once(messages, cancel_event, event_callback, stream_tokens=True)
    except asyncio.CancelledError:
        logger.info("[LLM] CancelledError — interrupted")
        return
    except Exception as e:
        logger.error(f"[LLM] {e}")
        await event_callback("llm_error", str(e))
        return

    if cancel_event.is_set():
        return

    if final_text and not output_matches_language(final_text, response_language):
        logger.warning(
            f"[LLM] Language mismatch (wanted {response_language}): {final_text[:80]!r} — retrying"
        )
        retry_content = f"{RETRY_INSTRUCTIONS[response_language]}\n{transcript}"
        retry_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": retry_content},
        ]
        try:
            retry_text = await _complete_once(retry_messages, cancel_event, event_callback, stream_tokens=False)
            if retry_text and output_matches_language(retry_text, response_language):
                final_text = retry_text
            else:
                logger.warning(f"[LLM] Retry still mismatched: {retry_text[:80]!r}")
        except Exception as e:
            logger.error(f"[LLM] Retry failed: {e}")

    if not final_text:
        logger.warning("[LLM] No answer — using fallback")
        fallbacks = {"english": FALLBACK_EN, "hindi": FALLBACK_HI, "hinglish": FALLBACK_HINGLISH}
        final_text = fallbacks.get(response_language, FALLBACK_EN)

    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": final_text})
    conversation_history[:] = conversation_history[-20:]

    logger.info(f"AI: {final_text}")
    await event_callback("llm_end", final_text, response_language)


async def close():
    await _http_client.aclose()
