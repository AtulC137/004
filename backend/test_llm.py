"""
test_llm.py — run this inside your backend container to diagnose LLM issues.

Usage:
    docker compose exec backend python test_llm.py

Tests every model + reasoning_effort combo and prints:
  - Time to first token (TTFT)
  - Full response text
  - Total time
  - Any errors
"""

import asyncio
import json
import os
import time

import httpx

SARVAM_API_KEY  = os.environ.get("SARVAM_API_KEY", "")
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

TEST_PROMPT = "Where is the Adobe event being held? One sentence only."

SYSTEM_PROMPT = (
    "You are a voice receptionist for an Adobe event. Answer immediately and briefly. "
    "ONE sentence only. Maximum 20 words."
)

# Every combo to test
TESTS = [
    {"model": "sarvam-m",   "reasoning_effort": None,     "label": "sarvam-m   | no-think (reasoning_effort=None)"},
    {"model": "sarvam-m",   "reasoning_effort": "low",    "label": "sarvam-m   | think low"},
    {"model": "sarvam-30b", "reasoning_effort": None,     "label": "sarvam-30b | no-think (reasoning_effort=None)"},
    {"model": "sarvam-30b", "reasoning_effort": "low",    "label": "sarvam-30b | think low"},
]

_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=45.0, write=10.0, pool=5.0),
    headers={
        "Authorization":        f"Bearer {SARVAM_API_KEY}",
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json",
    }
)


async def test_one(label: str, model: str, reasoning_effort):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": TEST_PROMPT},
        ],
        "max_tokens":  256,
        "temperature": 0.2,
        "stream":      True,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    full_text   = ""
    ttft        = None
    t_start     = time.monotonic()
    in_think    = False

    try:
        async with _client.stream("POST", SARVAM_CHAT_URL, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(f"  ERROR HTTP {response.status_code}: {body.decode()}")
                return

            async for raw_line in response.aiter_lines():
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
                    now = time.monotonic() - t_start

                    # Track think block
                    currently_in_think = full_text.count("<think>") > full_text.count("</think>")

                    if not currently_in_think and in_think:
                        print(f"  [think block ended at {now:.2f}s]")
                    in_think = currently_in_think

                    if not in_think and ttft is None:
                        ttft = now
                        print(f"  TTFT (first visible token): {ttft:.2f}s")

                except Exception as e:
                    print(f"  PARSE ERROR: {e}")
                    continue

    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return

    total = time.monotonic() - t_start

    # Strip think blocks for display
    import re
    clean = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
    clean = re.sub(r"<think>.*",          "", clean,     flags=re.DOTALL).strip()

    think_len = len(full_text) - len(clean)

    print(f"  RESPONSE : {clean!r}")
    print(f"  TTFT     : {ttft:.2f}s" if ttft else "  TTFT     : no visible tokens!")
    print(f"  TOTAL    : {total:.2f}s")
    print(f"  THINK    : {think_len} chars in <think> blocks")


async def check_connectivity():
    """Step 1: Can we reach api.sarvam.ai at all?"""
    print("\n[STEP 1] DNS + connectivity check...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get("https://api.sarvam.ai")
            print(f"  OK — HTTP {r.status_code}")
    except httpx.ConnectError as e:
        print(f"  FAIL — ConnectError: {e}")
        print("  → DNS cannot resolve api.sarvam.ai from inside Docker.")
        print("  → Check docker-compose network / DNS settings.")
        return False
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        return False
    return True


async def check_auth():
    """Step 2: Is the API key valid? Hit a simple non-streaming endpoint."""
    print("\n[STEP 2] Auth check (non-streaming)...")
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                SARVAM_CHAT_URL,
                headers={
                    "Authorization":        f"Bearer {SARVAM_API_KEY}",
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type":         "application/json",
                },
                json={
                    "model":       "sarvam-m",
                    "messages":    [{"role": "user", "content": "hi"}],
                    "max_tokens":  10,
                    "temperature": 0.2,
                    "stream":      False,   # non-streaming — returns immediately
                    "reasoning_effort": None,
                }
            )
            print(f"  HTTP {r.status_code}")
            body = r.json()
            if r.status_code == 200:
                reply = body["choices"][0]["message"]["content"]
                print(f"  OK — reply: {reply!r}")
                return True
            else:
                print(f"  FAIL — body: {json.dumps(body, indent=2)}")
                return False
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        return False


async def main():
    if not SARVAM_API_KEY:
        print("ERROR: SARVAM_API_KEY env var not set!")
        return

    print(f"SARVAM_API_KEY: {SARVAM_API_KEY[:8]}...{SARVAM_API_KEY[-4:]}")
    print(f"Prompt: {TEST_PROMPT!r}")

    # Run connectivity and auth checks first
    ok = await check_connectivity()
    if not ok:
        return

    ok = await check_auth()
    if not ok:
        print("\nAuth failed — fix API key / account before running streaming tests.")
        return

    print("\n[STEP 3] Streaming tests...")
    for t in TESTS:
        await test_one(t["label"], t["model"], t["reasoning_effort"])

    await _client.aclose()
    print("\n\nDONE. Look at TTFT values — use the model+setting with lowest TTFT.")


asyncio.run(main())