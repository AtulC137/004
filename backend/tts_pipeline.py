"""Sentence chunking for streaming LLM → TTS pipeline."""

import re

_MIN_SENTENCE_LEN = 10
_MAX_BUFFER_BEFORE_FLUSH = 120
_FIRST_SENTENCE = re.compile(r"^[^.!?]*[.!?]")


def drain_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete sentences from the start of buffer.
    Returns (complete_sentences, remainder).
    """
    sentences: list[str] = []
    rest = buffer

    while rest:
        match = _FIRST_SENTENCE.match(rest)
        if match:
            sentence = match.group(0).strip()
            rest = rest[match.end() :].lstrip()
            if len(sentence) >= _MIN_SENTENCE_LEN:
                sentences.append(sentence)
            elif sentence:
                rest = f"{sentence} {rest}".strip() if rest else sentence
                break
        else:
            if len(rest) >= _MAX_BUFFER_BEFORE_FLUSH:
                sentences.append(rest.strip())
                rest = ""
            break

    return sentences, rest
