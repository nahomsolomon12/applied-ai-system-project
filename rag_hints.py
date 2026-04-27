"""Hybrid RAG hint selection for the number guessing game.

The module always retrieves from markdown docs first, then optionally uses an
LLM to render a one-line dynamic hint grounded in retrieved context.
"""

from __future__ import annotations

import logging
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_HINT = (
    "Use the midpoint of the remaining range to narrow the search fastest."
)

CORPUS_DIR = Path(__file__).resolve().parent / "docs" / "probability"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
MAX_HINT_WORDS = 24


@dataclass(frozen=True)
class HintDocument:
    path: Path
    title: str
    content: str
    hint_line: str
    tokens: frozenset[str]


def build_hint_state(
    *,
    difficulty: str,
    low: int,
    high: int,
    attempt_limit: int,
    attempts_used: int,
    outcome: str,
    history: list,
):
    """Create a normalized state payload for hint selection."""

    range_size = max(high - low, 0)
    attempts_remaining = max(attempt_limit - attempts_used, 0)
    return {
        "difficulty": difficulty,
        "low": low,
        "high": high,
        "range_size": range_size,
        "attempt_limit": attempt_limit,
        "attempts_used": attempts_used,
        "attempts_remaining": attempts_remaining,
        "outcome": outcome,
        "history_size": len(history),
    }


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _title_from_path(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()


def _extract_hint_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("hint:"):
            return stripped.split(":", 1)[1].strip()

    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped

    return DEFAULT_HINT


@lru_cache(maxsize=1)
def load_hint_corpus(corpus_dir: str | None = None) -> tuple[HintDocument, ...]:
    """Load the markdown hint corpus from disk."""

    corpus_path = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    if not corpus_path.exists():
        logger.warning("Hint corpus directory missing: %s", corpus_path)
        return ()

    documents: list[HintDocument] = []
    for file_path in sorted(corpus_path.glob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Failed to read hint document: %s", file_path)
            continue

        documents.append(
            HintDocument(
                path=file_path,
                title=_title_from_path(file_path),
                content=content,
                hint_line=_extract_hint_line(content),
                tokens=_tokenize(f"{file_path.stem} {content}"),
            )
        )

    logger.info("Loaded %d hint documents from %s", len(documents), corpus_path)
    return tuple(documents)


def _score_document(document: HintDocument, state: dict) -> int:
    score = 0
    tokens = document.tokens

    if state["range_size"] >= 30:
        score += 10 if tokens & {"binary", "search", "midpoint", "halve", "split"} else 0
    if state["range_size"] <= 10:
        score += 10 if tokens & {"narrow", "precision", "final", "tight"} else 0

    if state["attempts_remaining"] <= 3:
        score += 8 if tokens & {"expected", "value", "efficient", "fewest", "optimal"} else 0

    if state["difficulty"].lower() == "hard":
        score += 5 if tokens & {"expected", "value", "efficient"} else 0

    if state["outcome"] == "Too High":
        score += 12 if tokens & {"lower", "down", "reduce", "narrow"} else 0
    elif state["outcome"] == "Too Low":
        score += 12 if tokens & {"upper", "higher", "raise", "narrow"} else 0

    score += len(tokens & _tokenize(state["difficulty"] + " range strategy probability guess"))
    return score


def _format_hint(document: HintDocument, state: dict) -> str:
    if document.path.stem == "range-narrowing":
        if state["outcome"] == "Too High":
            return "Use the lower half of the remaining range next to cut the search space fastest."
        if state["outcome"] == "Too Low":
            return "Use the upper half of the remaining range next to cut the search space fastest."

    if document.path.stem == "binary-search":
        return "A binary-search split is the fastest way to eliminate the most numbers right now."

    if document.path.stem == "expected-value":
        return "Choose the guess that removes the most remaining possibilities, not the closest-looking number."

    if document.path.stem == "difficulty-tuning":
        if state["outcome"] == "Too High":
            return "With fewer attempts left, still favor the lower half because it removes the most possibilities per guess."
        if state["outcome"] == "Too Low":
            return "With fewer attempts left, still favor the upper half because it removes the most possibilities per guess."
        return "With fewer attempts left, prioritize the split that gives you the most information per guess."

    if document.path.stem == "basics":
        return "Start with the midpoint of the current range so each guess removes the most possibilities."

    return document.hint_line or DEFAULT_HINT


def _get_top_documents(documents: tuple[HintDocument, ...], state: dict, limit: int = 3) -> tuple[HintDocument, ...]:
    ranked_documents = sorted(
        documents,
        key=lambda document: (_score_document(document, state), document.path.name),
        reverse=True,
    )
    return tuple(ranked_documents[:limit])


def _llm_enabled() -> bool:
    return os.getenv("RAG_HINT_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _call_chat_completions_api(
    *,
    api_key: str,
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    timeout_seconds: int = 15,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 80,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))

    choices = body.get("choices", [])
    if not choices:
        raise ValueError("LLM response did not include choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise ValueError("LLM response did not include message content.")

    return content


def _build_llm_messages(state: dict, top_documents: tuple[HintDocument, ...]) -> list[dict[str, str]]:
    context_lines = []
    for index, document in enumerate(top_documents, start=1):
        context_lines.append(
            f"{index}. {document.title}: {document.hint_line}"
        )

    user_prompt = (
        "Game state:\n"
        f"- Difficulty: {state['difficulty']}\n"
        f"- Range: {state['low']} to {state['high']}\n"
        f"- Range size: {state['range_size']}\n"
        f"- Attempts used: {state['attempts_used']}\n"
        f"- Attempts remaining: {state['attempts_remaining']}\n"
        f"- Last outcome: {state['outcome']}\n"
        "\n"
        "Retrieved strategy context:\n"
        + "\n".join(context_lines)
        + "\n\n"
        "Return exactly one line."
    )

    return [
        {
            "role": "system",
            "content": (
                "You generate one short strategy hint for a number guessing game. "
                "Ground the hint in provided context. Never reveal a secret number "
                "or claim to know it. Keep it under 24 words and actionable."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _sanitize_dynamic_hint(candidate_hint: str, state: dict) -> str | None:
    if not candidate_hint:
        return None

    one_line = " ".join(candidate_hint.strip().split())
    one_line = one_line.strip('"\'')

    lowered = one_line.lower()
    blocked_phrases = {
        "the answer is",
        "secret number is",
        "exact number is",
    }
    if any(phrase in lowered for phrase in blocked_phrases):
        return None

    words = one_line.split()
    if len(words) > MAX_HINT_WORDS:
        one_line = " ".join(words[:MAX_HINT_WORDS]).rstrip(".,;:") + "."

    if state["outcome"] == "Too High" and not re.search(r"\blower\b|\bdown\b", lowered):
        one_line = "Focus on the lower half next to remove the most possibilities."
    elif state["outcome"] == "Too Low" and not re.search(r"\bupper\b|\bhigher\b|\braise\b", lowered):
        one_line = "Focus on the upper half next to remove the most possibilities."

    return one_line


def _generate_dynamic_hint(state: dict, top_documents: tuple[HintDocument, ...]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("RAG_HINT_LLM_ENABLED is true but OPENAI_API_KEY is missing.")
        return None

    model = os.getenv("RAG_HINT_LLM_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL
    base_url = os.getenv("RAG_HINT_LLM_BASE_URL", OPENAI_BASE_URL).strip() or OPENAI_BASE_URL
    messages = _build_llm_messages(state, top_documents)

    try:
        raw_hint = _call_chat_completions_api(
            api_key=api_key,
            model=model,
            base_url=base_url,
            messages=messages,
        )
        safe_hint = _sanitize_dynamic_hint(raw_hint, state)
        if safe_hint:
            logger.info("Generated dynamic hint from LLM model=%s", model)
        else:
            logger.warning("Dynamic hint rejected by guardrails.")
        return safe_hint
    except (urllib.error.URLError, TimeoutError, ValueError):
        logger.exception("Dynamic hint generation failed; fallback will be used.")
        return None
    except Exception:
        logger.exception("Unexpected error during dynamic hint generation; fallback will be used.")
        return None


def generate_strategic_hint(state: dict, corpus: Iterable[HintDocument] | None = None) -> str:
    """Return one concise hint grounded in the markdown corpus.

    When enabled via environment variables, this uses retrieval + LLM
    generation with guardrails. Otherwise it falls back to deterministic
    retrieval formatting.
    """

    try:
        documents = tuple(corpus) if corpus is not None else load_hint_corpus()
        if not documents:
            logger.warning("No hint documents available; falling back to default hint.")
            return DEFAULT_HINT

        top_documents = _get_top_documents(documents, state, limit=3)
        chosen_document = top_documents[0]
        logger.info(
            "Selected hint document %s for state=%s",
            chosen_document.path.name,
            state,
        )

        if _llm_enabled():
            dynamic_hint = _generate_dynamic_hint(state, top_documents)
            if dynamic_hint:
                return dynamic_hint

        return _format_hint(chosen_document, state)
    except Exception:
        logger.exception("Hint generation failed; using default fallback hint.")
        return DEFAULT_HINT
