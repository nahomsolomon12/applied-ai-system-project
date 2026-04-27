# Number Guessing Game with RAG Hints

## Overview

This project is a Streamlit-based number guessing game designed to help players find a secret number in as few guesses as possible.

The twist is the hint system: instead of only returning a basic "higher" or "lower" response, the game uses a retrieval-augmented generation flow backed by premade markdown documentation on statistical reasoning, probability, and search strategy. The goal is to generate a single concise hint that helps the player make the next best guess and converge on the answer with minimal tries.

## How It Works

1. The game selects a secret number inside a defined range.
2. The player submits guesses through the Streamlit interface.
3. The app keeps the game state stable across interactions.
4. A RAG layer reads curated markdown files containing probability concepts and optimal guessing strategies.
5. The retrieved context is used to produce one short, useful hint that nudges the player toward the most efficient next guess.

## RAG Hint Design

The hint engine is intended to use prewritten markdown files such as:

- Statistical elimination strategies
- Midpoint and range narrowing methods
- Probability-based decision making
- Expected-value thinking for fewer guesses
- Adaptive advice for early, middle, and late game states

Each response should stay focused and actionable. The output should be a single line hint that explains what strategy to use next without revealing the answer directly.

The current implementation is hybrid:

- Retrieval: the app reads premade markdown files from `docs/probability/` and ranks them against the current game state.
- Generation: if enabled, an LLM uses only the retrieved context plus game state to produce a one-line dynamic hint.
- Guardrails: hints are sanitized to one line, restricted from revealing answers, and forced to remain directionally consistent with the latest high/low outcome.
- Fallback: if LLM configuration is missing or the model call fails, the app automatically returns a deterministic strategy hint.

Example hint styles:

- "Use the midpoint of your current range to cut the search space in half."
- "Your best move is to test the upper half of the remaining interval next."
- "A binary-search approach now gives you the fastest path to the target."

## Setup

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the app:
   `python -m streamlit run app.py`

## Optional Dynamic Hint Mode (Hybrid RAG + LLM)

By default, hints run in deterministic retrieval mode. To enable dynamic LLM hints, set:

- `RAG_HINT_LLM_ENABLED=true`
- `OPENAI_API_KEY=<your_api_key>`

Optional overrides:

- `RAG_HINT_LLM_MODEL` (default: `gpt-4o-mini`)
- `RAG_HINT_LLM_BASE_URL` (default: `https://api.openai.com/v1`)

Example (PowerShell):

`$env:RAG_HINT_LLM_ENABLED="true"`
`$env:OPENAI_API_KEY="your_key_here"`
`python -m streamlit run app.py`

## Hint Corpus

The RAG hints are built from markdown files in `docs/probability/`.

- `basics.md`
- `binary-search.md`
- `expected-value.md`
- `range-narrowing.md`
- `difficulty-tuning.md`

These documents should stay focused on statistical reasoning and search strategy so the hint stays short, safe, and useful.

## Project Goal

The main objective is to make the game feel smarter than a standard number guessing app by combining gameplay with retrieval-backed guidance. The markdown corpus provides the reasoning, and the app turns that reasoning into a short hint that helps the player reach the secret number in the fewest possible guesses.

## Suggested Content Structure for Markdown Docs

If you expand the hint corpus, keep the files organized around strategy topics rather than raw answers.

- `docs/probability/basics.md`
- `docs/probability/binary-search.md`
- `docs/probability/expected-value.md`
- `docs/probability/range-narrowing.md`
- `docs/probability/difficulty-tuning.md`

## Notes

- The hint system should never expose the secret number directly.
- The hint should always reflect the current game state.
- The best hint is short, specific, and mathematically useful.

## Demo

Add a screenshot of the working game and example hint output here once the RAG flow is implemented.
