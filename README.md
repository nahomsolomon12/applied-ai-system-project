# Number Guessing Game

This project is a Streamlit number guessing game built to help players reach the secret number in the fewest possible tries. Its core goal is to combine stable game logic (difficulty ranges, attempts, scoring, and history) with smarter hints than a simple higher or lower message. The current system uses a hybrid RAG design: it retrieves strategy guidance from premade markdown files and can optionally use an LLM to generate a one-line dynamic hint grounded in that retrieved context.

## Architecture Overview

The architecture is shown in the assets folder with a class-style diagram and a sequence diagram. The class diagram (`assets/classDiagrams.png`) shows the responsibilities split across Streamlit UI orchestration (`app.py`), game rules (`logic_utils.py`), state tracking (`st.session_state`), and the hint engine (`rag_hints.py`) with a markdown corpus under `docs/probability/`. The sequence diagram (`assets/sequenceDiagrams.png`) illustrates the runtime path: player input is parsed and scored first, then the hint engine retrieves the most relevant strategy docs, optionally calls an LLM for a single-line hint, applies guardrails, and returns a safe output to the UI.

## Setup Instructions

1. Open a terminal in the project root.
2. Install dependencies:
    `pip install -r requirements.txt`
3. Run tests (recommended before launching):
    `python -m pytest`
4. Start the app:
    `python -m streamlit run app.py`
5. In the browser UI, select difficulty, enter guesses, and enable `Show hint` to see strategic hints.

### Optional: Enable Dynamic LLM Hints

By default, the game uses deterministic retrieval-only hints. To enable hybrid retrieval + LLM generation:

1. Set environment variables:
    `RAG_HINT_LLM_ENABLED=true`
    `OPENAI_API_KEY=<your_api_key>`
2. Optional overrides:
    `RAG_HINT_LLM_MODEL=gpt-4o-mini`
    `RAG_HINT_LLM_BASE_URL=https://api.openai.com/v1`
3. Run the app again with Streamlit.

PowerShell example:

`$env:RAG_HINT_LLM_ENABLED="true"`
`$env:OPENAI_API_KEY="your_key_here"`
`python -m streamlit run app.py`

## Sample Interactions

### Example 1: Deterministic RAG hint after a high guess

- Input state:
   Difficulty: Normal, Range: 1-100, Guess: 70, Outcome: Too High
- AI output hint:
   `Use the lower half of the remaining range next to cut the search space fastest.`

### Example 2: Deterministic RAG hint after a low guess

- Input state:
   Difficulty: Normal, Range: 1-100, Guess: 30, Outcome: Too Low
- AI output hint:
   `Use the upper half of the remaining range next to cut the search space fastest.`

### Example 3: Hybrid mode dynamic hint (LLM enabled)

- Input state:
   Difficulty: Hard, Attempts remaining: 2, Outcome: Too High
- AI output hint (sample):
   `Favor the lower half and choose its midpoint to maximize information from this turn.`

## Design Decisions

- Why this design:
   The game keeps game rules deterministic and testable in `logic_utils.py`, while treating hinting as a separate retrieval/generation concern in `rag_hints.py`. This separation improves reliability and makes it easier to test core gameplay independently of AI behavior.
- Retrieval-first grounding:
   The hint pipeline always retrieves strategy text from curated markdown docs first, so generated hints stay anchored to your intended probability concepts.
- Guardrails and fallbacks:
   Dynamic outputs are constrained to one line, filtered for unsafe answer-reveal phrases, and checked for directional consistency with the latest outcome. If LLM config is missing or an API call fails, the system falls back to deterministic hints.
- Trade-offs:
   Deterministic hints are more predictable but less expressive; LLM hints are more flexible and natural but require stronger safety controls, network access, and key management.

## Testing Summary

- What worked:
   The game logic tests and RAG hint tests pass, including retrieval behavior, fallback behavior, and hybrid dynamic-hint guardrail behavior.
- What did not work initially:
   One hint-ranking path returned a generic strategy instead of explicitly reflecting high/low direction. This was fixed by enforcing directional consistency in hint formatting and guardrails.
- What we learned:
   Retrieval grounding plus strict post-processing is essential when adding LLM generation to gameplay advice. It preserves relevance and prevents unsafe or misleading hints while keeping the user experience responsive.

## Reflection

Initially, it was difficult to come up with an update to make for such a run-of-the-mill game. 

But I arrived at a practical solution to have a better set of hints utilizing RAG to give targetted hints. 

Initially, Copilot thought it best to use a deterministic set of hints for potential inputs

After some deliberation, I chose a dynamic approach instead and implemented an LLM powered RAG system that gives dynamic hints to arrive at the answer before the number of tries run out.

## Loom Video Here: 
[Loom](https://www.loom.com)