from rag_hints import build_hint_state, generate_strategic_hint, load_hint_corpus


def test_hint_corpus_loads_markdown_documents():
    corpus = load_hint_corpus()

    assert corpus
    assert any(document.path.stem == "binary-search" for document in corpus)


def test_generate_hint_prefers_lower_half_after_high_guess():
    corpus = load_hint_corpus()
    state = build_hint_state(
        difficulty="Normal",
        low=1,
        high=100,
        attempt_limit=8,
        attempts_used=2,
        outcome="Too High",
        history=[60],
    )

    hint = generate_strategic_hint(state, corpus)

    assert "lower half" in hint.lower()


def test_generate_hint_prefers_upper_half_after_low_guess():
    corpus = load_hint_corpus()
    state = build_hint_state(
        difficulty="Normal",
        low=1,
        high=100,
        attempt_limit=8,
        attempts_used=2,
        outcome="Too Low",
        history=[40],
    )

    hint = generate_strategic_hint(state, corpus)

    assert "upper half" in hint.lower()


def test_generate_hint_falls_back_when_corpus_is_missing():
    state = build_hint_state(
        difficulty="Easy",
        low=1,
        high=20,
        attempt_limit=6,
        attempts_used=1,
        outcome="Too Low",
        history=[8],
    )

    hint = generate_strategic_hint(state, corpus=[])

    assert "midpoint" in hint.lower()


def test_generate_hint_uses_dynamic_llm_hint_when_enabled(monkeypatch):
    corpus = load_hint_corpus()
    state = build_hint_state(
        difficulty="Normal",
        low=1,
        high=100,
        attempt_limit=8,
        attempts_used=3,
        outcome="Too High",
        history=[75, 60],
    )

    monkeypatch.setenv("RAG_HINT_LLM_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "rag_hints._call_chat_completions_api",
        lambda **kwargs: "Shift to the lower half and pick its midpoint to maximize information gain.",
    )

    hint = generate_strategic_hint(state, corpus)

    assert "lower half" in hint.lower()
    assert "information" in hint.lower()


def test_generate_hint_rejects_unsafe_dynamic_text_and_falls_back(monkeypatch):
    corpus = load_hint_corpus()
    state = build_hint_state(
        difficulty="Normal",
        low=1,
        high=100,
        attempt_limit=8,
        attempts_used=2,
        outcome="Too Low",
        history=[25],
    )

    monkeypatch.setenv("RAG_HINT_LLM_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "rag_hints._call_chat_completions_api",
        lambda **kwargs: "The answer is 42.",
    )

    hint = generate_strategic_hint(state, corpus)

    assert "answer is" not in hint.lower()
    assert "upper half" in hint.lower()
