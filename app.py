import random
import streamlit as st

#FIX: Refactored core game logic into logic_utils.py using Copilot Agent mode
#      (keeps app.py focused on Streamlit UI code)
from logic_utils import (
    get_range_for_difficulty,
    parse_guess,
    check_guess,
    reset_game_state,
    update_score,
)
from rag_hints import build_hint_state, generate_strategic_hint, load_hint_corpus

st.set_page_config(page_title="Glitchy Guesser", page_icon="🎮")

st.title("🎮 Game Glitch Investigator")
st.caption("An AI-generated guessing game. Something is off.")

st.sidebar.header("Settings")

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["Easy", "Normal", "Hard"],
    index=1,
)

attempt_limit_map = {
    "Easy": 6,
    "Normal": 8,
    "Hard": 5,
}
attempt_limit = attempt_limit_map[difficulty]

low, high = get_range_for_difficulty(difficulty)

st.sidebar.caption(f"Range: {low} to {high}")
st.sidebar.caption(f"Attempts allowed: {attempt_limit}")


@st.cache_resource(show_spinner=False)
def get_hint_corpus():
    return load_hint_corpus()


hint_corpus = get_hint_corpus()

if "secret" not in st.session_state:
    st.session_state.secret = random.randint(low, high)

if "attempts" not in st.session_state:
    st.session_state.attempts = 1

if "score" not in st.session_state:
    st.session_state.score = 0

if "status" not in st.session_state:
    st.session_state.status = "playing"

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Make a guess")

st.info(
    f"Guess a number between 1 and 100. "
    f"Attempts left: {attempt_limit - st.session_state.attempts}"
)

with st.expander("Developer Debug Info"):
    st.write("Secret:", st.session_state.secret)
    st.write("Attempts:", st.session_state.attempts)
    st.write("Score:", st.session_state.score)
    st.write("Difficulty:", difficulty)
    st.write("History:", st.session_state.history)

raw_guess = st.text_input(
    "Enter your guess:",
    key=f"guess_input_{difficulty}"
)

col1, col2, col3 = st.columns(3)
with col1:
    submit = st.button("Submit Guess 🚀")
with col2:
    new_game = st.button("New Game 🔁")
with col3:
    show_hint = st.checkbox("Show hint", value=True)

if new_game:
    # Reset game state so the UI returns to "playing" and the game flow restarts.
    # We use a helper so this behavior can be unit-tested.
    reset_game_state(st.session_state, low, high)

    st.success("New game started.")
    st.rerun()

if st.session_state.status != "playing":
    if st.session_state.status == "won":
        st.success("You already won. Start a new game to play again.")
    else:
        st.error("Game over. Start a new game to try again.")
    st.stop()

if submit:
    st.session_state.attempts += 1

    ok, guess_int, err = parse_guess(raw_guess)

    if not ok:
        st.session_state.history.append(raw_guess)
        st.error(err)
    else:
        st.session_state.history.append(guess_int)

        # Determine the game outcome (win / too high / too low).
        # The helper returns an outcome token; user-facing hint text is
        # mapped separately so the logic is easy to test and reason about.
        outcome = check_guess(guess_int, st.session_state.secret)

        hint_message = {
            "Win": "🎉 Correct!",
            "Too High": "📉 Go LOWER!",  # If guess is too high, tell player to lower it.
            "Too Low": "📈 Go HIGHER!",  # If guess is too low, tell player to raise it.
        }.get(outcome, "")

        if show_hint and hint_message:
            st.warning(hint_message)

        if show_hint and outcome != "Win":
            hint_state = build_hint_state(
                difficulty=difficulty,
                low=low,
                high=high,
                attempt_limit=attempt_limit,
                attempts_used=st.session_state.attempts,
                outcome=outcome,
                history=st.session_state.history,
            )
            strategic_hint = generate_strategic_hint(hint_state, hint_corpus)
            st.info(strategic_hint)

        st.session_state.score = update_score(
            current_score=st.session_state.score,
            outcome=outcome,
            attempt_number=st.session_state.attempts,
        )

        if outcome == "Win":
            st.balloons()
            st.session_state.status = "won"
            st.success(
                f"You won! The secret was {st.session_state.secret}. "
                f"Final score: {st.session_state.score}"
            )
        else:
            if st.session_state.attempts >= attempt_limit:
                st.session_state.status = "lost"
                st.error(
                    f"Out of attempts! "
                    f"The secret was {st.session_state.secret}. "
                    f"Score: {st.session_state.score}"
                )

st.divider()
st.caption("Built by an AI that claims this code is production-ready.")
