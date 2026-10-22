import copy
import streamlit as st

from tools.pdf_utils import extract_text_from_pdf
from tools.profile_parser import parse_profile
from agents.interviewer import Interviewer
from agents.evaluator import Evaluator

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")

# ---------- Session state ----------
# Streamlit re-runs this whole script on every click, so anything that must
# survive a click is stored in st.session_state.
DEFAULTS = {
    "stage": "setup",        # setup -> interview -> done
    "profile": None,         # CandidateProfile
    "interviewer": None,     # Interviewer agent (holds the conversation memory)
    "evaluator": None,       # Evaluator agent
    "current_q": None,       # Question waiting for an answer
    "messages": [],          # chat log shown on screen
    "evaluations": [],       # Evaluation objects, one per answered question
    "total_questions": 5,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)


def reset_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(value)


# ---------- Display helpers ----------
def show_profile(p):
    st.subheader("Your profile")
    if p.target_role or p.company:
        st.write(f"**Target role:** {p.target_role or 'not stated'} at {p.company or 'not stated'}")
    st.write("**Skills:** " + ", ".join(p.skills))
    with st.expander("Experience"):
        for item in p.experience:
            st.markdown(f"- {item}")
    st.write("**Gaps compared with the job description**")
    for gap in p.gaps_vs_jd:
        st.markdown(f"- {gap}")


def show_evaluation(ev):
    st.markdown(f"**Score: {ev.overall}/5**")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [("Relevance", ev.relevance), ("Depth", ev.depth),
                                          ("Clarity", ev.clarity), ("Structure", ev.structure)]):
        col.metric(label, f"{value}/5")
    if ev.strengths:
        st.markdown("**Strengths**")
        for s in ev.strengths:
            st.markdown(f"- {s}")
    st.markdown("**To improve**")
    for w in ev.weaknesses:
        st.markdown(f"- {w}")
    # Labelled as a suggestion because LLM rewrites can embellish (known limitation)
    st.markdown("**Suggested rewrite** (edit it so it is true to your own experience)")
    st.info(ev.improved_answer)


def render_messages():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("evaluation") is not None:
                show_evaluation(m["evaluation"])


# ---------- Actions ----------
def ask_next_question() -> bool:
    """Ask the Interviewer for the next question. Returns False if the LLM call failed."""
    try:
        with st.spinner("Preparing the next question..."):
            q = st.session_state.interviewer.next_question()
    except Exception as e:
        st.error(f"Could not generate a question: {e}")
        st.session_state.current_q = None
        return False
    st.session_state.current_q = q
    st.session_state.messages.append({"role": "assistant", "content": q.text})
    return True


def start_interview(profile, total, difficulty):
    s = st.session_state
    s.interviewer = Interviewer(profile, mode="hr", difficulty=difficulty, total_questions=total)
    s.evaluator = Evaluator(profile)
    s.total_questions = total
    s.messages = []
    s.evaluations = []
    s.current_q = None
    s.stage = "interview"
    ask_next_question()


def handle_answer(answer):
    s = st.session_state
    q = s.current_q
    with st.chat_message("user"):      # show the answer immediately, before the slow LLM call
        st.write(answer)
    try:
        with st.spinner("Evaluating your answer..."):
            evaluation = s.evaluator.evaluate(q, answer)
    except Exception as e:
        st.error(f"Evaluation failed: {e}. Please send your answer again.")
        return
    s.interviewer.record_answer(q, answer)          # short-term memory for follow-up questions
    s.evaluations.append(evaluation)
    s.messages.append({"role": "user", "content": answer})
    s.messages.append({"role": "assistant", "content": "Here is feedback on your answer:",
                       "evaluation": evaluation})
    s.current_q = None
    if len(s.evaluations) >= s.total_questions:
        s.stage = "done"
    else:
        ask_next_question()
    st.rerun()


# ---------- Pages ----------
def render_setup():
    st.title("🎤 AI Interview Coach")
    st.caption("Upload your resume, paste the job description, and practise a mock interview.")

    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Job description", height=220,
                           placeholder="Paste the full job description here...")

    if st.button("Build my profile", type="primary"):
        if resume_file is None or not jd_text.strip():
            st.warning("Please upload a resume PDF and paste a job description.")
        else:
            try:
                with st.spinner("Reading your resume and the job description..."):
                    resume_text = extract_text_from_pdf(resume_file)
                    st.session_state.profile = parse_profile(resume_text, jd_text)
            except Exception as e:
                st.error(f"Could not build the profile: {e}")

    profile = st.session_state.profile
    if profile is not None:
        show_profile(profile)
        st.subheader("Interview settings")
        st.selectbox("Interview type", ["HR"], help="Technical mode is added in a later step.")
        total = st.slider("Number of questions", 3, 6, 5)
        difficulty = st.slider("Starting difficulty", 1, 5, 3)
        if st.button("Start interview", type="primary"):
            start_interview(profile, total, difficulty)
            st.rerun()


def render_interview():
    s = st.session_state
    total = s.total_questions
    answered = len(s.evaluations)

    st.title("Mock interview")
    st.progress(answered / total)
    st.caption(f"Question {min(answered + 1, total)} of {total}")

    with st.sidebar:
        st.write(f"Answered: {answered}/{total}")
        if st.button("End interview now"):
            s.stage = "done"
            st.rerun()

    render_messages()

    if s.current_q is None:                 # the previous question call failed
        st.warning("The next question could not be generated.")
        if st.button("Try again") and ask_next_question():
            st.rerun()
        return

    answer = st.chat_input("Type your answer here...")
    if answer and answer.strip():
        handle_answer(answer.strip())


def render_done():
    s = st.session_state
    st.title("Interview complete")
    evs = s.evaluations
    if not evs:
        st.info("You ended the interview before answering any question.")
    else:
        avg = sum(e.overall for e in evs) / len(evs)
        st.metric("Average score", f"{avg:.2f} / 5")
        st.subheader("Question by question")
        for i, (turn, ev) in enumerate(zip(s.interviewer.history, evs), 1):
            with st.expander(f"Q{i}: {turn['question']}  ({ev.overall}/5)"):
                st.markdown(f"**Your answer:** {turn['answer']}")
                show_evaluation(ev)
    st.divider()
    if st.button("Start a new interview"):
        reset_all()
        st.rerun()


# ---------- Router ----------
if st.session_state.stage == "setup":
    render_setup()
elif st.session_state.stage == "interview":
    render_interview()
else:
    render_done()