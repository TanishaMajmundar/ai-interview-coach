import pandas as pd
import streamlit as st

from tools.pdf_utils import extract_text_from_pdf
from tools.profile_parser import parse_profile
from tools.session_store import list_sessions, load_session
from agents.interviewer import Interviewer
from agents.evaluator import Evaluator
from app.graph import build_graph
from app import guardrails as gr

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")
MODES = {"HR": "hr", "Technical": "technical", "Both": "both"}
for k, v in {"g": None, "profile": None, "flags": [], "flash": None}.items():
    st.session_state.setdefault(k, v)


@st.cache_resource
def get_graph():
    return build_graph()


def run(**updates):
    """One workflow step: pass the state into the LangGraph app and keep the result."""
    g = st.session_state.g
    g.update(updates)
    with st.spinner("Working on it..."):
        st.session_state.g = get_graph().invoke(g)
    return st.session_state.g


# ---------- display helpers ----------
def show_profile(p):
    st.subheader("Your profile")
    st.write(f"**Target role:** {p.target_role or 'not stated'} at {p.company or 'not stated'}")
    st.write("**Skills:** " + ", ".join(p.skills))
    with st.expander("Experience"):
        for x in p.experience:
            st.markdown(f"- {x}")
    st.write("**Gaps compared with the job description**")
    for x in p.gaps_vs_jd:
        st.markdown(f"- {x}")


def show_evaluation(ev):
    st.markdown(f"**Score: {ev.overall}/5**")
    for col, (lab, val) in zip(st.columns(4), [("Relevance", ev.relevance), ("Depth", ev.depth),
                                               ("Clarity", ev.clarity), ("Structure", ev.structure)]):
        col.metric(lab, f"{val}/5")
    if ev.strengths:
        st.markdown("**Strengths**")
        for x in ev.strengths:
            st.markdown(f"- {x}")
    st.markdown("**To improve**")
    for x in ev.weaknesses:
        st.markdown(f"- {x}")
    st.markdown("**Suggested rewrite** (edit it so it is true to your own experience)")
    st.info(ev.improved_answer)


def show_resources(resources):
    if resources:
        st.markdown("**Suggested resources**")
        for r in resources:
            st.markdown(f"- [{r['title']}]({r['link']}): {r['note']}")


def show_report(report, turns):
    if not turns:
        st.info("No questions were answered.")
        return
    st.metric("Average score", f"{report['average']} / 5")
    st.bar_chart(pd.Series(report["dimensions"]))
    c1, c2 = st.columns(2)
    c1.subheader("Top strengths")
    for x in dict.fromkeys(report["strengths"]):
        c1.markdown(f"- {x}")
    c2.subheader("Main areas to improve")
    for x in dict.fromkeys(report["improve"]):
        c2.markdown(f"- {x}")
    st.subheader("Recommended resources")
    if report["resources"]:
        show_resources(report["resources"])
    else:
        st.write("No resources were needed: all answers scored well.")
    st.subheader("Question by question")
    for i, t in enumerate(turns, 1):
        with st.expander(f"Q{i} ({t['mode']}): {t['question']}  [{t['evaluation'].overall}/5]"):
            st.markdown(f"**Your answer:** {t['answer']}")
            show_evaluation(t["evaluation"])
            show_resources(t["resources"])


# ---------- pages ----------
def page_setup():
    st.title("AI Interview Coach")
    st.caption("Upload your resume, paste the job description, and practise a mock interview.")
    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Job description", height=220, placeholder="Paste the full job description here...")
    if st.button("Build my profile", type="primary"):
        err = gr.validate_resume_file(resume_file) or gr.validate_text(jd_text, "job description", gr.MAX_JD_CHARS)
        if err:
            st.warning(err)
        else:
            try:
                with st.spinner("Reading your resume and the job description..."):
                    resume_text = extract_text_from_pdf(resume_file)
                    err = gr.validate_text(resume_text, "resume text", gr.MAX_RESUME_CHARS)
                    if err:
                        raise ValueError(err)
                    resume_text, f1 = gr.scan_injection(resume_text)
                    jd_clean, f2 = gr.scan_injection(jd_text)
                    st.session_state.flags = f1 + f2
                    st.session_state.profile = parse_profile(resume_text, jd_clean)
            except Exception as e:
                st.error(f"Could not build the profile: {e}")
    if st.session_state.flags:
        st.warning(f"{len(st.session_state.flags)} instruction-like phrase(s) were found in your documents "
                   "and removed before analysis.")
    profile = st.session_state.profile
    if profile is None:
        return
    show_profile(profile)
    st.subheader("Interview settings")
    itype = st.selectbox("Interview type", list(MODES))
    total = st.slider("Number of questions", 3, 6, 5)
    difficulty = st.slider("Starting difficulty", 1, 5, 3)
    if st.button("Start interview", type="primary"):
        st.session_state.g = {
            "profile": profile, "interviewer": Interviewer(profile, mode=MODES[itype], difficulty=difficulty,
                                                           total_questions=total),
            "evaluator": Evaluator(profile), "current_q": None, "answer": "", "turns": [], "total": total,
            "status": "new", "message": "", "end_early": False, "interview_type": itype}
        run()
        st.rerun()


def page_interview():
    g = st.session_state.g
    done, total = len(g["turns"]), g["total"]
    st.title("Mock interview")
    st.progress(done / total)
    st.caption(f"Question {min(done + 1, total)} of {total}")
    with st.sidebar:
        st.write(f"Answered: {done}/{total}")
        st.write(f"Current difficulty: {g['interviewer'].difficulty}/5")
        if st.button("End interview now"):
            run(end_early=True)
            st.rerun()
    if st.session_state.flash:
        st.warning(st.session_state.flash)
        st.session_state.flash = None
    for t in g["turns"]:
        with st.chat_message("assistant"):
            st.write(t["question"])
            st.caption(f"{t['mode'].upper()} · difficulty {t['difficulty']}/5")
        with st.chat_message("user"):
            st.write(t["answer"])
        with st.chat_message("assistant"):
            show_evaluation(t["evaluation"])
            show_resources(t["resources"])
            if t["new_d"] != t["old_d"]:
                st.caption(f"Difficulty {'raised' if t['new_d'] > t['old_d'] else 'lowered'}: "
                           f"{t['old_d']} → {t['new_d']}")
    q = g.get("current_q")
    if q is None:
        st.error(g.get("message") or "The next question could not be generated.")
        if st.button("Try again"):
            run()
            st.rerun()
        return
    with st.chat_message("assistant"):
        st.write(q.text)
        st.caption(f"{q.mode.upper()} · difficulty {q.difficulty}/5")
    ans = st.chat_input("Type your answer here...")
    if ans is not None:
        out = run(answer=ans)
        if out["status"] in ("rejected", "error"):
            st.session_state.flash = out["message"]
        st.rerun()


def page_done():
    g = st.session_state.g
    st.title("Interview complete")
    show_report(g["report"], g["turns"])
    if st.button("Start a new interview"):
        st.session_state.g = None
        st.rerun()


def page_history():
    st.title("History")
    rows = list_sessions()
    if not rows:
        st.info("No saved sessions yet. Finish an interview to see it here.")
        return
    st.dataframe([{"Date": r["created_at"], "Type": r["interview_type"], "Role": r["target_role"],
                   "Company": r["company"], "Questions": r["n_questions"], "Average": r["average"],
                   "ID": r["id"]} for r in rows])
    label = {r["id"]: f"{r['created_at']} · {r['interview_type']} · {r['average']}/5" for r in rows}
    sid = st.selectbox("Reopen a session", list(label), format_func=lambda x: label[x])
    _, turns, report = load_session(sid)
    show_report(report, turns)


# ---------- router ----------
if st.sidebar.radio("Navigate", ["Interview coach", "History"]) == "History":
    page_history()
elif st.session_state.g is None:
    page_setup()
elif st.session_state.g["status"] == "done":
    page_done()
else:
    page_interview()