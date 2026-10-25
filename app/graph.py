from typing import Any, TypedDict
from langgraph.graph import StateGraph, START, END
from app.guardrails import check_answer, is_off_topic
from tools.retriever import retrieve_resources
from tools.session_store import save_session as store_session

RETRIEVE_BELOW = 3.5   # Evaluator's overall score below this triggers the Retriever


class State(TypedDict, total=False):
    profile: Any
    interviewer: Any
    evaluator: Any
    current_q: Any
    last_eval: Any
    answer: str
    turns: list
    total: int
    status: str          # new | asking | ok | rejected | error | done
    message: str
    end_early: bool
    report: dict
    session_id: str
    interview_type: str


# ---------- nodes ----------
def ask_question(s):
    try:
        return {"current_q": s["interviewer"].next_question(), "status": "asking", "message": ""}
    except Exception as e:
        return {"status": "error", "message": f"Could not generate the next question: {e}"}


def guard(s):
    ok, msg, clean = check_answer(s.get("answer", ""))
    return {"answer": clean, "status": "ok"} if ok else {"status": "rejected", "message": msg}


def evaluate(s):
    try:
        ev = s["evaluator"].evaluate(s["current_q"], s["answer"])
    except Exception as e:
        return {"status": "error", "message": f"Evaluation failed: {e}. Please send your answer again."}
    if is_off_topic(ev, s["answer"]):
        return {"status": "rejected",
                "message": "That does not look like an answer to the question. Please answer the question asked."}
    return {"last_eval": ev, "status": "ok"}


def record(s):
    iv, q, ev, ans = s["interviewer"], s["current_q"], s["last_eval"], s["answer"]
    iv.record_answer(q, ans)                       # short-term memory
    old, new = iv.difficulty, iv.adjust_difficulty(ev.overall)
    turn = {"question": q.text, "mode": q.mode, "difficulty": q.difficulty, "answer": ans,
            "evaluation": ev, "resources": [], "old_d": old, "new_d": new}
    return {"turns": s.get("turns", []) + [turn], "current_q": None, "last_eval": None}


def retrieve(s):
    turns = list(s["turns"])
    t = turns[-1]
    try:
        query = t["question"] + " " + " ".join(t["evaluation"].weaknesses)
        turns[-1] = {**t, "resources": retrieve_resources(query)}
    except Exception:
        pass                                        # retrieval failure must not stop the interview
    return {"turns": turns}


def summarize(s):
    return {"report": build_report(s.get("turns", [])), "status": "done"}


def save(s):
    turns = s.get("turns", [])
    if not turns:
        return {"message": ""}
    p = s["profile"]
    sid = store_session({"interview_type": s["interview_type"], "target_role": p.target_role,
                         "company": p.company}, turns, s["report"])
    return {"session_id": sid}


def build_report(turns):
    if not turns:
        return {"average": None}
    evs = [t["evaluation"] for t in turns]
    dims = {d: round(sum(getattr(e, d) for e in evs) / len(evs), 2)
            for d in ("relevance", "depth", "clarity", "structure")}
    ranked = sorted(turns, key=lambda t: t["evaluation"].overall)
    seen, resources = set(), []
    for t in turns:
        for r in t["resources"]:
            if r["id"] not in seen:
                seen.add(r["id"])
                resources.append(r)
    return {"average": round(sum(e.overall for e in evs) / len(evs), 2), "dimensions": dims,
            "strengths": [x for t in ranked[::-1][:2] for x in t["evaluation"].strengths[:2]],
            "improve": [w for t in ranked[:2] for w in t["evaluation"].weaknesses[:2]],
            "resources": resources}


# ---------- routing (conditional edges) ----------
def entry(s):
    if s.get("end_early"):
        return "summarize"
    return "ask_question" if s.get("current_q") is None else "guard"


def after_check(s):
    return "go" if s.get("status") == "ok" else "stop"


def needs_retrieval(s):
    return "retrieve" if s["turns"][-1]["evaluation"].overall < RETRIEVE_BELOW else "skip"


def ask_or_finish(s):
    return "finish" if len(s["turns"]) >= s["total"] else "ask"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("ask_question", ask_question), ("guard", guard), ("evaluate", evaluate),
                     ("record", record), ("retrieve", retrieve), ("router", lambda s: {"message": ""}),
                     ("summarize", summarize), ("save_session", save)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, entry, {"ask_question": "ask_question", "guard": "guard",
                                           "summarize": "summarize"})
    g.add_conditional_edges("guard", after_check, {"go": "evaluate", "stop": END})
    g.add_conditional_edges("evaluate", after_check, {"go": "record", "stop": END})
    g.add_conditional_edges("record", needs_retrieval, {"retrieve": "retrieve", "skip": "router"})
    g.add_edge("retrieve", "router")
    g.add_conditional_edges("router", ask_or_finish, {"ask": "ask_question", "finish": "summarize"})
    g.add_edge("ask_question", END)
    g.add_edge("summarize", "save_session")
    g.add_edge("save_session", END)
    return g.compile()