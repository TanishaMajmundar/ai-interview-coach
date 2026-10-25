import json, sqlite3, uuid
from datetime import datetime
from pathlib import Path
from app.models import Evaluation

DB = Path("data/sessions.db")


def _conn():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at TEXT, "
                "interview_type TEXT, target_role TEXT, company TEXT, n_questions INTEGER, "
                "average REAL, report_json TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS answers (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT, idx INTEGER, mode TEXT, difficulty INTEGER, question TEXT, "
                "answer TEXT, overall REAL, evaluation_json TEXT, resources_json TEXT)")
    return con


def save_session(meta: dict, turns: list, report: dict) -> str:
    sid = uuid.uuid4().hex[:8]
    con = _conn()
    try:
        with con:
            con.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                        (sid, datetime.now().strftime("%Y-%m-%d %H:%M"), meta["interview_type"],
                         meta.get("target_role"), meta.get("company"), len(turns),
                         report.get("average"), json.dumps(report)))
            for i, t in enumerate(turns, 1):
                con.execute("INSERT INTO answers (session_id, idx, mode, difficulty, question, answer, "
                            "overall, evaluation_json, resources_json) VALUES (?,?,?,?,?,?,?,?,?)",
                            (sid, i, t["mode"], t["difficulty"], t["question"], t["answer"],
                             t["evaluation"].overall, t["evaluation"].model_dump_json(),
                             json.dumps(t["resources"])))
    finally:
        con.close()
    return sid


def list_sessions() -> list:
    con = _conn()
    try:
        return [dict(r) for r in con.execute("SELECT * FROM sessions ORDER BY created_at DESC")]
    finally:
        con.close()


def load_session(sid: str):
    con = _conn()
    try:
        meta = dict(con.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone())
        rows = con.execute("SELECT * FROM answers WHERE session_id=? ORDER BY idx", (sid,)).fetchall()
    finally:
        con.close()
    turns = [{"question": r["question"], "answer": r["answer"], "mode": r["mode"],
              "difficulty": r["difficulty"],
              "evaluation": Evaluation.model_validate_json(r["evaluation_json"]),
              "resources": json.loads(r["resources_json"])} for r in rows]
    return meta, turns, json.loads(meta["report_json"])