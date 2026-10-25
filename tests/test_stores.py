from pathlib import Path
from tools.retriever import retrieve_resources
import tools.session_store as ss
from app.models import Evaluation

for q in ["How would you get up to speed with Docker containers?", "overfitting and cross validation",
          "conflict with a teammate, answer lacks structure and result", "best pizza in town"]:
    print("\nQUERY:", q)
    for r in retrieve_resources(q, k=3, max_distance=2.0):
        print(f'  {r["distance"]}  {r["title"]}')

ss.DB = Path("data/test_sessions.db")
ev = Evaluation(relevance=4, depth=3, clarity=4, structure=3, strengths=["a"], weaknesses=["b"], improved_answer="c")
turn = {"question": "Q?", "answer": "A.", "mode": "hr", "difficulty": 3, "evaluation": ev, "resources": []}
sid = ss.save_session({"interview_type": "HR", "target_role": "ML Engineer", "company": "X"}, [turn], {"average": 3.5})
meta, turns, report = ss.load_session(sid)
assert turns[0]["evaluation"].overall == 3.5 and report["average"] == 3.5 and len(ss.list_sessions()) == 1
ss.DB.unlink()
print("\nSession store check passed")