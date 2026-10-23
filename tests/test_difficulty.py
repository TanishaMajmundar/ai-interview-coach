from app.models import CandidateProfile
from agents.interviewer import Interviewer, build_plan

profile = CandidateProfile.model_validate_json(
    open("data/sample_profile.json", encoding="utf-8").read()
)

# 1) Inspect the question plan for each mode (no LLM call)
for mode in ("hr", "technical", "both"):
    print(f"\nPLAN for {mode}:")
    for i, (m, topic) in enumerate(build_plan(mode), 1):
        print(f"  Q{i} [{m}] {topic}")

# 2) Difficulty rule, step by step (no LLM call)
iv = Interviewer(profile, mode="both", difficulty=3)
scores = [4.5, 4.25, 4.8, 3.0, 1.5, 1.0, 2.0, 1.0, 1.0, 3.9]
expected = [4, 5, 5, 5, 4, 3, 2, 1, 1, 1]
actual = [iv.adjust_difficulty(s) for s in scores]
print("\nscores     :", scores)
print("difficulty :", actual)
assert actual == expected, f"unexpected difficulty sequence: {actual}"
print("Difficulty check passed (up at >= 4, down at <= 2, clamped to 1..5)")