from app.models import CandidateProfile, Question
from agents.evaluator import Evaluator

profile = CandidateProfile.model_validate_json(
    open("data/sample_profile.json", encoding="utf-8").read()
)

question = Question(
    text="Tell me about a time you worked with others to build something they would actually use.",
    mode="hr",
    difficulty=3,
)

answers = {
    "good": (
        "At Brightpath, the sales team spent hours every week compiling reports in Excel. "
        "I met two sales managers, sketched a layout with them, then cleaned about 200K rows "
        "with Python and SQL and built a Power BI dashboard showing revenue by region. "
        "After they asked for a product filter I added it within a day. They started opening "
        "the dashboard every Monday instead of requesting reports."
    ),
    "average": (
        "During my internship I worked with the sales team to develop Power BI dashboards. "
        "My contribution was cleaning the data, preparing the analysis, and helping design "
        "the dashboard based on the team's requirements."
    ),
    "bad": "I just did what I was told. It was fine.",
    "casual_good": (
        "So the sales guys were making their reports in Excel every week and it took them "
        "forever. I sat down with two of the managers, we drew what they wanted on paper, and "
        "then I cleaned the data and built it in Power BI. After they saw the first version "
        "they wanted a product filter, so I added that the next day. Now they just open it "
        "every Monday."
    ),
}

# Inspect the inputs BEFORE any LLM call
print("QUESTION:", question.text)
for label, text in answers.items():
    print(f"\n[{label.upper()}] ({len(text)} chars)\n{text}")
print("-" * 40)

evaluator = Evaluator(profile)
results = {}
for label, text in answers.items():
    ev = evaluator.evaluate(question, text)
    results[label] = ev
    print(f"\n===== {label.upper()} =====")
    print(ev.model_dump_json(indent=2))
    print("OVERALL:", ev.overall)

print("\n===== SUMMARY =====")
for label, ev in results.items():
    print(f"{label:8s} rel={ev.relevance} depth={ev.depth} clarity={ev.clarity} "
          f"struct={ev.structure} overall={ev.overall}")

assert results["good"].overall > results["average"].overall > results["bad"].overall, \
    "Scores are not ordered good > average > bad"
assert results["good"].overall - results["average"].overall >= 1.0, \
    "Good and average are too close: rubric is too generous"
assert results["casual_good"].overall > results["average"].overall, \
    "Casual but specific answer should beat the generic one"
print("Casual check passed: casual_good > average")
print("\nOrdering check passed: good > average > bad")