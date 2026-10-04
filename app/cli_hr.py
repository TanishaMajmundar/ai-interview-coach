import sys
from pathlib import Path
from tools.pdf_utils import extract_text_from_pdf
from tools.profile_parser import parse_profile
from app.models import CandidateProfile
from agents.interviewer import Interviewer

DEBUG = "--debug" in sys.argv
PROFILE_CACHE = Path("data/sample_profile.json")


def load_profile() -> CandidateProfile:
    """Parse once, then reuse the saved profile (saves API calls while testing)."""
    if PROFILE_CACHE.exists():
        return CandidateProfile.model_validate_json(PROFILE_CACHE.read_text(encoding="utf-8"))
    resume = extract_text_from_pdf("data/sample_resume.pdf")
    jd = Path("data/sample_jd.txt").read_text(encoding="utf-8")
    profile = parse_profile(resume, jd)
    PROFILE_CACHE.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def main():
    profile = load_profile()
    print(f"\nProfile loaded for: {profile.name}")
    print("Type your answer and press Enter. Type 'quit' to end early.\n")

    interviewer = Interviewer(profile, mode="hr", total_questions=5, debug=DEBUG)

    for i in range(interviewer.total_questions):
        question = interviewer.next_question()
        print(f"[Q{i+1}/{interviewer.total_questions}] (difficulty {question.difficulty})")
        print(question.text)

        answer = input("\nYour answer: ").strip()
        if answer.lower() == "quit":
            print("Interview ended early.")
            break
        if not answer:
            answer = "(no answer given)"
        interviewer.record_answer(question, answer)
        print()

    print("\n===== Transcript =====")
    for i, t in enumerate(interviewer.history, 1):
        print(f"Q{i}: {t['question']}\nA{i}: {t['answer']}\n")


if __name__ == "__main__":
    main()