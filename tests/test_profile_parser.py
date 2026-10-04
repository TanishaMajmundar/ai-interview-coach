from pathlib import Path
from tools.pdf_utils import extract_text_from_pdf
from tools.profile_parser import parse_profile

resume = extract_text_from_pdf("data/sample_resume.pdf")
jd = Path("data/sample_jd.txt").read_text(encoding="utf-8")

# Inspect the inputs BEFORE calling the LLM
print("RESUME CHARS:", len(resume))
print(resume[:300])
print("JD CHARS:", len(jd))
print(jd[:300])
print("-" * 40)

profile = parse_profile(resume, jd)
print(profile.model_dump_json(indent=2))