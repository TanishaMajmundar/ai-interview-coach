import re

MAX_PDF_MB = 5
MAX_RESUME_CHARS = 15000
MAX_JD_CHARS = 8000
MAX_ANSWER_CHARS = 2500

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous |prior |above |earlier )?(instructions|prompts|rules)",
    r"forget (all |your |the )?(previous |prior )?(instructions|rules)",
    r"you are now\b",
    r"(reveal|show|print|repeat) (me )?(your |the )?(system )?(prompt|instructions)",
    r"system prompt",
    r"(give|assign|award) (this candidate|me|the candidate) (a )?(perfect|full|top|maximum|highest)",
    r"override (the )?(rules|instructions|scoring)",
    r"<\s*/?\s*(system|assistant|candidate_profile|answer|resume|job_description)\s*>",
]
_RX = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def scan_injection(text: str):
    """Rule-based layer: replace instruction-like phrases and report which rules fired."""
    flags, clean = [], text
    for rx in _RX:
        if rx.search(clean):
            flags.append(rx.pattern)
            clean = rx.sub("[removed]", clean)
    return clean, flags


def validate_resume_file(file):
    if file is None:
        return "Please upload a resume PDF."
    if not file.name.lower().endswith(".pdf"):
        return "The resume must be a PDF file."
    if file.size > MAX_PDF_MB * 1024 * 1024:
        return f"The PDF is larger than {MAX_PDF_MB} MB."
    return None


def validate_text(text, label, max_chars, min_chars=50):
    t = (text or "").strip()
    if len(t) < min_chars:
        return f"The {label} is too short."
    if len(t) > max_chars:
        return f"The {label} is too long (limit {max_chars} characters)."
    return None


def check_answer(answer):
    """Returns (ok, message, cleaned_answer). Rejected answers do not use up a question."""
    a = (answer or "").strip()
    if not a:
        return False, "Please type an answer.", a
    if len(a) > MAX_ANSWER_CHARS:
        return False, f"Your answer is too long (limit {MAX_ANSWER_CHARS} characters).", a
    if scan_injection(a)[1]:
        return False, "Your answer contains instruction-like text. Please answer the question in your own words.", a
    return True, "", a


def is_off_topic(ev, answer: str) -> bool:
    """Off-topic = a real attempt (12+ words) that the Evaluator scored 1 for relevance and depth."""
    return len(answer.split()) >= 12 and ev.relevance == 1 and ev.depth == 1