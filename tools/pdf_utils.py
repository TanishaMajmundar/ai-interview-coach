from pypdf import PdfReader


def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a text-based PDF."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No text found. The PDF may be a scanned image.")
    return text