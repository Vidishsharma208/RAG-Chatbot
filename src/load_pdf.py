import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    # Remove repeated spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file page by page.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            cleaned = clean_text(page_text)
            pages_text.append(cleaned)

    full_text = "\n\n".join(pages_text)
    return full_text


if __name__ == "__main__":
    from src.config import PDF_PATH

    text = extract_text_from_pdf(PDF_PATH)
    print(text[:2000])
    print(f"\nTotal characters extracted: {len(text)}")