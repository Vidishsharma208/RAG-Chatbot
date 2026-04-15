import json
import os
import re
from typing import List, Dict

from nltk.tokenize import sent_tokenize

from src.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_PATH
from src.load_pdf import extract_text_from_pdf


def sentence_word_count(sentence: str) -> int:
    return len(sentence.split())


def detect_section_name(text: str) -> str:
    patterns = [
        r"(\d+\.\s+[A-Za-z][A-Za-z\s&;/,\-]+)",
        r"(Section\s+\d+[\.:]?\s+[A-Za-z][A-Za-z\s&;/,\-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return "Unknown Section"


def build_chunks(sentences: List[str], chunk_size_words: int, overlap_words: int) -> List[str]:
    chunks = []
    current_chunk = []
    current_word_count = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = sentence_word_count(sentence)

        if current_word_count + sentence_words <= chunk_size_words:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            i += 1
            continue

        if not current_chunk:
            chunks.append(sentence.strip())
            i += 1
            continue

        chunk_text = " ".join(current_chunk).strip()
        chunks.append(chunk_text)

        overlap_chunk = []
        overlap_count = 0
        for sent in reversed(current_chunk):
            sent_words = sentence_word_count(sent)
            if overlap_count + sent_words <= overlap_words:
                overlap_chunk.insert(0, sent)
                overlap_count += sent_words
            else:
                break

        if overlap_chunk == current_chunk:
            overlap_chunk = []

        current_chunk = overlap_chunk
        current_word_count = sum(sentence_word_count(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def make_chunk_records(chunks: List[str]) -> List[Dict]:
    records = []
    for idx, chunk in enumerate(chunks):
        records.append({
            "chunk_id": idx,
            "text": chunk,
            "word_count": len(chunk.split()),
            "section": detect_section_name(chunk[:300])
        })
    return records


def save_chunks(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def process_pdf_to_chunks(pdf_path: str, output_path: str = CHUNKS_PATH) -> List[Dict]:
    print("Reading PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting into sentences...")
    sentences = sent_tokenize(text)

    print("Building chunks...")
    chunks = build_chunks(sentences, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    print("Creating records...")
    records = make_chunk_records(chunks)

    print("Saving chunks...")
    save_chunks(records, output_path)

    print(f"Done. Created {len(records)} chunks.")
    return records


if __name__ == "__main__":
    from src.config import PDF_PATH
    process_pdf_to_chunks(PDF_PATH)