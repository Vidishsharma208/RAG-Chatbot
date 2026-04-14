import json
import os
import re
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize

from src.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_PATH
from src.load_pdf import extract_text_from_pdf

nltk.download("punkt", quiet=True)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences safely.
    """
    return sent_tokenize(text)


def sentence_word_count(sentence: str) -> int:
    return len(sentence.split())


def build_chunks(sentences: List[str], chunk_size_words: int, overlap_words: int) -> List[str]:
    """
    Build sentence-aware chunks of roughly chunk_size_words,
    with some overlap for context continuity.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = sentence_word_count(sentence)

        # Add sentence if chunk not too large
        if current_word_count + sentence_words <= chunk_size_words or not current_chunk:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            i += 1
        else:
            chunk_text = " ".join(current_chunk).strip()
            chunks.append(chunk_text)

            # Create overlap from end of current chunk
            overlap_chunk = []
            overlap_count = 0
            for sent in reversed(current_chunk):
                sent_words = sentence_word_count(sent)
                if overlap_count + sent_words <= overlap_words:
                    overlap_chunk.insert(0, sent)
                    overlap_count += sent_words
                else:
                    break

            current_chunk = overlap_chunk
            current_word_count = sum(sentence_word_count(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def detect_section_name(text: str) -> str:
    """
    Very simple section detection from early chunk text.
    """
    match = re.search(r"(\d+\.\s+[A-Za-z][A-Za-z\s&;/,-]+)", text)
    if match:
        return match.group(1).strip()
    return "Unknown Section"


def make_chunk_records(chunks: List[str]) -> List[Dict]:
    """
    Convert chunk strings to structured JSON records.
    """
    records = []
    for idx, chunk in enumerate(chunks):
        records.append({
            "chunk_id": idx,
            "text": chunk,
            "word_count": len(chunk.split()),
            "section": detect_section_name(chunk)
        })
    return records


def save_chunks(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def process_pdf_to_chunks(pdf_path: str, output_path: str = CHUNKS_PATH) -> List[Dict]:
    """
    Full pipeline: PDF -> text -> sentences -> chunks -> JSON
    """
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    chunks = build_chunks(sentences, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    records = make_chunk_records(chunks)
    save_chunks(records, output_path)
    return records


if __name__ == "__main__":
    from src.config import PDF_PATH

    chunk_records = process_pdf_to_chunks(PDF_PATH)
    print(f"Created {len(chunk_records)} chunks.")
    print(chunk_records[0])