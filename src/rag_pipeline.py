from typing import Dict, List, Generator

from src.retrieve import retrieve_relevant_chunks
from src.generate import generate_answer_stream, generate_full_answer


def answer_query(query: str) -> Dict:
    """
    Full RAG flow without streaming.
    """
    retrieved_chunks = retrieve_relevant_chunks(query)
    answer = generate_full_answer(query, retrieved_chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": retrieved_chunks
    }


def answer_query_stream(query: str) -> tuple[Generator[str, None, None], List[Dict]]:
    """
    Full RAG flow with streaming.
    """
    retrieved_chunks = retrieve_relevant_chunks(query)
    stream_generator = generate_answer_stream(query, retrieved_chunks)
    return stream_generator, retrieved_chunks


if __name__ == "__main__":
    query = "Can buyers cancel an order on eBay?"
    result = answer_query(query)

    print("\nQuestion:", result["query"])
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- Chunk {src['chunk_id']} | Section: {src.get('section', 'Unknown')}")