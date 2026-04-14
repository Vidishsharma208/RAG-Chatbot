from typing import List, Dict, Generator

from groq import Groq

from src.config import GROQ_API_KEY, LLM_MODEL_NAME


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into prompt context.
    """
    formatted_parts = []
    for chunk in chunks:
        formatted_parts.append(
            f"[Chunk ID: {chunk['chunk_id']} | Section: {chunk.get('section', 'Unknown')}]\n{chunk['text']}"
        )
    return "\n\n".join(formatted_parts)


def build_prompt(user_query: str, retrieved_chunks: List[Dict]) -> str:
    context = format_context(retrieved_chunks)

    prompt = f"""
You are a helpful AI assistant for document question-answering.

Rules:
1. Answer ONLY from the provided context.
2. Do not add outside knowledge.
3. If the answer is not present in the context, say:
   "I could not find this in the provided document."
4. Keep the answer clear and grounded.
5. After the answer, mention which chunk IDs were used.

Context:
{context}

User Question:
{user_query}

Answer:
"""
    return prompt.strip()


def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return Groq(api_key=GROQ_API_KEY)


def generate_answer_stream(user_query: str, retrieved_chunks: List[Dict]) -> Generator[str, None, None]:
    """
    Stream response token-by-token from Groq.
    """
    client = get_groq_client()
    prompt = build_prompt(user_query, retrieved_chunks)

    stream = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You answer questions only from the provided document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_full_answer(user_query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Non-streaming version for testing.
    """
    client = get_groq_client()
    prompt = build_prompt(user_query, retrieved_chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You answer questions only from the provided document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=False
    )

    return response.choices[0].message.content