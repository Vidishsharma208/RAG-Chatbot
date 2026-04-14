import os
import json
import streamlit as st

from src.chunk_text import process_pdf_to_chunks
from src.embed_store import main as build_vector_db
from src.rag_pipeline import answer_query_stream
from src.config import (
    PDF_PATH,
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
)

st.set_page_config(page_title="Amlgo RAG Chatbot", page_icon="🤖", layout="wide")


def knowledge_base_ready():
    return os.path.exists(CHUNKS_PATH) and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)


def build_knowledge_base():
    if not os.path.exists(CHUNKS_PATH):
        process_pdf_to_chunks(PDF_PATH)
    if not os.path.exists(FAISS_INDEX_PATH):
        build_vector_db()


def get_chunk_count():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data)
    return 0


if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 Fine-Tuned RAG Chatbot")
st.caption("Answers questions from the uploaded training document using retrieval + LLM generation.")

with st.sidebar:
    st.header("Settings / Info")
    st.write(f"**LLM Model:** {LLM_MODEL_NAME}")
    st.write(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")

    if knowledge_base_ready():
        st.success(f"Knowledge base ready ({get_chunk_count()} chunks)")
    else:
        st.warning("Knowledge base not built yet")

    if st.button("🔄 Reset Chat"):
        st.session_state.history = []
        st.rerun()

st.markdown("## Setup")

if not knowledge_base_ready():
    st.info("First build the knowledge base. This may take a few minutes the first time.")
    if st.button("Build Knowledge Base"):
        try:
            with st.spinner("Building chunks, embeddings, and vector database..."):
                build_knowledge_base()
            st.success("Knowledge base built successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"Setup error: {e}")
else:
    st.success("Knowledge base is ready.")

st.markdown("---")
st.markdown("## Ask a Question")

user_query = st.text_area(
    "Type your question here:",
    placeholder="Example: Can eBay suspend a user account?",
    height=120
)

send_clicked = st.button("Send")

if send_clicked:
    if not knowledge_base_ready():
        st.warning("Please build the knowledge base first.")
    elif not user_query.strip():
        st.warning("Please enter a question first.")
    else:
        st.markdown("### Your Question")
        st.write(user_query)

        try:
            stream_generator, sources = answer_query_stream(user_query)

            st.markdown("### Answer")
            response_box = st.empty()
            final_answer = ""

            for token in stream_generator:
                final_answer += token
                response_box.markdown(final_answer)

            if not final_answer.strip():
                final_answer = "No response was generated."
                response_box.markdown(final_answer)

            st.session_state.history.append(("You", user_query))
            st.session_state.history.append(("Assistant", final_answer))

            with st.expander("📚 Source Chunks Used", expanded=True):
                for src in sources:
                    st.markdown(f"**Chunk ID:** {src['chunk_id']}")
                    st.markdown(f"**Section:** {src.get('section', 'Unknown')}")
                    st.markdown(f"**Word Count:** {src.get('word_count', 'N/A')}")
                    st.write(src["text"])
                    st.markdown("---")

        except Exception as e:
            st.error(f"Generation error: {e}")

st.markdown("---")
st.markdown("## Chat History")

if not st.session_state.history:
    st.info("No messages yet.")
else:
    for role, content in st.session_state.history:
        st.markdown(f"**{role}:**")
        st.write(content)
        st.markdown("---")