# 🤖 RAG Chatbot (Amlgo Labs Assignment)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers user queries based on a provided document (eBay User Agreement).

Instead of relying on general knowledge, the chatbot retrieves relevant sections from the document and generates accurate, grounded responses using an LLM.

---

## 🚀 Features

* 📄 PDF document ingestion
* ✂️ Sentence-aware chunking (100–300 words)
* 🧠 Semantic embeddings using SentenceTransformers
* ⚡ Fast similarity search using FAISS vector database
* 🤖 LLM-based answer generation (Llama 3 via Groq API)
* 💬 Streamlit-based interactive UI
* 🔄 Real-time streaming responses
* 📚 Source chunk display for transparency

---

## 🧠 System Architecture

```
PDF → Text Cleaning → Chunking → Embeddings → FAISS Index
     → User Query → Similarity Search → LLM → Final Answer
```

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit**
* **FAISS (Vector DB)**
* **SentenceTransformers**
* **Groq API (Llama 3)**
* **PyPDF**

---

## 📂 Project Structure

```
RAG-Chatbot/
│
├── data/          # Input PDF document
├── chunks/        # Processed text chunks
├── vectordb/      # FAISS index and metadata
├── src/           # Core pipeline code
│   ├── load_pdf.py
│   ├── chunk_text.py
│   ├── embed_store.py
│   ├── retrieve.py
│   ├── generate.py
│   └── rag_pipeline.py
│
├── app.py         # Streamlit app
├── requirements.txt
├── .env
└── README.md
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Vidishsharma208/RAG-Chatbot.git
cd RAG-Chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 5️⃣ Run Application

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

---

## 🧪 Example Queries

* Can eBay suspend a user account?
* Can buyers cancel orders?
* Does eBay guarantee vehicle safety?
* What happens if a seller violates policies?
* What is arbitration in this agreement?

---

## 📊 Output Behavior

* ✅ If answer is found → grounded response from document
* ❌ If not found →
  **"I could not find this in the provided document."**

---

## ⚠️ Limitations

* Depends on embedding quality
* May fail for very vague queries
* Initial setup (embedding + FAISS) takes time
* Limited to provided document only

---

## 🎥 Demo

👉 (Add your demo video link here)

---

## 📌 Assignment Requirements Covered

* ✔️ Document preprocessing & chunking
* ✔️ Embedding generation
* ✔️ Vector database (FAISS)
* ✔️ RAG pipeline (Retriever + Generator)
* ✔️ Streamlit chatbot with streaming responses
* ✔️ Source grounding

---

## 🙌 Author

**Vidish Sharma**

---
