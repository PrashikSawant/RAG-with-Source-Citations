# 📎 Day 14 — RAG with Source Citations

Every AI answer grounded with numbered inline citations,
page numbers, and a full reference list.
Built with ChromaDB, Sentence Transformers, Groq, and Streamlit.

## 💡 What It Does
- Upload PDF and TXT documents
- Ask questions and get answers with [1], [2], [3] inline citations
- Every claim references its exact source
- References section shows filename, page number, chunk, similarity
- Preview each source chunk directly in the UI
- Persistent ChromaDB storage across sessions

## 🛠️ Tech Stack
- Python 3.10+
- ChromaDB — persistent vector database
- Sentence Transformers (all-MiniLM-L6-v2) — embeddings
- LangChain Text Splitters — recursive chunking
- PyMuPDF — PDF extraction with page tracking
- Groq API (LLaMA 3.3 70B) — cited answer generation
- Streamlit — web interface
- python-dotenv — API key management

## 🚀 Setup & Run

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/day14-rag-citations
cd day14-rag-citations

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your API key
Create a .env file:
GROQ_API_KEY=your_key_here

### 4. Run
streamlit run app.py

## 🧠 Key Concepts
- Page-aware chunking — page number stored in metadata
- Numbered citation system — [1], [2], [3] inline
- Structured reference list with similarity scores
- Prompt engineering for forced attribution
- Rich metadata — filename, page, chunk, char count

## 📁 Project Structure
day14-rag-citations/
├── app.py              # Streamlit UI, citation rendering
├── rag_engine.py       # RAG logic, citation formatting
├── requirements.txt    # Dependencies
├── .env               # API key (not committed)
├── .gitignore         # Ignores .env, chroma_db, cache
└── chroma_db/         # Auto-created, gitignored

## 🔗 Part of 30-Day AI Engineering Bootcamp
Day 14 of 30 | RAG & Vector Databases Phase
