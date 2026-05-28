import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import fitz
import os
import hashlib

load_dotenv()

# initialize
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="rag_citations",
    metadata={"hnsw:space": "cosine"}
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text_with_pages(uploaded_file):
    """Extract text with page numbers for rich citations"""
    filename = uploaded_file.name
    pages_content = []

    if filename.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
        pages_content.append({"page": 1, "text": text})

    elif filename.endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages_content.append({
                    "page": page_num + 1,
                    "text": text
                })
    return pages_content

def get_doc_id(filename):
    return hashlib.md5(filename.encode()).hexdigest()[:8]

def document_exists(filename):
    results = collection.get(where={"filename": filename})
    return len(results["ids"]) > 0

def add_document(uploaded_file):
    """Extract, chunk, embed with rich metadata"""
    filename = uploaded_file.name

    if document_exists(filename):
        return f"'{filename}' already exists."

    pages_content = extract_text_with_pages(uploaded_file)
    if not pages_content:
        return "Could not extract text from this file."

    doc_id = get_doc_id(filename)
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_index = 0

    # chunk per page — preserves page number in metadata
    for page_data in pages_content:
        page_num = page_data["page"]
        page_text = page_data["text"]
        chunks = splitter.split_text(page_text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "filename": filename,
                "page": page_num,
                "chunk": chunk_index,
                "char_count": len(chunk)
            })
            all_ids.append(f"{doc_id}_chunk_{chunk_index}")
            chunk_index += 1

    embeddings = embedding_model.encode(all_chunks).tolist()
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )
    return f"✅ '{filename}' added — {chunk_index} chunks from {len(pages_content)} pages."

def get_documents_list():
    if collection.count() == 0:
        return []
    results = collection.get()
    return list(set(m["filename"] for m in results["metadatas"]))

def delete_document(filename):
    results = collection.get(where={"filename": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return f"✅ '{filename}' deleted."
    return "Document not found."

def semantic_search(query, top_k=5):
    """Search and return chunks with full metadata"""
    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metadatas, distances))

def format_citations(results):
    """Build a numbered reference list"""
    citations = []
    for i, (doc, meta, distance) in enumerate(results):
        similarity = round((1 - distance) * 100, 1)
        citation = {
            "number": i + 1,
            "filename": meta["filename"],
            "page": meta["page"],
            "chunk": meta["chunk"],
            "similarity": similarity,
            "preview": doc[:150] + "..." if len(doc) > 150 else doc
        }
        citations.append(citation)
    return citations

def answer_with_citations(query, results):
    """Generate answer with inline citation numbers"""
    if not results:
        return "No relevant content found.", []

    citations = format_citations(results)

    # build numbered context
    context = ""
    for c in citations:
        context += (
            f"[{c['number']}] {c['filename']} "
            f"(Page {c['page']}):\n{c['preview']}\n\n"
        )

    prompt = f"""You are a precise research assistant.
Answer the question using ONLY the context below.
Use inline citation numbers like [1], [2], [3] 
throughout your answer to reference sources.
Every factual claim MUST have a citation number.
End your answer with a one-line summary.

Context:
{context}

Question: {query}

Answer (use [1], [2], [3] inline):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )

    return response.choices[0].message.content, citations