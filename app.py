import streamlit as st
from rag_engine import (
    add_document,
    get_documents_list,
    delete_document,
    semantic_search,
    answer_with_citations,
    collection
)

st.set_page_config(page_title="RAG Citations", page_icon="📎")
st.title("📎 RAG with Source Citations")
st.caption("Every answer grounded with numbered citations")

# sidebar
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        if st.button("➕ Add to Knowledge Base"):
            with st.spinner("Processing..."):
                msg = add_document(uploaded_file)
            st.write(msg)
            st.rerun()

    st.divider()

    st.header("📄 Documents")
    docs = get_documents_list()

    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {doc}")
            with col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    delete_document(doc)
                    st.rerun()
    else:
        st.info("No documents yet.")

    st.divider()
    st.metric("Total Chunks", collection.count())
    st.caption("Day 14 — 30 Day AI Bootcamp")

# main area
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # render citations if they exist
        if "citations" in msg and msg["citations"]:
            st.divider()
            st.markdown("**📚 References:**")
            for c in msg["citations"]:
                st.markdown(
                    f"**[{c['number']}]** `{c['filename']}` — "
                    f"Page {c['page']} | "
                    f"Chunk {c['chunk']} | "
                    f"{c['similarity']}% match"
                )
                with st.expander(f"Preview [{c['number']}]"):
                    st.write(c["preview"])

# chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Searching and citing..."):
            results = semantic_search(question)
            answer, citations = answer_with_citations(question, results)

        st.write(answer)

        # render citations
        if citations:
            st.divider()
            st.markdown("**📚 References:**")
            for c in citations:
                st.markdown(
                    f"**[{c['number']}]** `{c['filename']}` — "
                    f"Page {c['page']} | "
                    f"Chunk {c['chunk']} | "
                    f"{c['similarity']}% match"
                )
                with st.expander(f"Preview [{c['number']}]"):
                    st.write(c["preview"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })