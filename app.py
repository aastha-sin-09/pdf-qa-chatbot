import streamlit as st
from rag_pipeline import (
    load_pdf,
    chunk_text,
    get_embeddings,
    build_vector_store,
    retrieve_chunks,
    generate_answer
)
import os

st.set_page_config(page_title="PDF Q&A Chatbot", layout="centered")
st.title("PDF Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it. Powered by sentence-transformers + Groq (Llama 3).")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Upload a PDF
    2. It gets split into chunks
    3. Each chunk is embedded (converted to a vector)
    4. Your question is matched to the most relevant chunks
    5. An LLM generates an answer from those chunks
    """)
    st.divider()
    if "chunks" in st.session_state:
        st.metric("Chunks indexed", len(st.session_state.chunks))
    if "chat_history" in st.session_state:
        st.metric("Questions asked", len(st.session_state.chat_history))
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is None:
    st.info("Upload a PDF above to get started.")
    st.stop()  # Don't render anything below until a file is uploaded

# ── Index the PDF (only when a new file is uploaded) ─────────────────────────
if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:

    temp_path = "data/temp_uploaded.pdf"
    os.makedirs("data", exist_ok=True)  # make sure data/ folder exists

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("Reading and indexing your PDF... this may take a moment."):
            text = load_pdf(temp_path)

            if not text.strip():
                st.error("Could not extract text from this PDF. It may be scanned or image-based.")
                st.stop()

            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            index = build_vector_store(embeddings)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.chat_history = []

        st.success(f"Ready! Indexed {len(chunks)} chunks from '{uploaded_file.name}'")

    except Exception as e:
        st.error(f"Something went wrong while processing the PDF: {e}")
        st.stop()

# ── Q&A Section ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Ask a Question")

question = st.text_input(
    "Type your question",
    placeholder="What is this document about?",
    key="question_input"
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        try:
            with st.spinner("Finding relevant content and generating answer..."):
                relevant_chunks = retrieve_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )
                answer = generate_answer(question, relevant_chunks)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })

        except Exception as e:
            st.error(f"Could not generate an answer: {e}")

# ── Chat History ──────────────────────────────────────────────────────────────
if "chat_history" in st.session_state and st.session_state.chat_history:
    st.divider()
    st.subheader("Conversation")

    for entry in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])