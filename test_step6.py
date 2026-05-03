from rag_pipeline import load_pdf, chunk_text, get_embeddings, build_vector_store, retrieve_chunks, generate_answer

# Full RAG pipeline
text = load_pdf("data/chap_11.pdf")
chunks = chunk_text(text)
embeddings = get_embeddings(chunks)
index = build_vector_store(embeddings)

question = "What is this document about?"  # use a real question about your PDF

print(f"Question: {question}\n")
relevant_chunks = retrieve_chunks(question, index, chunks)
answer = generate_answer(question, relevant_chunks)

print(f"Answer:\n{answer}")