from rag_pipeline import load_pdf, chunk_text, get_embeddings, build_vector_store, retrieve_chunks

text = load_pdf("data/chap_11.pdf")
chunks = chunk_text(text)
embeddings = get_embeddings(chunks)
index = build_vector_store(embeddings)

question = "What is this document about?"  # change to something relevant to your PDF
results = retrieve_chunks(question, index, chunks)

print(f"\nTop 3 chunks retrieved for: '{question}'\n")
for i, chunk in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()