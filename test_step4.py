from rag_pipeline import load_pdf, chunk_text, get_embeddings

text = load_pdf("data/your_file.pdf")
chunks = chunk_text(text)

print("Generating embeddings... (first run downloads the model)")
embeddings = get_embeddings(chunks)

print(f"\nNumber of embeddings: {len(embeddings)}")
print(f"Shape of one embedding: {embeddings[0].shape}")
print(f"First 5 numbers of first embedding: {embeddings[0][:5]}")