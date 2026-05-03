from rag_pipeline import load_pdf, chunk_text

text = load_pdf("data/chap_11.pdf")  # replace with your actual filename
print(f"Total characters extracted: {len(text)}")
print(f"\nFirst 300 characters:\n{text[:300]}")

chunks = chunk_text(text)
print(f"\nTotal chunks created: {len(chunks)}")
print(f"\nFirst chunk:\n{chunks[0]}")
print(f"\nSecond chunk (notice overlap with first):\n{chunks[1]}")