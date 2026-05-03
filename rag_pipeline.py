import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size = 500, overlap = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar = True)
    return embeddings

def build_vector_store(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)
    print(f"Vector store build with {index.ntotal} vectors.")
    return index

def retrieve_chunks(query, index, chunks, k=3):
    query_vector = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )

    result = response.json()

    # This helps us debug if the API returns an error
    if "choices" not in result:
        print("API Error:", result)
        return "Error: Could not get answer from LLM."

    return result["choices"][0]["message"]["content"]