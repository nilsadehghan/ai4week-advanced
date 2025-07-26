from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF
from sklearn.preprocessing import normalize
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn
import nest_asyncio
import threading

# Enable nested event loops (needed in Colab)
nest_asyncio.apply()

# Extract text from PDF
def extract_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

article_text = extract_text("/content/article.pdf")

#Split text into smaller chunks
def chunk_text(txt, max_chunk_size=1500):
    chunks = []
    while len(txt) > max_chunk_size:
        split_index = max(txt[:max_chunk_size].rfind("."), txt[:max_chunk_size].rfind("\n"))
        if split_index == -1:
            split_index = max_chunk_size
        chunks.append(txt[:split_index].strip())
        txt = txt[split_index:]
    if txt.strip():
        chunks.append(txt.strip())
    return chunks

chunks = chunk_text(article_text)

#Load sentence transformer and compute embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
embeddings = normalize(embeddings)

#Create FAISS index for similarity search
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Define FastAPI app
app = FastAPI()

class InputText(BaseModel):
    question: str

@app.post("/query/")
def get_query(data: InputText):
    query_embedding = normalize(model.encode([data.question]))
    D, I = index.search(query_embedding, k=2)

    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "text": chunks[idx],
            "score": float(score)
        })

    return {
        "question": data.question,
        "results": results
    }

# Start ngrok tunnel
ngrok.set_auth_token("30PeMD17bws7quN0t8Ak2FlCU2J_nSRGkAuN3D6nfXQa1T5m")
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

#Run the FastAPI app in a separate thread
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_app).start()
