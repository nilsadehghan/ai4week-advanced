📄 PDF Semantic Search API
A FastAPI app that extracts text from a PDF, chunks it, creates semantic embeddings using SentenceTransformers, indexes them with FAISS, and provides a query endpoint for semantic search over the document!

🚀 Features
Extracts text from PDF using PyMuPDF (fitz)

Splits long text into manageable chunks

Embeds chunks with sentence-transformers/all-MiniLM-L6-v2

Builds a FAISS similarity search index

Exposes a FastAPI endpoint for semantic question answering

Publicly accessible via ngrok tunnel

⚙️ Setup & Run
1. Install dependencies:
bash
Copy
Edit
pip install sentence-transformers faiss-cpu pymupdf scikit-learn fastapi pydantic pyngrok uvicorn nest_asyncio
2. Place your PDF
Put your PDF at the path:

css
Copy
Edit
/content/article.pdf
3. Run the script
bash
Copy
Edit
python rag_api.py
4. Use the API
After running, a public URL will be printed in the terminal. You can query it like this:

http
Copy
Edit
POST {public_url}/query/
Content-Type: application/json

{
  "question": "Your query here"
}
🧩 How it Works
Extracts all text from the PDF file

Splits text into chunks (max 1500 characters) at sentence or newline boundaries

Encodes each chunk using MiniLM embeddings

Normalizes and indexes embeddings with FAISS for fast similarity search

Runs a FastAPI server to accept queries

Returns top 2 most relevant chunks with similarity scores

Uses ngrok to expose the local API to the internet

🔑 Notes
Replace the ngrok auth token with your own

Change the PDF path to your actual document location

You can adjust:

max_chunk_size in chunk_text()

k (number of results returned) in index.search()

🧑‍💻 Author
Nilsa Dehghan
🌐 GitHub