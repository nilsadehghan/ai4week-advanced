📄 PDF Semantic Search API using Sentence Transformers, FAISS & FastAPI
This project creates a REST API that allows users to ask questions about the content of a PDF file. It uses:

Sentence Transformers for generating text embeddings,

FAISS for efficient semantic search,

FastAPI for the web API,

Ngrok for exposing the local API to the internet (useful for testing in environments like Google Colab),

PyMuPDF (fitz) for reading PDFs.

🚀 Features
Extracts text from a PDF document

Splits the text into manageable chunks

Embeds the chunks using sentence-transformers

Indexes them with FAISS for similarity search

Hosts a FastAPI endpoint to accept natural language queries

Returns top relevant chunks with similarity scores

🧱 Tech Stack
sentence-transformers (for encoding text)

faiss-cpu (for similarity search)

PyMuPDF (to extract text from PDFs)

FastAPI (to create the API)

uvicorn (ASGI server)

ngrok (for external access to your local server)

sklearn.preprocessing.normalize (for normalized embeddings)

📦 Installation
bash
Copy
Edit
pip install sentence-transformers faiss-cpu pymupdf fastapi pyngrok uvicorn nest_asyncio scikit-learn

🔍 API Endpoint
POST /query/
Description: Accepts a question and returns the most relevant chunks from the PDF.

Request:

json
Copy
Edit
{
  "question": "What is the main argument of the article?"
}
Response:

json
Copy
Edit
{
  "question": "What is the main argument of the article?",
  "results": [
    {
      "text": "First matching text chunk...",
      "score": 0.84
    },
    {
      "text": "Second matching text chunk...",
      "score": 0.79
    }
  ]
}
📌 Notes
The script uses nest_asyncio to allow asynchronous FastAPI + ngrok in environments like Jupyter or Colab.

The PDF is split into chunks of ~1500 characters to maintain context while embedding.

You can change the k=2 value in the index.search() call to retrieve more or fewer results.

📄 License
MIT License. Use freely with attribution.
