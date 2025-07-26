PDF Text Search with Sentence Transformers and FAISS 📄🔍
This project demonstrates how to extract text from a PDF, split it into manageable chunks, convert those chunks into vector embeddings using a pre-trained Sentence Transformer model, and perform fast similarity searches using FAISS.

Features ✨
Extract text from PDF files using PyMuPDF (fitz) 📚

Split long text into smaller chunks intelligently by sentences or newlines ✂️📝

Convert text chunks to embeddings with a pre-trained Sentence Transformer (all-MiniLM-L6-v2) 🤖

Normalize embeddings for cosine similarity search 🎯

Build a FAISS index for fast similarity search over text chunks ⚡️

Query the index with natural language questions to find the most relevant text chunks ❓💡

Requirements 🛠️
Python 3.7+

Install required packages:

bash
Copy
Edit
pip install sentence-transformers faiss-cpu pymupdf scikit-learn 
Usage 🚀
Place your PDF file (e.g. article.pdf) in the project directory or update the file path accordingly in the code.

Run the Python script:

python
Copy
Edit
from sentence_transformers import SentenceTransformer
import faiss
import fitz
from sklearn.preprocessing import normalize

# ... (full script as provided) ...
The script will:

Extract text from the PDF 🕵️‍♂️

Split it into chunks of up to 1500 characters (preferably at sentence boundaries) ✂️

Generate embeddings for each chunk 🧠

Create a FAISS index to search for similar chunks ⚡️

Run a sample query ("What is AI?") ❓

Print the most relevant text chunks and their similarity scores 🏆

Customization 🎨
Chunk size: Adjust max_chunk_size in the chunk_text() function to control how big each text chunk can be.

Model: Replace 'sentence-transformers/all-MiniLM-L6-v2' with any other Sentence Transformer model you prefer.

Number of results: Change k in index.search(query_embedding, k=2) to get more or fewer search results.

Query: Replace the query string to test different questions or search phrases.

Notes 📝
FAISS uses inner product similarity on normalized embeddings to approximate cosine similarity.

This approach works well for searching large documents or sets of documents efficiently.

For very large PDF files, consider more advanced chunking or indexing strategies.

License 📜
This project is provided as-is under the MIT License.