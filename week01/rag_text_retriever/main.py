from sentence_transformers import SentenceTransformer
import faiss
import fitz
from sklearn.preprocessing import normalize


# Function to extract all text from a PDF file using PyMuPDF (fitz)
def extract_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from the PDF located at the specified path
article_text = extract_text("/content/article.pdf")

# Function to split the long text into smaller chunks of max_chunk_size characters
# The function tries to split on the last '.' or newline character before the max_chunk_size limit
def chunk_text(txt, max_chunk_size=1500):
    chunks = []
    while len(txt) > max_chunk_size:
        split_index = max(txt[:max_chunk_size].rfind("."), txt[:max_chunk_size].rfind("\n"))
        if split_index == -1:
            split_index = max_chunk_size  # If no suitable split point found, just split at max_chunk_size
        chunks.append(txt[:split_index].strip())
        txt = txt[split_index:]
    if txt.strip():
        chunks.append(txt.strip())
    return chunks

# Split the extracted article text into smaller chunks
chunks = chunk_text(article_text)

# Load a pre-trained sentence transformer model to convert text chunks into embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for each text chunk
embeddings = model.encode(chunks)

# Normalize embeddings to unit length for cosine similarity using inner product
embeddings = normalize(embeddings)

# Set up FAISS index for fast similarity search using inner product (cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Add all chunk embeddings to the FAISS index
index.add(embeddings)

# Define a query string
query = "What is AI?"

# Generate embedding for the query
query_embedding = model.encode([query])

# Normalize query embedding
query_embedding = normalize(query_embedding)

# Search the FAISS index for the top 2 most similar chunks to the query
D, I = index.search(query_embedding, k=2)

# Print the query and the most relevant chunks with their similarity scores
print("Query:", query)
print("Results:")
for idx, score in zip(I[0], D[0]):
    print(f"- {chunks[idx]}  (score: {score:.4f})")
