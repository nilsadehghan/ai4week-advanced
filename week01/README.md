# 🧠 Poetic LLM Agent with RAG + FastAPI API

This is the Week 1 project of a 4-week AI/Data career roadmap. It combines a **custom LLM-based poetic chatbot** and a **Retrieval-Augmented Generation (RAG)** pipeline using FAISS and SentenceTransformers and deployed via **FastAPI**.

---

## 📌 Project Features

- 🗣️ **Poetic Chatbot**: A Chainlit chatbot powered by a fine-tuned Persian language model that responds poetically.
- 📄 **Document Q&A (RAG)**: Ask questions based on a PDF document using vector search + LLM.
- 🚀 **FastAPI Server**: Exposes the RAG system as a live API endpoint (accessible via ngrok).


---

## 🧰 Tech Stack

| Task                    | Tools / Libraries                                       |
|-------------------------|----------------------------------------------------------|
| Language Model (LLM)    | Huggingface Transformers (`rahiminia/manshoorai`)       |
| Chat UI                 | Chainlit                                                 |
| Vector Search (RAG)     | FAISS + SentenceTransformers                             |
| PDF Processing          | PyMuPDF (`fitz`)                                         |
| API Deployment          | FastAPI + Uvicorn + ngrok                                |
| Embedding Model         | `all-MiniLM-L6-v2`                                       |
| Python Environment      | Python 3.9+, `pip`                                       |

---


