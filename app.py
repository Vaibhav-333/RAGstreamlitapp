import os
import nltk
import torch
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document
from nltk.tokenize import sent_tokenize
from langdetect import detect

# Download punkt if not already available
nltk.download('punkt')

st.set_page_config(page_title="Chat with Your Documents", layout="wide")

st.title("📄 Chat with Your Documents")

# Sidebar settings
with st.sidebar:
    st.header("📚 Upload documents")
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    show_debug = st.checkbox("Show debug info")

# Initialize models
@st.cache_resource
def load_models():
    return {
        "qa": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
        "summarization": pipeline("summarization", model="facebook/bart-large-cnn"),
        "embedding": SentenceTransformer("all-MiniLM-L6-v2")
    }

models = load_models()

# Extract text from uploaded documents
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    return ""

# Chunk text with FAISS for vector-based retrieval
def vectorize_text(text, chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += " " + sent
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    embeddings = models["embedding"].encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    return chunks, index

# Detect language for multilingual support
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

if uploaded_files:
    full_text = "\n".join(extract_text(f) for f in uploaded_files)
    chunks, index = vectorize_text(full_text)

    # Generate document summary
    summary = models["summarization"](full_text[:2000], max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    st.markdown(f"### 📌 Document Summary:\n**{summary}**")

    st.success("✅ Documents processed. Ask your questions!")

    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        with st.spinner("Thinking..."):
            query_embedding = models["embedding"].encode([user_query])
            _, closest_chunk_idx = index.search(np.array(query_embedding, dtype=np.float32), 1)
            best_chunk = chunks[closest_chunk_idx[0][0]]

            try:
                result = models["qa"](question=user_query, context=best_chunk)
                st.markdown(f"### ✅ Answer:\n**{result['answer']}**")
            except Exception as e:
                st.error(f"Error during QA: {e}")

        # Language detection feedback
        detected_lang = detect_language(user_query)
        st.markdown(f"🌍 Detected Language: `{detected_lang}`")

        # Debug Info
        if show_debug:
            st.markdown("#### 🔍 Debug Info")
            st.write(result)
            st.write(f"Retrieved Chunk: {best_chunk}")
