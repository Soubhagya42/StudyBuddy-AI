import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---- PAGE CONFIG ----
st.set_page_config(page_title="StudyBuddy AI", page_icon="🧠")
st.title("🧠 StudyBuddy AI")
st.subheader("Upload a PDF and chat with it")

# ---- PDF UPLOAD ----
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Read the PDF
    pdf_reader = PdfReader(uploaded_file)
    
    # Extract text from every page
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    
    # Show success message
    st.success(f"✅ PDF loaded! {len(pdf_reader.pages)} pages found.")
    
    # Show a preview of extracted text
    st.subheader("📄 Text Preview (first 500 characters):")
    st.write(raw_text[:500])

    # ---- SPLIT TEXT INTO CHUNKS ----
if uploaded_file is not None:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # each chunk = 1000 characters
        chunk_overlap=200,    # 200 characters overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    
    st.subheader("📦 Text Chunks:")
    st.write(f"Your PDF was split into {len(chunks)} chunks")
    st.write("First chunk preview:")
    st.write(chunks[0])

    # ---- CREATE EMBEDDINGS ----
if uploaded_file is not None:
    # Load free embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert chunks to embeddings (numbers)
    embeddings = model.encode(chunks)
    
    st.subheader("🔢 Embeddings:")
    st.write(f"Each chunk converted to {len(embeddings[0])} numbers")
    st.write("First chunk embedding preview (first 10 numbers):")
    st.write(embeddings[0][:10])

    # ---- STORE IN FAISS ----
if uploaded_file is not None:
    # Convert embeddings to numpy array (format FAISS needs)
    embeddings_array = np.array(embeddings)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings_array)
    
    st.subheader("🗄️ Vector Database:")
    st.write(f"✅ Stored {index.ntotal} chunks in FAISS database")
    st.write("Your PDF is now searchable by meaning, not just keywords!")